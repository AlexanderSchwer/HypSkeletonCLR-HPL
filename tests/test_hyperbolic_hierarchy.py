import unittest
import itertools
import math

import geoopt
import torch

from tools.hyperbolic_hierarchy import (
    HIERARCHY_OBJECTIVES,
    _lca_depth_hyp,
    hierarchy_triplet_loss_hyp,
    prototype_affinity_hyp,
    sample_triplets_from_affinity,
    update_affinity_ema,
)
from tools.hyperbolic_geometry import make_hyperbolic_geometry
from tools.sinkhorn import sinkhorn_balanced_probabilities


class HyperbolicHierarchyTest(unittest.TestCase):
    def setUp(self):
        self.manifold = geoopt.PoincareBall(1.0)

    def test_lca_depth_matches_inner_collinear_point(self):
        inner = self.manifold.expmap0(torch.tensor([[0.2, 0.0]]))
        outer = self.manifold.expmap0(torch.tensor([[1.0, 0.0]]))

        depth = _lca_depth_hyp(inner, outer, self.manifold)

        torch.testing.assert_close(depth, self.manifold.dist0(inner), atol=1e-6, rtol=1e-6)

    def test_prototype_affinity_prefers_nearby_prototypes(self):
        proto_h = self.manifold.expmap0(
            torch.tensor([[0.2, 0.0], [0.22, 0.0], [-0.8, 0.0]])
        )

        affinity = prototype_affinity_hyp(proto_h, curvature=1.0)

        self.assertGreater(affinity[0, 1].item(), affinity[0, 2].item())
        self.assertEqual(affinity.diag().count_nonzero().item(), 0)
        torch.testing.assert_close(affinity.sum(), torch.tensor(1.0))

    def test_affinity_ema_and_triplet_sampling(self):
        affinity = torch.tensor(
            [[0.0, 0.8, 0.1], [0.8, 0.0, 0.1], [0.1, 0.1, 0.0]]
        )
        updated = update_affinity_ema(None, affinity)
        triplets = sample_triplets_from_affinity(updated, 32)

        self.assertEqual(tuple(triplets.shape), (32, 3))
        self.assertTrue(torch.all(triplets[:, 0] != triplets[:, 1]))
        self.assertTrue(torch.all(triplets[:, 0] != triplets[:, 2]))
        self.assertTrue(torch.all(triplets[:, 1] != triplets[:, 2]))

    def test_hierarchy_loss_has_finite_gradients(self):
        raw = torch.randn(4, 3, requires_grad=True)
        proto_h = self.manifold.expmap0(raw / (1.0 + raw.norm(dim=1, keepdim=True)))
        triplets = torch.tensor([[0, 1, 2], [1, 2, 3]])

        loss = hierarchy_triplet_loss_hyp(proto_h, triplets, curvature=1.0)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(raw.grad).all())

    def test_lorentz_backend_maps_tangent_to_ambient_hyperboloid(self):
        geometry = make_hyperbolic_geometry("lorentz", curvature=1.0)
        tangent = torch.randn(5, 3) * 0.1

        points = geometry.expmap0(tangent)
        pairwise = geometry.dist(points.unsqueeze(1), points.unsqueeze(0))

        self.assertEqual(tuple(points.shape), (5, 4))
        self.assertTrue(torch.isfinite(points).all())
        self.assertTrue(torch.isfinite(pairwise).all())
        torch.testing.assert_close(pairwise.diag(), torch.zeros(5), atol=1e-3, rtol=1e-5)

    def test_lorentz_hierarchy_loss_has_finite_gradients(self):
        geometry = make_hyperbolic_geometry("lorentz", curvature=1.0)
        raw = torch.randn(4, 3, requires_grad=True)
        proto_h = geometry.expmap0(raw / (1.0 + raw.norm(dim=1, keepdim=True)))
        triplets = torch.tensor([[0, 1, 2], [1, 2, 3]])

        loss = hierarchy_triplet_loss_hyp(
            proto_h,
            triplets,
            curvature=1.0,
            geometry_model="lorentz",
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(raw.grad).all())

    def test_ranking_ce_preserves_legacy_values_and_gradients(self):
        for model in ("poincare", "lorentz"):
            with self.subTest(model=model):
                geometry = make_hyperbolic_geometry(model, 1.0)
                raw = torch.tensor([[0.2, 0.1], [0.4, 0.2], [-0.3, 0.2]],
                                   dtype=torch.float64, requires_grad=True)
                points = geometry.expmap0(raw)
                triplets = torch.tensor([[0, 1, 2], [2, 0, 1]])
                # The pre-existing implementation, before objective dispatch.
                anc, pos, neg = (points[triplets[:, column]] for column in range(3))
                s_ap = _lca_depth_hyp(anc, pos, geometry)
                s_an = _lca_depth_hyp(anc, neg, geometry) + 0.05
                s_pn = _lca_depth_hyp(pos, neg, geometry) + 0.05
                expected = torch.nn.functional.cross_entropy(
                    torch.stack([s_ap, s_an, s_pn], dim=1),
                    torch.zeros(2, dtype=torch.long),
                )
                expected_grad = torch.autograd.grad(expected, raw, retain_graph=True)[0]
                for kwargs in ({}, {"objective": "ranking_ce"}):
                    actual = hierarchy_triplet_loss_hyp(points, triplets, 1.0,
                                                       geometry_model=model, **kwargs)
                    actual_grad = torch.autograd.grad(actual, raw, retain_graph=True)[0]
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    @staticmethod
    def _weighted_affinity():
        return torch.tensor([[0., 0.4, 0.05], [0.4, 0., 0.05], [0.05, 0.05, 0.]],
                            dtype=torch.float64)

    def test_weighted_loss_matches_collinear_calculation_and_ignores_margin(self):
        points = self.manifold.expmap0(torch.tensor([[0.2, 0.], [0.6, 0.], [-0.3, 0.]],
                                                    dtype=torch.float64))
        # Depths are (0.4, 0, 0), and scaled weights are (2.4, 0.3, 0.3).
        expected = 3.0 - (2.4 * math.exp(0.4) + 0.6) / (math.exp(0.4) + 2.0)
        for margin in (0., 0.05, 10.):
            loss = hierarchy_triplet_loss_hyp(
                points, torch.tensor([[0, 1, 2]]), 1.0, margin=margin,
                objective="similarity_weighted", affinity=self._weighted_affinity(),
            )
            self.assertAlmostEqual(loss.item(), expected, places=8)

    def test_weighted_loss_prefers_highest_affinity_pair_and_is_permutation_invariant(self):
        points = self.manifold.expmap0(torch.tensor([[0.2, 0.1], [0.4, 0.2], [-0.3, 0.2]],
                                                    dtype=torch.float64))
        def loss_for(order, triplet):
            return hierarchy_triplet_loss_hyp(
                points[order], torch.tensor([triplet]), 1.0,
                objective="similarity_weighted", affinity=self._weighted_affinity(),
            )
        preferred = loss_for([0, 1, 2], [0, 1, 2])
        self.assertLess(preferred.item(), loss_for([0, 2, 1], [0, 1, 2]).item())
        for triplet in itertools.permutations(range(3)):
            torch.testing.assert_close(loss_for([0, 1, 2], triplet), preferred)

    def test_weighted_loss_has_finite_gradients_and_detaches_affinity(self):
        for model in ("poincare", "lorentz"):
            with self.subTest(model=model):
                geometry = make_hyperbolic_geometry(model, 1.0)
                raw = torch.tensor([[0.2, 0.1], [0.4, 0.3], [-0.3, 0.2]],
                                   dtype=torch.float64, requires_grad=True)
                affinity = self._weighted_affinity().requires_grad_()
                loss = hierarchy_triplet_loss_hyp(
                    geometry.expmap0(raw), torch.tensor([[0, 1, 2]]), 1.0,
                    geometry_model=model, objective="similarity_weighted", affinity=affinity,
                )
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(raw.grad).all())
                self.assertGreater(raw.grad.abs().sum().item(), 0)
                self.assertIsNone(affinity.grad)

    def test_equal_or_zero_affinities_give_no_geometric_preference(self):
        for value in (0., 1. / 6.):
            raw = torch.tensor([[0.2, 0.1], [0.4, 0.3], [-0.3, 0.2]],
                               dtype=torch.float64, requires_grad=True)
            affinity = torch.full((3, 3), value, dtype=torch.float64)
            affinity.fill_diagonal_(0.)
            loss = hierarchy_triplet_loss_hyp(
                self.manifold.expmap0(raw), torch.tensor([[0, 1, 2]]), 1.0,
                objective="similarity_weighted", affinity=affinity,
            )
            loss.backward()
            self.assertAlmostEqual(loss.item(), 12. * value, places=12)
            torch.testing.assert_close(raw.grad, torch.zeros_like(raw), atol=1e-12, rtol=0)

    def test_empty_triplets_and_invalid_weighted_inputs(self):
        points = self.manifold.expmap0(torch.randn(3, 2) * 0.1)
        for objective in HIERARCHY_OBJECTIVES:
            empty = hierarchy_triplet_loss_hyp(points, torch.empty((0, 3), dtype=torch.long),
                                              1.0, objective=objective)
            self.assertEqual(empty.shape, torch.Size([]))
            self.assertEqual(empty.item(), 0.)
        with self.assertRaisesRegex(ValueError, "Unknown hierarchy objective"):
            hierarchy_triplet_loss_hyp(points, torch.tensor([[0, 1, 2]]), 1.0, objective="unknown")
        for affinity in (None, torch.ones(2, 2), torch.ones(3, 3, dtype=torch.long),
                         -torch.ones(3, 3), torch.full((3, 3), float("nan")),
                         torch.full((3, 3), float("inf"))):
            with self.subTest(affinity=affinity), self.assertRaises(ValueError):
                hierarchy_triplet_loss_hyp(points, torch.tensor([[0, 1, 2]]), 1.0,
                                          objective="similarity_weighted", affinity=affinity)

    def test_sinkhorn_approximately_balances_columns(self):
        torch.manual_seed(0)
        probabilities = torch.softmax(torch.rand(64, 16), dim=1)

        assignments = sinkhorn_balanced_probabilities(
            probabilities,
            n_iters=50,
            exponent=10.0,
        )

        torch.testing.assert_close(assignments.sum(dim=1), torch.ones(64), atol=1e-5, rtol=1e-5)
        expected_column_mass = torch.full((16,), 4.0)
        torch.testing.assert_close(
            assignments.sum(dim=0), expected_column_mass, atol=2e-2, rtol=2e-2
        )

    def test_sinkhorn_returns_probabilities_after_optimal_transport(self):
        torch.manual_seed(1)
        p = torch.softmax(torch.rand(32, 8), dim=1)

        q = sinkhorn_balanced_probabilities(
            p,
            n_iters=30,
            exponent=5.0,
        )

        self.assertEqual(tuple(q.shape), tuple(p.shape))
        torch.testing.assert_close(q.sum(dim=1), torch.ones(32), atol=1e-6, rtol=1e-6)
        self.assertTrue((q >= 0).all())


if __name__ == "__main__":
    unittest.main()
