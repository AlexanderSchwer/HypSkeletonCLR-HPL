"""Scientific checks for original-space assignments and induced reference trees."""
from itertools import combinations
import json
import unittest
import numpy as np
import torch
from tools.experiments._common import ROOT
from tools.experiments._prototype_plots import (
    assignment_table, leaf_paths, nearest_assignments, prune_tree, subset_indices,
)
from tools.experiments.hierarchy_triplets import closest_pairs, reference_depths
from tools.action_label_hierarchy import get_hierarchy, hierarchy_leaf_paths
from tools.hyperbolic_geometry import make_hyperbolic_geometry


class FocusedPrototypeTests(unittest.TestCase):
    def test_hyperbolic_prediction_and_checkpoint_index_order(self):
        geometry = make_hyperbolic_geometry(curvature=1)
        tangent = torch.tensor([[0.8, 0.0], [-0.4, 0.0], [0.0, 0.8]])
        proto = geometry.tangent_proto_to_manifold(tangent)
        samples = proto[[2, 0, 1]] * 0.95
        distances = geometry.dist(samples[:, None], proto[None])
        nearest, counts = nearest_assignments(distances.numpy())
        np.testing.assert_array_equal(nearest, [2, 0, 1])
        np.testing.assert_array_equal(nearest, torch.softmax(-distances / 0.1, dim=1).argmax(1))
        np.testing.assert_array_equal(counts, [1, 1, 1])

    def test_ties_are_not_arbitrarily_assigned_or_dropped_from_heatmaps(self):
        distances = np.array([[1.0, 1.0 + 1e-7, 3], [3, 2, 1], [0.1, 3, 4]])
        assignment, ties = nearest_assignments(distances)
        np.testing.assert_array_equal(assignment, [-1, 2, 0])
        np.testing.assert_array_equal(ties, [2, 1, 1])
        counts, fractions = assignment_table(np.array([7, 7, 9]), assignment, [9, 7], 3)
        np.testing.assert_array_equal(counts, [[1, 0, 0, 0], [0, 0, 1, 1]])
        np.testing.assert_allclose(fractions.sum(axis=1), 1)
        self.assertEqual(fractions[1, -1], 0.5)
        with self.assertRaises(ValueError):
            nearest_assignments(np.array([[np.nan, 1]]))

    def test_pruned_topology_preserves_every_reference_triplet_and_polytomy(self):
        config = json.loads((ROOT / "config/experiments/prototype_subsets.json").read_text())
        tree = get_hierarchy(config["reference"])["tree"]
        before = json.dumps(tree)
        original_paths = hierarchy_leaf_paths(config["reference"])
        for subset in config["subsets"]:
            ids = [i - 1 for i in subset["classes"]]
            pruned = prune_tree(tree, set(ids))
            paths = leaf_paths(pruned)
            self.assertEqual(set(paths), set(ids))
            triples = np.array(list(combinations(range(len(ids)), 3)))
            np.testing.assert_array_equal(
                closest_pairs(reference_depths(paths, ids), triples, reference=True),
                closest_pairs(reference_depths(original_paths, ids), triples, reference=True))
        self.assertEqual(json.dumps(tree), before)
        posture = prune_tree(tree, {7, 8, 13, 14, 26, 42})
        self.assertEqual(len(posture["children"]), 3)

    def test_subset_keeps_within_class_cache_order_for_fixed_links(self):
        labels = np.array([14, 13, 14, 13, 14, 13, 0])
        indices, _, order = subset_indices(labels, {"classes": [15, 14]}, "hypskeletonclr_ward")
        self.assertEqual(order, [13, 14])
        np.testing.assert_array_equal(indices, [1, 3, 5, 0, 2, 4])


if __name__ == "__main__":
    unittest.main()
