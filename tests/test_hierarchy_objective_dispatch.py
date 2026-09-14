import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from processor.pretrain_skeletonclr import SkeletonCLR_Processor
from processor.pretrain_skeletonclr_3views import SkeletonCLR_3views_Processor
from tools.hyperbolic_geometry import make_hyperbolic_geometry
from tools.hyperbolic_hierarchy import HIERARCHY_OBJECTIVES, hierarchy_triplet_loss_hyp


class HierarchyObjectiveDispatchTest(unittest.TestCase):
    def test_yaml_selection_and_cli_override(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            config.write_text("hier_objective: similarity_weighted\n", encoding="utf-8")
            processor = SkeletonCLR_Processor.__new__(SkeletonCLR_Processor)
            processor.load_arg(["--config", str(config)])
            self.assertEqual(processor.arg.hier_objective, "similarity_weighted")
            processor.load_arg(["--config", str(config), "--hier_objective", "ranking_ce"])
            self.assertEqual(processor.arg.hier_objective, "ranking_ce")

    def test_parser_default_and_choices(self):
        for processor_type in (SkeletonCLR_Processor, SkeletonCLR_3views_Processor):
            parser = processor_type.get_parser()
            self.assertEqual(parser.parse_args([]).hier_objective, "ranking_ce")
            for objective in HIERARCHY_OBJECTIVES:
                args = parser.parse_args(["--hier_objective", objective])
                self.assertEqual(vars(args)["hier_objective"], objective)
            with patch("sys.stderr"), self.assertRaises(SystemExit):
                parser.parse_args(["--hier_objective", "unknown"])

    def test_both_processors_dispatch_the_selected_objective(self):
        for processor_type in (SkeletonCLR_Processor, SkeletonCLR_3views_Processor):
            for objective in HIERARCHY_OBJECTIVES:
                with self.subTest(processor=processor_type.__name__, objective=objective):
                    processor = processor_type.__new__(processor_type)
                    processor.arg = processor_type.get_parser().parse_args([
                        "--hier_objective", objective, "--cluster_warmup_steps", "0",
                        "--hier_warmup_steps", "0", "--hier_update_interval", "1",
                        "--hier_triplets", "6",
                    ])
                    processor.dev = torch.device("cpu")
                    processor.model = SimpleNamespace(geometry=make_hyperbolic_geometry())
                    processor.global_step = 1
                    processor.cluster_affinity = None
                    processor._cluster_metrics = Mock(return_value={})
                    raw = torch.tensor([[0.2, 0.1], [0.4, 0.3], [-0.3, 0.2]], requires_grad=True)
                    points = make_hyperbolic_geometry().expmap0(raw)
                    posteriors = torch.softmax(torch.tensor([[1., 2., 3.], [3., 1., 2.]]), dim=1)
                    with patch("processor.pretrain_skeletonclr.hierarchy_triplet_loss_hyp",
                               wraps=hierarchy_triplet_loss_hyp) as loss_call:
                        _, loss, _ = processor._compute_cluster_losses(
                            {"p_q": posteriors, "q_k": posteriors, "proto_h": points})
                    self.assertEqual(loss_call.call_args.kwargs["objective"], objective)
                    self.assertIs(loss_call.call_args.kwargs["affinity"], processor.cluster_affinity)
                    self.assertFalse(processor.cluster_affinity.requires_grad)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    self.assertTrue(torch.isfinite(raw.grad).all())


if __name__ == "__main__":
    unittest.main()
