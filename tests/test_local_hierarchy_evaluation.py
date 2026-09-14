"""Small scientific correctness checks, independent of data and checkpoints."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from tools.experiments.hierarchy_triplets import (
    agreement, closest_pairs, permuted_references, reference_depths,
)
from tools.experiments.export_embeddings import select_samples
from tools.experiments._common import digest, load_caches, validate_embeddings, write_json, write_npz


class TripletTests(unittest.TestCase):
    def setUp(self):
        self.triplets = np.array([[0, 1, 2]])
        self.paths = {10: ["root", "pair", "a"], 30: ["root", "pair", "b"], 90: ["root", "c"]}
        self.depths = reference_depths(self.paths, [10, 30, 90])
        self.reference = closest_pairs(self.depths, self.triplets, reference=True)

    def test_identical_and_conflicting_trees(self):
        matching = np.array([[0, 1, 3], [1, 0, 3], [3, 3, 0]])
        conflicting = np.array([[0, 3, 1], [3, 0, 3], [1, 3, 0]])
        self.assertEqual(agreement(closest_pairs(matching, self.triplets), self.reference)["agreement"], 1)
        self.assertEqual(agreement(closest_pairs(conflicting, self.triplets), self.reference)["agreement"], 0)

    def test_unresolved_reference_is_excluded(self):
        star = closest_pairs(np.ones((3, 3)), self.triplets, reference=True)
        self.assertEqual(star.tolist(), [-1])
        self.assertEqual(agreement(np.array([0, 0]), np.array([0, -1]))["eligible_triplets"], 1)
        with self.assertRaises(ValueError):
            agreement(np.array([0]), star)

    def test_prediction_tie_is_incorrect(self):
        tied = closest_pairs(np.ones((3, 3)), self.triplets)
        stats = agreement(tied, self.reference)
        self.assertEqual(stats["agreement"], 0)
        self.assertEqual(stats["predicted_ties_on_eligible"], 1)
        near_tie = np.array([[0, 1, 1 + 1e-7], [1, 0, 2], [1 + 1e-7, 2, 0]])
        self.assertEqual(closest_pairs(near_tie, self.triplets).tolist(), [-1])

    def test_label_order_and_permutations(self):
        reordered = reference_depths(self.paths, [90, 30, 10])
        self.assertEqual(closest_pairs(reordered, self.triplets, reference=True).tolist(), [2])
        first = list(permuted_references(self.depths, self.triplets, count=30, seed=42))
        second = list(permuted_references(self.depths, self.triplets, count=30, seed=42))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(set(np.concatenate(first)), {0, 1, 2})


class CacheTests(unittest.TestCase):
    def make_cache(self, directory, name, ids, curvature=1):
        path = directory / f"{name}.npz"
        write_npz(path, embeddings=np.zeros((60, 32), dtype=np.float32), labels=np.arange(60),
                  sample_ids=ids, sample_names=np.array([str(i) for i in range(60)]))
        write_json(path.with_suffix(".json"), {"npz_sha256": digest(path), "signature": name,
                                              "curvature": curvature})
        return {"name": name, "title": name, "signature": name}

    def test_cache_hash_and_cross_model_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            models = [self.make_cache(directory, name, np.arange(60)) for name in ("A", "B")]
            write_json(directory / "manifest.json", {"models": models, "samples_per_class": 1})
            load_caches(directory)
            self.make_cache(directory, "B", np.arange(60)[::-1])
            with self.assertRaisesRegex(ValueError, "different sample_ids"):
                load_caches(directory)
            self.make_cache(directory, "B", np.arange(60))
            with (directory / "B.npz").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "Stale or modified"):
                load_caches(directory)

    def test_cache_curvature_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            models = [self.make_cache(directory, "A", np.arange(60)),
                      self.make_cache(directory, "B", np.arange(60), curvature=2)]
            write_json(directory / "manifest.json", {"models": models, "samples_per_class": 1})
            with self.assertRaisesRegex(ValueError, "different curvatures"):
                load_caches(directory)

    def test_balanced_reproducible_sampling(self):
        labels = np.repeat(np.arange(60), 25)
        ids = select_samples(labels, 20, 42)
        np.testing.assert_array_equal(ids, select_samples(labels, 20, 42))
        self.assertEqual(len(np.unique(ids)), 1200)
        np.testing.assert_array_equal(np.bincount(labels[ids]), np.full(60, 20))

    def test_reject_invalid_embeddings(self):
        embeddings = np.zeros((60, 32), dtype=np.float32)
        validate_embeddings(embeddings, np.arange(60), np.arange(60), 1.0, 1)
        embeddings[0, 0] = 1
        with self.assertRaises(ValueError):
            validate_embeddings(embeddings, np.arange(60), np.arange(60), 1.0, 1)
        embeddings[0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_embeddings(embeddings, np.arange(60), np.arange(60), 1.0, 1)


if __name__ == "__main__":
    unittest.main()
