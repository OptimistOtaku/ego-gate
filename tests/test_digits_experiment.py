import unittest

import numpy as np
import torch
from torch.nn import functional as F

from experiments.run_digits import (
    Config,
    MLP,
    aggregate,
    kcenter_indices,
    score_examples,
    select_indices,
    stratified_random_indices,
)


class SelectionTests(unittest.TestCase):
    def test_stratified_random_is_balanced(self):
        labels = np.repeat(np.arange(5), 10)
        selected = stratified_random_indices(labels, 20, np.random.default_rng(7))
        counts = np.bincount(labels[selected], minlength=5)
        np.testing.assert_array_equal(counts, np.full(5, 4))

    def test_kcenter_returns_unique_pool_members(self):
        features = np.random.default_rng(1).normal(size=(30, 4))
        candidates = np.arange(5, 25)
        selected = kcenter_indices(features, 8, candidates)
        self.assertEqual(len(selected), len(set(selected)))
        self.assertTrue(set(selected).issubset(set(candidates)))

    def test_all_bounded_conditions_obey_budget(self):
        labels = np.repeat(np.arange(5), 20)
        rng = np.random.default_rng(2)
        scores = {
            "doubt": rng.normal(size=100),
            "curiosity": rng.normal(size=100),
            "egogate": rng.normal(size=100),
            "embedding": rng.normal(size=(100, 8)),
        }
        cfg = Config(buffer_size=17)
        for condition in (
            "random",
            "stratified_random",
            "doubt",
            "curiosity",
            "egogate",
            "embedding_kcenter",
            "egogate_diverse",
        ):
            selected = select_indices(condition, labels, scores, cfg, seed=0)
            self.assertEqual(len(selected), 17, condition)
            self.assertEqual(len(set(selected)), 17, condition)


class GradientNormTests(unittest.TestCase):
    def test_analytic_gradient_norm_matches_autograd(self):
        torch.manual_seed(4)
        model = MLP(hidden_size=7)
        x = torch.rand(3, 64)
        y = torch.tensor([0, 2, 4])
        analytic = score_examples(model, x, y, "cpu")["curiosity"]

        expected = []
        for sample, label in zip(x, y):
            model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(sample.unsqueeze(0)), label.unsqueeze(0))
            loss.backward()
            expected.append(sum(p.grad.square().sum().item() for p in model.parameters()))
        np.testing.assert_allclose(analytic, np.asarray(expected), rtol=1e-5, atol=1e-6)


class StatisticsTests(unittest.TestCase):
    def test_preplanned_comparisons_and_holm_values_are_emitted(self):
        rows = []
        for seed in range(3):
            for condition, forgetting in (
                ("random", 0.4 + seed * 0.01),
                ("egogate", 0.3 + seed * 0.02),
                ("doubt", 0.2 + seed * 0.005),
            ):
                rows.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "task_a_pre": 0.9,
                        "task_a_post": 0.9 - forgetting,
                        "forgetting": forgetting,
                        "task_b_post": 0.8,
                    }
                )
        _, comparisons = aggregate(rows, ("random", "egogate", "doubt"))
        pairs = {(row["baseline"], row["condition"]) for row in comparisons}
        self.assertIn(("random", "egogate"), pairs)
        self.assertIn(("egogate", "doubt"), pairs)
        self.assertTrue(all(0.0 <= row["holm_p_value"] <= 1.0 for row in comparisons))


if __name__ == "__main__":
    unittest.main()
