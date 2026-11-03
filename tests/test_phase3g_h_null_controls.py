"""CPU tests for the structure-preserving h-probe permutation null; no GPU, VM or store."""

import unittest
from collections import Counter

from scripts import phase3g_h_null_controls as null


def examples():
    rows = []
    for base, true_relation in ((1, "before"), (2, "after"), (3, "before")):
        for condition in (null.LOW, null.TEMPORAL):
            for view, relations in (("archived", ("before", "after")), ("option_order_permuted", ("after", "before"))):
                for relation in relations:
                    rows.append({"base_sample_id": base, "condition": condition, "view": view,
                                 "relation_word": relation, "y_statement_true": int(relation == true_relation)})
    return rows


class NullTests(unittest.TestCase):
    def test_flip_keeps_pairing_and_inverts_whole_bases_only(self):
        rows = examples()
        flipped = null.flipped(rows, {2})
        for before, after in zip(rows, flipped):
            expected = 1-before["y_statement_true"] if before["base_sample_id"] == 2 else before["y_statement_true"]
            self.assertEqual(after["y_statement_true"], expected)
        for base in (1, 2, 3):
            for condition in (null.LOW, null.TEMPORAL):
                for view in ("archived", "option_order_permuted"):
                    labels = [r["y_statement_true"] for r in flipped
                              if (r["base_sample_id"], r["condition"], r["view"]) == (base, condition, view)]
                    self.assertEqual(sorted(labels), [0, 1])
        self.assertEqual(rows[0]["y_statement_true"], 1)

    def test_flip_sets_are_deterministic_and_vary(self):
        bases = list(range(50))
        first, second = null.flip_sets(bases, 20), null.flip_sets(bases, 20)
        self.assertEqual(first, second)
        self.assertGreater(len({tuple(f) for f in first}), 15)
        self.assertTrue(all(set(f) <= set(bases) for f in first))

    def test_within_fold_flips_keep_each_fold_composition(self):
        rows = examples()+[{**r, "base_sample_id": r["base_sample_id"]+10} for r in examples()]
        folds = [{"outer_fold": 0, "test_scene_seeds": [1, 2, 11]}, {"outer_fold": 1, "test_scene_seeds": [3, 12, 13]}]

        def before_true(labelled, fold):
            return sum(r["relation_word"] == "before" and r["y_statement_true"] and r["base_sample_id"] in fold["test_scene_seeds"]
                       for r in labelled)

        sets = null.within_fold_flip_sets(rows, folds, 30)
        self.assertGreater(len({tuple(f) for f in sets}), 3)
        for flips in sets:
            permuted = null.flipped(rows, set(flips))
            for fold in folds:
                self.assertEqual(before_true(permuted, fold), before_true(rows, fold))

    def test_mechanism_counts_fold_composition_and_called_relation(self):
        rows = examples()
        folds = [{"outer_fold": 0, "test_scene_seeds": [1, 2]}, {"outer_fold": 1, "test_scene_seeds": [3]}]
        # A relation rule "before is true" scores 1 on before-true bases and 0 on after-true bases.
        predictions = [{**r, "probe_prediction": int(r["relation_word"] == "before")} for r in rows]
        result = null.mechanism(rows, folds, predictions)
        self.assertEqual([r["test_before_true"] for r in result], [1, 1])
        self.assertEqual([r["train_before_true"] for r in result], [1, 1])
        self.assertEqual(result[0]["share_called_true"], {"before": 1.0, "after": 0.0})
        self.assertEqual([r["accuracy"] for r in result], [0.5, 1.0])

    def test_null_summary_tail_probabilities(self):
        summary = null.null_summary(0.9, [0.1, 0.2, 0.3, 0.9])
        self.assertAlmostEqual(summary["p_upper"], 2/5)
        self.assertAlmostEqual(summary["p_lower"], 1.0)
        self.assertEqual(summary["draws"], 4)
        self.assertEqual(Counter(null.accuracies([
            {"condition": null.LOW, "probe_prediction": 1, "y_statement_true": 1},
            {"condition": null.TEMPORAL, "probe_prediction": 0, "y_statement_true": 1}]).values()), Counter({1.0: 1, 0.0: 1}))


if __name__ == "__main__":
    unittest.main()
