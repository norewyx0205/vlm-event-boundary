import unittest

from scripts import select_attention_cases as selector


def paired_rows(base_id, condition, original_correct, swapped_correct):
    rows = []
    for variant, correct in (
        ("original", original_correct),
        ("swapped", swapped_correct),
    ):
        eval_id = f"sample_{base_id}_{condition}_{variant}"
        rows.append({
            "eval_id": eval_id,
            "pairing_id": f"sample_{base_id}_{condition}",
            "base_sample_id": base_id,
            "condition": condition,
            "prompt_variant": variant,
            "prediction": "A" if variant == "original" else "B",
            "is_correct": correct,
        })
    return rows


class AttentionCaseSelectionTest(unittest.TestCase):
    def test_temporal_visual_contrast_is_selected_as_atomic_bundle(self):
        temporal = paired_rows(1, "temporal_boundary", True, True)
        visual = paired_rows(1, "visual_boundary", False, False)
        main_rows = temporal + visual
        annotations = [dict(row) for row in main_rows]

        selected = selector.select_cases(
            annotations,
            main_rows,
            perturbation_rows=[],
            max_video_pairs=2,
        )

        self.assertEqual(len(selected), 4)
        self.assertEqual(
            {row["condition"] for row in selected},
            {"temporal_boundary", "visual_boundary"},
        )
        self.assertEqual(
            {row["attention_case_label"] for row in selected},
            {"temporal_stable_visual_both_wrong"},
        )
        self.assertEqual(
            {row["attention_case_bundle_size"] for row in selected},
            {2},
        )
        self.assertEqual(
            len({row["attention_case_bundle_id"] for row in selected}),
            1,
        )

    def test_atomic_bundle_is_not_partially_selected(self):
        temporal = paired_rows(1, "temporal_boundary", True, True)
        visual = paired_rows(1, "visual_boundary", False, False)
        main_rows = temporal + visual

        selected = selector.select_cases(
            [dict(row) for row in main_rows],
            main_rows,
            perturbation_rows=[],
            max_video_pairs=1,
        )

        self.assertEqual(selected, [])


if __name__ == "__main__":
    unittest.main()
