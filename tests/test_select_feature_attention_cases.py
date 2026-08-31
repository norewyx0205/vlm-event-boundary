import unittest

from scripts import select_feature_attention_cases as selector


FEATURES = selector.DEFAULT_FEATURES
CONDITIONS = selector.DEFAULT_CONDITIONS


def annotation(feature, base_id, condition, variant, target_1_first):
    starts = (10, 20) if target_1_first else (20, 10)
    video_id = f"l5_{feature}_sample_{base_id:03d}_{condition}.mp4"
    return {
        "eval_id": f"l5_{feature}_sample_{base_id:03d}_{condition}_{variant}",
        "pairing_id": f"l5_feature_sample_{base_id:03d}_{condition}",
        "feature_variant": feature,
        "base_sample_id": base_id,
        "condition": condition,
        "prompt_variant": variant,
        "video_id": video_id,
        "video_path": f"data/l5_feature_ablation_v1/{feature}/videos/{video_id}",
        "fps": 15,
        "duration_sec": 18,
        "total_frames": 270,
        "first_object_id": 1 if target_1_first else 2,
        "second_object_id": 2 if target_1_first else 1,
        "event_timing": {"first_event_start_frame": 10},
        "boundary_timing": {"gap_frames": 0, "visual_marker": "none"},
        "target_objects": [
            {
                "id": 1,
                "direction": "left",
                "from": [1, 2],
                "to": [3, 2],
                "start_frame": starts[0],
                "end_frame": starts[0] + 10,
            },
            {
                "id": 2,
                "direction": "up",
                "from": [4, 5],
                "to": [4, 3],
                "start_frame": starts[1],
                "end_frame": starts[1] + 10,
            },
        ],
        "distractors": [{
            "id": 1,
            "motion_kind": "static",
            "motion_timing": "none",
            "direction": "none",
            "from": [8, 8],
            "to": [8, 8],
        }],
    }


def make_rows(base_ids=(1, 2, 3, 4)):
    annotations = {feature: [] for feature in FEATURES}
    results = []
    outcome_pattern = {
        1: (True, True),
        2: (True, False),
        3: (False, False),
        4: (True, True),
    }
    for feature in FEATURES:
        for base_id in base_ids:
            for condition in CONDITIONS:
                correct_pair = outcome_pattern[base_id]
                for index, variant in enumerate(selector.PAIR_VARIANTS):
                    row = annotation(
                        feature,
                        base_id,
                        condition,
                        variant,
                        target_1_first=base_id % 2 == 1,
                    )
                    annotations[feature].append(row)
                    results.append({
                        "eval_id": row["eval_id"],
                        "prediction": "A" if variant == "original" else "B",
                        "is_correct": correct_pair[index],
                        "raw_response": variant,
                        "prompt_variant": variant,
                    })
    return annotations, results


class MatchedFeatureCaseSelectionTest(unittest.TestCase):
    def test_structure_signature_ignores_feature_identity(self):
        full = annotation("full", 1, "low_boundary", "original", True)
        color = annotation("color_only", 1, "low_boundary", "original", True)
        full["target_objects"][0]["shape"] = "square"
        color["target_objects"][0]["shape"] = "circle"
        full["target_objects"][0]["color"] = "orange"
        color["target_objects"][0]["color"] = "blue"
        self.assertEqual(
            selector.matched_structure_signature(full),
            selector.matched_structure_signature(color),
        )

    def test_builds_complete_matched_manifest(self):
        annotations, results = make_rows()
        index, base_ids = selector.annotation_index(annotations, FEATURES, CONDITIONS)
        audit = selector.validate_matched_structures(index, base_ids, FEATURES, CONDITIONS)
        expected_ids = {row["eval_id"] for rows in annotations.values() for row in rows}
        archived = selector.result_index(results, expected_ids)
        profiles = [
            selector.candidate_profile(base_id, index, archived, FEATURES, CONDITIONS)
            for base_id in base_ids
        ]
        selected, score, combinations = selector.choose_profiles(profiles, 4)
        manifest = selector.build_manifest(
            index, archived, selected, FEATURES, CONDITIONS
        )

        self.assertEqual(audit["cross_feature_condition_groups_checked"], 16)
        self.assertEqual(audit["mirrored_video_pairs_checked"], 64)
        self.assertEqual(combinations, 1)
        self.assertEqual(score[0], 0)
        self.assertEqual(len(manifest), 128)
        self.assertEqual(
            {row["attention_selection_first_mover"] for row in manifest},
            {"target_1_first", "target_2_first"},
        )
        pair_counts = {}
        for row in manifest:
            key = row["attention_pairing_id"]
            pair_counts[key] = pair_counts.get(key, 0) + 1
        self.assertEqual(set(pair_counts.values()), {2})
        self.assertEqual(
            {row["attention_case_bundle_pair_count"] for row in manifest},
            {16},
        )
        self.assertEqual(
            {row["attention_case_bundle_row_count"] for row in manifest},
            {32},
        )
        self.assertTrue(
            all("attention_case_bundle_size" not in row for row in manifest)
        )

    def test_structure_validation_detects_trajectory_mismatch(self):
        annotations, _results = make_rows(base_ids=(1,))
        for row in annotations["size_only"]:
            if row["condition"] == "low_boundary":
                row["target_objects"][0]["to"] = [99, 99]
        index, base_ids = selector.annotation_index(annotations, FEATURES, CONDITIONS)
        with self.assertRaisesRegex(ValueError, "not structurally matched"):
            selector.validate_matched_structures(index, base_ids, FEATURES, CONDITIONS)

    def test_structure_validation_detects_mirrored_video_mismatch(self):
        annotations, _results = make_rows(base_ids=(1,))
        swapped = next(
            row
            for row in annotations["full"]
            if row["condition"] == "low_boundary"
            and row["prompt_variant"] == "swapped"
        )
        swapped["video_path"] = "data/unexpected_duplicate.mp4"
        index, base_ids = selector.annotation_index(annotations, FEATURES, CONDITIONS)
        with self.assertRaisesRegex(ValueError, "Mirrored prompts"):
            selector.validate_matched_structures(index, base_ids, FEATURES, CONDITIONS)


if __name__ == "__main__":
    unittest.main()
