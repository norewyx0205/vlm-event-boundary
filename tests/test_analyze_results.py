import unittest

from scripts import analyze_results as analysis


def result_row(variant, perturbation, correct):
    return {
        "_source_file": "results/model/dataset/run/raw_results.jsonl",
        "dataset_name": "dataset",
        "dataset_version": "test",
        "difficulty_level": 5,
        "condition": "temporal_boundary",
        "video_id": "sample.mp4",
        "source_video_id": "sample.mp4",
        "prompt_variant": variant,
        "perturbation_type": perturbation,
        "is_correct": correct,
    }


class PerturbationAnalysisTest(unittest.TestCase):
    def test_reencode_control_can_be_primary_prompt_and_strict_baseline(self):
        rows = [
            result_row("original", "original", True),
            result_row("swapped", "original", True),
            result_row("original", "reencode_control", True),
            result_row("swapped", "reencode_control", True),
            result_row("original", "mask_target_1", False),
            result_row("swapped", "mask_target_1", True),
        ]
        summary, details = analysis.paired_perturbation_comparison(
            rows, baseline="reencode_control"
        )
        self.assertTrue(summary)
        target_rows = [row for row in details if row["perturbation_type"] == "mask_target_1"]
        self.assertEqual([row["difference"] for row in target_rows], [-1, 0])

        _, swap_details = analysis.swap_consistency(rows)
        strict_summary, strict_details = analysis.paired_perturbation_strict_comparison(
            swap_details, baseline="reencode_control"
        )
        self.assertTrue(strict_summary)
        target_strict = [
            row for row in strict_details if row["perturbation_type"] == "mask_target_1"
        ]
        self.assertEqual(target_strict[0]["difference"], -1)

        codec_summary, codec_details = analysis.codec_prediction_consistency(rows)
        self.assertEqual(len(codec_details), 2)
        self.assertEqual(
            next(row for row in codec_summary if row["scope"] == "overall")[
                "prediction_match_rate"
            ],
            1.0,
        )

    def test_model_input_summary_deduplicates_mirrored_prompts(self):
        metadata = {
            "video_grid_thw": [[4, 32, 32]],
            "visual_token_count_from_grid_thw": 4096,
            "video_token_count_from_mm_token_type_ids": 1024,
            "video_inputs": [{"shape": [8, 3, 512, 512], "frame_count_from_first_dim": 8}],
            "pixel_values_videos": {"shape": [8192, 1176]},
        }
        rows = []
        for variant in ("original", "swapped"):
            row = result_row(variant, "gap_removed", True)
            row.update({
                "video_path": "videos/gap_removed/sample.mp4",
                "duration_sec": 18.0,
                "input_metadata": metadata,
            })
            rows.append(row)

        summary = analysis.model_input_by_perturbation_condition(rows)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["unique_videos"], 1)
        self.assertEqual(summary[0]["sampled_frames_from_video_inputs"], "8")


if __name__ == "__main__":
    unittest.main()
