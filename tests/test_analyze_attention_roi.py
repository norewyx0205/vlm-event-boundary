import copy
import json
import math
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts import analyze_attention_roi as analysis


def result_fixture():
    profile = {
        "layer": 0,
        "visual_attention_fraction": 0.3,
        "spatial_roi": {
            "target_1": {
                "attention_mass": 0.03,
                "normalized_visual_attention": 0.1,
                "token_fraction": 0.05,
                "token_count": 5,
                "mean_attention_per_token": 0.006,
                "enrichment": 2.0,
            },
            "target_2": {
                "attention_mass": 0.06,
                "normalized_visual_attention": 0.2,
                "token_fraction": 0.2,
                "token_count": 20,
                "mean_attention_per_token": 0.003,
                "enrichment": 1.0,
            },
            "background": {
                "attention_mass": 0.21,
                "normalized_visual_attention": 0.7,
                "token_fraction": 0.75,
                "token_count": 75,
                "mean_attention_per_token": 0.0028,
                "enrichment": 0.7 / 0.75,
            },
        },
    }
    return {
        "eval_id": "sample_001_temporal_boundary_original",
        "base_sample_id": 1,
        "condition": "temporal_boundary",
        "prompt_variant": "original",
        "option_A": "The largest circle moves before the smallest circle.",
        "target_objects": [
            {
                "id": 1,
                "label": "the smallest circle",
                "reference_label": "smallest",
                "start_frame": 60,
                "end_frame": 90,
                "radius": 14,
            },
            {
                "id": 2,
                "label": "the largest circle",
                "reference_label": "largest",
                "start_frame": 30,
                "end_frame": 60,
                "radius": 48,
            },
        ],
        "layer_roi_profiles": [profile],
        "attention_semantics": "prompt_final_position_attention_predicting_first_answer_token",
        "roi_assignment_method": "overlap",
        "roi_padding": 8,
        "head_reduction": "mean",
        "source_frame_groups": [[0, 8]],
    }


class AttentionAnalysisTest(unittest.TestCase):
    def test_phase1_selection_metadata_reaches_analysis_rows(self):
        result = result_fixture()
        result.update({
            "attention_pairing_id": "size_only__001__temporal_boundary",
            "attention_case_bundle_id": "matched_feature_calibration_base_001",
            "attention_case_bundle_pair_count": 16,
            "attention_case_bundle_row_count": 32,
            "attention_archived_pair_outcome": "position_sensitive",
            "attention_selection_first_mover": "target_2_first",
        })

        metadata = analysis.base_metadata(Path("attention.json"), result)

        self.assertEqual(
            metadata["attention_archived_pair_outcome"],
            "position_sensitive",
        )
        self.assertEqual(metadata["attention_case_bundle_pair_count"], 16)
        self.assertEqual(metadata["attention_case_bundle_row_count"], 32)

    def test_archived_pair_outcome_survives_feature_summary_pipeline(self):
        layer_rows = []
        for variant in ("original", "swapped"):
            result = copy.deepcopy(result_fixture())
            result.update({
                "eval_id": f"sample_001_temporal_boundary_{variant}",
                "video_id": "sample_001_temporal_boundary.mp4",
                "feature_variant": "size_only",
                "prompt_variant": variant,
                "attention_pairing_id": "size_only__001__temporal_boundary",
                "attention_archived_pair_outcome": "position_sensitive",
            })
            layer_rows.extend(
                analysis.target_contrasts(Path("attention.json"), result)
            )

        stage_rows = analysis.stage_contrasts(layer_rows)
        paired_rows = analysis.paired_stage_contrasts(stage_rows)
        tables = analysis.feature_calibration_tables(stage_rows, paired_rows)
        outcome_rows = tables["feature_pair_outcome_stage_summary"]

        self.assertEqual(len(outcome_rows), 1)
        self.assertEqual(
            outcome_rows[0]["attention_archived_pair_outcome"],
            "position_sensitive",
        )
        self.assertNotEqual(
            outcome_rows[0]["attention_archived_pair_outcome"],
            "None",
        )

    def test_zip_archive_attention_json_is_discovered(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "output.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "analysis/attention/probe.json",
                    json.dumps([result_fixture()]),
                )
                archive.writestr(
                    "analysis/attention/probe_summary.json",
                    json.dumps({"rows": 1}),
                )

            rows, sources = analysis.read_attention_results(archive_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(len(sources), 1)
        self.assertIn("probe.json", sources[0])

    def test_roles_include_semantics_mover_and_prompt_subject(self):
        roles = analysis.target_roles(result_fixture())

        self.assertEqual(roles["target_1"]["mover_role"], "second mover")
        self.assertEqual(roles["target_2"]["mover_role"], "first mover")
        self.assertEqual(roles["target_1"]["subject_role"], "non-subject")
        self.assertEqual(roles["target_2"]["subject_role"], "subject")
        self.assertIn("smallest", roles["target_1"]["display_label"])

    def test_contrast_decomposes_mass_and_effective_area(self):
        rows = analysis.target_contrasts(Path("attention.json"), result_fixture())

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertAlmostEqual(
            row["delta_log_visual_mass_t1_over_t2"],
            math.log(0.5),
        )
        self.assertAlmostEqual(
            row["log_area_ratio_t1_over_t2"],
            math.log(0.25),
        )
        self.assertAlmostEqual(
            row["delta_log_enrichment_t1_over_t2"],
            math.log(2.0),
        )
        self.assertAlmostEqual(row["decomposition_residual"], 0.0)
        self.assertFalse(row["target_1_has_more_total_visual_mass"])
        self.assertTrue(row["target_1_has_higher_enrichment"])

    def test_legacy_metrics_are_exported_with_explicit_names(self):
        result = result_fixture()
        result["head_reduction"] = "max"
        result["layer_roi_profiles"][0]["spatial_roi"]["target_1"]["attention_mass"] = 0.91
        rows = analysis.flatten_roi_metrics(Path("attention.json"), result)
        target_1 = next(row for row in rows if row["roi"] == "target_1")

        self.assertEqual(target_1["metric_schema"], "legacy_v1_aliases")
        self.assertAlmostEqual(target_1["all_token_attention_share"], 0.03)
        self.assertEqual(
            target_1["all_token_attention_share_source"],
            "reconstructed_visual_fraction_x_visual_mass",
        )
        self.assertAlmostEqual(target_1["visual_normalized_attention_mass"], 0.1)
        self.assertAlmostEqual(target_1["effective_token_area_share"], 0.05)
        self.assertAlmostEqual(target_1["area_normalized_enrichment"], 2.0)

    def test_archive_audit_flags_missing_measurement_metadata(self):
        rows = [(Path("attention.json"), result_fixture())]
        audit = analysis.archive_audit(rows)

        self.assertEqual(len(audit), 1)
        self.assertTrue(audit[0]["measurement_metadata_complete"])
        legacy = result_fixture()
        legacy.pop("attention_semantics")
        legacy.pop("roi_assignment_method")
        legacy.pop("roi_padding")
        legacy_audit = analysis.archive_audit([(Path("legacy.json"), legacy)])
        self.assertFalse(legacy_audit[0]["measurement_metadata_complete"])

    def test_mirrored_prompts_are_averaged_before_group_summary(self):
        rows = []
        for base_id, values in ((1, (1.0, 3.0)), (2, (5.0, 7.0))):
            for variant, value in zip(("original", "swapped"), values):
                rows.append({
                    "source_attention_file": "attention.json",
                    "eval_id": f"sample_{base_id}_{variant}",
                    "video_id": f"sample_{base_id}.mp4",
                    "base_sample_id": base_id,
                    "feature_variant": "full",
                    "size_scene_variant": None,
                    "condition": "low_boundary",
                    "prompt_variant": variant,
                    "layer_stage": "early",
                    "target_1_mover_role": "first mover" if base_id == 1 else "second mover",
                    "mean_delta_log_visual_mass_t1_over_t2": value,
                    "mean_log_area_ratio_t1_over_t2": 0.0,
                    "mean_delta_log_enrichment_t1_over_t2": value,
                    "mean_decomposition_residual": 0.0,
                })

        paired = analysis.paired_stage_contrasts(rows)
        summary = analysis.summarize_contrasts(
            paired,
            ("feature_variant", "layer_stage"),
        )

        self.assertEqual(len(paired), 2)
        self.assertTrue(all(row["mirrored_pair_complete"] for row in paired))
        self.assertEqual(
            [row["mean_delta_log_visual_mass_t1_over_t2"] for row in paired],
            [2.0, 6.0],
        )
        self.assertEqual(summary[0]["n_base_samples"], 2)
        self.assertEqual(
            summary[0]["mean_delta_log_visual_mass_t1_over_t2_mean"],
            4.0,
        )


if __name__ == "__main__":
    unittest.main()
