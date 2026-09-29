import json
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np

from scripts import analyze_phase3b, phase3b_core, screen_phase3b_rescues, run_phase3b_screening
from scripts.run_phase3b_relocation_control import decoded_video_psnr, relocation_pairs
from scripts.select_phase3b_cases import choose_independent_cases, frozen_representatives
from scripts.run_phase3b_patching import patch_forward, validate_behavioral_category
from scripts.select_phase3b_cases import write_frozen_rows


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [1000 + ord(char) for char in text]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        payload = {"input_ids": self.encode(text)}
        if return_offsets_mapping:
            payload["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return payload


def annotation(base, condition, variant):
    correct = "A" if variant == "original" else "B"
    return {
        "eval_id": f"base_{base}_{condition}_{variant}",
        "base_sample_id": base, "feature_variant": "full", "condition": condition,
        "dataset_version": "test_phase3b",
        "prompt_variant": variant, "video_path": f"base_{base}_{condition}.mp4",
        "option_A": "The orange circle moves before the blue square." if variant == "original" else "The orange circle moves after the blue square.",
        "option_B": "The orange circle moves after the blue square." if variant == "original" else "The orange circle moves before the blue square.",
        "correct_option": correct, "correct_sentence": "before", "incorrect_sentence": "after",
        "fps": 15, "duration_sec": 18, "total_frames": 270,
        "target_objects": [
            {"id": 1, "label": "the orange circle", "shape": "circle", "color": "orange", "from": [1, 1], "to": [2, 1]},
            {"id": 2, "label": "the blue square", "shape": "square", "color": "blue", "from": [3, 3], "to": [4, 3]},
        ],
        "distractors": [], "first_object_id": 2,
    }


class Phase3BTest(unittest.TestCase):
    def test_relocation_uses_matched_event_two_displacement(self):
        low = annotation(1, "low_boundary", "original")
        temporal = annotation(1, "temporal_boundary", "original")
        for row in (low, temporal):
            row["phase3b_pair_id"] = "pair_1"
        low["event_timing"] = {
            "first_event_start_frame": 30, "first_event_end_frame": 60,
            "second_event_start_frame": 60, "second_event_end_frame": 90,
        }
        temporal["event_timing"] = {
            "first_event_start_frame": 30, "first_event_end_frame": 60,
            "second_event_start_frame": 105, "second_event_end_frame": 135,
        }
        self.assertEqual(relocation_pairs([low, temporal], 2)[0][1], 45)
        with self.assertRaisesRegex(ValueError, "differs from matched"):
            relocation_pairs([low, temporal], 2, requested_shift=15)
        with self.assertRaisesRegex(ValueError, "both boundary rows"):
            relocation_pairs([low], 2)

    def test_reencode_psnr_checks_frame_count_and_image_shape(self):
        original = np.zeros((4, 4, 3), dtype=np.uint8)
        reencoded = original.copy()
        reencoded[0, 0] = 10
        self.assertGreater(decoded_video_psnr([original], [reencoded]), 30)
        self.assertIsNone(decoded_video_psnr([original], [original]))
        with self.assertRaisesRegex(RuntimeError, "frame count"):
            decoded_video_psnr([original], [])

    def test_old_pool_provenance_is_checked_before_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_phase3b_screening.validate_pool_provenance(root, 42, 300)
            (root / "L5_full").mkdir()
            (root / "L5_full/annotations.jsonl").write_text("\n")
            with self.assertRaisesRegex(RuntimeError, "lacks its generation config"):
                run_phase3b_screening.validate_pool_provenance(root, 42, 300)
            (root / "phase3b_generation_config.json").write_text(json.dumps({
                "schema": "phase3b_rescue_pool_v1", "generator_code_sha256": {},
                "seed": 42, "max_new_bases": 300,
                "conditions": ["low_boundary", "temporal_boundary"],
            }))
            with self.assertRaisesRegex(RuntimeError, "generator_code_sha256"):
                run_phase3b_screening.validate_pool_provenance(root, 42, 300)

    def test_representatives_are_frozen_without_patch_outputs(self):
        selected = [
            {"base_sample_id": base, "first_object_id": mover, "prompt_variant": "original"}
            for mover in (1, 2) for base in (mover, mover + 2, mover + 4)
        ]
        self.assertEqual(frozen_representatives(selected), {
            "target_1_first": "phase3b_base_003_original",
            "target_2_first": "phase3b_base_004_original",
        })

    def test_analysis_uses_frozen_representative_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "analysis_case_manifest.jsonl"
            manifest.write_text("\n")
            (root / "case_selection_summary.json").write_text(json.dumps({
                "representative_pair_ids": {"target_1_first": "pair_2"}
            }))
            captures = {
                (pair, condition): {"analysis_stratum": "primary_rescue", "first_object_id": 1}
                for pair in ("pair_1", "pair_2")
                for condition in ("low_boundary", "temporal_boundary")
            }
            self.assertEqual(
                analyze_phase3b.representative_pairs(captures, manifest),
                {"target_1_first": "pair_2"},
            )
            (root / "case_selection_summary.json").unlink()
            with self.assertRaisesRegex(ValueError, "selection summary is missing"):
                analyze_phase3b.representative_pairs(captures, manifest)

    def test_section_scoped_text_groups_keep_query_and_options_separate(self):
        row = annotation(1, "low_boundary", "original")
        tokenizer = CharacterTokenizer()
        core = (
            "Which statement correctly describes the order of events?\n\n"
            f"A: {row['option_A']}\nB: {row['option_B']}\n\nAnswer with only A or B."
        )
        ids = tokenizer.encode("prefix\n\n" + core + "suffix")
        groups = phase3b_core.build_text_groups(row, ids, tokenizer)
        extracted = lambda name: "".join(chr(ids[position] - 1000) for position in groups[name])
        self.assertEqual(extracted("query_all"), "Which statement correctly describes the order of events?")
        self.assertEqual(extracted("option_temporal_relations"), "beforeafter")
        self.assertIn("orange circle", extracted("option_target_1_mentions"))
        self.assertIn("blue square", extracted("option_target_2_mentions"))
        self.assertFalse(set(groups["query_all"]) & set(groups["options_all"]))
        self.assertTrue(set(groups["option_temporal_relations"]) <= set(groups["options_all"]))

    def test_mover_roles_invert_when_target_two_moves_first(self):
        roles = phase3b_core.mover_roles(2)
        self.assertEqual(roles["first_mover_own_event"], "video_t2_e1")
        self.assertEqual(roles["first_mover_during_event_2"], "video_t2_e2")
        self.assertEqual(roles["second_mover_during_event_1"], "video_t1_e1")
        self.assertEqual(roles["second_mover_own_event"], "video_t1_e2")

    def test_event_relative_mapping_is_one_to_one_monotonic_and_audited(self):
        source = [
            {"position": 100 + i, "temporal_index": i, "progress": progress, "x": 3, "y": 4}
            for i, progress in enumerate((0.1, 0.5, 0.9))
        ]
        target = [
            {"position": 200 + i, "temporal_index": i + 4, "progress": progress, "x": 3, "y": 4}
            for i, progress in enumerate((0.08, 0.55, 0.92))
        ]
        mapped = phase3b_core.map_group(source, target)
        self.assertTrue(mapped["eligible"])
        self.assertEqual(mapped["source_positions"], [100, 101, 102])
        self.assertEqual(mapped["target_positions"], [200, 201, 202])
        self.assertEqual(mapped["source_coverage"], 1.0)
        self.assertEqual(mapped["target_coverage"], 1.0)
        distant = phase3b_core.map_group(
            source, [{**cell, "position": cell["position"] + 1000, "progress": 1.0}
                     for cell in target]
        )
        self.assertFalse(distant["eligible"])
        self.assertGreater(distant["rejected_temporal_bins"], 0)

    def test_screening_and_selection_preserve_base_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            annotation_path = Path(directory) / "annotations.jsonl"
            result_path = Path(directory) / "raw_results.jsonl"
            rows, results = [], []
            for base in (1, 2):
                for variant in ("original", "swapped"):
                    for condition in ("low_boundary", "temporal_boundary"):
                        item = annotation(base, condition, variant)
                        rows.append(item)
                        correct = condition == "temporal_boundary" or (base == 2 and variant == "swapped")
                        results.append({
                            "eval_id": item["eval_id"], "base_sample_id": base,
                            "feature_variant": "full", "condition": condition,
                            "video_path": item["video_path"], "correct_option": item["correct_option"],
                            "prediction": item["correct_option"] if correct else ("B" if item["correct_option"] == "A" else "A"),
                            "is_correct": correct,
                        })
            annotation_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            result_path.write_text("\n".join(json.dumps(row) for row in results) + "\n")
            screened, candidates, _, _ = screen_phase3b_rescues.screen([annotation_path], [result_path])
        self.assertEqual(len(screened), 4)
        self.assertEqual(len(candidates), 3)
        mappings = {(item["base_sample_id"], item["prompt_variant"]): {"eligible": True} for item in candidates}
        selected = choose_independent_cases(candidates, mappings, 2)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({item["base_sample_id"] for item in selected}), 2)
        self.assertEqual(selected[0]["prompt_variant"], "original")

    def test_aggregate_uses_one_base_per_location(self):
        rows = [{
            "base_sample_id": base, "patch_direction": "temporal_to_low",
            "token_group": "query_all", "layer": 4, "mover_role": None,
            "source_aligned_patch_effect": float(base), "categorical_flip": base == 2,
            "flip_toward_correct": base == 2, "flip_away_from_correct": False,
            "recovery": base / 2,
        } for base in (1, 2)]
        table = analyze_phase3b.aggregate_patches(rows, seed=42)
        self.assertEqual(table[0]["eligible_n"], 2)
        self.assertEqual(table[0]["mean_aligned_margin_effect"], 1.5)
        with self.assertRaisesRegex(ValueError, "Repeated base"):
            analyze_phase3b.aggregate_patches(rows + [rows[0]])

    def test_frozen_selection_cannot_be_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            write_frozen_rows(path, [{"base_sample_id": 1}])
            write_frozen_rows(path, [{"base_sample_id": 1}])
            with self.assertRaisesRegex(RuntimeError, "Frozen"):
                write_frozen_rows(path, [{"base_sample_id": 2}])

    def test_completeness_requires_both_capture_conditions_and_all_layers(self):
        captures = {("pair_1", "low_boundary"): {}}
        with self.assertRaisesRegex(ValueError, "Incomplete Phase 3B"):
            analyze_phase3b.validate_completeness([], [], captures, True)
        missing = analyze_phase3b.validate_completeness([], [], captures, False)
        self.assertEqual(missing[2], [("pair_1", "temporal_boundary")])

    def test_positionwise_hook_changes_logits_and_identity_is_noop(self):
        from tests.test_activation_patching import FakeModel, FakeProcessor, Inputs

        model, processor = FakeModel(), FakeProcessor()
        prepared = {
            "inputs": Inputs(input_ids=torch.tensor([[0, 1, 3]])),
            "groups": {"decision_position": [2]},
            "row": {"correct_option": "A"},
        }
        snapshots = []
        hook = model.model.layers[0].register_forward_hook(
            lambda _module, _args, output: snapshots.append(output.detach().clone())
        )
        baseline = model(**prepared["inputs"]).logits[0, -1]
        hook.remove()
        same, _ = patch_forward(model, processor, prepared, 0, [0], snapshots[0][0, [0]])
        self.assertTrue(torch.equal(baseline, same))
        different, _ = patch_forward(model, processor, prepared, 0, [0], snapshots[0][0, [1]])
        self.assertFalse(torch.equal(baseline, different))

    def test_case_freeze_keeps_mirrored_behavior_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations, results, mappings = [], [], []
            for base in (1, 2):
                for variant in ("original", "swapped"):
                    mappings.append({"base_sample_id": base, "prompt_variant": variant, "eligible": True})
                    for condition in ("low_boundary", "temporal_boundary"):
                        row = annotation(base, condition, variant)
                        annotations.append(row)
                        rescue = (base == 1 and variant == "original") or (base == 2 and variant == "swapped")
                        correct = condition == "temporal_boundary" if rescue else True
                        results.append({
                            "eval_id": row["eval_id"], "base_sample_id": base,
                            "feature_variant": "full", "condition": condition,
                            "video_path": row["video_path"], "correct_option": row["correct_option"],
                            "prediction": row["correct_option"] if correct else ("B" if row["correct_option"] == "A" else "A"),
                            "is_correct": correct,
                        })
            for name, rows in (("annotations", annotations), ("results", results), ("mappings", mappings)):
                (root / f"{name}.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            subprocess.run([
                sys.executable, "scripts/select_phase3b_cases.py",
                "--annotation_paths", str(root / "annotations.jsonl"),
                "--result_paths", str(root / "results.jsonl"),
                "--mapping_path", str(root / "mappings.jsonl"),
                "--output_dir", str(root / "selection"),
                "--primary_count", "2", "--reserve_count", "0", "--control_count", "0",
                "--mirrored_count", "2",
            ], check=True, capture_output=True, text=True)
            analysis = [json.loads(line) for line in (root / "selection/analysis_case_manifest.jsonl").read_text().splitlines()]
            summary = json.loads((root / "selection/case_selection_summary.json").read_text())
            self.assertEqual(len(summary["representative_pair_ids"]), 1)
            low = [row for row in analysis if row["condition"] == "low_boundary"]
            self.assertEqual(len(low), 4)
            self.assertEqual(sum(row["phase3b_analysis_stratum"] == "primary_rescue" for row in low), 2)
            mirrored = [row for row in low if row["phase3b_analysis_stratum"] == "mirrored_prompt_control"]
            self.assertEqual(len(mirrored), 2)
            self.assertTrue(all(row["phase3b_prompt_pair_behavior"] == "stable_both_correct" for row in mirrored))

    def test_prompt_pair_category_is_rechecked_from_live_margins(self):
        pair = {"low_boundary": {"phase3b_prompt_pair_behavior": "stable_both_correct"}}
        low = {"decision": {"margin": 0.5}}
        temporal = {"decision": {"margin": 1.0}}
        validate_behavioral_category(pair, low, temporal)
        with self.assertRaisesRegex(RuntimeError, "behavior changed"):
            validate_behavioral_category(pair, {"decision": {"margin": -0.5}}, temporal)

    def test_cpu_analysis_merges_complete_shard_and_writes_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text("\n".join(json.dumps({
                "phase3b_pair_id": "pair_1", "condition": condition,
                "base_sample_id": 1, "prompt_variant": "original", "correct_option": "A",
                "phase3b_analysis_stratum": "primary_rescue",
                "phase3b_prompt_pair_behavior": "temporal_rescue", "first_object_id": 1,
            }) for condition in ("low_boundary", "temporal_boundary")) + "\n")
            shard = root / "shards/shard_00"
            shard.mkdir(parents=True)
            config = {
                "schema": phase3b_core.SCHEMA, "repo_commit": "test",
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "mapping_sha256": "test", "shard_size": 5, "shard_index": 0,
                "model_name": "fake", "model_revision": "fake", "transformers_version": "fake",
                "qwen_vl_utils_version": "fake", "torch_version": "fake", "seed": 42,
                "video_fps": None, "video_num_frames": None, "video_max_pixels": None,
                "roi_padding": 8, "attn_implementation": "eager", "validate_controls": True,
                "run_fingerprint": "test-fingerprint", "pair_ids": ["pair_1"],
            }
            (shard / "run_config.json").write_text(json.dumps(config))
            for condition, margin in (("low_boundary", -1.0), ("temporal_boundary", 1.0)):
                capture = shard / "activations/pair_1" / condition
                capture.mkdir(parents=True)
                (capture / "index.json").write_text(json.dumps({
                    "run_fingerprint": "test-fingerprint", "analysis_stratum": "primary_rescue",
                    "decision": {"margin": margin}, "first_object_id": 1,
                    "mover_roles": phase3b_core.mover_roles(1),
                    "group_mean_norms_by_layer": {
                        str(layer): {group: float(layer + 1) for group in phase3b_core.GROUPS}
                        for layer in range(36)
                    },
                }))
            divergence = [
                {
                    "run_fingerprint": "test-fingerprint", "phase3b_pair_id": "pair_1",
                    "analysis_stratum": "primary_rescue", "token_group": group, "layer": layer,
                    "cosine_distance": 0.01 * (layer + 1), "relative_l2": 0.02 * (layer + 1),
                }
                for layer in range(36) for group in phase3b_core.GROUPS
            ]
            divergence_path = shard / "divergence/pair_1.jsonl"
            divergence_path.parent.mkdir(parents=True)
            divergence_path.write_text("\n".join(json.dumps(row) for row in divergence) + "\n")
            for direction in analyze_phase3b.DIRECTIONS:
                patches = [
                    {
                        "run_fingerprint": "test-fingerprint", "phase3b_pair_id": "pair_1",
                        "base_sample_id": 1, "analysis_stratum": "primary_rescue",
                        "patch_direction": direction, "token_group": group, "layer": layer,
                        "source_aligned_patch_effect": 0.01 * (layer + 1),
                        "categorical_flip": False, "flip_toward_correct": False,
                        "flip_away_from_correct": False, "recovery": 0.25,
                        "mover_role": next((role for role, literal in phase3b_core.mover_roles(1).items()
                                            if literal == group), None),
                    }
                    for group in phase3b_core.GROUPS
                    for layer in phase3b_core.PATCH_LAYERS
                    + ((35,) if group == "decision_position" else ())
                ]
                patch_path = shard / "patches/pair_1" / f"{direction}.jsonl"
                patch_path.parent.mkdir(parents=True, exist_ok=True)
                patch_path.write_text("\n".join(json.dumps(row) for row in patches) + "\n")
            controls = shard / "technical_controls/pair_1.json"
            controls.parent.mkdir(parents=True)
            controls.write_text(json.dumps({"phase3b_pair_id": "pair_1", "run_fingerprint": "test-fingerprint"}))
            env = dict(os.environ, MPLCONFIGDIR=str(root / "matplotlib"))
            subprocess.run([
                sys.executable, "scripts/analyze_phase3b.py",
                "--manifest_path", str(manifest), "--shards_root", str(root / "shards"),
                "--output_dir", str(root / "analysis"),
            ], check=True, capture_output=True, text=True, env=env)
            self.assertTrue((root / "analysis/plots/aggregate_divergence_heatmap.png").is_file())
            self.assertTrue((root / "analysis/plots/representative_target_1_first_activation_trajectories.png").is_file())
            self.assertEqual(json.loads((root / "analysis/aggregate_summary.json").read_text())["missing_patch_count"], 0)


if __name__ == "__main__":
    unittest.main()
