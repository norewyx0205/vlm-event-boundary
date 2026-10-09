import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from scripts import audit_phase3c_mappings, phase3c_core as core, prepare_phase3c


RUNTIME = {
    "transformers": "5.9.0", "torch": "2.11.0+cpu", "qwen_vl_utils": "0.0.14",
    "device": "cpu", "model_weights_loaded": False, "deepstack_decoder_injection_layers": [0, 1, 2],
}


def fixture_pair(base=1, mover=1, stratum="rescue", video_root=None):
    pair_id = f"phase3b_base_{base:03d}_original"
    ids = [123] * 60
    ids[1:25] = [151656] * 24
    metadata = {
        "merged_video_grid_thw": [4, 2, 3], "video_grid_thw": [4, 4, 6],
        "visual_token_count": 24, "source_frame_groups": [[0, 1], [2, 3], [4, 5], [6, 7]],
        "minimum_primary_roi_overlap": 0.1, "event_phase_dominance_threshold": 0.5,
    }
    candidate = {"pair_id": pair_id, "base_sample_id": base, "prompt_variant": "original",
        "first_object_id": mover, "stratum": stratum, "rows": {}, "capture_indices": {},
        "video_provenance": {}}
    records = {}
    for side, condition in enumerate(core.CONDITIONS):
        start = 7 + side * 6
        groups = {
            "video_t1_e2": [start, start + 6], "video_t2_e2": [start + 1, start + 7],
            "video_distractors_e2": [start + 2, start + 8],
            "options_all": [40, 41], "query_all": [30, 31],
        }
        margin = 1.0 if side or stratum == "stable" else -1.0
        prediction = "A" if margin > 0 else "B"
        path = str(Path(video_root or "/tmp") / f"base_{base}_{condition}.mp4")
        if video_root:
            Path(path).write_bytes(f"unit-test-video-{base}-{condition}".encode())
            video_hash = core.file_hash(path)
        else:
            video_hash = "a" * 64
        row = {"phase3b_pair_id": pair_id, "base_sample_id": base, "prompt_variant": "original",
            "first_object_id": mover, "correct_option": "A", "condition": condition,
            "phase3b_analysis_stratum": "primary_rescue" if stratum == "rescue" else "stable_both_correct_control",
            "phase3b_prompt_pair_behavior": "temporal_rescue" if stratum == "rescue" else "stable_both_correct",
            "eval_id": f"base_{base}_{condition}_original", "archived_prediction": prediction,
            "option_A": "before", "option_B": "after", "video_path": path, "total_frames": 10,
            "event_timing": {"first_event_start_frame": 0, "first_event_end_frame": 2,
                "second_event_start_frame": 2 + 2 * side, "second_event_end_frame": 6 + 2 * side}}
        index = {"eval_id": row["eval_id"], "run_fingerprint": "unit-test-source",
            "decision": {"margin": margin, "prediction": prediction, "correct_option": "A",
                "is_correct": margin > 0, "correct_logit": 2.0 + margin, "incorrect_logit": 2.0},
            "positions": sorted({value for positions in groups.values() for value in positions}),
            "group_positions": groups, "video_metadata": copy.deepcopy(metadata), "input_metadata": {},
            "standard_parity": {"first_token_match": True, "logits_allclose": True},
            "archived_input_parity": {"matches": True}}
        candidate["rows"][condition], candidate["capture_indices"][condition] = row, index
        candidate["video_provenance"][condition] = {"path": path, "sha256": video_hash}
        records[condition] = {"input_ids": ids[:], "attention_mask": [1] * 60,
            "prompt_token_count": 60, "prompt_input_ids_sha256": core.input_ids_hash(ids),
            "visual_positions": list(range(1, 25)), "video_metadata": copy.deepcopy(metadata),
            "sampled_frame_indices": list(range(8)),
            "group_positions": groups, "background_event2_positions": [start + offset for offset in (3, 4, 5, 9, 10, 11)],
            "distractors_event2_positions": groups["video_distractors_e2"],
            "video_path": path, "video_sha256": video_hash}
    mapping = {"pair_id": pair_id, "base_sample_id": base, "prompt_variant": "original",
        "first_object_id": mover, "eligible": True, "groups": {},
        "prompt_input_ids_sha256": core.input_ids_hash(ids),
        "low_video_metadata": metadata, "temporal_video_metadata": copy.deepcopy(metadata)}
    for group in core.ROI_GROUPS:
        mapping["groups"][group] = {
            "source_positions": records[core.CONDITIONS[0]]["group_positions"][group][:],
            "target_positions": records[core.CONDITIONS[1]]["group_positions"][group][:],
            "source_token_count": 2, "target_token_count": 2, "mapped_token_count": 2,
            "bin_pairs": [{"source_temporal_index": 1, "target_temporal_index": 2},
                          {"source_temporal_index": 2, "target_temporal_index": 3}],
        }
    candidate["archived_mapping"] = mapping
    return candidate, records


def fixture_archive(directory, pairs=12):
    source = Path(directory) / "source"
    video_root = source / "videos"
    video_root.mkdir(parents=True)
    candidates, live = [], {}
    for base in range(1, pairs + 1):
        candidate, records = fixture_pair(base, 1 if base % 2 else 2,
                                           "stable" if base <= 4 else "rescue", video_root)
        candidates.append(candidate)
        live[candidate["pair_id"]] = records
    manifest = [row for candidate in candidates for row in candidate["rows"].values()]
    mappings = [candidate["archived_mapping"] for candidate in candidates]
    core.atomic_write(source / "selection/analysis_case_manifest.jsonl", manifest, jsonl=True)
    core.atomic_write(source / "selection/selected_video_mappings.jsonl", mappings, jsonl=True)
    shard = source / "primary/checkpoints/shard_00"
    hashes, paths = {}, {}
    for candidate in candidates:
        for condition, row in candidate["rows"].items():
            core.atomic_write(shard / "activations" / candidate["pair_id"] / condition / "index.json",
                              candidate["capture_indices"][condition])
            paths[row["eval_id"]] = row["video_path"]
            hashes[row["eval_id"]] = candidate["video_provenance"][condition]["sha256"]
    core.atomic_write(shard / "run_config.json", {
        "run_fingerprint": "unit-test-source", "model_name": core.MODEL, "model_revision": core.REVISION,
        "transformers_version": "5.9.0", "video_sha256_by_eval_id": hashes,
    })
    config = {key: prepare_phase3c.DEFAULT_SETTINGS[key] for key in (
        "model_name", "model_revision", "expected_transformers_version", "dtype", "attn_implementation",
        "seed", "roi_padding", "video_fps", "video_num_frames", "video_max_pixels",
    )}
    config.update({"artifact_type": "real", "pipeline_fingerprint": "unit-test-source-pipeline",
        "manifest_sha256": {"full": core.file_hash(source / "selection/analysis_case_manifest.jsonl")},
        "mapping_sha256": core.file_hash(source / "selection/selected_video_mappings.jsonl"),
        "video_paths_by_eval_id": paths, "video_sha256": {paths[key]: value for key, value in hashes.items()}})
    core.atomic_write(source / "vm_run_config.json", config)
    return source, live


class Phase3CCoreTest(unittest.TestCase):
    def test_balance_is_deterministic_and_base_independent(self):
        candidates = [fixture_pair(base, 1 if base % 2 else 2, "stable" if base <= 4 else "rescue")[0]
                      for base in range(1, 13)]
        selected, missing = core.select_balanced(list(reversed(candidates)))
        self.assertFalse(missing)
        self.assertEqual([row["base_sample_id"] for row in selected], list(range(1, 13)))
        selected, missing = core.select_balanced(candidates + [copy.deepcopy(candidates[0])])
        self.assertEqual(len(selected), 12)
        self.assertEqual(len({row["base_sample_id"] for row in selected}), 12)

    def test_missing_mover_quota_is_not_silently_relaxed(self):
        candidates = [fixture_pair(base, 1, "stable" if base <= 4 else "rescue")[0] for base in range(1, 13)]
        _, missing = core.select_balanced(candidates)
        self.assertEqual(missing["rescue_target_2_first"], 4)
        self.assertEqual(missing["stable_target_2_first"], 2)

    def test_full_event_grid_requires_actual_positions(self):
        candidate, records = fixture_pair()
        result = core.prepare_support_audit(candidate, candidate["archived_mapping"], records)
        grid = result["supports"]["whole_event2"]
        self.assertEqual(grid["mapped_token_count"], 12)
        self.assertEqual(grid["directions"]["temporal_to_low"]["recipient_coverage"], 1.0)
        self.assertEqual(result["supports"]["both_targets_event2"]["mapped_token_count"], 4)
        self.assertTrue(result["capture_requirements"]["low_boundary"]["new_visual_positions_to_capture"])
        records["low_boundary"].pop("visual_positions")
        with self.assertRaises(KeyError):
            core.prepare_support_audit(candidate, candidate["archived_mapping"], records)

    def test_roi_count_mismatch_is_direction_specific(self):
        candidate, records = fixture_pair()
        record = records["temporal_boundary"]
        record["group_positions"]["video_t1_e2"].append(16)
        mapping = candidate["archived_mapping"]["groups"]["video_t1_e2"]
        mapping["target_token_count"] = 3
        record["background_event2_positions"].remove(16)
        layouts = [core.event_layout(candidate["rows"][condition], records[condition]["video_metadata"],
            records[condition]["visual_positions"], 60) for condition in core.CONDITIONS]
        support = core.reference_support(mapping, records["low_boundary"]["group_positions"]["video_t1_e2"],
            record["group_positions"]["video_t1_e2"], layouts)
        self.assertEqual(support["directions"]["temporal_to_low"]["recipient_coverage"], 1.0)
        self.assertEqual(support["directions"]["low_to_temporal"]["recipient_coverage"], 2 / 3)
        self.assertEqual(support["unmatched_temporal_positions"], [16])

    def test_grid_support_uses_real_cells_in_union_not_filled_vectors(self):
        candidate, records = fixture_pair()
        records["temporal_boundary"]["group_positions"]["video_t1_e2"].append(16)
        records["temporal_boundary"]["background_event2_positions"].remove(16)
        candidate["archived_mapping"]["groups"]["video_t1_e2"]["target_token_count"] = 3
        layouts = [core.event_layout(candidate["rows"][condition], records[condition]["video_metadata"],
            records[condition]["visual_positions"], 60) for condition in core.CONDITIONS]
        union = core.expanded_supports(core.event_bin_pairs(*layouts, max_progress_error=0.1),
            [records[condition]["group_positions"] for condition in core.CONDITIONS])["both_targets_event2"]
        self.assertEqual(union["scope"], "cross_condition_roi_union_with_observed_grid_cells")
        self.assertIn(10, union["low_positions"])
        self.assertIn(16, union["temporal_positions"])
        self.assertEqual(union["mapped_token_count"], 5)

    def test_equal_shapes_cannot_establish_correspondence(self):
        candidate, records = fixture_pair()
        records["temporal_boundary"]["visual_positions"][0:2] = [2, 1]
        with self.assertRaisesRegex(ValueError, "actual visual positions|Actual visual positions"):
            core.prepare_support_audit(candidate, candidate["archived_mapping"], records)

    def test_changed_token_ids_are_rejected(self):
        candidate, records = fixture_pair()
        records["low_boundary"]["input_ids"][0] = 124
        with self.assertRaisesRegex(ValueError, "token IDs"):
            core.prepare_support_audit(candidate, candidate["archived_mapping"], records)

    def test_padding_is_not_assumed_causally_visible(self):
        candidate, records = fixture_pair()
        records["low_boundary"]["attention_mask"][0] = 0
        with self.assertRaisesRegex(ValueError, "unpadded"):
            core.prepare_support_audit(candidate, candidate["archived_mapping"], records)

    def test_progress_error_and_unequal_bins_are_not_interpolated(self):
        candidate, records = fixture_pair()
        layouts = [core.event_layout(candidate["rows"][condition], records[condition]["video_metadata"],
                                    records[condition]["visual_positions"], 60) for condition in core.CONDITIONS]
        layouts[1]["bins"][0]["progress"] += 0.2
        with self.assertRaisesRegex(ValueError, "progress error"):
            core.event_bin_pairs(*layouts, max_progress_error=0.1)
        layouts[1]["bins"].pop()
        with self.assertRaisesRegex(ValueError, "Unequal"):
            core.event_bin_pairs(*layouts, max_progress_error=0.1)

    def test_half_event_bin_is_excluded(self):
        candidate, records = fixture_pair()
        item = records["low_boundary"]
        layout = core.event_layout(candidate["rows"]["low_boundary"], item["video_metadata"], item["visual_positions"], 60)
        self.assertEqual([row["temporal_index"] for row in layout["bins"]], [1, 2])

    def test_controls_match_query_key_event_and_causal_edge_budgets(self):
        candidate, records = fixture_pair()
        result = core.prepare_support_audit(candidate, candidate["archived_mapping"], records)
        control = result["knockout_controls"]["low_boundary"]["options_all"]["both_targets"]["background"]
        self.assertEqual(control["target_budget"], control["control_budget"])
        self.assertEqual(control["target_budget"]["visible_causal_edges_per_head"], 8)
        self.assertEqual(control["target_budget"]["all_head_visible_causal_edges_per_layer"], 256)
        self.assertEqual(control["key_counts_by_temporal_bin"], {"1": 2, "2": 2})
        optional = result["knockout_controls"]["low_boundary"]["options_all"]["both_targets"]["distractors"]
        self.assertFalse(optional["eligible"])
        self.assertEqual(optional["reason"], "insufficient_same_bin_control_keys")

    def test_background_objects_and_insufficient_budget_are_rejected(self):
        candidate, records = fixture_pair()
        records["low_boundary"]["background_event2_positions"].append(7)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            core.prepare_support_audit(candidate, candidate["archived_mapping"], records)
        records["low_boundary"]["background_event2_positions"] = [10, 16]
        with self.assertRaisesRegex(ValueError, "cannot match"):
            core.prepare_support_audit(candidate, candidate["archived_mapping"], records)

    def test_empty_future_and_all_blocked_edges_fail(self):
        for query, keys in (([], [1]), ([1], []), ([1], [2]), ([1], [0, 1])):
            with self.subTest(query=query, keys=keys), self.assertRaises(ValueError):
                core.edge_budget(query, keys, 10)

    def test_baseline_rejects_ties_nonfinite_and_prediction_disagreement(self):
        candidate, _ = fixture_pair()
        row = candidate["rows"]["low_boundary"]
        for margin in (0, float("nan"), float("inf"), 1):
            index = copy.deepcopy(candidate["capture_indices"]["low_boundary"])
            index["decision"]["margin"] = margin
            self.assertTrue(core.baseline_failures(row, index))

    def test_boundary_compression_keeps_the_two_changes_separate(self):
        hurt_temporal = core.boundary_outcomes(-1, 2, -1, 1)
        improved_low = core.boundary_outcomes(-1, 2, 0, 2)
        self.assertEqual(hurt_temporal["compression"], improved_low["compression"])
        self.assertEqual(hurt_temporal["delta_M_temporal"], -1)
        self.assertEqual(improved_low["delta_M_temporal"], 0)
        for item in (hurt_temporal, improved_low):
            self.assertEqual(item["compression"], item["delta_M_low"] - item["delta_M_temporal"])
        with self.assertRaises(ValueError):
            core.boundary_outcomes(-1, 2, float("nan"), 1)


class Phase3CArtifactTest(unittest.TestCase):
    def test_archive_audit_never_reads_patch_effects_and_remains_unfrozen(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _ = fixture_archive(directory)
            patch_results = source / "primary/analysis/patch_results.jsonl"
            patch_results.parent.mkdir()
            patch_results.write_text("invalid JSON deliberately: never read for selection")
            output = Path(directory) / "pilot"
            summary = prepare_phase3c.audit_archive(source, output)
            self.assertEqual(summary["archive_eligible_independent_bases"], 12)
            self.assertFalse(summary["case_ids_frozen"])
            self.assertFalse((output / "selection").exists())
            self.assertEqual(prepare_phase3c.audit_archive(source, output), summary)

    def test_source_and_output_trees_cannot_overlap(self):
        for source, output in (("/tmp/a", "/tmp/a"), ("/tmp/a", "/tmp/a/new"), ("/tmp/a/new", "/tmp/a")):
            with self.assertRaisesRegex(ValueError, "separate"):
                prepare_phase3c.safe_output(source, output)

    def test_explicit_real_provenance_and_revision_required(self):
        for key, value in (("artifact_type", None), ("artifact_type", "mock"), ("model_revision", "main")):
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as directory:
                source, _ = fixture_archive(directory)
                config = core.read_json(source / "vm_run_config.json")
                config[key] = value
                core.atomic_write(source / "vm_run_config.json", config)
                with self.assertRaises(ValueError):
                    prepare_phase3c.load_archive(source)

    def test_source_manifest_hash_and_capture_fingerprint_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _ = fixture_archive(directory)
            manifest = source / "selection/analysis_case_manifest.jsonl"
            manifest.write_text(manifest.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "manifest"):
                prepare_phase3c.load_archive(source)
        with tempfile.TemporaryDirectory() as directory:
            source, _ = fixture_archive(directory)
            path = next(source.glob("primary/checkpoints/*/activations/*/*/index.json"))
            index = core.read_json(path)
            index["run_fingerprint"] = "wrong"
            core.atomic_write(path, index)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                prepare_phase3c.load_archive(source)

    def test_baseline_failure_is_recorded_without_relabelling(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _ = fixture_archive(directory)
            path = source / "primary/checkpoints/shard_00/activations/phase3b_base_005_original/temporal_boundary/index.json"
            index = core.read_json(path)
            index["decision"]["margin"] = 0
            core.atomic_write(path, index)
            summary = prepare_phase3c.audit_archive(source, Path(directory) / "pilot")
            self.assertEqual(summary["archive_failures"], ["phase3b_base_005_original"])
            self.assertEqual(summary["preview_missing_quotas"], {"rescue_target_1_first": 1})

    def test_checkpoint_resume_and_independent_frozen_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            source, live = fixture_archive(directory)
            output = Path(directory) / "pilot"
            prepare_phase3c.audit_archive(source, output)
            def process(candidate, condition, *_args):
                return live[candidate["pair_id"]][condition]
            with patch.object(audit_phase3c_mappings, "processor_condition", side_effect=process) as mocked:
                first = audit_phase3c_mappings.run_audit(output, object(), RUNTIME, directory, {}, max_pairs=3)
                self.assertEqual(first["newly_audited"], 3)
                self.assertEqual(mocked.call_count, 6)
                with self.assertRaisesRegex(ValueError, "quotas"):
                    prepare_phase3c.freeze_selection(output)
                mocked.reset_mock()
                second = audit_phase3c_mappings.run_audit(output, object(), RUNTIME, directory, {})
                self.assertEqual(second["newly_audited"], 9)
                self.assertEqual(second["reused"], 3)
                self.assertEqual(mocked.call_count, 18)
            summary = prepare_phase3c.freeze_selection(output)
            self.assertEqual(summary["strata"], {"rescue": 8, "stable": 4})
            self.assertEqual(summary["evaluation_rows"], 24)
            self.assertFalse(summary["gpu_ready"])
            frozen_bytes = (output / "selection/frozen_config.json").read_bytes()
            self.assertEqual(prepare_phase3c.freeze_selection(output), summary)
            self.assertEqual((output / "selection/frozen_config.json").read_bytes(), frozen_bytes)

    def test_missing_processor_audit_blocks_freeze_with_clear_message(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _ = fixture_archive(directory)
            output = Path(directory) / "pilot"
            prepare_phase3c.audit_archive(source, output)
            with self.assertRaisesRegex(ValueError, "Processor-only audit is missing"):
                prepare_phase3c.freeze_selection(output)

    def test_failed_audits_are_not_retried_without_explicit_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            source, live = fixture_archive(directory)
            output = Path(directory) / "pilot"
            prepare_phase3c.audit_archive(source, output)
            with patch.object(audit_phase3c_mappings, "processor_condition", side_effect=ValueError("mapping unavailable")):
                first = audit_phase3c_mappings.run_audit(output, object(), RUNTIME, directory, {}, max_pairs=1)
            self.assertEqual(first["eligible_count"], 0)
            with patch.object(audit_phase3c_mappings, "processor_condition", side_effect=lambda candidate, condition, *_:
                              live[candidate["pair_id"]][condition]) as mocked:
                second = audit_phase3c_mappings.run_audit(output, object(), RUNTIME, directory, {}, max_pairs=1)
                self.assertEqual(second["reused"], 1)
                self.assertEqual(mocked.call_count, 2)
                mocked.reset_mock()
                third = audit_phase3c_mappings.run_audit(output, object(), RUNTIME, directory, {},
                                                       max_pairs=1, retry_failed=True)
                self.assertEqual(third["eligible_count"], 2)
                self.assertEqual(mocked.call_count, 2)

    def test_helper_code_changes_invalidate_the_preparation_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _ = fixture_archive(directory)
            output = Path(directory) / "pilot"
            prepare_phase3c.audit_archive(source, output)
            with patch.object(audit_phase3c_mappings, "file_hash", side_effect=lambda path:
                              "changed" if Path(path).name == "phase3b_core.py" else core.file_hash(path)):
                with self.assertRaisesRegex(ValueError, "Preparation code changed"):
                    audit_phase3c_mappings.validate_plan(output)

    def test_unaudited_earlier_candidates_prevent_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            source, live = fixture_archive(directory, pairs=14)
            output = Path(directory) / "pilot"
            prepare_phase3c.audit_archive(source, output)
            with patch.object(audit_phase3c_mappings, "processor_condition", side_effect=lambda candidate, condition, *_:
                              live[candidate["pair_id"]][condition]):
                audit_phase3c_mappings.run_audit(output, object(), RUNTIME, directory, {})
                # Run the later candidates as well, then remove one earlier audit record.
                candidates = core.read_jsonl(output / "candidate_manifest.jsonl")
                path = output / "processor_audit/mapping_audit.jsonl"
                rows = core.read_jsonl(path)
                for candidate in candidates[12:]:
                    processed = live[candidate["pair_id"]]
                    rows.append({"schema": core.SCHEMA, "pair_id": candidate["pair_id"],
                        "plan_fingerprint": rows[0]["plan_fingerprint"], "candidate_sha256": core.digest(candidate),
                        "eligible": True, "processor_records": processed,
                        "support_audit": core.prepare_support_audit(candidate, candidate["archived_mapping"], processed)})
                core.atomic_write(path, [row for row in rows if row["pair_id"] != "phase3b_base_005_original"], jsonl=True)
            with self.assertRaisesRegex(ValueError, "Earlier-ranked"):
                prepare_phase3c.freeze_selection(output)

    def test_changed_video_and_tampered_support_prevent_freeze(self):
        for mode in ("video", "support", "candidate"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                source, live = fixture_archive(directory)
                output = Path(directory) / "pilot"
                prepare_phase3c.audit_archive(source, output)
                with patch.object(audit_phase3c_mappings, "processor_condition", side_effect=lambda candidate, condition, *_:
                                  live[candidate["pair_id"]][condition]):
                    audit_phase3c_mappings.run_audit(output, object(), RUNTIME, directory, {})
                if mode == "video":
                    Path(live["phase3b_base_001_original"]["low_boundary"]["video_path"]).write_bytes(b"changed")
                elif mode == "support":
                    path = output / "processor_audit/mapping_audit.jsonl"
                    rows = core.read_jsonl(path)
                    rows[0]["support_audit"]["supports"]["whole_event2"]["mapped_token_count"] = 99
                    core.atomic_write(path, rows, jsonl=True)
                else:
                    path = output / "candidate_manifest.jsonl"
                    rows = core.read_jsonl(path)
                    rows[0]["first_object_id"] = 2
                    core.atomic_write(path, rows, jsonl=True)
                with self.assertRaises(ValueError):
                    prepare_phase3c.freeze_selection(output)
                self.assertFalse((output / "selection").exists())

    def test_changed_frozen_artifact_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            core.frozen_write(path, {"id": 1})
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "Frozen"):
                core.frozen_write(path, {"id": 2})
            self.assertEqual(path.read_bytes(), before)

    def test_cpu_entry_points_import_without_torch_or_transformers(self):
        result = subprocess.run([sys.executable, "-c",
            "import sys; from scripts import prepare_phase3c, audit_phase3c_mappings; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_processor_loader_hides_gpus_and_loads_no_model_weights(self):
        architecture = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=36, num_attention_heads=32),
                                       vision_config=SimpleNamespace(deepstack_visual_indexes=[8, 16, 24]),
                                       video_token_id=151656)
        processor = object()
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}), patch.dict(sys.modules, {
            "torch": SimpleNamespace(__version__="2.11.0+cpu"),
            "transformers": SimpleNamespace(__version__="5.9.0",
                AutoConfig=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: architecture),
                AutoProcessor=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: processor)),
        }), patch.object(audit_phase3c_mappings, "version", return_value="0.0.14"):
            actual, runtime = audit_phase3c_mappings.load_processor(prepare_phase3c.DEFAULT_SETTINGS)
            self.assertIs(actual, processor)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
            self.assertFalse(runtime["model_weights_loaded"])
            self.assertEqual(runtime["deepstack_decoder_injection_layers"], [0, 1, 2])

    def test_video_path_overrides_preserve_other_archived_mappings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paths.json"
            core.atomic_write(path, {"/old/pool": "/new/pool"})
            with patch.object(sys, "argv", ["audit_phase3c_mappings.py", "--plan_dir", directory,
                                           "--path_map", str(path)]), \
                    patch.object(audit_phase3c_mappings, "validate_plan", return_value={
                        "source_run_root": directory, "settings": prepare_phase3c.DEFAULT_SETTINGS}), \
                    patch.object(audit_phase3c_mappings, "read_json", return_value={"path_map": {
                        "/old/pool": "/archived/pool", "/old/controls": "/archived/controls"}}), \
                    patch.object(audit_phase3c_mappings, "load_processor", return_value=(object(), RUNTIME)), \
                    patch.object(audit_phase3c_mappings, "run_audit", return_value={"missing_quotas": {}}) as run:
                audit_phase3c_mappings.main()
                self.assertEqual(run.call_args.args[4], {
                    "/old/pool": "/new/pool", "/old/controls": "/archived/controls"})

    def test_video_path_override_rejects_relative_prefixes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paths.json"
            core.atomic_write(path, {"relative/pool": "/new/pool"})
            with self.assertRaisesRegex(ValueError, "absolute"):
                audit_phase3c_mappings.load_path_map(path)


if __name__ == "__main__":
    unittest.main()
