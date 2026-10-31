import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.phase3b_core import build_text_groups
from scripts.phase3c_core import read_json
from scripts.phase3e_option_features import (CompactOptionSites, FEATURE_NAMES, archived_readouts,
    audit_paired_readouts, option_readouts, statement_features)
from scripts.phase3e_option_probes import (control_features, crossfit, paired_analysis, validate_folds)
from scripts.phase3e_option_store import load_store, write_store, GATES
from scripts.phase3e_paired_design import build_plan, group_folds, labels_and_pairs
from scripts.phase3e_preflight import factorial_rows
from scripts.run_phase3e_option_probes import verify_contrast_projection, verify_job, write_job
from scripts import run_phase3e_paired_capture as capture_runner
from tests.test_phase3d_features import CharacterTokenizer
from tests.test_phase3e_preflight import Text


def token_mapping(row):
    tokenizer = CharacterTokenizer()
    text = ("system\nWhich statement correctly describes the order of events?\n\n"
            f"A: {row['option_A']}\nB: {row['option_B']}\n\nAnswer with only A or B.")
    ids = tokenizer(text)["input_ids"]
    record = {"input_ids": ids, "group_positions": build_text_groups(row, ids, tokenizer)}
    groups, audit = option_readouts(row, record, tokenizer)
    return record, groups, audit


class OptionReadoutTest(unittest.TestCase):
    def test_contextual_reference_not_cross_object_and_paired_alignment(self):
        _, rows = factorial_rows("/tmp/unused_options", [1])
        mappings = {}
        for row in rows:
            record, groups, audit = token_mapping(row)
            for option in ("A", "B"):
                reference = 3-row["subject_id"]
                label = next(o["label"] for o in row["target_objects"] if o["id"] == reference)
                reconstructed = "".join(chr(record["input_ids"][p]-256) for p in audit[option]["reference_positions"])
                self.assertEqual(reconstructed.casefold(), label.casefold())
                self.assertEqual(groups[f"option_{option}_reference_last"], [max(audit[option]["reference_positions"])])
                self.assertLess(max(audit[option]["relation_positions"]), min(audit[option]["reference_positions"]))
            mappings[row["eval_id"]] = {"option_readout_audit": audit}
        self.assertTrue(audit_paired_readouts(rows, mappings)["same_contextual_reference_tokens_and_positions"])
        changed = copy.deepcopy(mappings)
        changed[rows[0]["eval_id"]]["option_readout_audit"]["A"]["reference_last_position"] += 1
        with self.assertRaisesRegex(ValueError, "alignment"):
            audit_paired_readouts(rows, changed)

    def test_archived_option_sections_agree_with_actual_tokenizer(self):
        from scripts.phase3d_features import option_spans
        _, rows = factorial_rows("/tmp/unused_options", [1])
        row = rows[0]
        record, groups, audit = token_mapping(row)
        original = {"processor_record": record, "capture_groups": option_spans(
            row, record["input_ids"], CharacterTokenizer(), record["group_positions"])}
        self.assertEqual(archived_readouts(row, original), (groups, audit))
        altered = copy.deepcopy(record)
        altered["group_positions"]["options_all"].pop()
        with self.assertRaises(ValueError):
            option_readouts(row, altered, CharacterTokenizer())

    def test_compact_controller_is_noop_timing_checked_and_retains_only_four_raw_tokens(self):
        model = Text()
        groups = {f"option_{o}_{k}": [p] for o, p in (("A", 3), ("B", 5))
                  for k in ("reference_last", "reference_mean", "subject_mean")}
        groups.update(option_A_option_final=[4], option_B_option_final=[6], whole_event2=[0, 1, 2])
        inputs = torch.randn(1, 7, 4, dtype=torch.float16)
        plain = model(inputs)
        collector = CompactOptionSites(model, 7, groups)
        with collector.installed():
            observed = model(inputs)
        self.assertTrue(torch.equal(plain, observed))
        payload = collector.payload()
        self.assertEqual(payload["union_positions"], [3, 4, 5, 6])
        self.assertEqual(statement_features(payload, "A").shape, (len(FEATURE_NAMES), 4))
        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()))
        collector.raw["L17_attention_contribution"][0, 0] += 1
        with self.assertRaisesRegex(RuntimeError, "residual"):
            collector.validate()


class ProbeTest(unittest.TestCase):
    def setUp(self):
        self.seeds = list(range(10))
        _, self.rows = factorial_rows("/tmp/unused_options", self.seeds)
        self.examples = labels_and_pairs(self.rows)[0]
        self.folds = group_folds(self.seeds)
        self.settings = {"C_grid": [1.0], "bootstrap_replicates": 20}

    def test_nested_grouping_and_train_only_scaler(self):
        rng = np.random.default_rng(9)
        values = rng.normal(size=(len(self.examples), 4))
        values[:, 0] = [10*(2*r["y_statement_true"]-1) for r in self.examples]
        fitted = crossfit(self.examples, values, self.folds, self.settings)
        self.assertEqual(len(fitted["predictions"]), len(self.examples))
        for item, fold in zip(fitted["models"], self.folds):
            idx = [i for i, row in enumerate(self.examples) if row["scene_seed"] in fold["train_scene_seeds"]]
            np.testing.assert_allclose(item["model"].named_steps["scale"].mean_, values[idx].mean(0), atol=1e-14)
        changed = copy.deepcopy(self.folds)
        changed[0]["inner_folds"][0]["train_scene_seeds"].append(changed[0]["test_scene_seeds"][0])
        with self.assertRaisesRegex(ValueError, "leak"):
            validate_folds(self.examples, changed)

    def test_paired_scores_keep_original_levels_and_do_not_double_independent_sample_count(self):
        predictions = []
        for row in self.examples:
            magnitude = 3 if row["condition"] == "temporal_boundary" else 2
            z = magnitude*(2*row["y_statement_true"]-1)
            predictions.append({**row, "model_id": "same_frozen_fold", "probe_logit": z,
                "true_statement_evidence": magnitude})
        result = paired_analysis(predictions, self.rows, self.settings)
        boundary = result["boundary_summary"]
        self.assertEqual(boundary["evidence_low"]["mean"], 2)
        self.assertEqual(boundary["evidence_temporal"]["mean"], 3)
        self.assertEqual(boundary["evidence_temporal_minus_low"]["mean"], 1)
        self.assertEqual(boundary["evidence_temporal_minus_low"]["base_count"], 10)
        self.assertEqual(result["difference_in_differences"]["difference_in_differences"]["mean"], 2)
        changed = copy.deepcopy(predictions)
        changed[0]["model_id"] = "different_model"
        with self.assertRaisesRegex(ValueError, "different"):
            paired_analysis(changed, self.rows)

    def test_controls_are_label_balanced_for_identical_features(self):
        from collections import defaultdict, Counter
        from scripts.phase3c_core import digest
        for kind in ("lexical_only", "static_identity"):
            values = control_features(self.examples, self.rows, kind)
            buckets = defaultdict(Counter)
            for row, value in zip(self.examples, values):
                buckets[digest(value)][row["y_statement_true"]] += 1
            self.assertTrue(all(count[0] == count[1] for count in buckets.values()))
            fitted = crossfit(self.examples, values, self.folds, self.settings, categorical=True)
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_job(root, fitted, self.examples, self.rows)
                self.assertEqual(len(verify_job(root, self.examples, values, self.folds, True)), len(self.examples))

    def test_frozen_numeric_replay_and_tamper_rejection(self):
        values = np.asarray([[r["y_statement_true"], r["scene_seed"]] for r in self.examples], dtype=float)
        fitted = crossfit(self.examples, values, self.folds, self.settings)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_job(root, fitted, self.examples, self.rows)
            predictions = verify_job(root, self.examples, values, self.folds, False)
            paired = paired_analysis(predictions, self.rows, self.settings)
            self.assertEqual(verify_contrast_projection(root, self.examples, values, predictions, paired)[
                "order_and_boundary_contrasts_checked"], len(paired["order_pairs"])+len(paired["boundary_pairs"]))
            with (root / "outer_0.npz").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_job(root, self.examples, values, self.folds, False)

    def test_complete_store_shape_checksum_and_partial_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = build_plan(Path(directory) / "materials")
            examples = plan["statement_examples"]
            values = np.ones((len(examples), len(FEATURE_NAMES), 2), dtype=np.float32)
            gates = {**{name: True for name in GATES}, "prompt_cells": 640}
            decisions = {r["eval_id"]: {"correct_option": r["correct_option"], "prediction": None,
                "is_correct": False, "correct_logit": 1.0, "incorrect_logit": 1.0, "margin": 0.0}
                for r in plan["rows"]}
            root = Path(directory) / "store"
            write_store(root, plan, examples, values, decisions, gates, {"test_fixture": True})
            loaded = load_store(root / "manifest.json")
            np.testing.assert_array_equal(loaded[3], values)
            with self.assertRaisesRegex(ValueError, "immutable"):
                write_store(root, plan, examples, values, decisions, gates, {})
            with (root / "features.npz").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaises(ValueError):
                load_store(root / "manifest.json")
            smoke = Path(directory) / "smoke.json"
            from scripts.phase3c_core import atomic_write
            atomic_write(smoke, {"schema": "phase3e_option_export_smoke_v1", "complete": False})
            with self.assertRaisesRegex(ValueError, "not smoke"):
                load_store(smoke)

    def test_full_cpu_training_cli_and_independent_verification_on_synthetic_fixture(self):
        from scripts import phase3e_option_probes as probes
        from scripts.run_phase3e_option_probes import run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan(root / "materials")
            examples = plan["statement_examples"]
            values = np.zeros((1280, len(FEATURE_NAMES), 2), dtype=np.float32)
            values[:, :, 0] = np.asarray([2*r["y_statement_true"]-1 for r in examples])[:, None]
            values[:, :, 1] = np.asarray([r["subject_id"] for r in examples])[:, None]
            decisions = {r["eval_id"]: {"correct_option": r["correct_option"], "prediction": r["correct_option"],
                "is_correct": True, "correct_logit": 2., "incorrect_logit": 1., "margin": 1.}
                for r in plan["rows"]}
            write_store(root / "store", plan, examples, values, decisions,
                {**{name: True for name in GATES}, "prompt_cells": 640}, {"synthetic_unit_test_fixture": True})
            with patch.dict(probes.SETTINGS, self.settings):
                summary = run(root / "store/manifest.json", root / "probes")
                self.assertEqual(summary, run(root / "store/manifest.json", root / "probes", verify_only=True))
            self.assertEqual(summary["fitted_fold_models"], 65)
            self.assertEqual(summary["results"]["reference_last:L18_block_output"]["metrics"][0]["accuracy"], 1.)
            self.assertIn("Low evidence", (root / "probes/report.md").read_text())


class CaptureRunnerTest(unittest.TestCase):
    def test_sampling_contract_and_persistent_volume_boundary(self):
        budget = {"visual_tokens_per_prompt": 4608, "whole_event2_positions_per_prompt": 512,
                  "video_tensor_parity_across_text_views": True}
        self.assertEqual(capture_runner.budget_contract(budget), budget)
        with self.assertRaisesRegex(ValueError, "sampling"):
            capture_runner.budget_contract({**budget, "whole_event2_positions_per_prompt": 256})
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(capture_runner.os.path, "ismount", return_value=True):
                capture_runner.persistent_volume(Path(directory) / "run", directory)
                with self.assertRaisesRegex(ValueError, "persistent"):
                    capture_runner.persistent_volume(Path(directory).parent, directory)

    def test_raw_checkpoint_replay_then_tamper_rejection(self):
        from scripts.phase3c_core import atomic_write, digest, file_hash
        from scripts.run_phase3c_preflight import save_tensors
        _, rows = factorial_rows("/tmp/unused_options", [1])
        row = rows[0]
        model = Text()
        groups = {f"option_{o}_{kind}": [p] for o, p in (("A", 3), ("B", 5))
                  for kind in ("reference_last", "reference_mean", "subject_mean", "option_final")}
        groups.update(whole_event2=[0, 1], source_t1=[0], source_t2=[1])
        controller = CompactOptionSites(model, 7, groups)
        with controller.installed():
            model(torch.randn(1, 7, 4, dtype=torch.float16))
        payload = controller.payload()
        config = {"test_fixture": True}
        mapping = {"capture_groups": groups}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "captures/test.pt"
            logits = torch.tensor([2., 1.])
            save_tensors(path, {**payload, "eval_id": row["eval_id"], "config_sha256": digest(config),
                "baseline_logits": logits, "capture_logits": logits, "standard_logits": logits})
            record = {"eval_id": row["eval_id"], "config_sha256": digest(config),
                "mapping_sha256": digest(mapping), "tensor_path": "captures/test.pt", "tensor_sha256": file_hash(path),
                "timing_audit": payload["audit"], "decision": {"correct_token_id": 0, "incorrect_token_id": 1,
                    "correct_logit": 2., "incorrect_logit": 1., "margin": 1., "correct_option": row["correct_option"],
                    "predicted_first_token_id": 0, "prediction": row["correct_option"], "is_correct": True}}
            record["record_sha256"] = digest(record)
            atomic_write(root / "checkpoints" / (row["eval_id"]+".json"), record)
            self.assertEqual(capture_runner.verify_row(root, row, config, mapping)[0], record)
            changed = copy.deepcopy(mapping)
            changed["capture_groups"]["whole_event2"].append(2)
            with self.assertRaisesRegex(ValueError, "provenance"):
                capture_runner.verify_row(root, row, config, changed)
            with path.open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "hash"):
                capture_runner.verify_row(root, row, config, mapping)

    def test_full_capture_cannot_load_model_before_preflight_gate(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(capture_runner, "space"), \
                patch.object(capture_runner, "load_plan", return_value={"scene_seeds": [1, 2], "rows": []}), \
                patch.object(capture_runner, "config_for", return_value={"settings": {}}), \
                patch.object(capture_runner, "check_videos"), \
                patch.object(capture_runner, "verify", side_effect=ValueError("missing preflight")), \
                patch("scripts.run_phase3c_preflight.Engine") as engine:
            with self.assertRaisesRegex(ValueError, "preflight"):
                capture_runner.capture(directory, "unused", preflight=False)
            engine.assert_not_called()

    def test_storage_projection_backup_reserve_and_public_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "captures").mkdir()
            for i in range(32):
                (root / "captures" / f"{i}.pt").touch()
            (root / "processor_audit_preflight.json").write_text("{}")
            (root / ".phase3e_option.lock").touch()
            projection = capture_runner.storage_projection(root)
            self.assertEqual(projection["reserve_bytes"], 12*1024**3)
            self.assertGreater(projection["projected_remaining_and_backup_bytes"], projection["projected_run_bytes"])
            self.assertTrue(capture_runner.public_files(root))
            (root / "email.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Unexpected"):
                capture_runner.public_files(root)
            with patch.object(capture_runner.shutil, "disk_usage") as disk:
                disk.return_value.free = 1
                with self.assertRaisesRegex(RuntimeError, "Insufficient"):
                    capture_runner.space(root)

    def test_finish_refuses_pause_with_remaining_gpu_process(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("scripts.backup_phase3b.verify_backup", return_value={"schema": "phase3e_paired_option_backup_v1",
                    "source_run_root": str(Path(directory).resolve())}), \
                patch.object(capture_runner, "verify", return_value={}), \
                patch("scripts.run_phase3e_paired_unattended.assert_idle", side_effect=RuntimeError("GPU compute remains active")), \
                patch("scripts.surf_workspace.SurfClient") as surf, \
                patch("scripts.phase3b_email.EmailNotifier") as email:
            with self.assertRaisesRegex(RuntimeError, "GPU compute"):
                capture_runner.finish(directory, "unused")
            surf.assert_not_called()
            email.assert_not_called()


if __name__ == "__main__":
    unittest.main()
