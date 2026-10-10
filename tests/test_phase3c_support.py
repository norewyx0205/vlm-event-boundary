import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import phase3c_support as support
from scripts import run_phase3c_support as runner
from scripts.phase3c_core import CONDITIONS, atomic_write, digest, support_record


def fixture():
    rows, records = {}, {}
    for condition, start, offset in zip(CONDITIONS, (10, 20), (100, 140)):
        metadata = {"merged_video_grid_thw": [2, 2, 4], "source_frame_groups": [[start, start + 1], [start + 2, start + 3]],
                    "visual_token_count": 16}
        rows[condition] = {"event_timing": {"first_event_start_frame": 0, "first_event_end_frame": 5,
            "second_event_start_frame": start, "second_event_end_frame": start + 10}, "total_frames": 40,
            "eval_id": condition, "phase3c_analysis_stratum": "rescue", "base_sample_id": 1,
            "first_object_id": 2, "prompt_variant": "original", "correct_option": "A"}
        records[condition] = {"video_metadata": metadata, "visual_positions": list(range(offset, offset + 16)),
                              "prompt_token_count": 200}
    low, temporal = list(range(100, 116)), list(range(140, 156))
    target_low, target_temporal = [100, 108, 109], [140, 148, 149]
    mapping = {"processor_records": records, "support_audit": {"supports": {
        "whole_event2": support_record(low, temporal, low, temporal, "whole"),
        "both_targets_event2": support_record(target_low, target_temporal, target_low, target_temporal, "targets")},
        "event_bin_pairs": [{"low_temporal_index": 0, "temporal_temporal_index": 0},
                            {"low_temporal_index": 1, "temporal_temporal_index": 1}]}}
    return rows, mapping


class SupportAuditTest(unittest.TestCase):
    def test_partition_and_budget_match_both_directions(self):
        pair, mapping = fixture()
        original = copy.deepcopy(mapping)
        audit = support.audit_supports("case_1", pair, mapping)
        self.assertEqual(mapping, original)
        self.assertTrue(audit["partition_exact"])
        self.assertEqual((audit["whole_count"], audit["target_union_count"], audit["complement_count"]), (16, 3, 13))
        self.assertEqual(audit["matched_control_count"], 3)
        for row in audit["token_budget_by_bin"]:
            self.assertEqual(row["target_union_count"], row["control_count"])
        for side in ("low_positions", "temporal_positions"):
            whole = set(mapping["support_audit"]["supports"]["whole_event2"][side])
            target = set(mapping["support_audit"]["supports"]["both_targets_event2"][side])
            complement = set(audit["supports"][support.SUPPORTS[0]][side])
            control = set(audit["supports"][support.SUPPORTS[1]][side])
            self.assertFalse(target & complement)
            self.assertEqual(target | complement, whole)
            self.assertLessEqual(control, complement)
        for item in audit["supports"].values():
            self.assertEqual(len(item["low_positions"]), len(item["temporal_positions"]))
            self.assertTrue(all(row["recipient_coverage"] == 1 for row in item["directions"].values()))

    def test_selection_is_deterministic_and_not_margin_dependent(self):
        pair, mapping = fixture()
        before = support.audit_supports("case_1", pair, mapping)
        pair[CONDITIONS[0]]["margin"] = -100
        pair[CONDITIONS[1]]["margin"] = 100
        self.assertEqual(before, support.audit_supports("case_1", pair, mapping))

    def test_same_cells_are_matched_in_every_control_pair(self):
        pair, mapping = fixture()
        audit = support.audit_supports("case_1", pair, mapping)
        control = audit["supports"][support.SUPPORTS[1]]
        self.assertEqual([position - 100 for position in control["low_positions"]],
                         [position - 140 for position in control["temporal_positions"]])

    def test_reordered_target_correspondence_is_rejected(self):
        pair, mapping = fixture()
        mapping["support_audit"]["supports"]["both_targets_event2"]["temporal_positions"].reverse()
        with self.assertRaisesRegex(ValueError, "aligned subset"):
            support.audit_supports("case_1", pair, mapping)

    def test_insufficient_budget_pool_is_not_silently_relaxed(self):
        pair, mapping = fixture()
        low, temporal = list(range(100, 107)), list(range(140, 147))
        mapping["support_audit"]["supports"]["both_targets_event2"] = support_record(low, temporal, low, temporal, "targets")
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            support.audit_supports("case_1", pair, mapping)

    def test_wrong_event_relative_mapping_rejected(self):
        pair, mapping = fixture()
        whole = mapping["support_audit"]["supports"]["whole_event2"]
        whole["temporal_positions"] = whole["temporal_positions"][8:] + whole["temporal_positions"][:8]
        target = mapping["support_audit"]["supports"]["both_targets_event2"]
        target["temporal_positions"] = [148, 140, 141]
        with self.assertRaisesRegex(ValueError, "temporal-bin"):
            support.audit_supports("case_1", pair, mapping)

    def test_incomplete_whole_grid_rejected(self):
        pair, mapping = fixture()
        whole = mapping["support_audit"]["supports"]["whole_event2"]
        whole["low_positions"].pop()
        whole["temporal_positions"].pop()
        with self.assertRaisesRegex(ValueError, "complete audited grid"):
            support.audit_supports("case_1", pair, mapping)

    def test_fixed_unique_grid_does_not_repeat_old_supports(self):
        config = {"case_ids": [f"case_{i}" for i in range(12)], "preflight_case_ids": ["case_1", "case_2"]}
        for stage, count in (("patch", 288), ("preflight", 48)):
            tasks = support.task_grid(config, stage)
            self.assertEqual(len(tasks), count)
            self.assertEqual(len({row["task_id"] for row in tasks}), count)
            self.assertEqual({row["support"] for row in tasks}, set(support.SUPPORTS))
            self.assertEqual({row["layer"] for row in tasks}, {0, 4, 8, 12, 16, 20})


def result_fixture():
    pair, mapping = fixture()
    config = {"fingerprint": "frozen", "case_ids": ["case_1"], "preflight_case_ids": ["case_1"],
              "support_audit": {"case_1": support.audit_supports("case_1", pair, mapping)}}
    task = support.task_grid(config, "preflight")[0]
    before = {"margin": -1.0, "correct_logit": 0.0, "incorrect_logit": 1.0, "prediction": "B"}
    inputs = {"input_ids": {"sha256": "original"}}
    baselines = {condition: {"decision": before.copy(), "input_tensor_sha256": inputs} for condition in CONDITIONS}
    selected = config["support_audit"]["case_1"]["supports"][task["support"]]
    result = {"task_id": task["task_id"], "spec": task, "support_fingerprint": "frozen", "passed": True,
        "is_primary_effect_estimate": False, "baseline_decision": before, "decision": before.copy(),
        "hook_audit": {"block_output_sites": 36, "post_deepstack_layers": [0, 1, 2], "patch_applied_count": 1},
        "donor_capture_location": "block_output", "recipient_patch_location": "block_output",
        "noop_parity": {"exact_match": True, "max_abs_diff": 0}, "support_audit": selected,
        "donor_positions": selected["low_positions"], "recipient_positions": selected["low_positions"],
        "donor_condition": CONDITIONS[0], "recipient_condition": CONDITIONS[0], "input_tensor_sha256": inputs}
    return config, {"case_1": pair}, baselines, task, result


class SupportVerificationTest(unittest.TestCase):
    def test_actual_identity_control_validates(self):
        config, pairs, baselines, task, result = result_fixture()
        self.assertEqual(support.result_failures(result, task, config, pairs, baselines), [])

    def test_foreign_positions_and_changed_baselines_rejected(self):
        for field, value in (("recipient_positions", [1]), ("baseline_decision", {}),
                             ("input_tensor_sha256", {}), ("support_fingerprint", "other")):
            with self.subTest(field=field):
                config, pairs, baselines, task, result = result_fixture()
                result[field] = value
                self.assertTrue(support.result_failures(result, task, config, pairs, baselines))

    def test_nonexact_identity_rejected(self):
        config, pairs, baselines, task, result = result_fixture()
        result["noop_parity"] = {"exact_match": False, "max_abs_diff": 0.001}
        self.assertIn("failed_exact_noop", support.result_failures(result, task, config, pairs, baselines))

    def test_incomplete_checkpoint_cannot_establish_gate(self):
        config, pairs, baselines, _, _ = result_fixture()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                runner.saved_rows(root, "preflight", config, pairs, baselines, complete=True)

    def test_duplicate_foreign_checkpoint_rejected(self):
        config, pairs, baselines, _, result = result_fixture()
        with tempfile.TemporaryDirectory() as root:
            atomic_write(Path(root) / "preflight/task_checkpoints/foreign.json", result)
            with self.assertRaisesRegex(ValueError, "Foreign"):
                runner.saved_rows(root, "preflight", config, pairs, baselines)

    def test_passed_flag_does_not_override_corrupt_evidence(self):
        config, pairs, baselines, task, result = result_fixture()
        result["recipient_positions"] = [1]
        with tempfile.TemporaryDirectory() as root:
            atomic_write(Path(root) / "preflight/task_checkpoints" / f"{task['task_id']}.json", result)
            with self.assertRaisesRegex(ValueError, "claimed-passed"):
                runner.saved_rows(root, "preflight", config, pairs, baselines)

    def test_changed_code_binding_rejected(self):
        config = {"schema": support.SCHEMA, "artifact_type": "real", "followup_code_sha256": {"old": "code"}}
        config["fingerprint"] = digest(config)
        with self.assertRaisesRegex(ValueError, "binding"):
            support.validate_config(config)

    def test_complete_resume_does_not_load_weights(self):
        config, pairs, baselines, task, result = result_fixture()
        with tempfile.TemporaryDirectory() as root:
            atomic_write(Path(root) / "preflight/task_checkpoints" / f"{task['task_id']}.json", result)
            args = argparse_namespace(output_root=root, stage="preflight", retry_failed=False)
            with patch.object(runner, "task_grid", return_value=[task]), patch.object(runner, "Engine") as engine:
                summary = runner.run_stage(args, (config, {}, pairs, {}, baselines))
                self.assertTrue(summary["complete"])
                engine.assert_not_called()


class SupportLifecycleTest(unittest.TestCase):
    def test_other_support_worker_blocks_pause(self):
        from scripts import run_phase3c_unattended as idle
        process = argparse_namespace(stdout="999999 python scripts/run_phase3c_support.py --action run")
        with patch.object(idle, "legacy_idle"), patch.object(idle.subprocess, "run", return_value=process), \
             patch.object(idle.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "active"):
                runner.assert_quiescent()

    def run_job(self, root, child_code=0, stale=False, backup_fails=False):
        output = Path(root) / "followup"
        lifecycle = Path(str(output) + "_lifecycle")
        lifecycle.mkdir()
        (lifecycle / "job.log").write_text("public log", encoding="ascii")
        config = {"fingerprint": "frozen", "source_backup_manifest_sha256": "sourcehash"}
        args = argparse_namespace(output_root=str(output), source_backup="source_backup",
            backup_dir=str(Path(root) / "backups"), pause_after=True, retry_failed=False,
            project_root="repository", plan_dir="plan", source_run="source", storage_root=root,
            gpus="0,1", email_notify=True)
        order = []
        notifier, client = Mock(), Mock()
        notifier.send.side_effect = lambda *_: order.append("email") or {"smtp_accepted": True}
        client.request_pause.side_effect = lambda: order.append("pause") or {"request_accepted": True}

        def child(*_):
            order.append("child")
            if child_code == 0 and not stale:
                atomic_write(output / "last_attempt.json", {"attempt_id": args.attempt_id,
                    "complete": True, "support_fingerprint": "frozen"})
            return child_code

        def backup(*_, **__):
            order.append("backup")
            if backup_fails:
                raise RuntimeError("no space")
            return Path(root) / "package"

        def verify(*_):
            order.append("verify_backup")
            return {"complete": True, "reports_only": False, "support_fingerprint": "frozen"}

        with patch.object(runner, "context", return_value=(config, {}, {}, {}, {})), \
             patch.object(runner, "saved_rows"), patch.object(runner, "private_services", return_value=(notifier, client)), \
             patch.object(runner, "assert_quiescent", side_effect=lambda: order.append("idle")), \
             patch.object(runner, "run_child", side_effect=child), \
             patch.object(runner, "verify_analysis"), patch.object(runner, "create_backup", side_effect=backup), \
             patch.object(runner, "verify_backup", side_effect=verify):
            code = runner.job(args)
        return code, order, read_status(lifecycle)

    def test_success_notifies_after_verified_backup_before_pause(self):
        with tempfile.TemporaryDirectory() as root:
            code, order, status = self.run_job(root)
            self.assertEqual(code, 0)
            self.assertEqual(order, ["idle", "child", "backup", "verify_backup", "idle", "email", "pause"])
            self.assertTrue(status["experiment_success"])
            self.assertEqual(status["state"], "pause_requested")

    def test_experiment_failure_still_backs_up_before_pause(self):
        with tempfile.TemporaryDirectory() as root:
            code, order, status = self.run_job(root, child_code=1)
            self.assertEqual(code, 1)
            self.assertFalse(status["experiment_success"])
            self.assertLess(order.index("verify_backup"), order.index("pause"))

    def test_stale_success_report_does_not_establish_new_success(self):
        with tempfile.TemporaryDirectory() as root:
            code, _, status = self.run_job(root, stale=True)
            self.assertEqual(code, 1)
            self.assertFalse(status["experiment_success"])
            self.assertIn("verification_error_type", status)

    def test_backup_failure_blocks_pause(self):
        with tempfile.TemporaryDirectory() as root:
            code, order, status = self.run_job(root, backup_fails=True)
            self.assertEqual(code, 1)
            self.assertNotIn("pause", order)
            self.assertFalse(status["backup_verified"])
            self.assertEqual(status["state"], "needs_attention")


def read_status(lifecycle):
    import json
    return json.loads((lifecycle / "job_status.json").read_text())


def argparse_namespace(**fields):
    from types import SimpleNamespace
    return SimpleNamespace(**fields)


if __name__ == "__main__":
    unittest.main()
