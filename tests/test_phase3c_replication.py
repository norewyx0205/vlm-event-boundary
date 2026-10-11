import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import analyze_phase3c_footprint as footprint
from scripts import phase3c_replication as protocol
from scripts import run_phase3c_replication as runner
from scripts.phase3c_core import CONDITIONS, atomic_write, digest, file_hash, read_json, support_record
from tests.test_phase3c_budget import big_fixture


def expanded_fixture():
    pair, mapping = big_fixture()
    mapping["support_audit"]["supports"].update(
        video_t1_e2=support_record([100, 356], [700, 956], [100, 356], [700, 956], "single_roi_reference"),
        video_t2_e2=support_record([357], [957], [357], [957], "single_roi_reference"))
    candidate = {"pair_id": "phase3b_base_001_original", "base_sample_id": 1, "rows": pair,
                 "stratum": "rescue", "first_object_id": 2, "prompt_variant": "original"}
    return candidate, mapping


def cohort_fixture():
    plan = {"plan_fingerprint": "plan", "excluded_base_ids": [99], "selection_rule": "ascending unused IDs"}
    candidates, mappings, live = [], {}, {}
    base = 0
    for stratum in ("rescue", "stable"):
        for mover in (1, 2):
            for _ in range(4 if stratum == "rescue" else 2):
                base += 1
                candidate, mapping = expanded_fixture()
                candidate.update(pair_id=f"phase3b_base_{base:03d}_original", base_sample_id=base,
                                 stratum=stratum, first_object_id=mover)
                conditions = {}
                candidate["video_provenance"] = {}
                for condition in CONDITIONS:
                    correct = condition == CONDITIONS[1] or stratum == "stable"
                    prediction, margin = ("A", 1.0) if correct else ("B", -1.0)
                    row = candidate["rows"][condition]
                    row.update(eval_id=f"eval_{base}_{condition}", base_sample_id=base, first_object_id=mover,
                               correct_option="A", archived_prediction=prediction)
                    candidate["video_provenance"][condition] = {"sha256": "video"}
                    conditions[condition] = {"eval_id": row["eval_id"], "decision": {"margin": margin,
                        "prediction": prediction, "correct_option": "A", "is_correct": correct,
                        "correct_logit": 2.0 + margin, "incorrect_logit": 2.0},
                        "standard_parity": {"first_token_match": True, "logits_allclose": True},
                        "capture_noop_parity": {"exact_match": True, "max_abs_diff": 0},
                        "hook_audit": {"block_output_sites": 36, "post_deepstack_layers": [0, 1, 2]},
                        "logits_signature": {"sha256": "a" * 64, "dtype": "torch.float32", "shape": [151936]},
                        "archived_input_parity": {"matches": True}, "input_tensor_sha256": {"input": "signature"}}
                key = candidate["pair_id"]
                mappings[key] = {**mapping, "pair_id": key, "eligible": True}
                live[key] = {"pair_id": key, "candidate_sha256": digest(candidate), "plan_fingerprint": "plan",
                             "eligible": True, "conditions": conditions}
                candidates.append(candidate)
    return plan, candidates, mappings, live


class FootprintTest(unittest.TestCase):
    def test_energy_and_support_shares_are_distinct(self):
        value = footprint.energy_partition([1, 1, 4, 4], [0, 1])
        self.assertEqual(value["target_position_share"], 0.5)
        self.assertEqual(value["target_delta_energy_share"], 0.2)
        self.assertEqual(value["context_delta_energy_share"], 0.8)
        self.assertEqual(value["target_energy_enrichment"], 0.4)

    def test_zero_energy_is_not_called_no_causal_effect(self):
        value = footprint.energy_partition([0, 0], [0])
        self.assertIsNone(value["target_delta_energy_share"])
        self.assertEqual(value["whole_delta_energy"], 0)

    def test_invalid_energy_or_duplicate_correspondence_rejected(self):
        for values, indices in (([1, -1], [0]), ([1, float("nan")], [0]), ([1, 2], [0, 0]), ([1, 2], [3])):
            with self.assertRaises(ValueError):
                footprint.energy_partition(values, indices)

    def test_mapping_coverage_does_not_imply_complete_geometric_coverage(self):
        result = footprint.geometry_summary({1: 0.4, 2: 0.09, 3: 0, 4: 0.2}, [1, 4])
        self.assertEqual(result["omitted_target_touching_cell_count"], 1)
        self.assertLess(result["declared_union_retained_padded_target_overlap_mass_fraction"], 1)
        self.assertEqual(result["geometric_target_cell_count"], 3)


class ReplicationSelectionTest(unittest.TestCase):
    def test_excludes_base_not_only_prompt(self):
        candidate, _ = expanded_fixture()
        other = {**candidate, "base_sample_id": 2, "pair_id": "other"}
        swapped = {**candidate, "prompt_variant": "swapped"}
        self.assertEqual(protocol.filter_candidates([candidate, other, swapped], {1}), [other])

    def test_duplicate_remaining_base_rejected(self):
        candidate, _ = expanded_fixture()
        with self.assertRaisesRegex(ValueError, "one prompt"):
            protocol.filter_candidates([candidate, {**candidate, "prompt_variant": "swapped"}], set())

    def test_all_fixed_supports_preserve_mapping_and_grid_counts(self):
        candidate, mapping = expanded_fixture()
        before = copy.deepcopy(mapping)
        audit = protocol.expanded_audit(candidate, mapping)
        self.assertEqual(mapping, before)
        self.assertEqual(len(audit["supports"]), 17)
        self.assertEqual(set(protocol.ANCHORS), {name for name, spec in audit["support_specs"].items() if spec["family"] == "anchor"})
        config = {"case_ids": [f"case_{i}" for i in range(12)], "preflight_case_ids": ["case_0", "case_1"],
            "support_audit": {f"case_{i}": audit for i in range(12)}}
        for stage, count in (("replay", 24), ("preflight", 136), ("patch", 816)):
            tasks = protocol.budget.task_grid(config, stage)
            self.assertEqual(len(tasks), count)
            self.assertEqual(len({row["task_id"] for row in tasks}), count)
            self.assertFalse(any("knockout" in row["kind"] for row in tasks))

    def test_freeze_balanced_disjoint_before_interventions(self):
        plan, candidates, maps, live = cohort_fixture()
        with tempfile.TemporaryDirectory() as root, patch.object(protocol, "audited_candidates", return_value=(plan, candidates, maps)):
            frozen, pairs, _ = protocol.freeze_cohort(SimpleNamespace(output_root=root), live)
            self.assertEqual(len(pairs), 12)
            self.assertEqual(len(set(frozen["base_ids"])), 12)
            self.assertFalse(frozen["new_intervention_results_used"])
            self.assertEqual(len(frozen["preflight_case_ids"]), 2)
            self.assertEqual(frozen, protocol.freeze_cohort(SimpleNamespace(output_root=root), live)[0])

    def test_overlap_and_false_live_pass_are_blockers(self):
        for kind in ("overlap", "noop", "decision"):
            plan, candidates, maps, live = cohort_fixture()
            if kind == "overlap":
                plan["excluded_base_ids"] = [1]
            else:
                value = live[candidates[0]["pair_id"]]["conditions"][CONDITIONS[0]]
                if kind == "noop":
                    value["capture_noop_parity"]["exact_match"] = False
                else:
                    value["decision"]["margin"] = 0
            with tempfile.TemporaryDirectory() as root, patch.object(protocol, "audited_candidates", return_value=(plan, candidates, maps)):
                with self.assertRaises(ValueError):
                    protocol.freeze_cohort(SimpleNamespace(output_root=root), live)

    def test_earlier_eligible_candidate_cannot_be_skipped(self):
        plan, candidates, maps, live = cohort_fixture()
        extra = copy.deepcopy(candidates[0])
        extra.update(base_sample_id=0, pair_id="phase3b_base_000_original")
        candidates.insert(0, extra)
        maps[extra["pair_id"]] = {"eligible": True}
        with tempfile.TemporaryDirectory() as root, patch.object(protocol, "audited_candidates", return_value=(plan, candidates, maps)):
            with self.assertRaisesRegex(ValueError, "Earlier"):
                protocol.freeze_cohort(SimpleNamespace(output_root=root), live)

    def test_storage_estimate_reserves_backup_not_only_captures(self):
        _, _, maps, _ = cohort_fixture()
        for row in maps.values():
            for record in row["processor_records"].values():
                record["group_positions"] = {"dummy_group": [1, 2]}
        estimate = protocol.storage_estimate(maps)
        self.assertGreater(estimate["capture_backup_and_overhead_bytes"], 2 * estimate["capture_upper_bytes"])
        self.assertEqual(estimate["safety_reserve_bytes"], 12 * 1024**3)

    def test_source_roots_are_protected(self):
        args = SimpleNamespace(output_root="/evidence/subdir", source_run="/evidence", exclude_plan="/old", reference_run="/other")
        with self.assertRaisesRegex(ValueError, "read-only"):
            protocol.protected_output(args)


class ReplicationLifecycleTest(unittest.TestCase):
    def run_job(self, root, failure_stage=None, stale=False, backup_fails=False):
        output = Path(root) / "replication"
        lifecycle = Path(str(output) + "_lifecycle")
        lifecycle.mkdir()
        (lifecycle / "job.log").write_text("public log", encoding="ascii")
        args = SimpleNamespace(output_root=str(output), storage_root=root, pause_after=True, project_root="repo",
            source_run="source", exclude_plan="old", reference_run="prior", gpus="0,1", stage="interventions",
            backup_dir=str(Path(root) / "backups"), email_notify=True)
        order, config = [], {"fingerprint": "frozen"}
        notifier, client = Mock(), Mock()
        notifier.send.side_effect = lambda *_: order.append("email") or {"smtp_accepted": True}
        client.request_pause.side_effect = lambda: order.append("pause") or {"request_accepted": True}

        def child(_args, _command, label):
            order.append(label)
            if label == failure_stage:
                return 1
            if label == "replication_interventions" and not stale:
                atomic_write(output / "last_attempt.json", {"attempt_id": args.attempt_id, "complete": True, "support_fingerprint": "frozen"})
            return 0

        def backup(*_, **__):
            order.append("backup")
            if backup_fails:
                raise RuntimeError("no space")
            return Path(root) / "package"

        with patch.object(runner, "storage_gate"), patch.object(runner, "services", return_value=(notifier, client)), \
             patch.object(runner.protocol, "build_budget_context", return_value=(config, {}, {}, {}, {})), \
             patch.object(runner, "verify_analysis"), \
             patch.object(runner.shared, "assert_quiescent", side_effect=lambda: order.append("idle")), \
             patch.object(runner.shared, "run_child", side_effect=child), \
             patch.object(runner.shared, "create_backup", side_effect=backup), \
             patch.object(runner.shared, "verify_backup", side_effect=lambda *_: order.append("verify") or
                          {"complete": True, "reports_only": False, "schema": "phase3c_replication_backup_v1",
                           "available_activation_file_count": 24, "selected_videos_included": True}):
            code = runner.job(args)
        return code, order, read_json(lifecycle / "job_status.json")

    def test_success_gates_and_backup_email_pause_order(self):
        with tempfile.TemporaryDirectory() as root:
            code, order, status = self.run_job(root)
            self.assertEqual(code, 0)
            self.assertEqual(order, ["idle", "replication_screen", "replication_capture", "replication_interventions", "backup", "verify", "idle", "email", "pause"])
            self.assertTrue(status["experiment_success"])

    def test_failure_stops_before_interventions_but_preserves_backup(self):
        with tempfile.TemporaryDirectory() as root:
            code, order, status = self.run_job(root, failure_stage="replication_screen")
            self.assertEqual(code, 1)
            self.assertNotIn("replication_interventions", order)
            self.assertLess(order.index("verify"), order.index("pause"))
            self.assertFalse(status["experiment_success"])

    def test_stale_success_is_not_reused_as_current_completion(self):
        with tempfile.TemporaryDirectory() as root:
            code, _, status = self.run_job(root, stale=True)
            self.assertEqual(code, 1)
            self.assertFalse(status["experiment_success"])

    def test_backup_failure_blocks_pause(self):
        with tempfile.TemporaryDirectory() as root:
            code, order, status = self.run_job(root, backup_fails=True)
            self.assertEqual(code, 1)
            self.assertNotIn("pause", order)
            self.assertEqual(status["state"], "needs_attention")

    def test_selected_videos_are_in_full_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "replication"
            video = Path(tmp) / "source.mp4"
            video.write_bytes(b"video fixture")
            atomic_write(root / "selection/mappings.json", {"case": {"processor_records": {
                condition: {"video_path": str(video), "video_sha256": file_hash(video)} for condition in CONDITIONS}}})
            files = runner.backup_files(root, False)
            self.assertIn(video, files.values())
            video.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                runner.backup_files(root, False)

    def test_other_replication_worker_blocks_pause(self):
        from scripts import run_phase3c_unattended as idle
        process = SimpleNamespace(stdout="999999 python scripts/run_phase3c_replication.py --action run")
        with patch.object(idle, "legacy_idle"), patch.object(idle.subprocess, "run", return_value=process), \
             patch.object(idle.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "active"):
                runner.shared.assert_quiescent()


if __name__ == "__main__":
    unittest.main()
