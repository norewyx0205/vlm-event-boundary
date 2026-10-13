import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from scripts import phase3c_scale as protocol, run_phase3c_scale as runner
from scripts.phase3c_core import CONDITIONS, atomic_write, file_hash, read_json, support_record
from tests.test_phase3c_preflight import ToyText, toy_inputs


class ScaleProtocolTest(unittest.TestCase):
    def test_all_existing_bases_kept_once_with_prompts(self):
        candidates = [{"base_sample_id": i, "stratum": "rescue", "prompt_variant": "swapped" if i == 7 else "original"}
                      for i in range(50)]
        selected = protocol.select_cohort(list(reversed(candidates)) + [{"base_sample_id": 100, "stratum": "stable"}])
        self.assertEqual([r["base_sample_id"] for r in selected], list(range(50)))
        self.assertEqual(selected[7]["prompt_variant"], "swapped")
        for bad in (candidates[:-1], candidates[:-1] + [candidates[0]]):
            with self.assertRaises(ValueError):
                protocol.select_cohort(bad)

    def test_exact_grid_no_roi_or_knockout(self):
        plan = {"fingerprint": "fixed", "case_ids": [f"pair{i}" for i in range(50)], "representatives": ["pair0", "pair1"]}
        for stage, count in (("baseline", 100), ("identity", 8), ("patch", 200)):
            tasks = protocol.tasks(plan, stage)
            self.assertEqual(len(tasks), count)
            self.assertEqual(len({r["task_id"] for r in tasks}), count)
            self.assertEqual({r["support"] for r in tasks}, {"whole_event2"})
            self.assertEqual({r["condition"] for r in tasks}, set(CONDITIONS))
            self.assertEqual({r["location"] for r in tasks}, {"block_output"})
        self.assertEqual({r["layer"] for r in protocol.tasks(plan, "patch")}, {12, 16})
        with self.assertRaises(ValueError):
            protocol.tasks(plan, "knockout")

    def test_prior_effect_requires_prediction_and_both_logits(self):
        prior = {"patched_margin": 1., "patched_correct_logit": 2., "patched_incorrect_logit": 1., "patched_prediction": "A"}
        decision = {k.removeprefix("patched_"): v for k, v in prior.items()}
        self.assertTrue(protocol.source_replay_matches({"decision": decision}, prior))
        for key in decision:
            changed = {**decision, key: "B" if key == "prediction" else 3.}
            self.assertFalse(protocol.source_replay_matches({"decision": changed}, prior))

    def test_storage_accounts_for_backup_and_reserve(self):
        estimate = protocol.capture_estimate({"capture_reuse_count": 32},
            {"pair": {"support_audit": {"supports": {"whole_event2": {"low_positions": list(range(512))}}}}})
        self.assertGreater(estimate["required_free_bytes"], estimate["new_capture_upper_bytes"] * 2)
        self.assertEqual(estimate["safety_reserve_bytes"], 12 * 1024**3)

    def test_source_stage_schemas_require_positive_exact_completion(self):
        common = {"expected": 2, "completed": 2, "failures": [], "missing_task_ids": []}
        pilot = {**common, "schema": "phase3c_mechanistic_pilot_v1", "artifact_type": "real",
                 "passed": True, "task_manifest_sha256": "sha"}
        replication = {**common, "schema": "phase3c_budget_followup_v1", "complete": True, "manifest_sha256": "sha"}
        self.assertEqual(protocol.source_stage_manifest_field(pilot), "task_manifest_sha256")
        self.assertEqual(protocol.source_stage_manifest_field(replication), "manifest_sha256")
        for value in (pilot, replication):
            for changed in ({"completed": 1}, {"failures": ["failed"]}, {"missing_task_ids": ["missing"]},
                            {"schema": "unknown"}, {"complete": False, "passed": False}):
                with self.assertRaises(ValueError):
                    protocol.source_stage_manifest_field({**value, **changed})


class CompactCaptureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_two_site_capture_exactly_matches_all_layer_capture(self):
        model, inputs = ToyText(), toy_inputs()
        full = runner.ResidualSites(model, 6, [1, 2], [1, 2])
        compact = runner.CompactSites(model, 6, [1, 2], [1, 2])
        with full.installed():
            a = model(*inputs)
        with compact.installed():
            b = model(*inputs)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(len(full.states), 39)
        self.assertEqual(len(compact.states), 2)
        for layer in (12, 16):
            self.assertTrue(torch.equal(full.states[(layer, "block_output")]["vectors"], compact.states[(layer, "block_output")]["vectors"]))
        self.assertEqual(compact.validate()["post_deepstack_layers"], [0, 1, 2])
        self.assertEqual(compact.validate()["block_output_sites"], 36)
        compact.states.pop((12, "block_output"))
        with self.assertRaises(RuntimeError):
            compact.validate()
        self.assertNotIn("_deepstack_process", model.__dict__)

    def test_capture_checksum_is_enforced_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensor = root/"states.pt"
            runner.save_tensors(tensor, {"states": {}, "logits": torch.ones(2)})
            index = root/"index.json"
            atomic_write(index, {"artifact_type": "real", "eval_id": "eval", "input_tensor_sha256": {},
                "vectors_path": str(tensor), "vectors_sha256": file_hash(tensor)})
            row = {"capture_index_path": str(index), "capture_index_sha256": file_hash(index),
                   "eval_id": "eval", "input_tensor_sha256": {}}
            self.assertTrue(torch.equal(runner.checked_payload(row, torch)["logits"], torch.ones(2)))
            runner.save_tensors(tensor, {"states": {}, "logits": torch.zeros(2)})
            with self.assertRaises(ValueError):
                runner.checked_payload(row, torch)

    def test_checkpoint_edit_failed_or_foreign_rejected(self):
        row = runner.seal({"spec": {}, "plan_fingerprint": "fixed", "passed": False})
        with self.assertRaises(ValueError):
            runner.validate_row(row, {}, {"fingerprint": "fixed"}, {}, {}, None, {})
        row["passed"] = True
        with self.assertRaises(ValueError):
            runner.validate_row(row, {}, {"fingerprint": "fixed"}, {}, {}, None, {})

    def test_end_to_end_baseline_identity_patch_and_resume_without_forwards(self):
        with tempfile.TemporaryDirectory() as directory:
            root, inputs, model = Path(directory), toy_inputs(), ToyText()
            plan = {"fingerprint": "frozen", "case_ids": ["pair"], "representatives": ["pair"]}
            pair = {condition: {"condition": condition, "eval_id": condition, "phase3c_pair_id": "pair", "correct_option": "A",
                    "archived_prediction": "B" if condition == CONDITIONS[0] else "A"} for condition in CONDITIONS}
            mapping = {"processor_records": {c: {"input_ids": list(range(6)), "visual_positions": [1, 2]} for c in CONDITIONS},
                       "support_audit": {"supports": {"whole_event2": support_record([1, 2], [1, 2], [1, 2], [1, 2], "whole_event2")}}}
            context = (plan, {"pair": pair}, {"pair": mapping}, {}, {})
            engine = Mock()
            engine.text, engine.model, engine.torch = model, model, torch
            engine.prepare.side_effect = lambda row, _: {"row": row, "inputs": inputs, "input_tensor_sha256": {"condition": row["condition"]},
                "archived_input_parity": {"matches": True}, "input_metadata": {}}

            def forward(live):
                engine.live = live
                hidden = model(*inputs)
                value = hidden[0, -1].sum() / 100
                return torch.stack((value + (-2 if live["row"]["condition"] == CONDITIONS[0] else 2), torch.tensor(0.)))

            def decide(logits, row):
                correct, incorrect = map(float, logits)
                prediction = "A" if correct > incorrect else "B"
                return {"margin": correct-incorrect, "correct_logit": correct, "incorrect_logit": incorrect,
                        "prediction": prediction, "is_correct": prediction == "A", "correct_option": "A",
                        "predicted_first_token_id": int(logits.argmax())}

            engine.forward.side_effect, engine.decide.side_effect = forward, decide
            engine.technical.side_effect = lambda *args: runner.shared.Engine.technical(engine, *args)
            args = SimpleNamespace(output_root=str(root))
            with patch("scripts.probe_attention_roi.standard_first_token", side_effect=lambda *_: (
                int(forward(engine.live).argmax()), forward(engine.live))):
                runner.run_stage(args, context, "baseline", engine)
            self.assertEqual(len(list((root/"captures").rglob("*.pt"))), 2)
            runner.run_stage(args, context, "identity", engine)
            runner.run_stage(args, context, "patch", engine)
            before = engine.forward.call_count
            for stage in ("baseline", "identity", "patch"):
                runner.run_stage(args, context, stage, engine)
            self.assertEqual(engine.forward.call_count, before)
            self.assertTrue(read_json(root/"patch/summary.json")["complete"])


class LifecycleTest(unittest.TestCase):
    def lifecycle(self, code=0, backup_error=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"run"
            lifecycle = Path(str(root)+"_lifecycle")
            lifecycle.mkdir()
            (lifecycle/"job.log").touch()
            args = SimpleNamespace(output_root=str(root), backup_dir=str(Path(directory)/"backups"), pause_after=True)
            notifier, client = Mock(), Mock()
            events = []

            def child(args, *_):
                atomic_write(root/"last_attempt.json", {"attempt_id": args.attempt_id, "complete": True})
                events.append("experiment")
                return code

            def backup(*_, **__):
                events.append("backup")
                if backup_error:
                    raise RuntimeError("disk full")
                return Path(directory)/"bundle"

            def verify(_):
                events.append("verify")
                return {"complete": True, "reports_only": False, "schema": "phase3c_scale50_backup_v1", "available_activation_file_count": 68}

            notifier.send.side_effect = lambda *_: events.append("email")
            client.request_pause.side_effect = lambda: events.append("pause")
            with patch.object(protocol, "load", return_value=({}, {}, {}, {}, {})), \
                 patch.object(runner, "storage_gate"), patch.object(runner, "assert_idle"), \
                 patch.object(runner, "command", return_value=["child"]), \
                 patch.object(runner, "services", return_value=(notifier, client)), \
                 patch.object(runner.shared, "run_child", side_effect=child), \
                 patch.object(runner, "verify_analysis"), \
                 patch.object(runner.shared, "create_backup", side_effect=backup), \
                 patch.object(runner.shared, "verify_backup", side_effect=verify):
                result = runner.job(args)
            return result, events, read_json(lifecycle/"job_status.json")

    def test_success_backed_up_then_email_then_pause(self):
        code, events, status = self.lifecycle()
        self.assertEqual(code, 0)
        self.assertTrue(status["experiment_success"])
        self.assertEqual(events[-4:], ["backup", "verify", "email", "pause"])

    def test_failure_still_backed_up_notified_and_paused(self):
        code, events, status = self.lifecycle(code=1)
        self.assertEqual(code, 1)
        self.assertFalse(status["experiment_success"])
        self.assertEqual(events[-4:], ["backup", "verify", "email", "pause"])

    def test_failed_backup_blocks_pause(self):
        code, events, status = self.lifecycle(backup_error=True)
        self.assertEqual(code, 1)
        self.assertNotIn("pause", events)
        self.assertFalse(status["backup_verified"])

    def test_launch_command_retains_email_pause_and_scope(self):
        args = SimpleNamespace(**{name: name for name in ("source_run", "pilot_plan", "pilot_run", "replication_run", "output_root",
            "project_root", "storage_root", "gpus", "backup_dir", "email_config", "surf_config")},
            stage="patch", email_notify=True, pause_after=True, confirm_exclusive_workspace=True)
        command = runner.command(args, "job")
        self.assertIn("--email_notify", command)
        self.assertIn("--pause_after", command)
        self.assertIn("--confirm_exclusive_workspace", command)
        self.assertNotIn("--retry_failed", command)

    def test_backup_does_not_import_previous_tensor_stores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root/"outside.mp4"
            video.touch()
            run = root/"run"
            atomic_write(run/"mappings.json", {"pair": {"processor_records": {"low": {
                "video_path": str(video), "video_sha256": file_hash(video)}}}})
            atomic_write(run/"source_capture_refs.json", {"old_vectors_path": "/old/states.pt"})
            files = runner.backup_files(run, False)
            self.assertTrue(any(name.startswith("source_videos/") for name in files))
            self.assertFalse(any(path == Path("/old/states.pt") for path in files.values()))

    def test_low_storage_fail_before_model(self):
        args = SimpleNamespace(output_root="unused", storage_root="/data")
        with patch.object(protocol, "capture_estimate", return_value={"required_free_bytes": 100}), \
             patch.object(runner.shutil, "disk_usage", return_value=SimpleNamespace(free=99)), \
             patch.object(runner, "atomic_write"):
            with self.assertRaises(ValueError):
                runner.storage_gate(args, ({}, {}, {}))


if __name__ == "__main__":
    unittest.main()
