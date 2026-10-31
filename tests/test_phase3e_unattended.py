import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import run_phase3e_paired_capture as capture
from scripts import run_phase3e_paired_unattended as runner
from scripts.backup_phase3b import verify_backup
from scripts.phase3c_core import atomic_write, read_json


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        storage = Path(self.temporary.name).resolve()
        self.root = storage / "run"
        self.root.mkdir()
        atomic_write(self.root / "plan.json", {"synthetic_lifecycle_fixture": True})
        (self.root / "captures").mkdir()
        (self.root / "captures/test.pt").write_bytes(b"synthetic backup fixture, not model activations")
        self.args = SimpleNamespace(output_root=str(self.root), storage_root=str(storage),
            backup_dir=str(storage / "backups"), project_root=str(Path(runner.__file__).resolve().parents[1]),
            source_plan=str(storage / "source.json"), stage="preflight", action="job",
            attempt_id="1"*32, email_notify=True, email_config=str(storage / "private/email.json"),
            pause_after=True, confirm_exclusive_workspace=True, surf_config=str(storage / "private/surf.json"),
            gpus="0,1", session_name="phase3e-test")
        self.notifier, self.client = Mock(), Mock()
        self.notifier.send.return_value = {"smtp_accepted": True}
        self.client.request_pause.return_value = {"accepted": True, "confirmed_paused": False}
        self.plan = {"fingerprint": "plan-sha"}

    def execute(self, child=None, verify_stage=None, pack=None, verify=None, idle=None, precheck=None):
        with patch.object(runner, "precheck", side_effect=precheck,
                          return_value=(self.plan, {"synthetic_fixture": True})), \
                patch.object(runner, "snapshot_code"), \
                patch.object(runner, "verify_stage", side_effect=verify_stage,
                             return_value={"plan_fingerprint": "plan-sha"}), \
                patch.object(capture, "space"):
            kwargs = {"child": child or Mock(return_value=0), "idle": idle or Mock(),
                "notifier_factory": lambda args: self.notifier, "client_factory": lambda args: self.client}
            if pack is not None:
                kwargs["pack"] = pack
            if verify is not None:
                kwargs["verify"] = verify
            code = runner.job(self.args, **kwargs)
        status = read_json(runner.lifecycle_dir(self.args) / "job_status.json")
        return code, status

    def test_preflight_success_backs_up_then_notifies_then_requests_pause(self):
        events = []
        self.notifier.send.side_effect = lambda *args: events.append("email") or {"smtp_accepted": True}
        self.client.request_pause.side_effect = lambda: events.append("pause") or {"accepted": True}
        child = Mock(return_value=0)
        code, status = self.execute(child=child)
        self.assertEqual(code, 0)
        self.assertEqual(events, ["email", "pause"])
        self.assertTrue(status["backup_verified"])
        self.assertFalse(status["full_experiment_complete"])
        self.assertTrue(status["pause_not_confirmed"])
        self.assertEqual(status["state"], "pause_requested")
        self.assertEqual(child.call_count, 1)
        self.assertIn("preflight", child.call_args.args[1])
        manifest = verify_backup(status["backup_dir"])
        self.assertEqual(manifest["schema"], runner.BACKUP_SCHEMA)
        self.assertTrue(manifest["stage_verified"])
        self.assertIn("Full 40-family capture complete: False", self.notifier.send.call_args.args[1])

    def test_full_success_runs_capture_export_verify_but_no_new_ko_or_probe(self):
        self.args.stage = "full"
        child = Mock(return_value=0)
        code, status = self.execute(child=child)
        self.assertEqual(code, 0)
        self.assertTrue(status["full_experiment_complete"])
        commands = [call.args[1] for call in child.call_args_list]
        self.assertEqual([c[c.index("--stage")+1] for c in commands], ["capture", "export", "verify"])
        self.assertTrue(all("knockout" not in c and "probes" not in c for c in commands))

    def test_failed_child_stops_grid_and_preserves_partial_backup_before_pause(self):
        self.args.stage = "full"
        child = Mock(return_value=7)
        code, status = self.execute(child=child)
        self.assertEqual(code, 7)
        self.assertEqual(child.call_count, 1)
        self.assertFalse(status["experiment_success"])
        self.assertTrue(status["backup_verified"])
        self.client.request_pause.assert_called_once()
        manifest = verify_backup(status["backup_dir"])
        self.assertFalse(manifest["stage_verified"])
        self.assertFalse(manifest["full_experiment_complete"])
        self.assertIn("run/failure.json", {r["path"] for r in manifest["files"]})
        self.assertIn("FAILED", self.notifier.send.call_args.args[0])

    def test_child_exception_does_not_leak_exception_text_or_retry(self):
        child = Mock(side_effect=RuntimeError("test secret must not appear in status or mail"))
        code, status = self.execute(child=child)
        self.assertEqual(code, 1)
        child.assert_called_once()
        self.assertEqual(status["child_error_type"], "RuntimeError")
        self.assertNotIn("test secret", json.dumps(status))
        self.assertNotIn("test secret", self.notifier.send.call_args.args[1])

    def test_verification_failure_is_not_reported_as_success(self):
        code, status = self.execute(verify_stage=ValueError("bad result"))
        self.assertEqual(code, 1)
        self.assertFalse(status["experiment_success"])
        self.assertEqual(status["verification_error_type"], "ValueError")
        self.assertTrue(status["backup_verified"])

    def test_backup_failure_sends_alert_and_never_pauses(self):
        code, status = self.execute(pack=Mock(side_effect=RuntimeError("disk error")))
        self.assertEqual(code, 1)
        self.assertEqual(status["state"], "needs_attention")
        self.assertFalse(status["backup_verified"])
        self.client.request_pause.assert_not_called()
        self.notifier.send.assert_called_once()

    def test_stale_attempt_backup_is_rejected(self):
        def stale(bundle):
            result = verify_backup(bundle)
            return {**result, "attempt_id": "2"*32}
        code, status = self.execute(verify=stale)
        self.assertEqual(code, 1)
        self.assertFalse(status["backup_verified"])
        self.client.request_pause.assert_not_called()

    def test_remaining_gpu_process_after_backup_blocks_pause(self):
        idle = Mock(side_effect=[None, None, RuntimeError("other GPU process")])
        code, status = self.execute(idle=idle)
        self.assertEqual(code, 1)
        self.assertTrue(status["backup_verified"])
        self.assertEqual(status["state"], "needs_attention")
        self.client.request_pause.assert_not_called()

    def test_smtp_failure_still_allows_verified_quiescent_pause(self):
        self.notifier.send.side_effect = RuntimeError("SMTP unavailable")
        code, status = self.execute()
        self.assertEqual(code, 0)
        self.assertEqual(status["email"], {"error_type": "RuntimeError"})
        self.client.request_pause.assert_called_once()

    def test_precheck_failure_does_not_start_experiment_or_pause(self):
        child = Mock(return_value=0)
        code, status = self.execute(child=child, precheck=ValueError("missing preflight"))
        self.assertEqual(code, 1)
        self.assertNotIn("experiment_success", status)
        self.assertIn("NOT STARTED", self.notifier.send.call_args.args[0])
        child.assert_not_called()
        self.client.request_pause.assert_not_called()

    def test_no_pause_request_when_pause_disabled(self):
        self.args.pause_after = False
        code, status = self.execute()
        self.assertEqual(code, 0)
        self.assertEqual(status["state"], "done_without_pause")
        self.client.request_pause.assert_not_called()

    def test_exclusive_workspace_confirmation_required(self):
        self.args.confirm_exclusive_workspace = False
        with patch.object(capture, "persistent_volume"), self.assertRaisesRegex(ValueError, "exclusive"):
            runner.validate_args(self.args)
        self.args.confirm_exclusive_workspace = True
        self.args.backup_dir = str(self.root / "backup")
        with patch.object(capture, "persistent_volume"), self.assertRaisesRegex(ValueError, "separate"):
            runner.validate_args(self.args)

    def test_launch_uses_tmux_and_duplicate_session_is_rejected(self):
        with patch.object(runner.shutil, "which", return_value="/usr/bin/tmux"), \
                patch.object(runner.subprocess, "run") as command, \
                patch.object(runner, "precheck"), patch.object(runner, "make_notifier"), \
                patch.object(runner, "make_client"), patch.object(runner, "assert_idle"), \
                patch.object(runner, "snapshot_code"):
            command.return_value.returncode = 0
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                runner.launch(self.args)
            command.reset_mock()
            command.side_effect = [SimpleNamespace(returncode=1), SimpleNamespace(returncode=0)]
            result = runner.launch(self.args)
            self.assertEqual(result["status"], "detached_job_started")
            launched = command.call_args_list[-1].args[0]
            self.assertEqual(launched[:2], ["tmux", "new-session"])
            self.assertIn("--action job", launched[-1])
            self.assertIn("--confirm_exclusive_workspace", launched[-1])

    def test_full_precheck_requires_saved_preflight_before_any_child(self):
        self.args.stage = "full"
        atomic_write(self.root / "preflight_summary.json", {"stale": True})
        with patch.object(capture, "load_plan", return_value=self.plan), \
                patch.object(capture, "config_for", return_value={}), patch.object(capture, "space"), \
                patch.object(capture, "verify", return_value={"fresh": True}), \
                patch.object(capture, "storage_projection") as projection:
            with self.assertRaisesRegex(ValueError, "preflight"):
                runner.precheck(self.args)
            projection.assert_not_called()

    def test_capture_package_uses_real_supported_backup_prefix(self):
        with patch.object(capture, "verify", return_value={"plan_fingerprint": "plan-sha"}), \
                patch("scripts.phase3e_option_store.load_store"), patch.object(capture, "space"):
            result = capture.package(self.root, self.args.backup_dir)
        bundle = Path(result["backup_directory"])
        self.assertTrue(bundle.name.startswith("phase3c_backup_"))
        self.assertEqual(verify_backup(bundle)["schema"], "phase3e_paired_option_backup_v1")

    def test_code_snapshot_rejects_changed_sources(self):
        project = Path(self.args.storage_root) / "project"
        (project / "scripts").mkdir(parents=True)
        (project / "docs").mkdir()
        (project / "scripts/sample.py").write_text("pass\n")
        (project / "docs/phase3e_b_paired_option_protocol.md").write_text("test fixture\n")
        args = copy.copy(self.args)
        args.project_root = str(project)
        runner.snapshot_code(args)
        (project / "scripts/sample.py").write_text("raise RuntimeError()\n")
        with self.assertRaisesRegex(ValueError, "snapshot"):
            runner.snapshot_code(args)

    def test_shared_idle_guard_rejects_cpu_workers_with_no_gpu_compute(self):
        for cmd in ("python scripts/run_phase3e_option_probes.py --output_root /tmp/out",
                    "python -m scripts.run_phase3e_option_probes --output_root /tmp/out",
                    "python scripts/audit_phase3e_interactions.py --output_root /tmp/out"):
            with patch("scripts.run_phase3d_single_layer_ko.assert_idle"), \
                    patch.object(runner.subprocess, "run", side_effect=[SimpleNamespace(stdout=""),
                        SimpleNamespace(stdout=f"{os.getpid()+100} {cmd}\n")]), \
                    patch.object(runner.os, "readlink", return_value="/usr/bin/python3"):
                with self.assertRaisesRegex(RuntimeError, "Another Phase 3E process"):
                    runner.assert_idle()

    def test_standalone_finish_reuses_shared_guard_and_rejects_cpu_task(self):
        with patch("scripts.backup_phase3b.verify_backup", return_value={"schema": "phase3e_paired_option_backup_v1",
                    "source_run_root": self.args.output_root}), patch.object(capture, "verify", return_value={}), \
                patch.object(runner, "assert_idle", side_effect=RuntimeError("CPU task active")), \
                patch("scripts.phase3b_email.EmailNotifier") as email, patch("scripts.surf_workspace.SurfClient") as surf:
            with self.assertRaisesRegex(RuntimeError, "CPU task"):
                capture.finish(self.root, self.args.backup_dir)
            email.assert_not_called()
            surf.assert_not_called()

    def test_standalone_rechecks_cpu_idle_after_email_and_records_blocked_pause(self):
        with patch("scripts.backup_phase3b.verify_backup", return_value={"schema": "phase3e_paired_option_backup_v1",
                    "source_run_root": self.args.output_root}), patch.object(capture, "verify", return_value={}), \
                patch.object(runner, "assert_idle", side_effect=[None, RuntimeError("CPU task started")]), \
                patch("scripts.phase3b_email.load_private_config", return_value={}), \
                patch("scripts.phase3b_email.EmailNotifier") as email, patch("scripts.surf_workspace.SurfClient") as surf:
            email.return_value.send.return_value = {"sent": True}
            with self.assertRaisesRegex(RuntimeError, "CPU task started"):
                capture.finish(self.root, self.args.backup_dir)
            email.return_value.send.assert_called_once()
            surf.assert_not_called()
            receipt = read_json(self.root.parent / (self.root.name+"_completion.json"))
            self.assertTrue(receipt["pause_blocked"])
            self.assertFalse(receipt["workspace_quiescent"])
            self.assertTrue(receipt["pause_not_confirmed"])


if __name__ == "__main__":
    unittest.main()
