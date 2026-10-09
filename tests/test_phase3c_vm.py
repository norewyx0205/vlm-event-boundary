import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import test_phase3c_primary as fixtures
from scripts import analyze_phase3c, backup_phase3c as backup, phase3c_core as core
from scripts import launch_phase3c_vm as launch, run_phase3c_vm as vm, run_phase3c_unattended as job
from scripts import run_phase3c_preflight as runner


PROJECT = Path(__file__).resolve().parents[1]


class Phase3CVmTest(unittest.TestCase):
    def fixture(self, directory, stage="baseline", policy="off", email=False):
        with contextlib.redirect_stdout(io.StringIO()):
            data = list(fixtures.Phase3CPrimaryTest().setup_run(directory))
        plan, root, frozen, _, _, config, _ = data
        (root / "execution_config.json").unlink()
        data[5] = runner.bind_execution(root, frozen, config["runtime"], PROJECT, {})
        parser = argparse.ArgumentParser()
        job.add_job_arguments(parser)
        args = job.validate_job_args(parser.parse_args([
            "--stage", stage, "--plan_dir", str(plan), "--output_root", str(root),
            "--storage_root", directory, "--project_root", str(PROJECT),
            "--backup_dir", str(Path(directory) / "backups"), "--bootstrap_repeats", "100", "--no_plots",
            "--pause_policy", policy, "--email_config", str(Path(directory) / "private/email.json"),
            "--surf_config", str(Path(directory) / "private/surf.json"),
            *(["--confirm_exclusive_workspace"] if policy != "off" else []), *(["--email_notify"] if email else [])]))
        return args, data

    def scientific_child(self, data, fail=None):
        plan, root, frozen, pairs, mappings, config, engine = data
        def child(args, command, label):
            if label == fail:
                return 7
            if label == "analyze":
                analyze_phase3c.analyze(plan, root, repeats=100, plots=False)
            else:
                result = runner.run_stage(root, frozen, pairs, mappings, engine, config, label)
                self.assertTrue(result["passed"])
            return 0
        return child

    def save_binding(self, args):
        config, _ = vm.build_plan(args)
        vm.save_plan(args, config)
        core.atomic_write(Path(args.output_root) / "vm_last_status.json",
                          {"complete": False, "attempt_id": "failed-test-attempt", "stage": args.stage})
        return config

    def test_plan_is_read_only_and_full_requires_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            args, data = self.fixture(directory)
            root = data[1]
            before = sorted(str(path) for path in root.rglob("*"))
            config, steps = vm.build_plan(args)
            self.assertEqual(steps, ["baseline"])
            self.assertEqual(config["tasks"], {"patch": 696, "routing": 24, "knockout": 2592})
            self.assertEqual(before, sorted(str(path) for path in root.rglob("*")))
            args.stage = "full"
            with self.assertRaises(FileNotFoundError):
                vm.build_plan(args)
            self.assertEqual(data[6].calls, [])

    def test_vm_full_executes_exact_grid_and_cpu_analysis(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            args, data = self.fixture(directory, "full")
            fixtures.Phase3CPrimaryTest().gates(data)
            self.assertEqual(vm.execute(args, self.scientific_child(data)), 0)
            status = core.read_json(data[1] / "vm_last_status.json")
            self.assertTrue(status["complete"])
            self.assertEqual(status["completed_steps"], ["patch", "routing", "knockout", "analyze"])
            self.assertEqual(job.verify_attempt(args, "old-attempt")["attempt_id"], status["attempt_id"])
            with self.assertRaises(ValueError):
                job.verify_attempt(args, status["attempt_id"])
            self.assertEqual(len(data[6].calls), 24 + 4 + 24 * 3)

    def test_failed_stage_stops_chain_and_never_inherits_success(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            args, data = self.fixture(directory, "full")
            fixtures.Phase3CPrimaryTest().gates(data)
            core.atomic_write(data[1] / "vm_last_status.json", {"complete": True, "attempt_id": "old"})
            labels = []
            def child(_args, _command, label):
                labels.append(label)
                return 1
            self.assertEqual(vm.execute(args, child), 1)
            status = core.read_json(data[1] / "vm_last_status.json")
            self.assertFalse(status["complete"])
            self.assertNotEqual(status["attempt_id"], "old")
            self.assertEqual(labels, ["patch"])
            args.gpu_weight_budget_gib = 11
            self.assertEqual(vm.execute(args, child), 1)
            self.assertIsNone(core.read_json(data[1] / "vm_last_status.json")["vm_fingerprint"])
            self.assertEqual(labels, ["patch"])

    def test_binding_and_snapshot_cannot_change_or_follow_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            args, data = self.fixture(directory)
            self.save_binding(args)
            args.bootstrap_repeats = 200
            with self.assertRaisesRegex(ValueError, "binding changed"):
                vm.build_plan(args)
            path = data[0] / "candidate_summary.json"
            original = path.read_bytes()
            path.unlink()
            target = Path(directory) / "other.json"
            target.write_bytes(original)
            path.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlink"):
                vm.snapshot_files(data[0])

    def test_locks_refuse_duplicates_and_no_other_root_is_adopted(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(directory)
            with vm.vm_lock(args.output_root):
                with self.assertRaisesRegex(RuntimeError, "owns"):
                    with vm.vm_lock(args.output_root):
                        pass
            core.atomic_write(Path(args.output_root) / "vm_last_status.json", {"attempt_id": "prior"})
            with vm.run_lock(args.output_root):
                with self.assertRaises(RuntimeError):
                    vm.execute(args, child=Mock())
            self.assertEqual(core.read_json(Path(args.output_root) / "vm_last_status.json"), {"attempt_id": "prior"})
            args.output_root = args.plan_dir
            with self.assertRaises(ValueError):
                job.validate_job_args(args)

    def test_backup_contains_snapshots_selected_videos_states_and_failed_status(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            args, data = self.fixture(directory)
            config = self.save_binding(args)
            states = data[1] / "captures/unit-test/states.pt"
            states.parent.mkdir(parents=True)
            states.write_bytes(b"not-real-model-states")
            bundle = backup.create_backup(data[1], args.backup_dir, part_bytes=1000)
            manifest = backup.verify_backup(bundle)
            paths = {row["path"] for row in manifest["files"]}
            self.assertIn("run/captures/unit-test/states.pt", paths)
            self.assertIn("run/preparation_snapshot/selection/frozen_config.json", paths)
            self.assertEqual(len([path for path in paths if path.startswith("source_videos/")]), 24)
            self.assertEqual(manifest["vm_fingerprint"], config["vm_fingerprint"])
            self.assertFalse(manifest["run_status"]["complete"])
            self.assertFalse(manifest["phase3b_source_backup_included"])
            self.assertFalse(manifest["local_backup_confirmed"])
            first = bundle / manifest["archives"][0]["name"]
            first.write_bytes(b"corrupt")
            with self.assertRaises(RuntimeError):
                backup.verify_backup(bundle)

    def test_backup_rejects_changed_source_and_private_files_reports_only_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            args, data = self.fixture(directory)
            config = self.save_binding(args)
            package = backup.create_backup(data[1], args.backup_dir, reports_only=True)
            manifest = backup.verify_backup(package)
            self.assertTrue(manifest["reports_only"])
            self.assertFalse(manifest["activation_tensors_included"])
            private = data[1] / "email.json"
            private.write_text("{}")
            with self.assertRaisesRegex(ValueError, "Unexpected"):
                backup.backup_files(data[1], False)
            private.unlink()
            video = Path(next(iter(config["video_sha256"])))
            video.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "video"):
                backup.backup_files(data[1], False)

    def lifecycle(self, args, data, *, failure=False, backup_error=False, busy_after=False, email_error=False, pause_error=False):
        events = []
        config = self.save_binding(args)
        prior = core.read_json(data[1] / "vm_last_status.json")
        def child(_args):
            events.append("experiment")
            core.atomic_write(data[1] / "vm_last_status.json", {**prior, "attempt_id": "new", "complete": not failure})
            return 1 if failure else 0
        def package(*_args, **kwargs):
            events.append("backup")
            self.assertFalse(kwargs["reports_only"])
            if backup_error:
                raise RuntimeError("not a real provider response")
            return "test-bundle"
        def verify(_bundle):
            events.append("verify")
            return {"schema": "phase3c_local_backup_v1", "complete": True, "reports_only": False,
                    "activation_tensors_included": True, "vm_fingerprint": config["vm_fingerprint"],
                    "run_status": core.read_json(data[1] / "vm_last_status.json")}
        def idle():
            events.append("idle")
            if busy_after and events.count("idle") > 1:
                raise RuntimeError("busy")
        client = Mock()
        client.check.return_value = {"workspace_id": "test-id"}
        def pause():
            events.append("pause")
            if pause_error:
                raise TimeoutError("ambiguous private API response")
            return {"billing_stop_confirmed": False}
        client.request_pause.side_effect = pause
        notifier = Mock()
        def send(_subject, _body):
            events.append("email")
            if email_error:
                raise RuntimeError("private provider detail")
            return {"smtp_accepted": True}
        notifier.send.side_effect = send
        def run():
            with patch.object(job, "verify_attempt", return_value={"attempt_id": "new"}):
                return job.execute_job(args, client, child, package, verify, idle, notifier)
        return events, client, notifier, run

    def test_success_and_failure_email_after_backup_before_pause(self):
        for failed in (False, True):
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                args, data = self.fixture(directory, policy="finished", email=True)
                events, client, notifier, run = self.lifecycle(args, data, failure=failed)
                self.assertEqual(run(), 1 if failed else 0)
                self.assertEqual(events, ["idle", "experiment", "backup", "verify", "idle", "email", "pause"])
                status = core.read_json(job.lifecycle_directory(args) / "job_status.json")
                self.assertTrue(status["backup_verified"])
                self.assertFalse(status["billing_stop_confirmed"])
                self.assertFalse(status["local_backup_confirmed"])
                self.assertIn("NOT establish a completed", notifier.send.call_args.args[1])
                self.assertEqual(client.request_pause.call_count, 1)

    def test_email_failure_does_not_block_pause_and_pause_timeout_is_not_retried(self):
        for options in ({"email_error": True}, {"pause_error": True}):
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                args, data = self.fixture(directory, policy="finished", email=True)
                _, client, _, run = self.lifecycle(args, data, **options)
                self.assertEqual(run(), 1 if options.get("pause_error") else 0)
                self.assertEqual(client.request_pause.call_count, 1)
                status = core.read_json(job.lifecycle_directory(args) / "job_status.json")
                self.assertNotIn("private provider", json.dumps(status))
                self.assertNotIn("private API", json.dumps(status))

    def test_low_disk_or_other_active_job_blocks_pause(self):
        for options in ({"backup_error": True}, {"busy_after": True}):
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                args, data = self.fixture(directory, policy="finished", email=True)
                _, client, notifier, run = self.lifecycle(args, data, **options)
                self.assertEqual(run(), 1)
                client.request_pause.assert_not_called()
                self.assertIn("NEEDS ATTENTION", notifier.send.call_args.args[0])

    def test_reports_only_or_stale_status_cannot_authorize_pause(self):
        for stale in (True, False):
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                args, data = self.fixture(directory, policy="finished", email=True)
                events, client, _, _ = self.lifecycle(args, data)
                def child(_args):
                    if not stale:
                        core.atomic_write(data[1] / "vm_last_status.json", {"attempt_id": "new", "stage": args.stage, "complete": False})
                    return 1
                self.assertEqual(job.execute_job(args, client, child, backup=lambda *_a, **_k: "bad",
                    verify=lambda _: {"complete": True, "reports_only": True}, idle_check=lambda: None, notifier=Mock()), 1)
                client.request_pause.assert_not_called()

    def test_unattended_real_backup_on_baseline_and_success_gate(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            args, data = self.fixture(directory)
            child = lambda requested: vm.execute(requested, self.scientific_child(data))
            self.assertEqual(job.execute_job(args, child=child), 0)
            status = core.read_json(job.lifecycle_directory(args) / "job_status.json")
            manifest = backup.verify_backup(status["backup_dir"])
            self.assertTrue(status["experiment_success"])
            self.assertIn("run/unattended_run_status.json", {row["path"] for row in manifest["files"]})
            self.assertEqual(manifest["run_status"]["attempt_id"], status["attempt_id"])

    def test_dry_run_has_no_gpu_api_tmux_or_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            args, data = self.fixture(directory)
            before = sorted(str(path) for path in Path(directory).rglob("*"))
            command = [sys.executable, "scripts/launch_phase3c_vm.py", "--dry_run", *job.vm_options(args), "--backup_dir", args.backup_dir]
            result = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True, check=True)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["steps"], ["baseline"])
            self.assertIn("scripts/run_phase3c_unattended.py", plan["command"][2])
            self.assertEqual(before, sorted(str(path) for path in Path(directory).rglob("*")))

    def test_duplicate_tmux_or_failed_artifact_plan_cannot_send_mail_or_launch(self):
        for duplicate in (True, False):
            with tempfile.TemporaryDirectory() as directory:
                args, _ = self.fixture(directory, email=True)
                argv = ["launch", *job.vm_options(args), "--backup_dir", args.backup_dir, "--email_notify", "--email_config", args.email_config]
                with patch.object(sys, "argv", argv), patch.object(launch.shutil, "which", return_value="/usr/bin/tmux"), \
                     patch.object(launch.subprocess, "run", return_value=SimpleNamespace(returncode=0 if duplicate else 1)) as tmux, \
                     patch.object(launch, "build_plan", side_effect=ValueError("blocked")), \
                     patch.object(launch, "EmailNotifier") as notifier:
                    with self.assertRaises(SystemExit if duplicate else ValueError):
                        launch.main()
                    self.assertEqual(tmux.call_count, 1)
                    notifier.assert_not_called()

    def test_launcher_starts_only_after_plan_and_auth_checks(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            args, _ = self.fixture(directory, email=True)
            argv = ["launch", *job.vm_options(args), "--backup_dir", args.backup_dir, "--email_notify", "--email_config", args.email_config]
            with patch.object(sys, "argv", argv), patch.object(launch.shutil, "which", return_value="/usr/bin/tmux"), \
                 patch.object(launch.subprocess, "run", side_effect=[SimpleNamespace(returncode=1), SimpleNamespace(returncode=0)]) as tmux, \
                 patch.object(launch, "load_email_config", return_value={}), patch.object(launch, "EmailNotifier") as notifier:
                launch.main()
                notifier.return_value.check.assert_called_once()
                self.assertEqual(tmux.call_count, 2)
                command = tmux.call_args.args[0]
                self.assertEqual(command[:3], ["tmux", "new-session", "-d"])
                self.assertIn("run_phase3c_unattended.py", command[-1])
                self.assertTrue((job.lifecycle_directory(args) / "launch_plan.json").exists())

    def test_bad_email_auth_blocks_experiment_and_pause(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            args, _ = self.fixture(directory, policy="finished", email=True)
            notifier, child, client = Mock(), Mock(), Mock()
            notifier.check.side_effect = RuntimeError("private SMTP response")
            self.assertEqual(job.execute_job(args, client=client, child=child, notifier=notifier), 1)
            child.assert_not_called()
            client.request_pause.assert_not_called()
            status = core.read_json(job.lifecycle_directory(args) / "job_status.json")
            self.assertNotIn("private SMTP", json.dumps(status))

    def test_pause_and_backup_paths_require_explicit_safe_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(directory)
            args.pause_policy = "finished"
            with self.assertRaisesRegex(ValueError, "whole VM"):
                job.validate_job_args(args)
            args.pause_policy = "off"
            args.backup_dir = args.plan_dir
            with self.assertRaisesRegex(ValueError, "separate sibling"):
                job.validate_job_args(args)

    def test_idle_check_detects_other_lifecycle_but_not_itself(self):
        own = f"{os.getpid()} python scripts/run_phase3c_unattended.py"
        with patch.object(job, "legacy_idle"), patch.object(job.subprocess, "run", return_value=SimpleNamespace(stdout=own)):
            job.assert_quiescent()
        other = f"{os.getpid() + 100} python scripts/run_phase3c_unattended.py"
        with patch.object(job, "legacy_idle"), patch.object(job.subprocess, "run", return_value=SimpleNamespace(stdout=own + "\n" + other)):
            with self.assertRaisesRegex(RuntimeError, "active"):
                job.assert_quiescent()


if __name__ == "__main__":
    unittest.main()
