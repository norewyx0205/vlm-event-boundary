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
from unittest.mock import MagicMock, Mock, patch
from urllib.error import HTTPError

from scripts import surf_workspace as surf, run_phase3b_unattended as job, launch_phase3b_vm as launch
from scripts.run_phase3b_vm import write_json

WORKSPACE = "11111111-2222-4333-8444-555555555555"


def config():
    return {"schema": "phase3b_surf_private_v1", "workspace_id": WORKSPACE,
            "workspace_name": "Test VM", "token": "unit-test-token-not-a-real-credential"}


def api_response(status="running", allowed=None, **overrides):
    return {"id": WORKSPACE, "name": "Test VM", "status": status,
            "allowed_actions": ["pause"] if allowed is None else allowed, **overrides}


def arguments(directory, policy="success"):
    root = Path(directory)
    return job.validate_job_args(SimpleNamespace(
        stage="full", output_root=str(root / "run"), storage_root=str(root), project_root=str(Path.cwd()),
        backup_dir=str(root / "backups"), surf_config=str(root / "private/surf.json"),
        pause_policy=policy, confirm_exclusive_workspace=True,
        runner_args=["--", "--selection_dir", str(root / "selection"), "--rescue_pool_root", str(root / "pool")],
    ))


class SurfTest(unittest.TestCase):
    def test_private_file_permissions_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "private/surf.json")
            surf.save_private_config(path, config())
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(path.parent.stat().st_mode & 0o777, 0o700)
            self.assertEqual(surf.load_private_config(path), config())
            with self.assertRaises(FileExistsError):
                surf.save_private_config(path, config())
            path.chmod(0o644)
            with self.assertRaises(PermissionError):
                surf.load_private_config(path)

    def test_config_cannot_be_in_backup_or_repo_or_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                surf.save_private_config(root / "repo/surf.json", config(), (root / "repo",))
            path = root / "private/surf.json"
            surf.save_private_config(path, config())
            link = root / "link.json"
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                surf.load_private_config(link)

    def test_exact_read_and_pause_api_contract_and_no_delete(self):
        responses = [api_response(), api_response(status="pausing")]
        requests = []
        def open_request(request, timeout):
            requests.append(request)
            response = Mock()
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            response.read.return_value = json.dumps(responses.pop(0)).encode()
            return response
        client = surf.SurfClient(config(), SimpleNamespace(open=open_request))
        result = client.request_pause()
        self.assertEqual([r.get_method() for r in requests], ["GET", "POST"])
        self.assertEqual(requests[1].full_url, surf.API_BASE + f"workspaces/{WORKSPACE}/actions/")
        self.assertEqual(json.loads(requests[1].data), [{"action": "pause", "parameters": {}}])
        self.assertEqual(requests[0].get_header("Authorization"), config()["token"])
        self.assertTrue(result["request_accepted"])
        self.assertFalse(result["billing_stop_confirmed"])
        self.assertNotIn(config()["token"], json.dumps(result))

    def test_wrong_target_and_missing_permission_fail_closed(self):
        client = surf.SurfClient(config())
        with patch.object(client, "_request", return_value=api_response(allowed=[])):
            with self.assertRaisesRegex(RuntimeError, "Pause permission"):
                client.check(require_pause=True)
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value.read.return_value = json.dumps(api_response(id="other")).encode()
        client = surf.SurfClient(config(), opener)
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            client.check()

    def test_api_error_does_not_echo_token_or_response(self):
        opener = Mock()
        opener.open.side_effect = HTTPError("https://example", 403, config()["token"], {}, io.BytesIO(config()["token"].encode()))
        with self.assertRaises(RuntimeError) as error:
            surf.SurfClient(config(), opener).check()
        self.assertIn("HTTP 403", str(error.exception))
        self.assertNotIn(config()["token"], str(error.exception))
        self.assertIsNone(surf.NoRedirects().redirect_request(None, None, 302, "", {}, "https://example"))

    def test_ambiguous_pause_timeout_is_not_retried(self):
        client = surf.SurfClient(config())
        with patch.object(client, "check", return_value={"status": "running", "pause_allowed": True}), \
                patch.object(client.opener, "open", side_effect=TimeoutError) as request:
            with self.assertRaisesRegex(RuntimeError, "may already be queued"):
                client.request_pause()
        self.assertEqual(request.call_count, 1)

    def test_already_pausing_does_not_submit_duplicate(self):
        client = surf.SurfClient(config())
        with patch.object(client, "_request", return_value=api_response(status="pausing")) as request:
            result = client.request_pause()
        self.assertEqual(request.call_count, 1)
        self.assertTrue(result["already_paused_or_pausing"])

    def test_setup_requires_terminal_before_requesting_token(self):
        with patch.object(sys, "argv", ["surf", "setup", "--workspace_id", WORKSPACE, "--workspace_name", "Test VM"]), \
                patch.object(sys.stdin, "isatty", return_value=False), patch.object(surf.getpass, "getpass") as prompt, \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                surf.main()
            prompt.assert_not_called()


class UnattendedTest(unittest.TestCase):
    def run_fixture(self, directory, *, return_code=0, policy="success", backup_error=None, busy_after=False):
        args = arguments(directory, policy)
        events = []
        client = Mock()
        client.check.return_value = {"id": WORKSPACE, "name": "Test VM", "status": "running", "pause_allowed": True}
        client.request_pause.side_effect = lambda: events.append("pause") or {"request_accepted": True, "billing_stop_confirmed": False}
        def child(_args):
            events.append("experiment")
            write_json(Path(args.output_root, "vm_run_config.json"), {"video_sha256": {}})
            write_json(Path(args.output_root, "vm_last_status.json"), {"stage": "full", "complete": return_code == 0})
            Path(args.output_root, "layer.pt").write_bytes(b"saved activation")
            return return_code
        def backup(*_args, **kwargs):
            events.append("backup")
            self.assertFalse(kwargs["reports_only"])
            if backup_error:
                raise RuntimeError(backup_error)
            return Path(directory, "bundle")
        def verify(_bundle):
            events.append("verify")
            return {"complete": True, "reports_only": False, "activation_tensors_included": True}
        def idle():
            events.append("idle")
            if busy_after and events.count("idle") > 1:
                raise RuntimeError("another GPU job")
        return args, events, client, lambda: job.execute_job(args, client, child, backup, verify, idle)

    def test_success_backups_then_verifies_then_pauses_without_claiming_local_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            args, events, client, run = self.run_fixture(directory)
            self.assertEqual(run(), 0)
            self.assertEqual(events, ["idle", "experiment", "backup", "verify", "idle", "pause"])
            status = json.loads((job.lifecycle_directory(args) / "job_status.json").read_text())
            self.assertTrue(status["backup_verified"])
            self.assertFalse(status["local_backup_confirmed"])
            self.assertEqual(status["state"], "pause_requested")
            self.assertFalse(status["pause_result"]["billing_stop_confirmed"])

    def test_failed_experiment_is_backed_up_but_success_policy_does_not_pause(self):
        with tempfile.TemporaryDirectory() as directory:
            args, events, client, run = self.run_fixture(directory, return_code=1)
            self.assertEqual(run(), 1)
            self.assertEqual(events, ["idle", "experiment", "backup", "verify"])
            client.request_pause.assert_not_called()

    def test_finished_policy_can_pause_failed_run_after_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            args, events, client, run = self.run_fixture(directory, return_code=1, policy="finished")
            self.assertEqual(run(), 1)
            self.assertEqual(events[-1], "pause")

    def test_backup_failure_or_other_jobs_prevent_pause(self):
        for options in ({"backup_error": "disk full"}, {"busy_after": True}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                args, events, client, run = self.run_fixture(directory, **options)
                with self.assertRaises(RuntimeError):
                    run()
                client.request_pause.assert_not_called()
                self.assertEqual(json.loads((job.lifecycle_directory(args) / "job_status.json").read_text())["state"], "needs_attention")

    def test_off_policy_has_no_network_or_idle_requirement_but_still_backs_up(self):
        with tempfile.TemporaryDirectory() as directory:
            args, events, client, run = self.run_fixture(directory, policy="off")
            self.assertEqual(run(), 0)
            self.assertEqual(events, ["experiment", "backup", "verify"])
            client.check.assert_not_called()
            client.request_pause.assert_not_called()

    def test_reports_only_or_unverified_backup_cannot_authorize_pause(self):
        for verified in (False, True):
            with self.subTest(verified=verified), tempfile.TemporaryDirectory() as directory:
                args, events, client, run = self.run_fixture(directory)
                def child(_args):
                    write_json(Path(args.output_root, "vm_run_config.json"), {"video_sha256": {}})
                    write_json(Path(args.output_root, "vm_last_status.json"), {"stage": "full", "complete": True})
                    return 0
                verify = Mock(return_value={"complete": True, "reports_only": True, "activation_tensors_included": False})
                if not verified:
                    verify.side_effect = RuntimeError("checksum mismatch")
                with self.assertRaises(RuntimeError):
                    job.execute_job(args, client, child, Mock(return_value=Path(directory, "bundle")), verify, Mock())
                client.request_pause.assert_not_called()

    def test_real_full_backup_includes_activations_and_status_but_not_private_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(directory, policy="off")
            surf.save_private_config(args.surf_config, config())
            def child(_args):
                write_json(Path(args.output_root, "vm_run_config.json"), {"video_sha256": {}})
                write_json(Path(args.output_root, "vm_last_status.json"), {"stage": "full", "complete": True})
                Path(args.output_root, "layer.pt").write_bytes(b"saved activation")
                return 0
            self.assertEqual(job.execute_job(args, child=child), 0)
            status = json.loads((job.lifecycle_directory(args) / "job_status.json").read_text())
            manifest = json.loads(Path(status["backup_dir"], "backup_manifest.json").read_text())
            self.assertIn("run/layer.pt", [entry["path"] for entry in manifest["files"]])
            self.assertIn("run/unattended_run_status.json", [entry["path"] for entry in manifest["files"]])
            self.assertNotIn(config()["token"], json.dumps(manifest))
            self.assertFalse(manifest["local_backup_confirmed"])

    def test_pause_failure_retains_verified_backup_and_pending_status(self):
        with tempfile.TemporaryDirectory() as directory:
            args, events, client, run = self.run_fixture(directory)
            client.request_pause.side_effect = RuntimeError("Pause timed out; check portal")
            with self.assertRaises(RuntimeError):
                run()
            status = json.loads((job.lifecycle_directory(args) / "job_status.json").read_text())
            self.assertTrue(status["backup_verified"])
            self.assertEqual(status["state"], "needs_attention")
            self.assertTrue(status["pause_not_confirmed"])
            client.request_pause.assert_called_once()

    def test_api_precheck_failure_does_not_start_expensive_work(self):
        with tempfile.TemporaryDirectory() as directory:
            args, events, client, run = self.run_fixture(directory)
            client.check.side_effect = RuntimeError("permission denied")
            with self.assertRaises(RuntimeError):
                run()
            self.assertEqual(events, [])

    def test_duplicate_lock_reserved_args_and_unconfirmed_pause_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(directory)
            with job.lifecycle_lock(job.lifecycle_directory(args)):
                with self.assertRaisesRegex(RuntimeError, "already owns"):
                    with job.lifecycle_lock(job.lifecycle_directory(args)):
                        pass
            args.confirm_exclusive_workspace = False
            with self.assertRaisesRegex(ValueError, "whole VM"):
                job.validate_job_args(args)
            args.confirm_exclusive_workspace = True
            args.runner_args = ["--stage=preflight"]
            with self.assertRaisesRegex(ValueError, "not again"):
                job.validate_job_args(args)

    def test_gpu_and_other_runner_guards(self):
        with patch.object(job.subprocess, "run", return_value=SimpleNamespace(stdout="123\n")):
            with self.assertRaisesRegex(RuntimeError, "GPU compute"):
                job.assert_quiescent()
        outputs = [SimpleNamespace(stdout=""), SimpleNamespace(stdout="123 python scripts/run_eval.py --test\n")]
        with patch.object(job.subprocess, "run", side_effect=outputs):
            with self.assertRaisesRegex(RuntimeError, "research runner"):
                job.assert_quiescent()

    def test_launcher_dry_run_needs_no_tmux_api_or_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(directory)
            command = [sys.executable, "scripts/launch_phase3b_vm.py", "--dry_run", "--output_root", args.output_root,
                       "--storage_root", args.storage_root, "--backup_dir", args.backup_dir,
                       "--pause_policy", "success", "--confirm_exclusive_workspace", "--", *args.runner_args]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            plan = json.loads(result.stdout)
            self.assertIn("scripts/run_phase3b_unattended.py", plan["command"][2])
            self.assertEqual(plan["command"][0], sys.executable)
            self.assertFalse(job.lifecycle_directory(args).exists())
            self.assertFalse(Path(args.output_root).exists())

    def test_launcher_starts_detached_job_and_refuses_duplicate_session(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(directory, policy="off")
            argv = ["launch", "--output_root", args.output_root, "--storage_root", args.storage_root,
                    "--backup_dir", args.backup_dir, "--", "--gpus", "0,1"]
            with patch.object(sys, "argv", argv), patch.object(launch.shutil, "which", return_value="/usr/bin/tmux"), \
                    patch.object(launch.subprocess, "run", side_effect=[SimpleNamespace(returncode=1), SimpleNamespace(returncode=0)]) as tmux, \
                    contextlib.redirect_stdout(io.StringIO()):
                launch.main()
            command = tmux.call_args_list[1].args[0]
            self.assertEqual(command[:3], ["tmux", "new-session", "-d"])
            self.assertIn(sys.executable, command[-1])
            self.assertIn("--gpus 0,1", command[-1])
            self.assertIn("2>&1", command[-1])
            self.assertNotIn(config()["token"], command[-1])
            self.assertFalse(Path(args.output_root).exists())
            with patch.object(sys, "argv", argv), patch.object(launch.shutil, "which", return_value="/usr/bin/tmux"), \
                    patch.object(launch.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as tmux, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    launch.main()
                self.assertEqual(tmux.call_count, 1)

    def test_notebook_background_command_keeps_wrapper_arguments_separate(self):
        notebook = json.loads(Path("notebooks/phase3b_vm.ipynb").read_text())
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(directory, policy="off")
            namespace = {"PHASE3B_VM_STAGE": "full", "PHASE3B_VM_BACKGROUND": True,
                         "PHASE3B_VM_CONFIRM_EXCLUSIVE": False, "PHASE3B_VM_PAUSE_POLICY": "off",
                         "PHASE3B_VM_OUTPUT_ROOT": Path(args.output_root), "STORAGE_ROOT": Path(args.storage_root),
                         "PROJECT_ROOT": Path(args.project_root), "PHASE3B_VM_BACKUP_DIR": Path(args.backup_dir),
                         "PHASE3B_VM_SURF_CONFIG": Path(args.surf_config),
                         "vm_runner_options": args.runner_args, "subprocess": subprocess, "sys": sys}
            with patch.object(subprocess, "run") as call:
                exec("".join(notebook["cells"][5]["source"]), namespace)
            command = call.call_args.args[0]
            parser = launch.argparse.ArgumentParser()
            parser.add_argument("--session_name")
            job.add_job_arguments(parser)
            validated = job.validate_job_args(parser.parse_args(command[3:]))
            self.assertEqual(validated.stage, "full")
            self.assertEqual(validated.output_root, args.output_root)
            self.assertEqual(validated.runner_args, args.runner_args)

    def test_notebook_background_last_cell_never_packages_running_job(self):
        notebook = json.loads(Path("notebooks/phase3b_vm.ipynb").read_text())
        with tempfile.TemporaryDirectory() as directory:
            lifecycle = Path(directory, "run_lifecycle")
            write_json(lifecycle / "job_status.json", {"state": "experiment_running", "backup_verified": False})
            namespace = {"PHASE3B_VM_BACKGROUND": True, "PHASE3B_VM_LIFECYCLE": lifecycle, "json": json}
            with patch("scripts.backup_phase3b.create_backup") as backup, \
                    patch("scripts.backup_phase3b.verify_backup") as verify, contextlib.redirect_stdout(io.StringIO()):
                exec("".join(notebook["cells"][7]["source"]), namespace)
            backup.assert_not_called()
            verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
