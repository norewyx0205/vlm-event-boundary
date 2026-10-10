import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_phase3c_vm as fixtures


PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT / "notebooks/phase3c_vm.ipynb"


class Phase3CNotebookTest(unittest.TestCase):
    def setUp(self):
        self.prior_cwd = Path.cwd()
        self.cells = json.loads(NOTEBOOK.read_text())["cells"]

    def tearDown(self):
        os.chdir(self.prior_cwd)

    def code(self, cell_id):
        return "".join(next(cell for cell in self.cells if cell["id"] == cell_id)["source"])

    def configured(self, directory):
        code = self.code("phase3c-config").replace("Path('/data/yuxuanstorage')", f"Path({directory!r})")
        code = code.replace("PROJECT_ROOT = STORAGE_ROOT / 'vlm-event-boundary'", f"PROJECT_ROOT = Path({str(PROJECT)!r})")
        namespace = {}
        exec(compile(code, str(NOTEBOOK), "exec"), namespace)
        return namespace

    def execute(self, cell_id, namespace):
        exec(compile(self.code(cell_id), str(NOTEBOOK), "exec"), namespace)

    def test_notebook_has_unique_cells_clean_outputs_and_compilable_code(self):
        notebook = json.loads(NOTEBOOK.read_text())
        self.assertEqual(notebook["nbformat"], 4)
        ids = [cell["id"] for cell in self.cells]
        self.assertEqual(len(ids), len(set(ids)))
        for cell in self.cells:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                compile("".join(cell["source"]), str(NOTEBOOK), "exec")

    def test_run_all_defaults_to_read_only_status_without_gpu_backup_or_network(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            namespace = self.configured(directory)
            before = list(Path(directory).rglob("*"))
            with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as calls:
                for cell_id in ("phase3c-readiness", "phase3c-prepare", "phase3c-execute", "phase3c-backup-status"):
                    self.execute(cell_id, namespace)
            self.assertEqual(namespace["PHASE3C_ACTION"], "status")
            self.assertEqual(calls.call_count, 1)
            self.assertEqual(calls.call_args.args[0][2], "scripts/inspect_phase3c.py")
            self.assertNotIn("--verify", calls.call_args.args[0])
            self.assertEqual(calls.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
            self.assertFalse(namespace["PHASE3C_CONFIRM_GPU_RUN"])
            self.assertFalse(namespace["PHASE3C_EMAIL_NOTIFY"])
            self.assertEqual(namespace["PHASE3C_PAUSE_POLICY"], "off")
            self.assertEqual(before, list(Path(directory).rglob("*")))

    def test_prepare_requires_confirmation_and_runs_cpu_steps_only(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            namespace = self.configured(directory)
            namespace["PHASE3C_ACTION"] = "prepare"
            with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as calls:
                with self.assertRaises(RuntimeError):
                    self.execute("phase3c-prepare", namespace)
                calls.assert_not_called()
                namespace["PHASE3C_CONFIRM_CPU_PREPARATION"] = True
                self.execute("phase3c-prepare", namespace)
                self.assertEqual(calls.call_count, 3)
                self.assertEqual([call.args[0][2] for call in calls.call_args_list],
                                 ["scripts/prepare_phase3c.py", "scripts/audit_phase3c_mappings.py", "scripts/prepare_phase3c.py"])
                for call in calls.call_args_list:
                    self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
                    self.assertEqual(call.kwargs["env"]["HF_HOME"], str(Path(directory) / "cache/huggingface"))

    def test_cpu_failure_stops_before_freeze_or_gpu(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            namespace = self.configured(directory)
            namespace.update(PHASE3C_ACTION="prepare", PHASE3C_CONFIRM_CPU_PREPARATION=True)
            with patch.object(subprocess, "run", side_effect=[subprocess.CompletedProcess([], 0), subprocess.CalledProcessError(1, "mapping")]) as calls:
                with self.assertRaises(subprocess.CalledProcessError):
                    self.execute("phase3c-prepare", namespace)
                self.assertEqual(calls.call_count, 2)

    def test_frozen_cohort_is_validated_and_reused_without_rewriting_preparation(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            args, data = fixtures.Phase3CVmTest().fixture(directory)
            namespace = self.configured(directory)
            namespace.update(PHASE3C_ACTION="prepare", PHASE3C_CONFIRM_CPU_PREPARATION=True,
                             PHASE3C_PLAN=data[0], PHASE3C_RUN=data[1])
            before = {str(path): path.stat().st_mtime_ns for path in Path(directory).rglob("*") if path.is_file()}
            with patch.object(subprocess, "run") as calls:
                self.execute("phase3c-prepare", namespace)
                calls.assert_not_called()
            self.assertEqual(before, {str(path): path.stat().st_mtime_ns for path in Path(directory).rglob("*") if path.is_file()})
            namespace["PHASE3C_ACTION"] = "mapping"
            with self.assertRaises(RuntimeError):
                self.execute("phase3c-prepare", namespace)

    def test_gpu_confirmation_and_review_are_independent_requirements(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            namespace = self.configured(directory)
            with patch.object(subprocess, "run") as calls:
                for stage in ("baseline", "preflight", "full"):
                    namespace["PHASE3C_ACTION"] = stage
                    with self.assertRaises(RuntimeError):
                        self.execute("phase3c-execute", namespace)
                namespace.update(PHASE3C_ACTION="full", PHASE3C_CONFIRM_GPU_RUN=True)
                with self.assertRaisesRegex(RuntimeError, "Review"):
                    self.execute("phase3c-execute", namespace)
                calls.assert_not_called()

    def test_explicit_full_uses_new_lifecycle_and_no_legacy_experiment(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            namespace = self.configured(directory)
            namespace.update(PHASE3C_ACTION="full", PHASE3C_CONFIRM_GPU_RUN=True, PHASE3C_PREFLIGHT_REVIEWED=True,
                             PHASE3C_EMAIL_NOTIFY=True, PHASE3C_PAUSE_POLICY="finished", PHASE3C_CONFIRM_EXCLUSIVE=True)
            with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as calls:
                self.execute("phase3c-execute", namespace)
                command = calls.call_args.args[0]
                self.assertEqual(command[2], "scripts/launch_phase3c_vm.py")
                self.assertEqual(command[command.index("--stage") + 1], "full")
                self.assertEqual(command[command.index("--gpus") + 1], "0,1")
                self.assertIn("--email_notify", command)
                self.assertIn("--confirm_exclusive_workspace", command)
                self.assertNotIn("run_phase3b_vm.py", " ".join(command))

    def test_dry_run_full_needs_no_gpu_confirmation_or_review_flag(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            namespace = self.configured(directory)
            namespace["PHASE3C_ACTION"] = "dry_run_full"
            with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as calls:
                self.execute("phase3c-execute", namespace)
                self.assertIn("--dry_run", calls.call_args.args[0])

    def test_foreground_cannot_bypass_lifecycle_for_pause_or_email(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            namespace = self.configured(directory)
            namespace.update(PHASE3C_ACTION="baseline", PHASE3C_BACKGROUND=False, PHASE3C_CONFIRM_GPU_RUN=True,
                             PHASE3C_EMAIL_NOTIFY=True)
            with patch.object(subprocess, "run") as calls:
                with self.assertRaises(ValueError):
                    self.execute("phase3c-execute", namespace)
                calls.assert_not_called()

    def test_backup_is_explicit_and_not_invoked_by_end_status_cell(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            namespace = self.configured(directory)
            namespace["PHASE3C_ACTION"] = "backup"
            with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as calls:
                with self.assertRaises(RuntimeError):
                    self.execute("phase3c-backup-status", namespace)
                calls.assert_not_called()
                namespace["PHASE3C_CONFIRM_BACKUP"] = True
                self.execute("phase3c-backup-status", namespace)
                self.assertEqual(calls.call_args.args[0][2], "scripts/backup_phase3c.py")
                self.assertNotIn("--reports_only", calls.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
