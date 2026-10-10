import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_phase3c_vm as fixtures
import test_phase3c_primary as primary_fixtures
from scripts import analyze_phase3c as analysis, inspect_phase3c as status
from scripts import phase3c_core as core, run_phase3c_preflight as runner


PROJECT = Path(__file__).resolve().parents[1]


def snapshot(root):
    return {str(path.relative_to(root)): (path.stat().st_mtime_ns, core.file_hash(path))
            for path in Path(root).rglob("*") if path.is_file()}


class Phase3CReadinessTest(unittest.TestCase):
    def fixture(self, directory):
        return fixtures.Phase3CVmTest().fixture(directory)

    def test_missing_artifacts_are_reported_without_creating_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "not-created"
            report = status.inspect(missing, verify=True)
            self.assertEqual(report["next_action"], "audit")
            self.assertFalse(missing.exists())
            self.assertFalse(report["full_launch_prerequisites_verified"])
            self.assertTrue(report["stages"]["audit"]["missing_paths"])
            self.assertEqual(report["diagnostics"], [])

    def test_default_status_is_lightweight_and_does_not_promote_recorded_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            _, data = self.fixture(directory)
            plan, root = data[:2]
            core.atomic_write(root / "baseline/summary.json", {"passed": True, "expected": 24, "completed": 24})
            core.atomic_write(root / "baseline/rows.jsonl", [], jsonl=True)
            core.atomic_write(root / "baseline/task_manifest.jsonl", [], jsonl=True)
            core.atomic_write(root / "baseline/progress_status.json", {"passed_tasks": 24, "elapsed_sec": 9})
            before = snapshot(Path(directory))
            with patch.object(status, "load_selection", side_effect=AssertionError("not lightweight")), \
                 patch.object(status, "verify_stage", side_effect=AssertionError("no heavy checks")):
                report = status.inspect(plan, root)
            self.assertEqual(report["stages"]["baseline"]["status"], "recorded_complete")
            self.assertFalse(report["stages"]["baseline"]["verified"])
            self.assertFalse(report["full_launch_prerequisites_verified"])
            self.assertEqual(report["stages"]["baseline"]["reported_progress"]["passed_tasks"], 24)
            self.assertEqual(snapshot(Path(directory)), before)

    def test_partial_freeze_is_not_repaired_by_read_only_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            _, data = self.fixture(directory)
            plan, root = data[:2]
            (plan / "selection/case_manifest.jsonl").unlink()
            before = snapshot(Path(directory))
            report = status.inspect(plan, root, True)
            self.assertFalse(report["stages"]["freeze"]["verified"])
            self.assertEqual(report["next_action"], "freeze")
            self.assertEqual(snapshot(Path(directory)), before)

    def test_complete_freeze_can_be_verified_without_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            _, data = self.fixture(directory)
            plan, root = data[:2]
            before = snapshot(Path(directory))
            report = status.inspect(plan, root, True)
            self.assertTrue(report["stages"]["freeze"]["verified"])
            self.assertTrue(report["stages"]["mapping"]["verified"])
            self.assertFalse(report["full_launch_prerequisites_verified"])
            self.assertEqual(report["next_action"], "baseline")
            self.assertEqual(snapshot(Path(directory)), before)

    def test_incomplete_valid_checkpoint_is_resumable_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            _, data = self.fixture(directory)
            plan, root, frozen, pairs, mappings, config, engine = data
            runner.run_stage(root, frozen, pairs, mappings, engine, config, "baseline", max_tasks=2)
            report = status.inspect(plan, root, True)
            self.assertEqual(report["stages"]["baseline"]["status"], "incomplete")
            self.assertEqual(report["next_action"], "baseline")
            self.assertFalse(report["full_launch_prerequisites_verified"])
            self.assertEqual(report["diagnostics"], [])

    def test_failed_baseline_requires_diagnosis_not_blind_resume(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            _, data = self.fixture(directory)
            plan, root, frozen, pairs, mappings, config, engine = data
            engine.fail = True
            runner.run_stage(root, frozen, pairs, mappings, engine, config, "baseline")
            report = status.inspect(plan, root, True)
            self.assertEqual(report["stages"]["baseline"]["status"], "recorded_failure")
            self.assertEqual(report["next_action"], "diagnose_and_preserve_artifacts")

    def test_gates_verified_but_full_result_still_pending_and_corruption_blocks(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            _, data = self.fixture(directory)
            plan, root, frozen, pairs, mappings, config, engine = data
            helper = primary_fixtures.Phase3CPrimaryTest()
            helper.gates(data)
            report = status.inspect(plan, root, True)
            self.assertTrue(report["full_launch_prerequisites_verified"])
            self.assertFalse(report["primary_grid_and_analysis_verified"])
            self.assertEqual(report["next_action"], "patch")
            first = next((root / "captures").rglob("test_fixture.json"))
            first.write_text("corrupt-test-only")
            report = status.inspect(plan, root, True)
            self.assertFalse(report["full_launch_prerequisites_verified"])
            self.assertEqual(report["stages"]["baseline"]["status"], "invalid")
            self.assertEqual(report["next_action"], "diagnose_and_preserve_artifacts")

    def test_fully_complete_fixture_requires_all_grid_and_analysis_checksums(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            _, data = self.fixture(directory)
            plan, root, frozen, pairs, mappings, config, engine = data
            primary_fixtures.Phase3CPrimaryTest().gates(data)
            for stage in ("patch", "routing", "knockout"):
                runner.run_stage(root, frozen, pairs, mappings, engine, config, stage)
            analysis.analyze(plan, root, repeats=100, plots=False)
            before = snapshot(Path(directory))
            report = status.inspect(plan, root, True)
            self.assertTrue(report["primary_grid_and_analysis_verified"])
            self.assertEqual(report["next_action"], "review_report_and_verify_local_backup")
            self.assertEqual(snapshot(Path(directory)), before)
            (root / "analysis/analysis_config.json").write_text("malformed-test-json")
            report = status.inspect(plan, root, True)
            self.assertFalse(report["primary_grid_and_analysis_verified"])
            self.assertEqual(report["stages"]["analyze"]["status"], "invalid")

    def test_modified_preparation_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, data = self.fixture(directory)
            plan, root = data[:2]
            config = core.read_json(plan / "plan_config.json")
            config["artifact_type"] = "mock"
            core.atomic_write(plan / "plan_config.json", config)
            report = status.inspect(plan, root, True)
            self.assertEqual(report["stages"]["audit"]["status"], "invalid")
            self.assertFalse(report["full_launch_prerequisites_verified"])

    def test_inspector_cli_emits_only_json_and_never_imports_model_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            guard = "\n".join([
                "import importlib.abc, runpy, sys",
                "class NoModels(importlib.abc.MetaPathFinder):",
                "    def find_spec(self, fullname, path=None, target=None):",
                "        if fullname.split('.')[0] in {'torch', 'transformers', 'qwen_vl_utils'}:",
                "            raise AssertionError('Inspector must not import model packages')",
                "sys.meta_path.insert(0, NoModels())",
                "sys.path.insert(0, 'scripts')",
                "sys.argv = ['scripts/inspect_phase3c.py', *sys.argv[1:]]",
                "runpy.run_path('scripts/inspect_phase3c.py', run_name='__main__')",
            ])
            command = [sys.executable, "-c", guard, "--plan_dir", str(Path(directory) / "absent"), "--json", "--verify"]
            result = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertTrue(report["read_only"])
            self.assertFalse(report["inspector_started_processor_or_gpu"])
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
