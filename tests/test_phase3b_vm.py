import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import backup_phase3b, run_phase3b_vm as vm, run_phase3b_patching, analyze_phase3b, run_eval
from scripts.phase3b_paths import load_path_map, resolve_video_path


def setup_selection(directory):
    root = Path(directory)
    selection, project, pool = root / "selection", root / "project", root / "pool"
    selection.mkdir()
    project.mkdir()
    pool.mkdir()
    rows = []
    for base in range(1, 51):
        for condition in ("low_boundary", "temporal_boundary"):
            relative = f"videos/{base}_{condition}.mp4"
            target = pool / relative
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(f"frozen video {base} {condition}".encode())
            rows.append({
                "phase3b_pair_id": f"pair_{base:03d}", "base_sample_id": base,
                "condition": condition, "eval_id": f"eval_{base}_{condition}",
                "first_object_id": 1 if base % 2 else 2,
                "phase3b_analysis_stratum": "primary_rescue",
                "video_path": "/content/drive/MyDrive/vlm_phase3b/rescue_pool/" + relative,
            })
    for filename, items in (("analysis_case_manifest.jsonl", rows), ("preflight_case_manifest.jsonl", rows[:4]),
                             ("selected_video_mappings.jsonl", [{"pair_id": f"pair_{base:03d}", "eligible": True} for base in range(1, 51)])):
        (selection / filename).write_text("".join(json.dumps(row) + "\n" for row in items))
    (selection / "case_selection_summary.json").write_text(json.dumps({"primary_bases": list(range(1, 51)), "selection_purpose": "formal_primary"}))
    return SimpleNamespace(selection_dir=str(selection), project_root=str(project), rescue_pool_root=str(pool),
                           storage_root=str(root), output_root=str(root / "run"), gpus=["0", "1"], stage="preflight",
                           model_name="Qwen/Qwen3-VL-8B-Instruct", model_revision="pinned", expected_transformers_version="5.9.0",
                           expected_torch_version="2.11.0", expected_qwen_vl_utils_version="0.0.14",
                           execution_mode="independent", gpu_weight_budget_gib=10,
                           seed=42, video_fps=None, video_num_frames=None, video_max_pixels=None)


class VMTest(unittest.TestCase):
    def test_path_map_uses_longest_component_prefix_and_preserves_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "video.mp4").write_bytes(b"video")
            row = {"video_path": "/old/pool/video.mp4"}
            self.assertEqual(resolve_video_path(row["video_path"], root, {"/old": "/wrong", "/old/pool": str(root)}), root / "video.mp4")
            self.assertEqual(row["video_path"], "/old/pool/video.mp4")
            with self.assertRaises(FileNotFoundError):
                resolve_video_path("/old/pool_extra/video.mp4", root, {"/old/pool": str(root)})
            with self.assertRaisesRegex(FileNotFoundError, "Mapped video"):
                resolve_video_path(str(root / "video.mp4"), root, {str(root): str(root / "missing")})
            self.assertEqual(resolve_video_path("video.mp4", root), root / "video.mp4")
            config = root / "paths.json"
            config.write_text('{"relative": "/absolute"}')
            with self.assertRaises(ValueError):
                load_path_map(config)

    def test_round_robin_is_disjoint_and_complete(self):
        schedule = vm.shard_schedule(65, ["0", "1"])
        self.assertEqual(schedule["0"], [0, 2, 4, 6, 8, 10, 12])
        self.assertEqual(schedule["1"], [1, 3, 5, 7, 9, 11])
        self.assertEqual(sorted(schedule["0"] + schedule["1"]), list(range(13)))
        with self.assertRaises(ValueError):
            vm.parse_gpus("0,0")

    def test_worker_is_single_gpu_and_persistent_cache(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1", "HF_HUB_CACHE": "/tmp/old"}):
            env = vm.worker_environment("1", "/data/yuxuanstorage")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(env["HF_HOME"], "/data/yuxuanstorage/cache/huggingface")
        self.assertNotIn("HF_HUB_CACHE", env)

    def test_plan_is_cpu_only_and_missing_video_fails_before_model(self):
        with tempfile.TemporaryDirectory() as directory:
            args = setup_selection(directory)
            before = Path(args.selection_dir, "analysis_case_manifest.jsonl").read_bytes()
            plan = vm.build_plan(args)
            self.assertEqual(plan["pair_count"], 50)
            with vm.run_lock(args.output_root):
                vm.save_plan(args.output_root, plan)
            with vm.run_lock(args.output_root):
                vm.save_plan(args.output_root, vm.build_plan(args))
            self.assertEqual(Path(args.selection_dir, "analysis_case_manifest.jsonl").read_bytes(), before)
            Path(args.rescue_pool_root, "videos/50_low_boundary.mp4").unlink()
            with self.assertRaises(FileNotFoundError):
                vm.build_plan(args)

    def test_standalone_plan_cli_runs_without_gpu_or_rescreening(self):
        with tempfile.TemporaryDirectory() as directory:
            args = setup_selection(directory)
            command = [sys.executable, "scripts/run_phase3b_vm.py", "--stage", "plan",
                       "--selection_dir", args.selection_dir, "--rescue_pool_root", args.rescue_pool_root,
                       "--output_root", args.output_root, "--project_root", args.project_root,
                       "--storage_root", args.storage_root, "--gpus", "0,1", "--execution_mode", "independent"]
            for _ in range(2):
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                self.assertIn("No model loaded", result.stdout)
            config = json.loads(Path(args.output_root, "vm_run_config.json").read_text())
            self.assertEqual(config["primary_count"], 50)
            self.assertEqual(config["schedule"]["0"], [0, 2, 4, 6, 8])
            self.assertFalse(Path(args.output_root, "logs").exists())

    def test_provenance_rejects_changed_videos_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            args = setup_selection(directory)
            with vm.run_lock(args.output_root):
                vm.save_plan(args.output_root, vm.build_plan(args))
                Path(args.rescue_pool_root, "videos/1_low_boundary.mp4").write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    vm.save_plan(args.output_root, vm.build_plan(args))

    def test_duplicate_runner_and_colab_preflight_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = setup_selection(directory)
            with vm.run_lock(args.output_root):
                with self.assertRaisesRegex(RuntimeError, "Another runner"):
                    with vm.run_lock(args.output_root):
                        pass
                with self.assertRaisesRegex(RuntimeError, "A10 VM"):
                    vm.require_vm_preflight(args, vm.build_plan(args))

    def test_shards_bind_distinct_workers_and_preserve_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            args = setup_selection(directory)
            calls, lock = [], threading.Lock()
            runner = SimpleNamespace(started=0, cancel=lambda: None)
            def record(command, label, gpu):
                with lock:
                    calls.append((command, label, gpu))
            runner.run = record
            vm.run_shards(args, runner, Path(args.selection_dir, "analysis_case_manifest.jsonl"),
                          lambda gpu: Path(args.output_root, "primary/checkpoints"), {"0": [0, 2], "1": [1]})
            self.assertEqual(len(calls), 6)
            for command, _, gpu in calls:
                shard = int(command[command.index("--shard_index") + 1])
                self.assertEqual(str(shard % 2), gpu)
                self.assertIn("--single_gpu", command)
                self.assertNotIn("--no_verify_standard_generation", command)
                self.assertNotIn("--no_validate_controls", command)

    def test_model_parallel_is_one_worker_with_both_gpus_and_all_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            args = setup_selection(directory)
            args.execution_mode = "model_parallel"
            self.assertEqual(vm.execution_schedule(args, 65), {"model_parallel": list(range(13))})
            plan = vm.build_plan(args)
            args.execution_mode = "independent"
            self.assertNotEqual(plan["pipeline_fingerprint"], vm.build_plan(args)["pipeline_fingerprint"])
            args.execution_mode = "model_parallel"
            calls = []
            runner = SimpleNamespace(started=0, cancel=lambda: None, run=lambda *call: calls.append(call))
            vm.run_shards(args, runner, Path(args.selection_dir, "preflight_case_manifest.jsonl"),
                          lambda worker: vm.preflight_directory(args, worker) / "checkpoints",
                          vm.execution_schedule(args, 2))
            self.assertEqual(len(calls), 2)
            self.assertEqual([call[0][call[0].index("--stage") + 1] for call in calls], ["capture", "patch"])
            for command, _, gpu in calls:
                self.assertEqual(gpu, "0,1")
                self.assertIn("--model_parallel", command)
                self.assertIn("--gpu_weight_budget_gib", command)
                self.assertNotIn("--single_gpu", command)
                self.assertNotIn("--no_validate_controls", command)
            args.stage = "full"
            calls.clear()
            vm.run_shards(args, runner, Path(args.selection_dir, "analysis_case_manifest.jsonl"),
                          lambda worker: Path(args.output_root, "primary/checkpoints"), {"model_parallel": [1]})
            for command, _, _ in calls:
                expected = command[command.index("--expected_gpu_hardware_path") + 1]
                self.assertIn("preflight/model_parallel/checkpoints/shard_00/run_config.json", expected)

    def test_model_parallel_plan_cli_never_initializes_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            args = setup_selection(directory)
            command = [sys.executable, "scripts/run_phase3b_vm.py", "--stage", "plan",
                       "--selection_dir", args.selection_dir, "--rescue_pool_root", args.rescue_pool_root,
                       "--output_root", args.output_root, "--project_root", args.project_root,
                       "--storage_root", args.storage_root]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertIn("No model loaded", result.stdout)
            config = json.loads(Path(args.output_root, "vm_run_config.json").read_text())
            self.assertEqual(config["execution_mode"], "model_parallel")
            self.assertEqual(config["schedule"], {"model_parallel": list(range(10))})

    def test_model_parallel_guard_and_weight_budget(self):
        cuda = run_phase3b_patching.torch.cuda
        model = SimpleNamespace(hf_device_map={"model.layers.0": 0, "model.layers.20": 1},
                                parameters=lambda: [SimpleNamespace(device="cuda:0"), SimpleNamespace(device="cuda:1")])
        with patch.object(cuda, "is_available", return_value=True), patch.object(cuda, "device_count", return_value=2), patch.object(cuda, "get_device_properties", return_value=SimpleNamespace(total_memory=22 * 1024 ** 3)):
            run_phase3b_patching.validate_model_parallel_model(model)
            model.hf_device_map["model.layers.20"] = "cpu"
            with self.assertRaisesRegex(RuntimeError, "offload"):
                run_phase3b_patching.validate_model_parallel_model(model)
            model.hf_device_map["model.layers.20"] = 1
            model.parameters = lambda: [SimpleNamespace(device="cuda:0")]
            with self.assertRaisesRegex(RuntimeError, "Both GPUs"):
                run_phase3b_patching.validate_model_parallel_model(model)
            with self.assertRaises(ValueError):
                run_phase3b_patching.validate_model_parallel_model(weight_budget_gib=22)
        with patch.object(cuda, "is_available", return_value=True), patch.object(cuda, "device_count", return_value=1):
            with self.assertRaisesRegex(RuntimeError, "exactly two"):
                run_phase3b_patching.validate_model_parallel_model()

    def test_model_parallel_loader_preserves_precision_and_eager(self):
        args = SimpleNamespace(model_parallel=True, gpu_weight_budget_gib=10, single_gpu=False,
                               model_name="Qwen/Qwen3-VL-8B-Instruct", model_revision="pinned", attn_implementation="eager")
        with patch.object(run_phase3b_patching, "validate_model_parallel_model") as validate, patch.object(run_phase3b_patching, "load_model", return_value=("model", "processor")) as load:
            self.assertEqual(run_phase3b_patching.load_phase3b_model(args), ("model", "processor"))
            self.assertEqual(validate.call_count, 2)
            self.assertEqual(load.call_args.kwargs["max_memory"], {0: 10 * 1024 ** 3, 1: 10 * 1024 ** 3, "cpu": 0})
            self.assertEqual(load.call_args.kwargs["device_map"], "balanced")
            self.assertEqual(load.call_args.kwargs["attn_implementation"], "eager")
            self.assertNotIn("load_in_4bit", load.call_args.kwargs)
        from unittest.mock import MagicMock
        auto_model, auto_processor = MagicMock(), MagicMock()
        with patch.object(run_eval, "AutoModelForImageTextToText", auto_model), patch.object(run_eval, "AutoProcessor", auto_processor):
            run_eval.load_model("checkpoint", device_map="balanced", max_memory={0: 10, 1: 10, "cpu": 0})
            self.assertEqual(auto_model.from_pretrained.call_args.kwargs["dtype"], run_eval.torch.float16)
            self.assertEqual(auto_model.from_pretrained.call_args.kwargs["device_map"], "balanced")
            run_eval.load_model("checkpoint")
            self.assertEqual(auto_model.from_pretrained.call_args.kwargs["device_map"], "auto")
            self.assertNotIn("max_memory", auto_model.from_pretrained.call_args.kwargs)

    def test_actual_placement_is_saved_and_must_match_preflight_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory, "run_config.json")
            config.write_text('{"run_fingerprint": "unchanged"}')
            model = SimpleNamespace(hf_device_map={"early": 0, "late": 1})
            run_phase3b_patching.record_model_placement(config, model)
            saved = json.loads(config.read_text())
            self.assertEqual(saved["model_device_map"], {"early": "0", "late": "1"})
            model.hf_device_map = {"early": 1, "late": 0}
            with self.assertRaisesRegex(RuntimeError, "placement differs"):
                run_phase3b_patching.record_model_placement(config, model)
            self.assertEqual(json.loads(config.read_text()), saved)
            fresh = Path(directory, "new_config.json")
            fresh.write_text('{}')
            with self.assertRaisesRegex(RuntimeError, "placement differs"):
                run_phase3b_patching.record_model_placement(fresh, model, config)

    def test_model_parallel_preflight_gate_does_not_accept_replica_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            args = setup_selection(directory)
            args.execution_mode = "model_parallel"
            plan = vm.build_plan(args)
            root = Path(args.output_root)
            root.mkdir()
            vm.write_json(root / "vm_preflight_summary.json", {"complete": True, "gpus": args.gpus, "pipeline_fingerprint": plan["pipeline_fingerprint"]})
            base = vm.preflight_directory(args, "model_parallel")
            vm.write_json(base / "checkpoints/shard_00/activations/pair/low_boundary/index.json",
                          {"decision": {"prediction": "A"}, "standard_parity": {"first_token_match": True, "logits_allclose": True}})
            vm.write_json(base / "analysis/aggregate_summary.json", {})
            vm.write_json(root / "relocation_control/relocation_summary.json", {})
            vm.require_vm_preflight(args, plan)
            args.execution_mode = "independent"
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                vm.require_vm_preflight(args, vm.build_plan(args))

    def test_preflight_model_parallel_orchestration_including_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            args = setup_selection(directory)
            calls = []
            def run(command, label, gpu=None):
                calls.append((command, label, gpu))
                output = Path(command[command.index("--output_dir") + 1])
                if "scripts/run_phase3b_patching.py" in command:
                    for pair_id, pair in vm.manifest_pairs(Path(args.selection_dir, "preflight_case_manifest.jsonl")).items():
                        for condition in pair:
                            vm.write_json(output / "shard_00/activations" / pair_id / condition / "index.json",
                                          {"decision": {"prediction": "A", "margin": 1},
                                           "standard_parity": {"first_token_match": True, "logits_allclose": True}})
                elif "scripts/analyze_phase3b.py" in command:
                    vm.write_json(output / "aggregate_summary.json", dict.fromkeys((
                        "missing_patch_count", "missing_capture_count", "missing_divergence_count", "missing_technical_control_count",
                    ), 0))
                else:
                    self.assertIn("scripts/run_phase3b_relocation_control.py", command)
                    self.assertIn("--expected_gpu_hardware_path", command)
                    vm.write_json(output / "relocation_summary.json", {})
            runner = SimpleNamespace(started=vm.time.perf_counter(), cancel=lambda: None, run=run)
            argv = ["runner", "--stage", "preflight", "--selection_dir", args.selection_dir,
                    "--rescue_pool_root", args.rescue_pool_root, "--output_root", args.output_root,
                    "--storage_root", args.storage_root, "--project_root", args.project_root]
            with patch.object(vm.sys, "argv", argv), patch.object(vm, "LoggedRunner", return_value=runner):
                vm.main()
            self.assertEqual(len(calls), 4)
            self.assertEqual([call[2] for call in calls], ["0,1", "0,1", None, "0,1"])
            summary = json.loads(Path(args.output_root, "vm_preflight_summary.json").read_text())
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["execution_mode"], "model_parallel")
            self.assertIsNone(summary["cross_gpu_margin_differences"])
            self.assertEqual(len(summary["capture_audits"]["model_parallel"]), 4)

    def test_hardware_gate_rejects_changed_weight_budget_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(expected_gpu_hardware_path=str(Path(directory, "config.json")),
                                   model_parallel=True, single_gpu=False, gpu_weight_budget_gib=10)
            vm.write_json(args.expected_gpu_hardware_path, {"gpu_hardware": {"devices": 2}, "single_gpu": False,
                                                           **run_phase3b_patching.model_placement_settings(args)})
            with patch.object(run_phase3b_patching, "gpu_hardware_metadata", return_value={"devices": 2}):
                run_phase3b_patching.validate_preflight_hardware(args)
                args.gpu_weight_budget_gib = 9
                with self.assertRaisesRegex(RuntimeError, "settings differ"):
                    run_phase3b_patching.validate_preflight_hardware(args)

    def test_cpu_merge_rejects_different_actual_model_placement(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, device in enumerate(("0", "1")):
                vm.write_json(Path(directory, f"shard_{index:02d}/run_config.json"), {
                    "run_fingerprint": str(index), "model_parallel": True,
                    "model_device_map": {"model.layers.0": device}, "shard_index": index, "pair_ids": [str(index)],
                })
            with self.assertRaisesRegex(ValueError, "incompatible"):
                analyze_phase3b.read_shards(directory)

    def test_logged_worker_surfaces_real_error_and_checkpoints_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_root=directory, storage_root=directory, project_root=directory)
            checkpoint = Path(directory, "saved.pt")
            checkpoint.write_bytes(b"saved")
            runner = vm.LoggedRunner(args)
            with self.assertRaisesRegex(RuntimeError, "full traceback"):
                runner.run([sys.executable, "-c", "import os; print(os.environ['CUDA_VISIBLE_DEVICES'], flush=True); raise RuntimeError('real failure')"], "failure", "1")
            output = next(Path(directory, "logs").glob("*.log")).read_text()
            self.assertIn("real failure", output)
            self.assertEqual(checkpoint.read_bytes(), b"saved")
            self.assertTrue(runner.stop.is_set())

    def test_capture_audit_reads_actual_standard_parity_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "shard_00/activations/pair/low_boundary/index.json")
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"decision": {"prediction": "A", "margin": 1},
                                        "standard_parity": {"first_token_match": True, "logits_allclose": True}}))
            self.assertEqual(len(vm.capture_audit(directory)), 1)
            path.write_text('{"standard_parity": null}')
            with self.assertRaisesRegex(RuntimeError, "parity"):
                vm.capture_audit(directory)

    def test_single_gpu_guard_rejects_cpu_offload_and_multiple_visible_gpus(self):
        with patch.object(run_phase3b_patching.torch.cuda, "is_available", return_value=True), patch.object(run_phase3b_patching.torch.cuda, "device_count", return_value=1):
            run_phase3b_patching.validate_single_gpu_model(SimpleNamespace(hf_device_map={"model": 0}))
            with self.assertRaisesRegex(RuntimeError, "offloaded"):
                run_phase3b_patching.validate_single_gpu_model(SimpleNamespace(hf_device_map={"model": "cpu"}))
        with patch.object(run_phase3b_patching.torch.cuda, "is_available", return_value=True), patch.object(run_phase3b_patching.torch.cuda, "device_count", return_value=2):
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                run_phase3b_patching.validate_single_gpu_model()

    def test_cpu_merge_rejects_mixed_gpu_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, gpu in enumerate(("A100", "A10")):
                path = Path(directory, f"shard_{index:02d}/run_config.json")
                path.parent.mkdir()
                path.write_text(json.dumps({"run_fingerprint": str(index), "gpu_hardware": {"name": gpu}, "shard_index": index, "pair_ids": [str(index)]}))
            with self.assertRaisesRegex(ValueError, "incompatible"):
                analyze_phase3b.read_shards(directory)

    def test_full_backup_includes_tensors_and_verifies_parts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory, "run")
            root.mkdir()
            (root / "layer_00.pt").write_bytes(b"saved activation")
            (root / "summary.json").write_text('{"complete": true}')
            bundle = backup_phase3b.create_backup(root, Path(directory, "backups"), part_bytes=10)
            manifest = backup_phase3b.verify_backup(bundle)
            self.assertFalse(manifest["local_backup_confirmed"])
            self.assertTrue(manifest["activation_tensors_included"])
            self.assertIn("run/layer_00.pt", [row["path"] for row in manifest["files"]])
            self.assertGreater(len(manifest["archives"]), 1)
            part = bundle / manifest["archives"][0]["name"]
            part.write_bytes(part.read_bytes() + b"damaged")
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                backup_phase3b.verify_backup(bundle)

    def test_backup_refuses_low_space_and_reports_only_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory, "run")
            root.mkdir()
            (root / "layer.pt").write_bytes(b"activation")
            (root / "summary.json").write_text("{}")
            with patch.object(backup_phase3b.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
                with self.assertRaisesRegex(RuntimeError, "Do not delete checkpoints"):
                    backup_phase3b.create_backup(root, Path(directory, "backups"))
            bundle = backup_phase3b.create_backup(root, Path(directory, "backups"), reports_only=True)
            manifest = backup_phase3b.verify_backup(bundle)
            self.assertFalse(manifest["activation_tensors_included"])
            self.assertNotIn("run/layer.pt", [row["path"] for row in manifest["files"]])
            manifest["complete"] = False
            (bundle / "backup_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                backup_phase3b.verify_backup(bundle)

    def test_notebook_cells_are_syntactically_valid(self):
        for name in ("phase3b_vm.ipynb", "colab_eval.ipynb"):
            notebook = json.loads(Path("notebooks", name).read_text())
            cells = notebook["cells"] if name == "phase3b_vm.ipynb" else notebook["cells"][-1:]
            for cell in cells:
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), name, "exec")


if __name__ == "__main__":
    unittest.main()
