"""Run frozen Phase 3B shards with isolated replicas or one two-GPU model; never re-screen cases."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

try:
    from .common import PROJECT_ROOT, read_jsonl
    from .phase3b_paths import resolve_video_path
except ImportError:
    from common import PROJECT_ROOT, read_jsonl
    from phase3b_paths import resolve_video_path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def run_lock(root):
    Path(root).mkdir(parents=True, exist_ok=True)
    with (Path(root) / ".pipeline.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another runner or backup is using this output root.") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def parse_gpus(value):
    gpus = value.split(",")
    if not gpus or any(not part.isdigit() for part in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError("Use distinct physical GPU indices, e.g. --gpus 0,1.")
    return gpus


def shard_schedule(pair_count, gpus, shard_size=5):
    if pair_count < 1 or shard_size < 1:
        raise ValueError("Pair count and shard size must be positive.")
    return {gpu: list(range(index, math.ceil(pair_count / shard_size), len(gpus))) for index, gpu in enumerate(gpus)}


def execution_schedule(args, pair_count):
    if args.execution_mode == "model_parallel":
        return {"model_parallel": list(range(math.ceil(pair_count / 5)))}
    return shard_schedule(pair_count, args.gpus)


def visible_devices(args, worker):
    return ",".join(args.gpus) if args.execution_mode == "model_parallel" else worker


def preflight_directory(args, worker):
    name = "model_parallel" if args.execution_mode == "model_parallel" else f"gpu_{worker}"
    return Path(args.output_root) / "preflight" / name


def worker_environment(gpu, storage_root):
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "PYTHONUNBUFFERED": "1", "HF_HOME": str(Path(storage_root) / "cache/huggingface"),
        "TORCH_HOME": str(Path(storage_root) / "cache/torch"),
    })
    # Old overrides must not redirect the model cache to an ephemeral root disk.
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        env.pop(name, None)
    return env


def manifest_pairs(path):
    result = {}
    ids = set()
    for row in read_jsonl(path):
        pair = result.setdefault(row["phase3b_pair_id"], {})
        if row["condition"] in pair or row["eval_id"] in ids:
            raise ValueError(f"Duplicate row in {path}.")
        pair[row["condition"]] = row
        ids.add(row["eval_id"])
    if not result or any(set(pair) != {"low_boundary", "temporal_boundary"} for pair in result.values()):
        raise ValueError(f"Missing complete low/temporal pairs in {path}.")
    return dict(sorted(result.items()))


def build_plan(args):
    selection = Path(args.selection_dir).resolve()
    summary = json.loads((selection / "case_selection_summary.json").read_text(encoding="utf-8"))
    if len(set(summary["primary_bases"])) != 50 or summary.get("selection_purpose", "formal_primary") != "formal_primary":
        raise ValueError("VM execution requires the already frozen formal 50-case selection.")
    manifests = {name: selection / filename for name, filename in (
        ("preflight", "preflight_case_manifest.jsonl"), ("full", "analysis_case_manifest.jsonl"),
    )}
    pairs = {name: manifest_pairs(path) for name, path in manifests.items()}
    primary = [pair["low_boundary"] for pair in pairs["full"].values() if pair["low_boundary"]["phase3b_analysis_stratum"] == "primary_rescue"]
    if len(primary) != 50 or {row["base_sample_id"] for row in primary} != set(summary["primary_bases"]):
        raise ValueError("Full manifest does not contain exactly the frozen 50 independent primary bases.")
    if len(pairs["preflight"]) != 2 or {int(pair["low_boundary"]["first_object_id"]) for pair in pairs["preflight"].values()} != {1, 2}:
        raise ValueError("Preflight must contain one rescue for each first-mover role.")
    if any(key not in pairs["full"] or pair != pairs["full"][key] for key, pair in pairs["preflight"].items()):
        raise ValueError("Preflight is not a subset of the frozen full manifest.")
    mapping_path = selection / "selected_video_mappings.jsonl"
    mapping_rows = read_jsonl(mapping_path)
    mappings = {row["pair_id"]: row for row in mapping_rows}
    if len(mappings) != len(mapping_rows):
        raise ValueError("Duplicate frozen mappings.")
    path_map = {
        "/content/drive/MyDrive/vlm_phase3b/rescue_pool": str(Path(args.rescue_pool_root).resolve()),
        "/content/vlm-event-boundary": str(Path(args.project_root).resolve()),
    }
    video_hashes, video_paths = {}, {}
    for key, pair in pairs["full"].items():
        if key not in mappings or not mappings[key]["eligible"]:
            raise ValueError(f"Missing/ineligible frozen mapping for {key}.")
        for row in pair.values():
            path = resolve_video_path(row["video_path"], args.project_root, path_map)
            video_paths[row["eval_id"]] = str(path)
            video_hashes[str(path)] = video_hashes.get(str(path)) or digest(path)
    weight_budget = float(args.gpu_weight_budget_gib)
    if weight_budget.is_integer():
        weight_budget = int(weight_budget)
    payload = {
        "schema": "phase3b_vm_execution_v1", "artifact_type": "real",
        "selection_dir": str(selection), "path_map": path_map,
        "selection_sha256": digest(selection / "case_selection_summary.json"),
        "manifest_sha256": {name: digest(path) for name, path in manifests.items()},
        "mapping_sha256": digest(mapping_path),
        "source_code_sha256": {name: digest(Path(__file__).parent / name) for name in (
            "run_phase3b_vm.py", "run_phase3b_patching.py", "run_phase3b_relocation_control.py",
            "phase3b_core.py", "phase3b_paths.py", "probe_attention_roi.py", "run_eval.py",
            "activation_patching_core.py", "analyze_phase3b.py",
        )},
        "model_name": args.model_name, "model_revision": args.model_revision,
        "expected_transformers_version": args.expected_transformers_version,
        "expected_torch_version": args.expected_torch_version,
        "expected_qwen_vl_utils_version": args.expected_qwen_vl_utils_version,
        "seed": args.seed, "dtype": "float16", "attn_implementation": "eager",
        "video_fps": args.video_fps, "video_num_frames": args.video_num_frames,
        "video_max_pixels": args.video_max_pixels, "roi_padding": 8, "shard_size": 5,
        "video_sha256": video_hashes, "video_paths_by_eval_id": video_paths,
        "pair_count": len(pairs["full"]), "primary_count": 50,
        "execution_mode": args.execution_mode,
        "gpu_weight_budget_gib": weight_budget if args.execution_mode == "model_parallel" else None,
        "gpus": args.gpus,
        "reuse_source_config_sha256": digest(Path(args.reuse_completed_from) / "vm_run_config.json") if getattr(args, "reuse_completed_from", None) else None,
        "reuse_source_root": str(Path(args.reuse_completed_from).resolve()) if getattr(args, "reuse_completed_from", None) else None,
        "baseline_gate_code_sha256": {name: digest(Path(__file__).parent / name) for name in (
            "phase3b_baseline.py", "run_phase3b_baseline.py", "phase3b_checkpoint_reuse.py",
        )},
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {**payload, "pipeline_fingerprint": fingerprint,
            "schedule": execution_schedule(args, len(pairs["full"]))}


def save_plan(root, plan):
    path = Path(root) / "vm_run_config.json"
    if path.is_file():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior["pipeline_fingerprint"] != plan["pipeline_fingerprint"]:
            raise RuntimeError("VM code, cohort, videos or settings changed. Preserve this run and use a new --output_root.")
    elif any(Path(root).iterdir()) and set(path.name for path in Path(root).iterdir()) != {".pipeline.lock"}:
        raise RuntimeError("Output root has artifacts but no VM provenance; use a fresh A10 run directory.")
    write_json(path, plan)
    write_json(Path(root) / "path_map.json", plan["path_map"])
    frozen = Path(root) / "selection"
    frozen.mkdir(exist_ok=True)
    for source in Path(plan["selection_dir"]).glob("*"):
        if source.is_file():
            target = frozen / source.name
            if target.is_file() and digest(target) != digest(source):
                raise RuntimeError(f"Saved selection differs: {target}.")
            if not target.exists():
                shutil.copy2(source, target)


class LoggedRunner:
    def __init__(self, args):
        self.args = args
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.active = set()
        self.started = time.perf_counter()

    def cancel(self):
        self.stop.set()
        with self.lock:
            for process in list(self.active):
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def run(self, command, label, gpu=None):
        if self.stop.is_set():
            raise RuntimeError("Pipeline stopped; saved checkpoints remain intact.")
        log = Path(self.args.output_root) / "logs" / f"{label}_{time.time_ns()}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = worker_environment(gpu, self.args.storage_root) if gpu is not None else {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        print(f"[{label}] starting; elapsed={(time.perf_counter() - self.started)/60:.1f} min; log={log}", flush=True)
        with self.lock:
            if self.stop.is_set():
                raise RuntimeError("Pipeline stopped before worker launch.")
            process = subprocess.Popen(command, cwd=self.args.project_root, env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                       start_new_session=True)
            self.active.add(process)
        try:
            with log.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({"command": command, "gpu": gpu}) + "\n")
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    print(f"[{label}] {line}", end="", flush=True)
            code = process.wait()
            if code:
                self.cancel()
                raise RuntimeError(f"{label} failed (exit {code}); full traceback is in {log}. Resume the same command after diagnosing it.")
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            process.stdout.close()
            with self.lock:
                self.active.discard(process)


def model_options(args):
    options = ["--model_name", args.model_name, "--model_revision", args.model_revision,
               "--expected_transformers_version", args.expected_transformers_version,
               "--expected_torch_version", args.expected_torch_version,
               "--expected_qwen_vl_utils_version", args.expected_qwen_vl_utils_version,
               "--seed", str(args.seed), "--project_root", args.project_root,
               "--path_map_path", str(Path(args.output_root) / "path_map.json")]
    if args.execution_mode == "model_parallel":
        options.extend(["--model_parallel", "--gpu_weight_budget_gib", str(args.gpu_weight_budget_gib)])
    else:
        options.append("--single_gpu")
    for name in ("video_fps", "video_num_frames", "video_max_pixels"):
        if getattr(args, name) is not None:
            options.extend([f"--{name}", str(getattr(args, name))])
    return options


def run_shards(args, runner, manifest, checkpoint, assignments):
    certificate = None
    if args.stage == "full":
        try:
            from .phase3b_checkpoint_reuse import validate_reuse
        except ImportError:
            from phase3b_checkpoint_reuse import validate_reuse
        certificate = validate_reuse(args.output_root, manifest=manifest)
    def worker(gpu, shards):
        for shard in shards:
            if certificate and f"shard_{shard:02d}" in certificate["shards"]:
                print(f"REUSE shard {shard:02d}: certified complete capture/patch; no GPU forwards.", flush=True)
                continue
            for stage in ("capture", "patch"):
                hardware_options = []
                if args.stage == "full":
                    hardware_options = ["--expected_gpu_hardware_path", str(preflight_directory(args, gpu) / "checkpoints/shard_00/run_config.json")]
                runner.run([
                    sys.executable, "-u", "scripts/run_phase3b_patching.py", "--stage", stage,
                    "--manifest_path", str(manifest), "--mapping_path", str(Path(args.selection_dir) / "selected_video_mappings.jsonl"),
                    "--output_dir", str(checkpoint(gpu)), "--shard_index", str(shard), "--shard_size", "5",
                    "--attn_implementation", "eager", "--roi_padding", "8", "--empty_cache_each_pair",
                    *model_options(args), *hardware_options,
                ], f"{gpu}_shard_{shard:02d}_{stage}", visible_devices(args, gpu))
                write_json(Path(args.output_root) / "progress" / f"gpu_{gpu}.json", {
                    "stage": args.stage, "worker": gpu, "visible_gpus": visible_devices(args, gpu),
                    "execution_mode": args.execution_mode, "shard": shard, "checkpoint_stage": stage,
                    "elapsed_sec": time.perf_counter() - runner.started,
                    "completed_shards": [i for i in shards if i < shard] + ([shard] if stage == "patch" else []),
                })
    with ThreadPoolExecutor(max_workers=len(assignments)) as pool:
        futures = [pool.submit(worker, gpu, shards) for gpu, shards in assignments.items()]
        try:
            for future in futures:
                future.result()
        except BaseException:
            runner.cancel()
            raise


def analyze(runner, manifest, checkpoint, output):
    certificate = Path(runner.args.output_root) / "checkpoint_reuse.json" if hasattr(runner, "args") else None
    reuse_options = ["--reuse_certificate", str(certificate)] if certificate and certificate.is_file() and "primary" in checkpoint.parts else []
    runner.run([sys.executable, "-u", "scripts/analyze_phase3b.py", "--manifest_path", str(manifest),
                "--shards_root", str(checkpoint), "--output_dir", str(output), *reuse_options], f"cpu_{output.name}")
    summary = json.loads((output / "aggregate_summary.json").read_text(encoding="utf-8"))
    if any(summary.get(key, 1) for key in (
        "missing_patch_count", "missing_capture_count", "missing_divergence_count", "missing_technical_control_count",
    )):
        raise RuntimeError("CPU merge is incomplete; do not report this as a completed run.")


def capture_audit(checkpoint):
    output = {}
    for path in sorted(Path(checkpoint).glob("shard_*/activations/*/*/index.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        parity = item.get("standard_parity")
        if not parity or not parity["first_token_match"] or not parity["logits_allclose"]:
            raise RuntimeError(f"Missing/failed standard parity: {path}.")
        output[str(path.relative_to(checkpoint))] = {
            "decision": item["decision"], "standard_generation_parity": parity,
            "archived_input_parity": item.get("archived_input_parity"),
        }
    if not output:
        raise RuntimeError("Preflight has no captured decisions.")
    return output


def require_vm_preflight(args, plan):
    try:
        from .phase3b_checkpoint_reuse import validate_reuse
    except ImportError:
        from phase3b_checkpoint_reuse import validate_reuse
    certificate = validate_reuse(args.output_root, plan)
    if certificate and certificate["preflight_reused"]:
        saved = json.loads((Path(args.output_root) / "reuse_source/vm_preflight_summary.json").read_text())
        if not saved.get("complete") or saved["pipeline_fingerprint"] != certificate["source_pipeline_fingerprint"]:
            raise RuntimeError("Certified source VM preflight is incomplete.")
        capture_audit(preflight_directory(args, "model_parallel") / "checkpoints")
        return
    path = Path(args.output_root) / "vm_preflight_summary.json"
    if not path.is_file():
        raise RuntimeError("Run --stage preflight on the A10 VM first; Colab A100 preflight does not satisfy the VM gate.")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if not saved.get("complete") or saved.get("pipeline_fingerprint") != plan["pipeline_fingerprint"] or not set(args.gpus) <= set(saved["gpus"]):
        raise RuntimeError("VM preflight does not match this code/cohort/GPU allocation.")
    for gpu in execution_schedule(args, 2):
        require = preflight_directory(args, gpu)
        capture_audit(require / "checkpoints")
        if not (require / "analysis/aggregate_summary.json").is_file():
            raise RuntimeError("VM preflight analysis is missing.")
    if not (Path(args.output_root) / "relocation_control/relocation_summary.json").is_file():
        raise RuntimeError("VM relocation control is missing.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "preflight", "baseline", "full", "analyze"), default="plan")
    parser.add_argument("--selection_dir", required=True)
    parser.add_argument("--rescue_pool_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--storage_root", default="/data/yuxuanstorage")
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--execution_mode", choices=("independent", "model_parallel"), default="model_parallel",
                        help="model_parallel: one FP16 model across two GPUs, reserving attention workspace.")
    parser.add_argument("--gpu_weight_budget_gib", type=float, default=10,
                        help="Model-parallel weight placement budget per GPU, not a total VRAM limit.")
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--model_revision", default="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
    parser.add_argument("--expected_transformers_version", default="5.9.0")
    parser.add_argument("--expected_torch_version", default="2.11.0")
    parser.add_argument("--expected_qwen_vl_utils_version", default="0.0.14")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video_fps", type=float, default=None)
    parser.add_argument("--video_num_frames", type=int, default=None)
    parser.add_argument("--video_max_pixels", type=int, default=None)
    parser.add_argument("--reuse_completed_from", help="Explicit immutable source VM run; only identical complete shards are reused.")
    args = parser.parse_args()
    args.gpus = parse_gpus(args.gpus)
    if args.execution_mode == "model_parallel" and (len(args.gpus) != 2 or not math.isfinite(args.gpu_weight_budget_gib) or args.gpu_weight_budget_gib <= 0):
        parser.error("Model parallel requires two GPU IDs and a positive finite weight budget.")
    if args.video_fps is not None and args.video_num_frames is not None:
        parser.error("Use only one temporal sampling control.")
    for name in ("output_root", "selection_dir", "rescue_pool_root", "project_root", "storage_root"):
        setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
    if not Path(args.output_root).is_relative_to(args.storage_root):
        parser.error("Outputs must be on persistent --storage_root (default /data/yuxuanstorage), not scratch.")
    if not Path(args.storage_root).is_dir():
        parser.error("Persistent storage is not mounted.")
    with run_lock(args.output_root):
        plan = build_plan(args)
        save_plan(args.output_root, plan)
        print(f"Frozen cohort: {plan['primary_count']} primary / {plan['pair_count']} total pairs; schedule={plan['schedule']}; storage={args.output_root}", flush=True)
        if args.stage == "plan":
            print("CPU-only plan complete. No model loaded. Next: --stage preflight.")
            return
        runner = LoggedRunner(args)
        root = Path(args.output_root)
        manifest = Path(args.selection_dir) / "analysis_case_manifest.jsonl"
        try:
            if args.reuse_completed_from:
                try:
                    from .phase3b_checkpoint_reuse import prepare_reuse, validate_reuse
                except ImportError:
                    from phase3b_checkpoint_reuse import prepare_reuse, validate_reuse
                if (root / "checkpoint_reuse.json").is_file():
                    validate_reuse(root, plan)
                else:
                    with run_lock(args.reuse_completed_from):
                        prepare_reuse(args, plan)
            if args.stage == "preflight":
                if (root / "checkpoint_reuse.json").is_file():
                    require_vm_preflight(args, plan)
                    write_json(root / "vm_last_status.json", {
                        "stage": "preflight", "complete": True, "reused": True,
                        "elapsed_sec": time.perf_counter() - runner.started,
                    })
                    print("Certified unchanged VM preflight reused. Next: --stage baseline; full not started.")
                    return
                manifest = Path(args.selection_dir) / "preflight_case_manifest.jsonl"
                workers = list(execution_schedule(args, 2))
                run_shards(args, runner, manifest, lambda gpu: preflight_directory(args, gpu) / "checkpoints", {gpu: [0] for gpu in workers})
                audits = {}
                for gpu in workers:
                    base = preflight_directory(args, gpu)
                    analyze(runner, manifest, base / "checkpoints", base / "analysis")
                    audits[gpu] = capture_audit(base / "checkpoints")
                reference = audits[workers[0]]
                for gpu, audit in audits.items():
                    if set(audit) != set(reference) or any(audit[key]["decision"]["prediction"] != reference[key]["decision"]["prediction"] for key in reference):
                        raise RuntimeError(f"Preflight predictions differ between GPUs: {gpu}.")
                runner.run([sys.executable, "-u", "scripts/run_phase3b_relocation_control.py",
                            "--manifest_path", str(manifest), "--output_dir", str(root / "relocation_control"),
                            "--expected_gpu_hardware_path", str(preflight_directory(args, workers[0]) / "checkpoints/shard_00/run_config.json"),
                            *model_options(args)], "gpu_relocation_control", visible_devices(args, workers[0]))
                write_json(root / "vm_preflight_summary.json", {
                    "complete": True, "pipeline_fingerprint": plan["pipeline_fingerprint"], "gpus": args.gpus,
                    "capture_audits": audits,
                    "execution_mode": args.execution_mode,
                    "cross_gpu_margin_differences": {gpu: {key: audit[key]["decision"]["margin"] - reference[key]["decision"]["margin"] for key in reference} for gpu, audit in audits.items()} if args.execution_mode == "independent" else None,
                    "elapsed_sec": time.perf_counter() - runner.started,
                    "interpretation": "Technical gate passed; inspect relocation effects before full run.",
                })
            else:
                if args.stage in ("baseline", "full"):
                    require_vm_preflight(args, plan)
                    try:
                        from .run_phase3b_baseline import require_gate
                    except ImportError:
                        from run_phase3b_baseline import require_gate
                    worker = list(execution_schedule(args, 2))[0]
                    runner.run([sys.executable, "-u", "scripts/run_phase3b_baseline.py",
                                "--run_root", str(root), "--hardware_config",
                                str(preflight_directory(args, worker) / "checkpoints/shard_00/run_config.json")],
                               "cohort_baseline_gate", visible_devices(args, worker))
                    require_gate(root, plan)
                    if args.stage == "full":
                        run_shards(args, runner, manifest, lambda gpu: root / "primary/checkpoints", plan["schedule"])
                if args.stage != "baseline":
                    analyze(runner, manifest, root / "primary/checkpoints", root / "primary/analysis")
                    write_json(root / "primary/capture_parity_audit.json", capture_audit(root / "primary/checkpoints"))
        except BaseException:
            runner.cancel()
            write_json(root / "vm_last_status.json", {"stage": args.stage, "complete": False, "elapsed_sec": time.perf_counter() - runner.started})
            raise
        write_json(root / "vm_last_status.json", {"stage": args.stage, "complete": True,
                   "finished_at": datetime.now(timezone.utc).isoformat(), "elapsed_sec": time.perf_counter() - runner.started})
        print("Saved on persistent storage. Run backup_phase3b.py and download the backup to your computer; VM storage alone is not a local backup.")


if __name__ == "__main__":
    main()
