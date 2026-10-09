"""Explicit, resumable Phase 3C VM orchestration; never screen or refreeze cases."""

import argparse
import fcntl
import hashlib
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

try:
    from .analyze_phase3c import verify_inputs
    from .phase3b_paths import load_path_map
    from .phase3c_core import atomic_write, digest, file_hash, frozen_write, read_json
    from .phase3c_execution import execution_hashes, load_selection, require_stage, technical_tasks
    from .phase3c_primary import PRIMARY_STAGES, primary_tasks
    from .run_phase3b_vm import parse_gpus, run_lock, worker_environment
    from .run_phase3c_preflight import check_execution_request
except ImportError:
    from analyze_phase3c import verify_inputs
    from phase3b_paths import load_path_map
    from phase3c_core import atomic_write, digest, file_hash, frozen_write, read_json
    from phase3c_execution import execution_hashes, load_selection, require_stage, technical_tasks
    from phase3c_primary import PRIMARY_STAGES, primary_tasks
    from run_phase3b_vm import parse_gpus, run_lock, worker_environment
    from run_phase3c_preflight import check_execution_request


STAGES = ("baseline", "preflight", "patch", "routing", "knockout", "full", "analyze")
SNAPSHOT_ROOTS = ("plan_config.json", "archive_audit.jsonl", "candidate_manifest.jsonl", "candidate_summary.json",
                  "processor_audit", "selection")


def add_arguments(parser):
    parser.add_argument("--stage", choices=("plan", *STAGES), default="plan")
    parser.add_argument("--target_stage", choices=STAGES, default="full", help="Stage to validate when --stage plan is used.")
    parser.add_argument("--plan_dir", required=True)
    parser.add_argument("--output_root", required=True, help="The same execution root used by baseline/preflight.")
    parser.add_argument("--storage_root", default="/data/yuxuanstorage")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--execution_mode", choices=("model_parallel", "single_gpu"), default="model_parallel")
    parser.add_argument("--gpu_weight_budget_gib", type=float, default=10)
    parser.add_argument("--path_map")
    parser.add_argument("--retry_failed", action="store_true")
    parser.add_argument("--bootstrap_repeats", type=int, default=2000)
    parser.add_argument("--no_plots", action="store_true")


@contextmanager
def vm_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".phase3c_vm.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another Phase 3C VM orchestrator or backup owns this root.") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def normalized_args(args):
    for name in ("plan_dir", "output_root", "storage_root", "project_root"):
        setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
    if args.path_map:
        args.path_map = str(Path(args.path_map).expanduser().resolve())
    storage, output, plan = (Path(getattr(args, name)) for name in ("storage_root", "output_root", "plan_dir"))
    if not storage.is_dir() or storage not in output.parents or storage not in plan.parents:
        raise ValueError("Preparation and execution artifacts must be beneath an existing persistent storage root.")
    if output == plan or output in plan.parents:
        raise ValueError("Execution must not overwrite preparation artifacts.")
    if (output / "vm_run_config.json").exists():
        raise ValueError("This is a Phase 3B execution root; Phase 3C must remain isolated.")
    if not Path(args.project_root, "scripts/run_phase3c.py").is_file():
        raise ValueError("Set --project_root to the Phase 3C repository.")
    if not math.isfinite(args.gpu_weight_budget_gib) or args.gpu_weight_budget_gib <= 0 or args.bootstrap_repeats < 100:
        raise ValueError("Use a finite positive weight budget and at least 100 bootstrap repeats.")
    if len(parse_gpus(args.gpus)) != (2 if args.execution_mode == "model_parallel" else 1):
        raise ValueError("GPU count differs from execution mode.")
    return args


def snapshot_files(plan_dir):
    root = Path(plan_dir)
    files = {}
    for name in SNAPSHOT_ROOTS:
        path = root / name
        if path.is_symlink():
            raise ValueError("Preparation snapshots refuse symlinks.")
        if not path.exists():
            raise FileNotFoundError(f"Missing preparation artifact: {path}")
        for item in sorted(path.rglob("*")) if path.is_dir() else [path]:
            if item.is_symlink():
                raise ValueError("Preparation snapshots refuse symlinks.")
            if item.is_file():
                files[item.relative_to(root).as_posix()] = item
    return files


def child_command(args, stage):
    if stage == "analyze":
        command = [sys.executable, "-u", "scripts/analyze_phase3c.py", "--plan_dir", args.plan_dir,
            "--execution_dir", args.output_root, "--bootstrap_repeats", str(args.bootstrap_repeats)]
        return command + (["--no_plots"] if args.no_plots else [])
    script = "run_phase3c_preflight.py" if stage in ("baseline", "preflight") else "run_phase3c.py"
    command = [sys.executable, "-u", f"scripts/{script}", "--stage", stage, "--plan_dir", args.plan_dir,
        "--output_dir", args.output_root, "--storage_root", args.storage_root, "--project_root", args.project_root,
        "--gpus", args.gpus, "--execution_mode", args.execution_mode, "--gpu_weight_budget_gib", str(args.gpu_weight_budget_gib)]
    if args.path_map:
        command.extend(["--path_map", args.path_map])
    if args.retry_failed:
        command.append("--retry_failed")
    return command


def require_prerequisites(args, frozen, pairs, mappings, stage):
    root = Path(args.output_root)
    audit = read_json(Path(args.plan_dir) / "processor_audit/config.json")
    path_map = {**audit["path_map"], **load_path_map(args.path_map)}
    check_execution_request(root, frozen, args.project_root, path_map, args.execution_mode,
                            args.gpu_weight_budget_gib, parse_gpus(args.gpus))
    if stage == "baseline":
        return
    execution = read_json(root / "execution_config.json")
    baselines = require_stage(root, execution, pairs, "baseline", {row["eval_id"] for pair in pairs.values() for row in pair.values()})
    if stage != "preflight":
        tasks = technical_tasks(frozen, mappings)
        require_stage(root, execution, pairs, "preflight", {task["task_id"] for task in tasks}, expected_tasks=tasks)
    if stage == "knockout":
        tasks = primary_tasks(frozen, mappings, "routing")
        require_stage(root, execution, pairs, "routing", {task["task_id"] for task in tasks}, mappings, baselines, tasks)
    if stage == "analyze":
        verify_inputs(args.plan_dir, root)


def build_plan(args):
    frozen, pairs, mappings = load_selection(args.plan_dir)
    source = Path(read_json(Path(args.plan_dir) / "plan_config.json")["source_run_root"]).resolve()
    output = Path(args.output_root)
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Phase 3B is read-only source evidence, not a Phase 3C output root.")
    stage = args.target_stage if args.stage == "plan" else args.stage
    require_prerequisites(args, frozen, pairs, mappings, stage)
    files = snapshot_files(args.plan_dir)
    videos = {record["video_path"]: record["video_sha256"] for mapping in mappings.values()
              for record in mapping["processor_records"].values()}
    for path, expected in videos.items():
        if file_hash(path) != expected:
            raise ValueError("Frozen selected video bytes changed.")
    config = {"schema": "phase3c_vm_execution_v1", "artifact_type": "real", "plan_dir": args.plan_dir,
        "output_root": args.output_root, "selection_fingerprint": frozen["selection_fingerprint"],
        "preparation_snapshot_sha256": {name: file_hash(path) for name, path in files.items()}, "video_sha256": videos,
        "execution_code_sha256": execution_hashes(), "orchestration_code_sha256": file_hash(__file__),
        "project_root": args.project_root, "storage_root": args.storage_root, "gpus": args.gpus,
        "execution_mode": args.execution_mode, "gpu_weight_budget_gib": args.gpu_weight_budget_gib,
        "path_map": {**read_json(Path(args.plan_dir) / "processor_audit/config.json")["path_map"], **load_path_map(args.path_map)},
        "analysis_settings": {"bootstrap_repeats": args.bootstrap_repeats, "plots": not args.no_plots},
        "case_count": len(pairs), "rescue_cases": 8, "stable_cases": 4,
        "tasks": {name: len(primary_tasks(frozen, mappings, name)) for name in PRIMARY_STAGES}}
    config["vm_fingerprint"] = digest(config)
    prior = output / "phase3c_vm_config.json"
    if prior.exists() and read_json(prior) != config:
        raise ValueError("VM orchestration binding changed; preserve it and use a separate execution root.")
    steps = ["patch", "routing", "knockout", "analyze"] if stage == "full" else [stage]
    return config, steps


def save_plan(args, config):
    frozen_write(Path(args.output_root) / "phase3c_vm_config.json", config)
    for name, path in snapshot_files(args.plan_dir).items():
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != config["preparation_snapshot_sha256"][name]:
            raise ValueError("Preparation bytes changed while saving their snapshot.")
        target = Path(args.output_root) / "preparation_snapshot" / name
        if target.exists() and file_hash(target) != config["preparation_snapshot_sha256"][name]:
            raise ValueError("Saved preparation snapshot changed.")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")
            with temporary.open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)


def run_child(args, command, label):
    log = Path(args.output_root) / "logs" / f"{label}_{time.time_ns()}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = worker_environment(args.gpus, args.storage_root)
    if label == "analyze":
        env["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("RESEARCH_CLOUD_TOKEN", "SURF_API_TOKEN", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        env.pop(name, None)
    print(f"[Phase 3C VM] {label} starting; log={log}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=args.project_root, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        previous, cancelled = {}, False
        def stop(signum, _frame):
            nonlocal cancelled
            cancelled = True
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, stop)
        try:
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                print(f"[{label}] {line}", end="", flush=True)
            code = process.wait()
            return 130 if cancelled else code
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()


def verify_stage(args, stage):
    frozen, pairs, mappings = load_selection(args.plan_dir)
    root = Path(args.output_root)
    if stage == "analyze":
        _, _, execution, _, _ = verify_inputs(args.plan_dir, root)
        summary = read_json(root / "analysis/aggregate_summary.json")
        config = read_json(root / "analysis/analysis_config.json")
        status = read_json(root / "analysis/analysis_status.json")
        if (not summary.get("complete") or not status.get("complete") or
                summary["analysis_fingerprint"] != config["analysis_fingerprint"] or
                status["analysis_fingerprint"] != config["analysis_fingerprint"] or
                config["analysis_fingerprint"] != digest({k: v for k, v in config.items() if k != "analysis_fingerprint"}) or
                config["execution_fingerprint"] != execution["execution_fingerprint"] or
                config["selection_fingerprint"] != frozen["selection_fingerprint"] or
                config["analysis_code_sha256"] != file_hash(Path(__file__).with_name("analyze_phase3c.py")) or
                config["case_bootstrap_repeats"] != args.bootstrap_repeats or config["plots"] != (not args.no_plots)):
            raise ValueError("CPU analysis did not complete for the current inputs.")
        if config["plots"] and config["visualization_code_sha256"] != file_hash(Path(__file__).with_name("visualize_phase3c.py")):
            raise ValueError("Visualization code changed; preserve the previous analysis.")
        for name, expected in config["input_sha256"].items():
            if file_hash(root / name) != expected:
                raise ValueError("Analysis input checksums are stale.")
        for name, expected in summary["output_sha256"].items():
            if file_hash(root / "analysis" / name) != expected:
                raise ValueError("Analysis output checksums changed.")
        return
    require_prerequisites(args, frozen, pairs, mappings, "preflight" if stage == "baseline" else stage)
    if stage == "baseline":
        return
    execution = read_json(root / "execution_config.json")
    if stage == "preflight":
        tasks = technical_tasks(frozen, mappings)
        require_stage(root, execution, pairs, stage, {task["task_id"] for task in tasks}, expected_tasks=tasks)
    else:
        baseline = require_stage(root, execution, pairs, "baseline", {row["eval_id"] for pair in pairs.values() for row in pair.values()})
        tasks = primary_tasks(frozen, mappings, stage)
        require_stage(root, execution, pairs, stage, {task["task_id"] for task in tasks}, mappings, baseline, tasks)


def execute(args, child=run_child):
    started, attempt = time.perf_counter(), uuid.uuid4().hex
    root = Path(args.output_root)
    with vm_lock(root):
        status = {"schema": "phase3c_vm_attempt_v1", "artifact_type": "real", "stage": args.stage,
            "attempt_id": attempt, "vm_fingerprint": None, "complete": False,
            "started_at": datetime.now(timezone.utc).isoformat(), "completed_steps": []}
        # A direct scientific CLI owns .pipeline.lock, not the VM orchestration lock.
        with run_lock(root):
            atomic_write(root / "vm_last_status.json", status)
        try:
            with run_lock(root):
                config, steps = build_plan(args)
                save_plan(args, config)
            status["vm_fingerprint"] = config["vm_fingerprint"]
            for stage in steps:
                status.update(current_step=stage, elapsed_sec=time.perf_counter() - started)
                atomic_write(root / "vm_last_status.json", status)
                code = child(args, child_command(args, stage), stage)
                if code:
                    raise RuntimeError(f"{stage} exited {code}; checkpoints preserved. No later stage was launched.")
                verify_stage(args, stage)
                status["completed_steps"].append(stage)
            status.update(complete=True, elapsed_sec=time.perf_counter() - started)
            atomic_write(root / "vm_last_status.json", status)
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            status.update(complete=False, error_type=type(exc).__name__, elapsed_sec=time.perf_counter() - started)
            atomic_write(root / "vm_last_status.json", status)
            print(f"Phase 3C VM blocked: {exc}", flush=True)
            return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = normalized_args(parser.parse_args())
    if args.stage == "plan":
        config, steps = build_plan(args)
        import json
        print(json.dumps({"config": config, "steps": steps,
                          "commands": [child_command(args, stage) for stage in steps]}, indent=2))
        return
    sys.exit(execute(args))


if __name__ == "__main__":
    main()
