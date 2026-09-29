"""Resume batched Phase 3B screening until the eligible target or budget cap."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import transformers

try:
    from .activation_patching_core import atomic_write_json
    from .common import PROJECT_ROOT, read_jsonl, slugify
    from .screen_phase3b_rescues import screen
except ImportError:
    from activation_patching_core import atomic_write_json
    from common import PROJECT_ROOT, read_jsonl, slugify
    from screen_phase3b_rescues import screen


def result_path(result_root, model_name, start, end):
    dataset = f"phase3b_screen_{start:03d}_{end:03d}"
    directory = Path(result_root) / slugify(model_name) / slugify(dataset)
    candidates = sorted(directory.glob("*/raw_results.jsonl"))
    return candidates[-1] if candidates else None


def validate_pool_provenance(
    pool_root, seed, max_new_bases, allow_budget_extension=False, output_root=None,
):
    pool_root = Path(pool_root)
    config_path = pool_root / "phase3b_generation_config.json"
    has_pool = (pool_root / "L5_full" / "annotations.jsonl").exists() or any(
        (pool_root / "batches").glob("batch_???_???"))
    if not config_path.exists():
        if has_pool:
            raise RuntimeError("Existing Phase 3B pool lacks its generation config; cannot reuse it.")
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_hashes = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ("generate_phase3b_rescue_pool.py", "generate_ladder_dataset.py")
    }
    expected = {
        "schema": "phase3b_rescue_pool_v1",
        "generator_code_sha256": expected_hashes,
        "seed": seed,
        "max_new_bases": max_new_bases,
        "conditions": ["low_boundary", "temporal_boundary"],
    }
    previous_budget = config.get("max_new_bases")
    mismatches = [
        key for key, value in expected.items()
        if key != "max_new_bases" and config.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            f"Existing Phase 3B pool provenance differs on {mismatches}; "
            "use the original code/config or a new pool directory."
        )
    if previous_budget == max_new_bases:
        return
    if not allow_budget_extension or not isinstance(previous_budget, int) or previous_budget >= max_new_bases:
        raise RuntimeError(
            f"Existing Phase 3B pool has max_new_bases={previous_budget}, requested={max_new_bases}. "
            "Only a larger budget with --extend_existing_budget can reuse this pool."
        )
    amendment_path = pool_root / "budget_amendments.json"
    amendments = json.loads(amendment_path.read_text(encoding="utf-8")) if amendment_path.is_file() else []
    progress_path = Path(output_root or pool_root.parent / "analysis") / "screening_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
    amendments.append({
        "amended_at_utc": datetime.now(timezone.utc).isoformat(),
        "previous_max_new_bases": previous_budget,
        "new_max_new_bases": max_new_bases,
        "eligible_rescue_bases_at_amendment": progress.get("eligible_rescue_base_count"),
        "reason": "Pre-specified screening budget exhausted before the eligible rescue target",
    })
    atomic_write_json(amendment_path, amendments)
    config["max_new_bases"] = max_new_bases
    atomic_write_json(config_path, config)
    print(
        f"Phase 3B screening budget amended from {previous_budget} to {max_new_bases} "
        f"new bases; recorded at {amendment_path}.", flush=True,
    )


def completed_result(path, annotation_path):
    if not path or not path.is_file():
        return False
    expected = {row["eval_id"] for row in read_jsonl(annotation_path)}
    actual = [row["eval_id"] for row in read_jsonl(path)]
    return len(actual) == len(expected) and set(actual) == expected


def result_signature(path):
    config_path = Path(path).parent / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"Evaluation config is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    environment = config.get("environment") or {}
    model_load = config.get("model_load") or {}
    return {
        "model_name": config.get("model_name"),
        "model_revision": model_load.get("model_revision"),
        "dtype": model_load.get("dtype"),
        "load_in_4bit": model_load.get("load_in_4bit"),
        "attn_implementation": model_load.get("attn_implementation"),
        "seed": config.get("seed"),
        "deterministic": config.get("deterministic"),
        "video_sampling_request": config.get("video_sampling_request"),
        "decoding": config.get("decoding"),
        "transformers_version": environment.get("transformers_version"),
        "qwen_vl_utils_version": environment.get("qwen_vl_utils_version"),
        "torch_version": environment.get("torch_version"),
    }


def validate_signatures(args, results):
    reference = result_signature(results[0])
    required = {
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "dtype": "float16",
        "load_in_4bit": False,
        "attn_implementation": "eager",
        "seed": args.seed,
        "deterministic": True,
        "video_sampling_request": {"fps": None, "num_frames": None, "max_pixels": None, "min_pixels": None},
        "decoding": {"do_sample": False, "num_beams": 1, "max_new_tokens": 10},
        "transformers_version": args.expected_transformers_version,
    }
    mismatches = [key for key, value in required.items() if reference.get(key) != value]
    if mismatches:
        raise RuntimeError(f"Existing L5_full evaluation differs from Phase 3B settings: {mismatches}.")
    try:
        qwen_utils_version = version("qwen-vl-utils")
    except PackageNotFoundError as exc:
        raise RuntimeError("qwen-vl-utils is required for Phase 3B.") from exc
    live_versions = {
        "transformers_version": transformers.__version__,
        "qwen_vl_utils_version": qwen_utils_version,
        "torch_version": torch.__version__,
    }
    changed_runtime = [key for key, value in live_versions.items() if reference.get(key) != value]
    if changed_runtime:
        raise RuntimeError(f"Current Colab runtime differs from the frozen L5_full run: {changed_runtime}.")
    for path in results[1:]:
        signature = result_signature(path)
        differences = [key for key in reference if signature.get(key) != reference[key]]
        if differences:
            raise RuntimeError(f"Batch evaluation {path} differs from frozen L5_full: {differences}.")


class ScreeningProgress:
    def __init__(self, output_root, max_new_bases, stop_eligible_bases, heartbeat_sec=60):
        self.path = Path(output_root) / "screening_status.json"
        self.started = time.perf_counter()
        self.heartbeat_sec = heartbeat_sec
        previous_path = Path(output_root) / "screening_progress.json"
        previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.is_file() else {}
        self.state = {
            "schema": "phase3b_screening_status_v1",
            "session_started_utc": datetime.now(timezone.utc).isoformat(),
            "max_new_bases": max_new_bases,
            "stop_eligible_bases": stop_eligible_bases,
            "eligible_rescue_bases_last_audit": previous.get("eligible_rescue_base_count"),
            "screened_bases_last_audit": previous.get("screened_base_count"),
            "evaluated_new_bases": 0,
            "current_batch": None,
        }

    def update(self, stage, *, heartbeat=False, **fields):
        self.state.update(fields)
        self.state["stage"] = stage
        self.state["session_elapsed_sec"] = round(time.perf_counter() - self.started, 1)
        self.state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(self.path, self.state)
        elapsed_min = self.state["session_elapsed_sec"] / 60
        print(
            f"[Phase3B screen {elapsed_min:.1f} min] {stage}"
            f"{' (still running)' if heartbeat else ''} | "
            f"batch={self.state['current_batch'] or '-'} | "
            f"evaluated_new={self.state['evaluated_new_bases']}/{self.state['max_new_bases']} | "
            f"eligible_last_audit={self.state['eligible_rescue_bases_last_audit']}/"
            f"{self.state['stop_eligible_bases']} | checkpoint={self.path}",
            flush=True,
        )


def run(command, progress=None):
    print("$", " ".join(map(str, command)), flush=True)
    started = time.perf_counter()
    process = subprocess.Popen([sys.executable, "-u", *map(str, command)], cwd=PROJECT_ROOT)
    try:
        while True:
            try:
                return_code = process.wait(timeout=progress.heartbeat_sec if progress else None)
                break
            except subprocess.TimeoutExpired:
                if progress:
                    progress.update(progress.state["stage"], heartbeat=True)
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    elapsed_sec = round(time.perf_counter() - started, 1)
    print(f"[Phase3B screen] subprocess exit={return_code} after "
          f"{elapsed_sec / 60:.1f} min", flush=True)
    if return_code:
        if progress:
            progress.update("subprocess failed", failed_phase=progress.state["stage"], exit_code=return_code)
        raise subprocess.CalledProcessError(return_code, process.args)
    if progress:
        progress.update(
            progress.state["stage"], last_step_elapsed_sec=elapsed_sec,
            last_step_command=str(command[0]),
        )


def evaluate_batch(args, batch, progress=None):
    annotation = batch / "annotations.jsonl"
    if not annotation.is_file():
        raise RuntimeError(f"Generated batch lacks annotations: {batch}")
    start, end = map(int, batch.name.removeprefix("batch_").split("_"))
    if completed_result(result_path(args.result_root, args.model_name, start, end), annotation):
        if progress:
            progress.update("reused completed evaluation", current_batch=batch.name)
        return
    run([
        "scripts/run_eval.py", "--annotation_path", annotation,
        "--model_name", args.model_name, "--model_revision", args.model_revision,
        "--dataset_name", f"phase3b_screen_{start:03d}_{end:03d}",
        "--output_dir", args.result_root, "--attn_implementation", "eager",
        "--seed", args.seed, "--deterministic",
    ], progress)


def all_paths(args, pool_root):
    annotations = [str(Path(args.existing_annotation_path))]
    results = [str(Path(args.existing_result_path))]
    for batch in sorted((pool_root / "batches").glob("batch_???_???")):
        annotation = batch / "annotations.jsonl"
        if not annotation.is_file():
            continue
        start, end = map(int, batch.name.removeprefix("batch_").split("_"))
        result = result_path(args.result_root, args.model_name, start, end)
        if not completed_result(result, annotation):
            raise RuntimeError(f"Batch {batch.name} lacks a complete evaluation; rerun that batch first.")
        annotations.append(str(annotation))
        results.append(str(result))
    return annotations, results


def audit(args, annotations, results, output_root, progress=None):
    validate_signatures(args, results)
    screen_dir = output_root / "screening"
    if progress:
        progress.update("classify behavioral rescues", current_batch=progress.state["current_batch"])
    run([
        "scripts/screen_phase3b_rescues.py", "--annotation_paths", *annotations,
        "--result_paths", *results, "--output_dir", screen_dir,
    ], progress)
    audit_dir = output_root / "mapping_audit"
    if progress:
        progress.update("audit processor-only mappings")
    run([
        "scripts/audit_phase3b_mappings.py", "--annotation_paths", *annotations,
        "--result_paths", *results, "--output_dir", audit_dir,
        "--project_root", PROJECT_ROOT, "--model_name", args.model_name,
        "--model_revision", args.model_revision,
        "--expected_transformers_version", args.expected_transformers_version,
        "--roi_padding", args.roi_padding,
        "--min_mapping_coverage", args.min_mapping_coverage,
    ], progress)
    mappings = read_jsonl(audit_dir / "video_mapping_manifest.jsonl")
    screened, rescue, _, _ = screen(annotations, results)
    eligible = {
        int(row["base_sample_id"]) for row in mappings
        if row.get("eligible") and any(
            item["base_sample_id"] == int(row["base_sample_id"])
            and item["prompt_variant"] == row["prompt_variant"]
            for item in rescue
        )
    }
    summary = {
        "schema": "phase3b_screening_progress_v1",
        "screened_base_count": len({row["base_sample_id"] for row in screened}),
        "eligible_rescue_base_count": len(eligible),
        "eligible_rescue_bases": sorted(eligible),
        "annotation_paths": annotations,
        "result_paths": results,
        "mapping_path": str(audit_dir / "video_mapping_manifest.jsonl"),
    }
    atomic_write_json(output_root / "screening_progress.json", summary)
    if progress:
        progress.update(
            "audit checkpoint saved",
            eligible_rescue_bases_last_audit=summary["eligible_rescue_base_count"],
            screened_bases_last_audit=summary["screened_base_count"],
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing_annotation_path", required=True)
    parser.add_argument("--existing_result_path", required=True)
    parser.add_argument("--output_root", default=str(PROJECT_ROOT / "analysis" / "phase3b"))
    parser.add_argument("--pool_root", default=str(PROJECT_ROOT / "data" / "phase3b_rescue_pool"))
    parser.add_argument("--result_root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--model_revision", default="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
    parser.add_argument("--expected_transformers_version", default="5.9.0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--max_new_bases", type=int, default=300)
    parser.add_argument("--extend_existing_budget", action="store_true",
                        help="Explicitly amend only the saved new-base cap and record the change.")
    parser.add_argument("--stop_eligible_bases", type=int, default=60)
    parser.add_argument("--roi_padding", type=int, default=8)
    parser.add_argument("--min_mapping_coverage", type=float, default=0.5)
    parser.add_argument("--only_existing", action="store_true",
                        help="Audit the current pool without generating or evaluating new samples.")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_bases < 1 or args.stop_eligible_bases < 1:
        parser.error("Batch size, budget, and stopping threshold must be positive.")
    pool_root = Path(args.pool_root)
    output_root = Path(args.output_root)
    progress_reporter = ScreeningProgress(output_root, args.max_new_bases, args.stop_eligible_bases)
    progress_reporter.update("validate existing results and pool")
    validate_signatures(args, [args.existing_result_path])
    validate_pool_provenance(
        pool_root, args.seed, args.max_new_bases,
        allow_budget_extension=args.extend_existing_budget,
        output_root=output_root,
    )
    if not args.only_existing:
        for batch in sorted((pool_root / "batches").glob("batch_???_???")):
            progress_reporter.update("resume existing batch", current_batch=batch.name)
            if not (batch / "annotations.jsonl").is_file():
                start, end = map(int, batch.name.removeprefix("batch_").split("_"))
                progress_reporter.update("generate missing batch")
                run([
                    "scripts/generate_phase3b_rescue_pool.py", "--output_root", pool_root,
                    "--start_base_id", start, "--count", end - start + 1,
                    "--max_new_bases", args.max_new_bases, "--seed", args.seed,
                ], progress_reporter)
            progress_reporter.update("evaluate existing batch")
            evaluate_batch(args, batch, progress_reporter)
            start, end = map(int, batch.name.removeprefix("batch_").split("_"))
            progress_reporter.update(
                "existing batch complete",
                evaluated_new_bases=progress_reporter.state["evaluated_new_bases"] + end - start + 1,
            )
    annotations, results = all_paths(args, pool_root)
    progress_reporter.update(
        "all completed batches accounted for",
        evaluated_new_bases=sum(len(read_jsonl(path)) // 4 for path in results[1:]),
    )
    progress = audit(args, annotations, results, output_root, progress_reporter)
    if args.only_existing:
        progress_reporter.update("existing-only audit complete")
        print(progress, flush=True)
        return
    end_cap = 30 + args.max_new_bases
    while progress["eligible_rescue_base_count"] < args.stop_eligible_bases:
        existing_batches = sorted((pool_root / "batches").glob("batch_???_???"))
        start = max((int(path.name.split("_")[2]) for path in existing_batches), default=30) + 1
        if start > end_cap:
            break
        end = min(start + args.batch_size - 1, end_cap)
        batch_name = f"batch_{start:03d}_{end:03d}"
        progress_reporter.update("generate new batch", current_batch=batch_name)
        run([
            "scripts/generate_phase3b_rescue_pool.py", "--output_root", pool_root,
            "--start_base_id", start, "--count", end - start + 1,
            "--max_new_bases", args.max_new_bases, "--seed", args.seed,
        ], progress_reporter)
        batch = pool_root / "batches" / f"batch_{start:03d}_{end:03d}"
        progress_reporter.update("evaluate new batch")
        evaluate_batch(args, batch, progress_reporter)
        progress_reporter.update(
            "batch evaluation checkpoint saved",
            evaluated_new_bases=progress_reporter.state["evaluated_new_bases"] + end - start + 1,
        )
        annotations, results = all_paths(args, pool_root)
        progress = audit(args, annotations, results, output_root, progress_reporter)
    progress["stopping_reason"] = (
        "mapping_eligible_target_reached" if progress["eligible_rescue_base_count"] >= args.stop_eligible_bases
        else "new_base_budget_exhausted"
    )
    progress["max_new_bases"] = args.max_new_bases
    progress["stop_eligible_bases"] = args.stop_eligible_bases
    progress["session_elapsed_sec"] = round(time.perf_counter() - progress_reporter.started, 1)
    progress["evaluated_new_bases"] = progress_reporter.state["evaluated_new_bases"]
    progress["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(output_root / "screening_progress.json", progress)
    progress_reporter.update("screen complete", stopping_reason=progress["stopping_reason"])
    print(progress, flush=True)


if __name__ == "__main__":
    main()
