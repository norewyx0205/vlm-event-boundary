"""Resume batched Phase 3B screening until 60 eligible bases or the fixed cap."""

import argparse
import json
import subprocess
import sys
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


def run(command):
    print("$", " ".join(map(str, command)), flush=True)
    subprocess.run([sys.executable, *map(str, command)], cwd=PROJECT_ROOT, check=True)


def evaluate_batch(args, batch):
    annotation = batch / "annotations.jsonl"
    if not annotation.is_file():
        raise RuntimeError(f"Generated batch lacks annotations: {batch}")
    start, end = map(int, batch.name.removeprefix("batch_").split("_"))
    if completed_result(result_path(args.result_root, args.model_name, start, end), annotation):
        return
    run([
        "scripts/run_eval.py", "--annotation_path", annotation,
        "--model_name", args.model_name, "--model_revision", args.model_revision,
        "--dataset_name", f"phase3b_screen_{start:03d}_{end:03d}",
        "--output_dir", args.result_root, "--attn_implementation", "eager",
        "--seed", args.seed, "--deterministic",
    ])


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


def audit(args, annotations, results, output_root):
    validate_signatures(args, results)
    screen_dir = output_root / "screening"
    run([
        "scripts/screen_phase3b_rescues.py", "--annotation_paths", *annotations,
        "--result_paths", *results, "--output_dir", screen_dir,
    ])
    audit_dir = output_root / "mapping_audit"
    run([
        "scripts/audit_phase3b_mappings.py", "--annotation_paths", *annotations,
        "--result_paths", *results, "--output_dir", audit_dir,
        "--project_root", PROJECT_ROOT, "--model_name", args.model_name,
        "--model_revision", args.model_revision,
        "--expected_transformers_version", args.expected_transformers_version,
        "--roi_padding", args.roi_padding,
        "--min_mapping_coverage", args.min_mapping_coverage,
    ])
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
    validate_signatures(args, [args.existing_result_path])
    if not args.only_existing:
        for batch in sorted((pool_root / "batches").glob("batch_???_???")):
            if not (batch / "annotations.jsonl").is_file():
                start, end = map(int, batch.name.removeprefix("batch_").split("_"))
                run([
                    "scripts/generate_phase3b_rescue_pool.py", "--output_root", pool_root,
                    "--start_base_id", start, "--count", end - start + 1,
                    "--max_new_bases", args.max_new_bases, "--seed", args.seed,
                ])
            evaluate_batch(args, batch)
    annotations, results = all_paths(args, pool_root)
    progress = audit(args, annotations, results, output_root)
    if args.only_existing:
        print(progress)
        return
    end_cap = 30 + args.max_new_bases
    while progress["eligible_rescue_base_count"] < args.stop_eligible_bases:
        existing_batches = sorted((pool_root / "batches").glob("batch_???_???"))
        start = max((int(path.name.split("_")[2]) for path in existing_batches), default=30) + 1
        if start > end_cap:
            break
        end = min(start + args.batch_size - 1, end_cap)
        run([
            "scripts/generate_phase3b_rescue_pool.py", "--output_root", pool_root,
            "--start_base_id", start, "--count", end - start + 1,
            "--max_new_bases", args.max_new_bases, "--seed", args.seed,
        ])
        batch = pool_root / "batches" / f"batch_{start:03d}_{end:03d}"
        evaluate_batch(args, batch)
        annotations, results = all_paths(args, pool_root)
        progress = audit(args, annotations, results, output_root)
        print(f"Eligible rescue bases: {progress['eligible_rescue_base_count']}", flush=True)
    progress["stopping_reason"] = (
        "mapping_eligible_target_reached" if progress["eligible_rescue_base_count"] >= args.stop_eligible_bases
        else "new_base_budget_exhausted"
    )
    atomic_write_json(output_root / "screening_progress.json", progress)
    print(progress)


if __name__ == "__main__":
    main()
