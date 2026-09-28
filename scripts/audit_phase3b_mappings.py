"""Processor-only eligibility audit for Phase 3B video token mappings."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import torch
import transformers
from transformers import AutoProcessor

try:
    from .activation_patching_core import atomic_write_json, atomic_write_jsonl
    from .common import PROJECT_ROOT, read_jsonl
    from .phase3b_core import MIN_ROI_OVERLAP, SCHEMA, VIDEO_GROUPS, build_pair_mappings, prepare_example
    from .screen_phase3b_rescues import screen
except ImportError:
    from activation_patching_core import atomic_write_json, atomic_write_jsonl
    from common import PROJECT_ROOT, read_jsonl
    from phase3b_core import MIN_ROI_OVERLAP, SCHEMA, VIDEO_GROUPS, build_pair_mappings, prepare_example
    from screen_phase3b_rescues import screen


def audit_pair(low, temporal, processor, args):
    prepared = {
        "low_boundary": prepare_example(
            low, processor, args.project_root, args.video_fps, args.video_num_frames,
            args.video_max_pixels, args.roi_padding,
        ),
        "temporal_boundary": prepare_example(
            temporal, processor, args.project_root, args.video_fps, args.video_num_frames,
            args.video_max_pixels, args.roi_padding,
        ),
    }
    ids = [prepared[key]["inputs"].input_ids[0] for key in ("low_boundary", "temporal_boundary")]
    same_ids = ids[0].shape == ids[1].shape and torch.equal(ids[0], ids[1])
    mappings = build_pair_mappings(
        prepared["low_boundary"], prepared["temporal_boundary"],
        args.min_mapping_coverage, args.max_progress_error,
    )
    failures = [name for name in VIDEO_GROUPS if not mappings[name]["eligible"]]
    if not same_ids:
        failures.append("prompt_token_ids_differ")
    return {
        "schema": SCHEMA,
        "base_sample_id": int(low["base_sample_id"]),
        "prompt_variant": low["prompt_variant"],
        "pair_id": f"phase3b_base_{int(low['base_sample_id']):03d}_{low['prompt_variant']}",
        "eligible": not failures,
        "ineligible_groups": failures,
        "same_prompt_token_ids": same_ids,
        "prompt_token_count": int(ids[0].numel()),
        "prompt_input_ids_sha256": hashlib.sha256(
            ids[0].detach().cpu().numpy().tobytes()
        ).hexdigest(),
        "first_object_id": int(low["first_object_id"]),
        "correct_option": low["correct_option"],
        "groups": mappings,
        "low_video_metadata": prepared["low_boundary"]["video_metadata"],
        "temporal_video_metadata": prepared["temporal_boundary"]["video_metadata"],
        "low_input_metadata": prepared["low_boundary"]["input_metadata"],
        "temporal_input_metadata": prepared["temporal_boundary"]["input_metadata"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation_paths", nargs="+", required=True)
    parser.add_argument("--result_paths", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--video_fps", type=float, default=None)
    parser.add_argument("--video_num_frames", type=int, default=None)
    parser.add_argument("--video_max_pixels", type=int, default=None)
    parser.add_argument("--roi_padding", type=int, default=8)
    parser.add_argument("--min_mapping_coverage", type=float, default=0.5)
    parser.add_argument("--max_progress_error", type=float, default=0.35)
    parser.add_argument("--expected_transformers_version", default=None)
    parser.add_argument("--control_bases", type=int, default=20)
    args = parser.parse_args()
    if args.expected_transformers_version and transformers.__version__ != args.expected_transformers_version:
        parser.error("Processor version differs from the archived evaluation runtime.")
    if not 0 < args.min_mapping_coverage <= 1:
        parser.error("--min_mapping_coverage must be in (0, 1].")
    if not 0 < args.max_progress_error <= 1:
        parser.error("--max_progress_error must be in (0, 1].")
    if args.video_fps is not None and args.video_num_frames is not None:
        parser.error("Select only one temporal sampling control.")
    screened, candidates, annotations, _ = screen(args.annotation_paths, args.result_paths)
    rescue_bases = {row["base_sample_id"] for row in candidates}
    controls = [
        row for row in screened
        if row["outcome"] == "stable_both_correct"
        and row["prompt_variant"] == "original"
        and row["base_sample_id"] not in rescue_bases
    ][:args.control_bases]
    candidate_keys = {(row["base_sample_id"], row["prompt_variant"]) for row in candidates}
    mirrored = [
        row for row in screened
        if row["base_sample_id"] in rescue_bases
        and (row["base_sample_id"], row["prompt_variant"]) not in candidate_keys
    ]
    audit_items = candidates + mirrored + controls
    outputs = []
    output_dir = Path(args.output_dir)
    output_path = output_dir / "video_mapping_manifest.jsonl"
    static_config = {
        "schema": SCHEMA, "model_name": args.model_name, "model_revision": args.model_revision,
        "mapping_code_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("phase3b_core.py", "probe_attention_roi.py", "run_eval.py")
        },
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "qwen_vl_utils_version": version("qwen-vl-utils"),
        "video_fps": args.video_fps, "video_num_frames": args.video_num_frames,
        "video_max_pixels": args.video_max_pixels, "roi_padding": args.roi_padding,
        "min_mapping_coverage": args.min_mapping_coverage,
        "max_progress_error": args.max_progress_error,
        "minimum_primary_roi_overlap": MIN_ROI_OVERLAP,
        "event_phase_dominance_threshold": 0.5,
    }
    config_path = output_dir / "video_mapping_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != static_config:
        raise ValueError("Existing mapping audit uses different processor or mapping settings.")
    atomic_write_json(config_path, static_config)
    prior = {row["pair_id"]: row for row in read_jsonl(output_path)} if output_path.exists() else {}
    kwargs = {"revision": args.model_revision} if args.model_revision else {}
    processor = AutoProcessor.from_pretrained(args.model_name, **kwargs)
    for candidate in audit_items:
        base_id, variant = candidate["base_sample_id"], candidate["prompt_variant"]
        pair_id = f"phase3b_base_{base_id:03d}_{variant}"
        if pair_id in prior:
            outputs.append(prior[pair_id])
            continue
        low = annotations[candidate["low_eval_id"]]
        temporal = annotations[candidate["temporal_eval_id"]]
        try:
            record = audit_pair(low, temporal, processor, args)
        except Exception as exc:
            record = {
                "schema": SCHEMA, "base_sample_id": base_id, "prompt_variant": variant,
                "pair_id": pair_id, "eligible": False,
                "ineligible_groups": ["preprocessing_failure"],
                "failure_type": type(exc).__name__, "failure_message": str(exc),
            }
        outputs.append(record)
        atomic_write_jsonl(output_path, outputs)
        print(f"Mapping {len(outputs)}/{len(audit_items)} {pair_id}: eligible={record['eligible']}", flush=True)
    config = {
        **static_config,
        "candidate_count": len(candidates), "eligible_prompt_pairs": sum(row["eligible"] for row in outputs),
        "eligible_rescue_bases": len({row["base_sample_id"] for row in outputs if row["eligible"] and row["base_sample_id"] in rescue_bases}),
    }
    atomic_write_json(output_dir / "video_mapping_coverage_summary.json", config)
    print(config)


if __name__ == "__main__":
    main()
