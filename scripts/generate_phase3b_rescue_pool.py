"""Generate a resumable Level-5 full-feature low/temporal screening batch."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

try:
    from .activation_patching_core import atomic_write_json
    from .activation_patching_core import atomic_write_jsonl
    from .common import read_jsonl
    from .common import PROJECT_ROOT
    from .generate_ladder_dataset import LEVELS, generate_level, make_target_identity_specs, parse_durations
except ImportError:
    from activation_patching_core import atomic_write_json
    from activation_patching_core import atomic_write_jsonl
    from common import read_jsonl
    from common import PROJECT_ROOT
    from generate_ladder_dataset import LEVELS, generate_level, make_target_identity_specs, parse_durations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default=str(PROJECT_ROOT / "data" / "phase3b_rescue_pool"))
    parser.add_argument("--start_base_id", type=int, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--max_new_bases", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--level_durations", type=parse_durations, default=parse_durations("10,12,14,16,18,20"))
    parser.add_argument("--event_duration_sec", type=float, default=2.0)
    parser.add_argument("--temporal_gap_sec", type=float, default=3.0)
    parser.add_argument("--visual_marker_sec", type=float, default=1.0)
    parser.add_argument("--audio_beep_duration_sec", type=float, default=0.35)
    parser.add_argument("--static_distractors", type=int, default=2)
    parser.add_argument("--moving_distractors", type=int, default=2)
    args = parser.parse_args()
    if args.start_base_id < 31 or args.count < 1:
        parser.error("New Phase 3B base IDs start at 31 and count must be positive.")
    if args.start_base_id + args.count - 1 > 30 + args.max_new_bases:
        parser.error("Requested range exceeds the pre-specified new-base budget.")
    if len(args.level_durations) < 5:
        parser.error("--level_durations must include Level 5.")

    root = Path(args.output_root)
    config_path = root / "phase3b_generation_config.json"
    config = {
        "schema": "phase3b_rescue_pool_v1",
        "generator_code_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("generate_phase3b_rescue_pool.py", "generate_ladder_dataset.py")
        },
        "feature_variant": "full",
        "size_contrast_condition": "main",
        "conditions": ["low_boundary", "temporal_boundary"],
        "counterbalance_prompt_subject": False,
        "seed": args.seed,
        "fps": args.fps,
        "level_durations": args.level_durations,
        "event_duration_sec": args.event_duration_sec,
        "temporal_gap_sec": args.temporal_gap_sec,
        "visual_marker_sec": args.visual_marker_sec,
        "audio_beep_duration_sec": args.audio_beep_duration_sec,
        "static_distractors": args.static_distractors,
        "moving_distractors": args.moving_distractors,
        "max_new_bases": args.max_new_bases,
    }
    if config_path.exists():
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        if prior != config:
            raise ValueError("Generation configuration differs from the existing Phase 3B pool.")
    elif (root / "L5_full" / "annotations.jsonl").exists():
        raise ValueError("An existing rescue pool has no generation config; refusing to append.")
    else:
        atomic_write_json(config_path, config)
    batch_seed = args.seed + args.start_base_id * 1009
    random.seed(batch_seed)
    np.random.seed(batch_seed)
    level = next(item for item in LEVELS if item["difficulty_level"] == 5)
    level = {**level, "difficulty_name": "L5_full", "sample_prefix": "l5_full"}
    args.dataset_version = "phase3b_rescue_pool_v1"
    args.samples_per_level = args.count
    args.base_id_start = args.start_base_id
    args.extra_first_mover = 1 if ((args.start_base_id - 31) // args.count) % 2 == 0 else 2
    args.conditions = ("low_boundary", "temporal_boundary")
    args.append = True
    args.feature_variant = "full"
    args.paired_feature_ablation = True
    args.counterbalance_prompt_subject = False
    args.disable_unrelated_later_motion = False
    args.target_radii = None
    args.distractor_radii = None
    args.static_count_override = None
    args.moving_count_override = None
    args.size_scene_variant = ""
    args.size_contrast_condition = "main"
    args.target_size_condition = ""
    args.distractor_count_condition = ""
    batch_end = args.start_base_id + args.count - 1
    cumulative_path = root / "L5_full" / "annotations.jsonl"
    existing_rows = read_jsonl(cumulative_path) if cumulative_path.is_file() else []
    existing_batch_rows = [
        row for row in existing_rows
        if args.start_base_id <= int(row["base_sample_id"]) <= batch_end
    ]
    if existing_batch_rows and len(existing_batch_rows) != args.count * 4:
        raise RuntimeError("Existing Phase 3B batch has incomplete annotations; inspect before resuming.")
    if not existing_batch_rows:
        generate_level(level, args, args.level_durations, make_target_identity_specs(args.count))
    batch_dir = root / "batches" / f"batch_{args.start_base_id:03d}_{batch_end:03d}"
    batch_rows = [
        row for row in read_jsonl(cumulative_path)
        if args.start_base_id <= int(row["base_sample_id"]) <= batch_end
    ]
    if len(batch_rows) != args.count * 4:
        raise RuntimeError(f"Expected {args.count * 4} batch evaluation rows, got {len(batch_rows)}.")
    missing_videos = [
        row["video_path"] for row in batch_rows
        if not (PROJECT_ROOT / row["video_path"]).is_file()
    ]
    if missing_videos:
        raise RuntimeError(f"Batch annotations reference missing videos: {missing_videos[:3]}")
    atomic_write_jsonl(batch_dir / "annotations.jsonl", batch_rows)

    batch_path = root / "generation_batches.jsonl"
    previous_batches = read_jsonl(batch_path) if batch_path.exists() else []
    if any(row["start_base_id"] == args.start_base_id for row in previous_batches):
        return
    with batch_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "start_base_id": args.start_base_id,
            "count": args.count,
            "batch_seed": batch_seed,
            "conditions": list(args.conditions),
        }) + "\n")


if __name__ == "__main__":
    main()
