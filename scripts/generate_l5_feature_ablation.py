import argparse
import json
import random
from pathlib import Path

import numpy as np

try:
    from .common import PROJECT_ROOT
    from .generate_ladder_dataset import (
        LEVELS,
        FEATURE_VARIANTS,
        generate_level,
        make_target_identity_specs,
        parse_durations,
    )
except ImportError:
    from common import PROJECT_ROOT
    from generate_ladder_dataset import (
        LEVELS,
        FEATURE_VARIANTS,
        generate_level,
        make_target_identity_specs,
        parse_durations,
    )


VARIANT_NAMES = {
    "full": "L5_full",
    "shape_only": "L5_shape_only",
    "color_only": "L5_color_only",
}


def write_readme(args):
    output_root = Path(args.output_root)
    lines = [
        "# Level 5 Feature Ablation",
        "",
        f"Dataset version: `{args.dataset_version}`",
        "",
        "The three variants use paired geometry, motion paths, event order, distractor timing, and boundary timing.",
        "Only the object feature encoding changes.",
        "",
        "| Variant | Feature encoding |",
        "| --- | --- |",
        "| `L5_full` | color and shape conjunction |",
        "| `L5_shape_only` | all objects black; shape identifies each target |",
        "| `L5_color_only` | all objects circles; color identifies each target |",
        "",
        "Each variant contains four boundary conditions and mirrored original/swapped prompts.",
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    config = {
        "dataset_version": args.dataset_version,
        "samples_per_variant": args.samples_per_variant,
        "seed": args.seed,
        "fps": args.fps,
        "level_durations": args.level_durations,
        "event_duration_sec": args.event_duration_sec,
        "temporal_gap_sec": args.temporal_gap_sec,
        "visual_marker_sec": args.visual_marker_sec,
        "audio_beep_duration_sec": args.audio_beep_duration_sec,
        "static_distractors": args.static_distractors,
        "moving_distractors": args.moving_distractors,
        "variants": list(FEATURE_VARIANTS),
    }
    (output_root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_version", default="l5_feature_ablation_v1")
    parser.add_argument("--samples_per_variant", type=int, default=30)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--level_durations", type=parse_durations, default=parse_durations("10,12,14,16,18,20"))
    parser.add_argument("--event_duration_sec", type=float, default=2.0)
    parser.add_argument("--temporal_gap_sec", type=float, default=3.0)
    parser.add_argument("--visual_marker_sec", type=float, default=1.0)
    parser.add_argument("--audio_beep_duration_sec", type=float, default=0.35)
    parser.add_argument("--static_distractors", type=int, default=2)
    parser.add_argument("--moving_distractors", type=int, default=2)
    parser.add_argument("--disable_unrelated_later_motion", action="store_true")
    args = parser.parse_args()

    if len(args.level_durations) < 5:
        raise SystemExit("--level_durations must include a duration for Level 5.")
    if args.output_root is None:
        args.output_root = str(PROJECT_ROOT / "data" / args.dataset_version)

    args.samples_per_level = args.samples_per_variant
    args.paired_feature_ablation = True
    base_level = next(level for level in LEVELS if level["difficulty_level"] == 5)

    for feature_variant in FEATURE_VARIANTS:
        random.seed(args.seed)
        np.random.seed(args.seed)
        target_identity_specs = make_target_identity_specs(args.samples_per_variant)
        args.feature_variant = feature_variant

        level = {
            **base_level,
            "difficulty_name": VARIANT_NAMES[feature_variant],
            "sample_prefix": VARIANT_NAMES[feature_variant].lower(),
            "description": f"Level 5 feature ablation: {feature_variant}",
        }
        generate_level(level, args, args.level_durations, target_identity_specs)

    write_readme(args)


if __name__ == "__main__":
    main()
