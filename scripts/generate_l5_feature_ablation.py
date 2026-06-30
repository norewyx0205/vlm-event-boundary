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
    "size_only": "L5_size_only",
}

SIZE_SCENE_CONFIGS = {
    "large_few": {
        "target_size_condition": "large",
        "distractor_count_condition": "few",
        "target_radii": (28, 50),
        "distractor_radii": (38,),
        "moving_distractors": 1,
    },
    "large_many": {
        "target_size_condition": "large",
        "distractor_count_condition": "many",
        "target_radii": (28, 50),
        "distractor_radii": (32, 36, 42, 46),
        "moving_distractors": 4,
    },
    "small_few": {
        "target_size_condition": "small",
        "distractor_count_condition": "few",
        "target_radii": (14, 28),
        "distractor_radii": (21,),
        "moving_distractors": 1,
    },
    "small_many": {
        "target_size_condition": "small",
        "distractor_count_condition": "many",
        "target_radii": (14, 28),
        "distractor_radii": (18, 21, 24, 26),
        "moving_distractors": 4,
    },
}

CLEAR_SIZE_SCENE_CONFIGS = {
    "clear_large_few": {
        "target_size_condition": "large",
        "distractor_count_condition": "few",
        "size_contrast_condition": "clear",
        "target_radii": (28, 72),
        "distractor_radii": (46,),
        "moving_distractors": 1,
    },
    "clear_large_many": {
        "target_size_condition": "large",
        "distractor_count_condition": "many",
        "size_contrast_condition": "clear",
        "target_radii": (28, 72),
        "distractor_radii": (40, 46, 52, 58),
        "moving_distractors": 4,
    },
    "clear_small_few": {
        "target_size_condition": "small",
        "distractor_count_condition": "few",
        "size_contrast_condition": "clear",
        "target_radii": (14, 48),
        "distractor_radii": (30,),
        "moving_distractors": 1,
    },
    "clear_small_many": {
        "target_size_condition": "small",
        "distractor_count_condition": "many",
        "size_contrast_condition": "clear",
        "target_radii": (14, 48),
        "distractor_radii": (24, 30, 36, 40),
        "moving_distractors": 4,
    },
}


def parse_variants(value):
    variants = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(variants) - set(FEATURE_VARIANTS))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown feature variants: {', '.join(unknown)}")
    return variants


def clone_args(args, **updates):
    values = vars(args).copy()
    values.update(updates)
    return argparse.Namespace(**values)


def feature_level(base_level, feature_variant):
    return {
        **base_level,
        "difficulty_name": VARIANT_NAMES[feature_variant],
        "sample_prefix": VARIANT_NAMES[feature_variant].lower(),
        "description": f"Level 5 feature ablation: {feature_variant}",
    }


def generate_feature_variants(args, base_level):
    if args.size_stress_only or args.size_clear_contrast_only:
        return

    for feature_variant in args.variants:
        random.seed(args.seed)
        np.random.seed(args.seed)
        target_identity_specs = make_target_identity_specs(args.samples_per_variant)
        variant_args = clone_args(
            args,
            samples_per_level=args.samples_per_variant,
            paired_feature_ablation=True,
            # Preserve the exact RNG sequence used by the existing full/shape/color datasets.
            # Their subject split is 16/14 while first-mover identity is 15/15.
            counterbalance_prompt_subject=False,
            feature_variant=feature_variant,
            target_radii=(20, 42) if feature_variant == "size_only" else None,
            distractor_radii=(28, 34) if feature_variant == "size_only" else None,
            static_count_override=None,
            moving_count_override=None,
            size_scene_variant="",
            size_contrast_condition="main",
            target_size_condition="",
            distractor_count_condition="",
        )
        generate_level(
            feature_level(base_level, feature_variant),
            variant_args,
            args.level_durations,
            target_identity_specs,
        )


def generate_size_stress_pilot(args, base_level):
    if args.skip_size_stress_pilot or args.size_clear_contrast_only:
        return

    pilot_root = Path(args.output_root) / "size_stress_pilot"
    for scene_name, config in SIZE_SCENE_CONFIGS.items():
        random.seed(args.seed)
        np.random.seed(args.seed)
        target_identity_specs = make_target_identity_specs(args.size_stress_samples_per_cell)
        scene_args = clone_args(
            args,
            output_root=str(pilot_root),
            samples_per_level=args.size_stress_samples_per_cell,
            paired_feature_ablation=False,
            counterbalance_prompt_subject=True,
            feature_variant="size_only",
            target_radii=config["target_radii"],
            distractor_radii=config["distractor_radii"],
            static_count_override=0,
            moving_count_override=config["moving_distractors"],
            size_scene_variant=scene_name,
            size_contrast_condition="original",
            target_size_condition=config["target_size_condition"],
            distractor_count_condition=config["distractor_count_condition"],
        )
        level = {
            **base_level,
            "difficulty_name": f"L5_size_only_{scene_name}",
            "sample_prefix": f"l5_size_only_{scene_name}",
            "description": (
                "Level 5 size-only 2x2 pilot: "
                f"{config['target_size_condition']} targets, "
                f"{config['distractor_count_condition']} distractors"
            ),
        }
        generate_level(level, scene_args, args.level_durations, target_identity_specs)


def generate_size_clear_contrast_pilot(args, base_level):
    if args.skip_size_clear_contrast_pilot or args.size_stress_only:
        return

    pilot_root = Path(args.output_root) / "size_clear_contrast_pilot"
    for scene_name, config in CLEAR_SIZE_SCENE_CONFIGS.items():
        random.seed(args.seed)
        np.random.seed(args.seed)
        target_identity_specs = make_target_identity_specs(args.size_clear_contrast_samples_per_cell)
        scene_args = clone_args(
            args,
            output_root=str(pilot_root),
            samples_per_level=args.size_clear_contrast_samples_per_cell,
            paired_feature_ablation=False,
            counterbalance_prompt_subject=True,
            feature_variant="size_only",
            target_radii=config["target_radii"],
            distractor_radii=config["distractor_radii"],
            static_count_override=0,
            moving_count_override=config["moving_distractors"],
            size_scene_variant=scene_name,
            size_contrast_condition=config["size_contrast_condition"],
            target_size_condition=config["target_size_condition"],
            distractor_count_condition=config["distractor_count_condition"],
        )
        level = {
            **base_level,
            "difficulty_name": f"L5_size_only_{scene_name}",
            "sample_prefix": f"l5_size_only_{scene_name}",
            "description": (
                "Level 5 size-only clear-contrast pilot: "
                f"{config['target_size_condition']} targets, "
                f"{config['distractor_count_condition']} distractors"
            ),
        }
        generate_level(level, scene_args, args.level_durations, target_identity_specs)


def write_readme(args):
    output_root = Path(args.output_root)
    lines = [
        "# Level 5 Feature Ablation",
        "",
        f"Dataset version: `{args.dataset_version}`",
        "",
        "The four main variants use paired geometry, motion paths, event order, distractor timing, and boundary timing.",
        "Only the object feature encoding changes.",
        "",
        "| Variant | Feature encoding |",
        "| --- | --- |",
        "| `L5_full` | color and shape conjunction |",
        "| `L5_shape_only` | all objects black; shape identifies each target |",
        "| `L5_color_only` | all objects circles; color identifies each target |",
        "| `L5_size_only` | all objects are black circles; prompts identify the smallest versus largest circle |",
        "",
        "Each main variant contains 30 base samples, four boundary conditions, and mirrored original/swapped prompts.",
        "",
        "## Size-only 2x2 stress pilot",
        "",
        "The `size_stress_pilot/` directory independently manipulates absolute target size and distractor count.",
        "",
        "| Scene variant | Target radii | Moving distractors |",
        "| --- | --- | ---: |",
    ]
    for scene_name, config in SIZE_SCENE_CONFIGS.items():
        lines.append(
            f"| `{scene_name}` | {config['target_radii'][0]}, {config['target_radii'][1]} | "
            f"{config['moving_distractors']} |"
        )
    lines.extend([
        "",
        "## Size-only clear-contrast pilot",
        "",
        "The `size_clear_contrast_pilot/` directory repeats the 2x2 size/crowding design with larger target-distractor size margins.",
        "",
        "| Scene variant | Target radii | Moving distractors |",
        "| --- | --- | ---: |",
    ])
    for scene_name, config in CLEAR_SIZE_SCENE_CONFIGS.items():
        lines.append(
            f"| `{scene_name}` | {config['target_radii'][0]}, {config['target_radii'][1]} | "
            f"{config['moving_distractors']} |"
        )
    lines.extend([
        "",
        "Every distractor radius lies strictly between the target radii, so `the smallest circle` and `the largest circle` each identify exactly one target.",
    ])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    config = {
        "dataset_version": args.dataset_version,
        "samples_per_variant": args.samples_per_variant,
        "size_stress_samples_per_cell": args.size_stress_samples_per_cell,
        "size_clear_contrast_samples_per_cell": args.size_clear_contrast_samples_per_cell,
        "seed": args.seed,
        "fps": args.fps,
        "level_durations": args.level_durations,
        "event_duration_sec": args.event_duration_sec,
        "temporal_gap_sec": args.temporal_gap_sec,
        "visual_marker_sec": args.visual_marker_sec,
        "audio_beep_duration_sec": args.audio_beep_duration_sec,
        "static_distractors": args.static_distractors,
        "moving_distractors": args.moving_distractors,
        "feature_variants": list(FEATURE_VARIANTS),
        "generated_variants": args.variants,
        "size_scene_variants": SIZE_SCENE_CONFIGS,
        "size_clear_contrast_variants": CLEAR_SIZE_SCENE_CONFIGS,
    }
    (output_root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_version", default="l5_feature_ablation_v1")
    parser.add_argument("--samples_per_variant", type=int, default=30)
    parser.add_argument("--size_stress_samples_per_cell", type=int, default=10)
    parser.add_argument("--size_clear_contrast_samples_per_cell", type=int, default=10)
    parser.add_argument("--variants", type=parse_variants, default=list(FEATURE_VARIANTS))
    parser.add_argument("--skip_size_stress_pilot", action="store_true")
    parser.add_argument("--skip_size_clear_contrast_pilot", action="store_true")
    parser.add_argument("--size_stress_only", action="store_true")
    parser.add_argument("--size_clear_contrast_only", action="store_true")
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

    base_level = next(level for level in LEVELS if level["difficulty_level"] == 5)
    generate_feature_variants(args, base_level)
    generate_size_stress_pilot(args, base_level)
    generate_size_clear_contrast_pilot(args, base_level)
    write_readme(args)


if __name__ == "__main__":
    main()
