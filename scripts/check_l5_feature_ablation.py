import argparse
from collections import defaultdict
from pathlib import Path

try:
    from .common import PROJECT_ROOT, read_jsonl
except ImportError:
    from common import PROJECT_ROOT, read_jsonl


VARIANTS = ("full", "shape_only", "color_only")
VARIANT_DIRS = {
    "full": "L5_full",
    "shape_only": "L5_shape_only",
    "color_only": "L5_color_only",
}


def structural_signature(row):
    targets = tuple(
        (
            obj["id"],
            obj["direction"],
            tuple(obj["from"]),
            tuple(obj["to"]),
            obj["start_frame"],
            obj["end_frame"],
        )
        for obj in sorted(row["target_objects"], key=lambda item: item["id"])
    )
    distractors = tuple(
        (
            item["id"],
            item["motion_kind"],
            item["motion_timing"],
            item["direction"],
            tuple(item["from"]),
            tuple(item["to"]),
        )
        for item in sorted(row["distractors"], key=lambda item: item["id"])
    )
    return (
        row["first_object_id"],
        row["second_object_id"],
        row["correct_relation"],
        targets,
        distractors,
        row["event_timing"],
        row["boundary_timing"],
    )


def identity_errors(row):
    errors = []
    variant = row["feature_variant"]
    targets = row["target_objects"]
    distractors = row["distractors"]
    all_objects = targets + distractors

    target_pairs = {(obj["color"], obj["shape"]) for obj in targets}
    distractor_pairs = {(obj["color"], obj["shape"]) for obj in distractors}
    if target_pairs & distractor_pairs:
        errors.append("distractor duplicates a target color-shape identity")

    if variant == "full":
        if len({obj["color"] for obj in targets}) != 2 or len({obj["shape"] for obj in targets}) != 2:
            errors.append("full targets must differ in both color and shape")
    elif variant == "shape_only":
        if {obj["color"] for obj in all_objects} != {"black"}:
            errors.append("shape_only contains a non-black object")
        if len({obj["shape"] for obj in targets}) != 2:
            errors.append("shape_only targets must have distinct shapes")
        if {obj["shape"] for obj in targets} & {obj["shape"] for obj in distractors}:
            errors.append("shape_only distractor duplicates a target shape")
    elif variant == "color_only":
        if {obj["shape"] for obj in all_objects} != {"circle"}:
            errors.append("color_only contains a non-circle object")
        if len({obj["color"] for obj in targets}) != 2:
            errors.append("color_only targets must have distinct colors")
        if {obj["color"] for obj in targets} & {obj["color"] for obj in distractors}:
            errors.append("color_only distractor duplicates a target color")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(PROJECT_ROOT / "data" / "l5_feature_ablation_v1"))
    args = parser.parse_args()

    root = Path(args.root)
    rows_by_variant = {}
    failures = []

    for variant in VARIANTS:
        variant_dir = root / VARIANT_DIRS[variant]
        annotation_path = variant_dir / "annotations.jsonl"
        rows = read_jsonl(annotation_path)
        rows_by_variant[variant] = rows

        video_ids = {row["video_id"] for row in rows}
        disk_videos = {path.name for path in (variant_dir / "videos").glob("*.mp4")}
        pair_counts = defaultdict(set)
        for row in rows:
            pair_counts[row["video_id"]].add(row["prompt_variant"])
            for error in identity_errors(row):
                failures.append(f"{row['eval_id']}: {error}")

        bad_pairs = [
            video_id
            for video_id, prompt_variants in pair_counts.items()
            if prompt_variants != {"original", "swapped"}
        ]
        if len(rows) != len(video_ids) * 2:
            failures.append(f"{variant}: expected two eval rows per video")
        if bad_pairs:
            failures.append(f"{variant}: incomplete mirrored pairs: {bad_pairs[:5]}")
        if disk_videos != video_ids:
            failures.append(
                f"{variant}: disk/annotation mismatch extra={len(disk_videos - video_ids)} "
                f"missing={len(video_ids - disk_videos)}"
            )

        print(
            f"{variant}: eval_rows={len(rows)} videos={len(video_ids)} "
            f"disk_videos={len(disk_videos)}"
        )

    signatures = defaultdict(dict)
    for variant, rows in rows_by_variant.items():
        for row in rows:
            if row["prompt_variant"] != "original":
                continue
            key = (str(row["base_sample_id"]), row["condition"])
            signatures[key][variant] = structural_signature(row)

    for key, variant_signatures in signatures.items():
        if set(variant_signatures) != set(VARIANTS):
            failures.append(f"{key}: missing paired feature variant")
            continue
        if len(set(repr(value) for value in variant_signatures.values())) != 1:
            failures.append(f"{key}: structural pairing differs across feature variants")

    print(f"paired_stimuli={len(signatures)}")
    print(f"feature_constraints_ok={not failures}")
    print(f"structural_pairing_ok={not failures}")

    if failures:
        for failure in failures[:20]:
            print(f"ERROR: {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
