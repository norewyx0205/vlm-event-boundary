import argparse
from collections import defaultdict
from pathlib import Path

try:
    from .common import PROJECT_ROOT, read_jsonl
except ImportError:
    from common import PROJECT_ROOT, read_jsonl


VARIANTS = ("full", "shape_only", "color_only", "size_only")
VARIANT_DIRS = {
    "full": "L5_full",
    "shape_only": "L5_shape_only",
    "color_only": "L5_color_only",
    "size_only": "L5_size_only",
}
SIZE_SCENE_VARIANTS = ("large_few", "large_many", "small_few", "small_many")
CLEAR_SIZE_SCENE_VARIANTS = (
    "clear_large_few",
    "clear_large_many",
    "clear_small_few",
    "clear_small_many",
)


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

    if variant != "size_only":
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
    elif variant == "size_only":
        if {obj["color"] for obj in all_objects} != {"black"}:
            errors.append("size_only contains a non-black object")
        if {obj["shape"] for obj in all_objects} != {"circle"}:
            errors.append("size_only contains a non-circle object")
        target_by_label = {obj.get("size_label"): obj for obj in targets}
        if set(target_by_label) != {"small", "large"}:
            errors.append("size_only targets must contain exactly one small and one large label")
            return errors
        small_radius = target_by_label["small"].get("radius")
        large_radius = target_by_label["large"].get("radius")
        if not isinstance(small_radius, int) or not isinstance(large_radius, int):
            errors.append("size_only target radius is missing")
            return errors
        if small_radius >= large_radius:
            errors.append("small target radius must be less than large target radius")
        distractor_radii = [obj.get("radius") for obj in distractors]
        if any(not isinstance(radius, int) for radius in distractor_radii):
            errors.append("size_only distractor radius is missing")
        elif any(not small_radius < radius < large_radius for radius in distractor_radii):
            errors.append("size_only distractor radius does not preserve target uniqueness")
        if any(obj.get("size_label") not in {"medium", "near_small", "near_large"} for obj in distractors):
            errors.append("size_only distractor has an invalid size label")
        if row.get("target_1_radius") != targets[0].get("radius"):
            errors.append("target_1_radius does not match target metadata")
        if row.get("target_2_radius") != targets[1].get("radius"):
            errors.append("target_2_radius does not match target metadata")
        references = {obj.get("reference_label") for obj in targets}
        if references != {"smallest", "largest"}:
            errors.append("size_only targets must have unique smallest/largest reference labels")
        if row.get("target_1_reference_label") != targets[0].get("reference_label"):
            errors.append("target_1_reference_label does not match target metadata")
        if row.get("target_2_reference_label") != targets[1].get("reference_label"):
            errors.append("target_2_reference_label does not match target metadata")
        prompt_text = f"{row.get('option_A', '')} {row.get('option_B', '')}".lower()
        if "smallest circle" not in prompt_text or "largest circle" not in prompt_text:
            errors.append("size_only prompt does not use unique smallest/largest references")

    return errors


def validate_dataset_dir(variant_dir, expected_variant, expected_base_samples, failures):
    annotation_path = variant_dir / "annotations.jsonl"
    if not annotation_path.exists():
        failures.append(f"{variant_dir}: missing annotations.jsonl")
        return []

    rows = read_jsonl(annotation_path)
    video_ids = {row["video_id"] for row in rows}
    disk_videos = {path.name for path in (variant_dir / "videos").glob("*.mp4")}
    pair_counts = defaultdict(set)
    for row in rows:
        if row.get("feature_variant") != expected_variant:
            failures.append(f"{row.get('eval_id')}: unexpected feature_variant")
        pair_counts[row["video_id"]].add(row["prompt_variant"])
        for error in identity_errors(row):
            failures.append(f"{row['eval_id']}: {error}")

    bad_pairs = [
        video_id
        for video_id, prompt_variants in pair_counts.items()
        if prompt_variants != {"original", "swapped"}
    ]
    expected_videos = expected_base_samples * 4
    expected_rows = expected_videos * 2
    if len(rows) != expected_rows:
        failures.append(f"{variant_dir.name}: expected {expected_rows} eval rows, found {len(rows)}")
    if len(video_ids) != expected_videos:
        failures.append(f"{variant_dir.name}: expected {expected_videos} videos, found {len(video_ids)}")
    if bad_pairs:
        failures.append(f"{variant_dir.name}: incomplete mirrored pairs: {bad_pairs[:5]}")
    if disk_videos != video_ids:
        failures.append(
            f"{variant_dir.name}: disk/annotation mismatch extra={len(disk_videos - video_ids)} "
            f"missing={len(video_ids - disk_videos)}"
        )
    print(
        f"{variant_dir.name}: eval_rows={len(rows)} videos={len(video_ids)} "
        f"disk_videos={len(disk_videos)}"
    )
    return rows


def balance_errors(rows, label):
    unique_samples = {}
    for row in rows:
        if row["prompt_variant"] == "original":
            unique_samples.setdefault(str(row["base_sample_id"]), row)
    original_rows = list(unique_samples.values())
    first_labels = []
    subject_labels = []
    for row in original_rows:
        targets = {obj["id"]: obj for obj in row["target_objects"]}
        first_labels.append(targets[row["first_object_id"]].get("size_label"))
        sentence = row["correct_sentence"].lower()
        subject_labels.append("small" if sentence.startswith("the smallest circle") else "large")

    errors = []
    for field_name, values in [("first mover", first_labels), ("prompt subject", subject_labels)]:
        counts = {value: values.count(value) for value in {"small", "large"}}
        if abs(counts["small"] - counts["large"]) > 2:
            errors.append(f"{label}: unbalanced {field_name}: {counts}")
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
        rows = validate_dataset_dir(variant_dir, variant, 30, failures)
        rows_by_variant[variant] = rows
        if variant == "size_only":
            failures.extend(balance_errors(rows, variant))

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

    pilot_root = root / "size_stress_pilot"
    if pilot_root.exists():
        for scene_name in SIZE_SCENE_VARIANTS:
            rows = validate_dataset_dir(
                pilot_root / f"L5_size_only_{scene_name}",
                "size_only",
                10,
                failures,
            )
            failures.extend(balance_errors(rows, scene_name))
            for row in rows:
                if row.get("size_scene_variant") != scene_name:
                    failures.append(f"{row.get('eval_id')}: wrong size_scene_variant")

    clear_root = root / "size_clear_contrast_pilot"
    if clear_root.exists():
        for scene_name in CLEAR_SIZE_SCENE_VARIANTS:
            rows = validate_dataset_dir(
                clear_root / f"L5_size_only_{scene_name}",
                "size_only",
                10,
                failures,
            )
            failures.extend(balance_errors(rows, scene_name))
            for row in rows:
                if row.get("size_scene_variant") != scene_name:
                    failures.append(f"{row.get('eval_id')}: wrong size_scene_variant")
                if row.get("size_contrast_condition") != "clear":
                    failures.append(f"{row.get('eval_id')}: wrong size_contrast_condition")

    print(f"feature_constraints_ok={not failures}")
    print(f"structural_pairing_ok={not failures}")

    if failures:
        for failure in failures[:20]:
            print(f"ERROR: {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
