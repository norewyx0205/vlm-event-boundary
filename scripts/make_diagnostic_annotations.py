import argparse
from pathlib import Path

try:
    from .common import PROJECT_ROOT, read_jsonl, write_jsonl
except ImportError:
    from common import PROJECT_ROOT, read_jsonl, write_jsonl


OPPOSITE_DIRECTION = {
    "left": "right",
    "right": "left",
    "up": "down",
    "down": "up",
}


def object_label(obj):
    if obj.get("reference_label") in {"smallest", "largest"}:
        return f"the {obj['reference_label']} circle"
    if obj.get("label"):
        return obj["label"]
    return f"the {obj.get('color', '').strip()} {obj.get('shape', '').strip()}".strip()


def source_rows(annotation_paths):
    seen = set()
    for path in annotation_paths:
        for row in read_jsonl(path):
            if row.get("prompt_variant") != "original":
                continue
            key = (str(path), row.get("pairing_id") or row.get("video_id"))
            if key in seen:
                continue
            seen.add(key)
            yield path, row


def mirror_rows(base_row, diagnostic_type, question, correct_sentence, incorrect_sentence):
    common = {
        **base_row,
        "diagnostic_type": diagnostic_type,
        "diagnostic_family": "recognize_track_order",
        "diagnostic_source_eval_id": base_row.get("eval_id", ""),
        "question": question,
        "correct_sentence": correct_sentence,
        "incorrect_sentence": incorrect_sentence,
    }
    eval_prefix = f"{base_row.get('pairing_id') or base_row.get('eval_id')}_{diagnostic_type}"
    return [
        {
            **common,
            "eval_id": f"{eval_prefix}_original",
            "prompt_variant": "original",
            "option_A": correct_sentence,
            "option_B": incorrect_sentence,
            "correct_option": "A",
        },
        {
            **common,
            "eval_id": f"{eval_prefix}_swapped",
            "prompt_variant": "swapped",
            "option_A": incorrect_sentence,
            "option_B": correct_sentence,
            "correct_option": "B",
        },
    ]


def make_first_mover_rows(row):
    targets = {obj["id"]: obj for obj in row["target_objects"]}
    first = targets[row["first_object_id"]]
    second = targets[row["second_object_id"]]
    return mirror_rows(
        row,
        "first_mover_identity",
        "Which statement correctly identifies the target object that moves first?",
        f"{object_label(first).capitalize()} moves first.",
        f"{object_label(second).capitalize()} moves first.",
    )


def make_motion_binding_rows(row):
    targets = sorted(row["target_objects"], key=lambda item: item["id"])
    if not targets:
        return []
    subject = targets[int(row.get("base_sample_id", 0)) % len(targets)]
    direction = subject.get("direction")
    if not direction:
        return []
    other_directions = [
        target.get("direction")
        for target in targets
        if target.get("id") != subject.get("id") and target.get("direction") != direction
    ]
    incorrect_direction = other_directions[0] if other_directions else OPPOSITE_DIRECTION.get(direction)
    if not incorrect_direction or incorrect_direction == direction:
        return []
    return mirror_rows(
        row,
        "motion_direction_binding",
        "Which statement correctly describes one target object's motion direction?",
        f"{object_label(subject).capitalize()} moves {direction}.",
        f"{object_label(subject).capitalize()} moves {incorrect_direction}.",
    )


def make_size_extreme_rows(row):
    if row.get("feature_variant") != "size_only":
        return []
    distractors = row.get("distractors") or []
    if not distractors:
        return []
    return mirror_rows(
        row,
        "size_extreme_identity",
        "Which statement correctly describes the relative size of a target object?",
        "The smallest circle is smaller than every distractor.",
        "The largest circle is smaller than every distractor.",
    )


DIAGNOSTIC_BUILDERS = {
    "first_mover_identity": make_first_mover_rows,
    "motion_direction_binding": make_motion_binding_rows,
    "size_extreme_identity": make_size_extreme_rows,
}


def parse_types(value):
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(names) - set(DIAGNOSTIC_BUILDERS))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown diagnostic types: {', '.join(unknown)}")
    return names


def annotation_paths(annotation_path, annotation_root):
    if annotation_path:
        return [Path(annotation_path)]
    root = Path(annotation_root)
    return sorted(root.rglob("annotations.jsonl"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", default=None)
    parser.add_argument("--annotation_root", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument(
        "--diagnostic_types",
        type=parse_types,
        default=list(DIAGNOSTIC_BUILDERS),
        help="Comma-separated diagnostic types.",
    )
    args = parser.parse_args()

    if not args.annotation_path and not args.annotation_root:
        raise SystemExit("Provide --annotation_path or --annotation_root.")

    paths = annotation_paths(args.annotation_path, args.annotation_root)
    if not paths:
        raise SystemExit("No annotations.jsonl files found.")

    output_path = Path(args.output_path) if args.output_path else (
        PROJECT_ROOT / "data" / "diagnostics" / "annotations.jsonl"
    )
    rows = []
    for _, row in source_rows(paths):
        for diagnostic_type in args.diagnostic_types:
            rows.extend(DIAGNOSTIC_BUILDERS[diagnostic_type](row))

    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} diagnostic eval rows to {output_path}")


if __name__ == "__main__":
    main()
