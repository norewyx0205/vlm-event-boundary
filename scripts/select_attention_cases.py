import argparse
from collections import defaultdict
from pathlib import Path

try:
    from .common import read_jsonl, write_jsonl
except ImportError:
    from common import read_jsonl, write_jsonl


PAIR_VARIANTS = {"original", "swapped"}


def load_raw_results(path):
    path = Path(path)
    files = [path] if path.is_file() else sorted(path.rglob("raw_results.jsonl"))
    rows = []
    for result_file in files:
        rows.extend(read_jsonl(result_file))
    return rows


def pair_outcome(rows):
    variants = {row.get("prompt_variant"): row for row in rows}
    if not PAIR_VARIANTS.issubset(variants):
        return None
    original = bool(variants["original"].get("is_correct"))
    swapped = bool(variants["swapped"].get("is_correct"))
    if original and swapped:
        return "both_correct"
    if not original and not swapped:
        return "both_wrong"
    return "position_sensitive"


def source_eval_id(row):
    if row.get("source_eval_id"):
        return row["source_eval_id"]
    eval_id = str(row.get("eval_id") or "")
    return eval_id.rsplit("_roi_", 1)[0] if "_roi_" in eval_id else eval_id


def select_cases(annotation_rows, main_rows, perturbation_rows, max_video_pairs):
    annotation_by_eval = {row["eval_id"]: row for row in annotation_rows}
    main_by_eval = {row["eval_id"]: row for row in main_rows if row.get("eval_id")}
    main_pairs = defaultdict(list)
    for row in main_rows:
        key = (str(row.get("base_sample_id")), row.get("condition"))
        main_pairs[key].append(row)
    main_outcomes = {key: pair_outcome(rows) for key, rows in main_pairs.items()}

    perturb_pairs = defaultdict(lambda: defaultdict(list))
    for row in perturbation_rows:
        source_id = source_eval_id(row)
        source_annotation = annotation_by_eval.get(source_id)
        if source_annotation is None:
            continue
        key = (
            str(source_annotation.get("base_sample_id")),
            source_annotation.get("condition"),
        )
        perturb_pairs[key][row.get("perturbation_type")].append(row)

    candidates_by_label = defaultdict(list)
    label_order = []
    seen_candidates = set()

    def add(key, label):
        if key in seen_candidates or main_outcomes.get(key) is None:
            return
        seen_candidates.add(key)
        if label not in candidates_by_label:
            label_order.append(label)
        candidates_by_label[label].append(key)

    for key, variants in sorted(perturb_pairs.items()):
        baseline_name = "reencode_control" if "reencode_control" in variants else "original"
        baseline = pair_outcome(variants.get(baseline_name, []))
        distractors = pair_outcome(variants.get("mask_distractors", []))
        target_outcomes = [
            pair_outcome(variants.get(name, []))
            for name in ("mask_target_1", "mask_target_2")
        ]
        if baseline != "both_correct" and distractors == "both_correct":
            add(key, "distractor_mask_repairs_strict_pair")
        if baseline == "both_correct" and any(value not in {None, "both_correct"} for value in target_outcomes):
            add(key, "target_mask_breaks_strict_pair")
        available = [
            pair_outcome(rows)
            for name, rows in variants.items()
            if name not in {"original", "reencode_control"}
        ]
        if baseline is not None and available and all(value == baseline for value in available):
            add(key, "perturbation_negative_control")

    base_ids = sorted({key[0] for key in main_outcomes})
    for base_id in base_ids:
        temporal = main_outcomes.get((base_id, "temporal_boundary"))
        visual = main_outcomes.get((base_id, "visual_boundary"))
        if temporal == "both_correct" and visual == "both_wrong":
            add((base_id, "temporal_boundary"), "temporal_stable_visual_both_wrong")
            add((base_id, "visual_boundary"), "temporal_stable_visual_both_wrong")
        elif temporal == "both_correct" and visual == "position_sensitive":
            add((base_id, "temporal_boundary"), "temporal_stable_visual_position_sensitive")
            add((base_id, "visual_boundary"), "temporal_stable_visual_position_sensitive")

    for outcome in ("both_correct", "position_sensitive", "both_wrong"):
        for key, value in sorted(main_outcomes.items()):
            if value == outcome:
                add(key, f"balanced_{outcome}")

    candidates = []
    round_index = 0
    while len(candidates) < max_video_pairs:
        added = False
        for label in label_order:
            values = candidates_by_label[label]
            if round_index < len(values):
                candidates.append((values[round_index], label))
                added = True
                if len(candidates) >= max_video_pairs:
                    break
        if not added:
            break
        round_index += 1

    selected = []
    for (base_id, condition), label in candidates:
        rows = [
            row
            for row in annotation_rows
            if str(row.get("base_sample_id")) == base_id and row.get("condition") == condition
        ]
        rows.sort(key=lambda row: (row.get("prompt_variant") != "original", row.get("eval_id")))
        for row in rows:
            archived = main_by_eval.get(row["eval_id"])
            if archived is None:
                raise ValueError(f"Main result is missing selected eval_id={row['eval_id']}")
            output = dict(row)
            output["attention_case_label"] = label
            output["archived_prediction"] = archived.get("prediction")
            output["archived_is_correct"] = archived.get("is_correct")
            output["archived_raw_response"] = archived.get("raw_response")
            selected.append(output)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", required=True)
    parser.add_argument("--main_results", required=True)
    parser.add_argument("--perturbation_results", default=None)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_video_pairs", type=int, default=4)
    args = parser.parse_args()
    if args.max_video_pairs <= 0:
        parser.error("--max_video_pairs must be positive.")

    annotations = read_jsonl(args.annotation_path)
    main_rows = load_raw_results(args.main_results)
    perturbation_rows = (
        load_raw_results(args.perturbation_results)
        if args.perturbation_results
        else []
    )
    selected = select_cases(
        annotations,
        main_rows,
        perturbation_rows,
        args.max_video_pairs,
    )
    output_path = Path(args.output_path)
    write_jsonl(output_path, selected)
    print(
        f"Wrote {len(selected)} attention evaluation rows "
        f"({len(selected) // 2} mirrored video pairs) to {output_path}"
    )


if __name__ == "__main__":
    main()
