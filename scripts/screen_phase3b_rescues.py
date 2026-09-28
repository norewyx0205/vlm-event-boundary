"""Classify matched low/temporal behaviour before Phase 3B activation work."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .activation_patching_core import atomic_write_json, atomic_write_jsonl
    from .common import read_jsonl
    from .select_activation_patching_cases import validate_matched_pair
except ImportError:
    from activation_patching_core import atomic_write_json, atomic_write_jsonl
    from common import read_jsonl
    from select_activation_patching_cases import validate_matched_pair


CONDITIONS = ("low_boundary", "temporal_boundary")
VARIANTS = ("original", "swapped")


def classify(low, temporal):
    if low["prediction"] not in {"A", "B"} or temporal["prediction"] not in {"A", "B"}:
        return "invalid_prediction"
    if not low["is_correct"] and temporal["is_correct"]:
        return "temporal_rescue"
    if low["is_correct"] and temporal["is_correct"]:
        return "stable_both_correct"
    if low["is_correct"] and not temporal["is_correct"]:
        return "temporal_degradation"
    return "both_wrong"


def combine_rows(paths, key, compatible_fields):
    combined = {}
    for path in paths:
        for row in read_jsonl(path):
            if row.get("feature_variant") != "full" or row.get("condition") not in CONDITIONS:
                continue
            identity = row[key]
            if identity in combined and any(
                combined[identity].get(field) != row.get(field)
                for field in compatible_fields
            ):
                raise ValueError(f"Conflicting duplicate {key}={identity} across input files.")
            combined[identity] = row
    return combined


def screen(annotation_paths, result_paths):
    annotations = combine_rows(
        annotation_paths,
        "eval_id",
        ("base_sample_id", "condition", "prompt_variant", "video_path", "option_A", "option_B"),
    )
    results = combine_rows(
        result_paths,
        "eval_id",
        ("prediction", "is_correct", "correct_option", "video_path"),
    )
    by_key = {}
    for row in annotations.values():
        key = (int(row["base_sample_id"]), row["prompt_variant"], row["condition"])
        if key in by_key:
            raise ValueError(f"Duplicate annotation for {key}.")
        by_key[key] = row
    screened = []
    candidates = []
    for base_id in sorted({key[0] for key in by_key}):
        for variant in VARIANTS:
            pair = [by_key.get((base_id, variant, condition)) for condition in CONDITIONS]
            if any(item is None for item in pair):
                raise ValueError(f"Missing low/temporal annotation for base {base_id}, {variant}.")
            validate_matched_pair(*pair)
            archived = [results.get(item["eval_id"]) for item in pair]
            if any(item is None for item in archived):
                raise ValueError(f"Missing result for base {base_id}, {variant}.")
            if any(result["correct_option"] != annotation["correct_option"] for result, annotation in zip(archived, pair)):
                raise ValueError(f"Correct option changed for base {base_id}, {variant}.")
            low, temporal = archived
            outcome = classify(low, temporal)
            record = {
                "base_sample_id": base_id,
                "prompt_variant": variant,
                "correct_option": pair[0]["correct_option"],
                "outcome": outcome,
                "low_eval_id": pair[0]["eval_id"],
                "temporal_eval_id": pair[1]["eval_id"],
                "low_prediction": low["prediction"],
                "temporal_prediction": temporal["prediction"],
                "low_is_correct": bool(low["is_correct"]),
                "temporal_is_correct": bool(temporal["is_correct"]),
                "first_object_id": pair[0]["first_object_id"],
                "dataset_version": pair[0]["dataset_version"],
            }
            screened.append(record)
            if outcome == "temporal_rescue":
                candidates.append(record)
    return screened, candidates, annotations, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation_paths", nargs="+", required=True)
    parser.add_argument("--result_paths", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    screened, candidates, _, _ = screen(args.annotation_paths, args.result_paths)
    output = Path(args.output_dir)
    atomic_write_jsonl(output / "screening_results.jsonl", screened)
    atomic_write_jsonl(output / "rescue_candidates.jsonl", candidates)
    outcome_counts = Counter(item["outcome"] for item in screened)
    rescue_by_variant = Counter(item["prompt_variant"] for item in candidates)
    summary = {
        "schema": "phase3b_behavioral_screen_v1",
        "independent_bases": len({item["base_sample_id"] for item in screened}),
        "prompt_pairs": len(screened),
        "outcome_counts": dict(outcome_counts),
        "rescue_by_variant": dict(rescue_by_variant),
        "independent_rescue_bases": len({item["base_sample_id"] for item in candidates}),
        "annotation_paths": args.annotation_paths,
        "result_paths": args.result_paths,
    }
    atomic_write_json(output / "screening_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
