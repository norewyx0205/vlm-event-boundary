import argparse
import itertools
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .common import read_jsonl, write_jsonl
    from .select_attention_cases import pair_outcome
except ImportError:
    from common import read_jsonl, write_jsonl
    from select_attention_cases import pair_outcome


FEATURE_DIRECTORIES = {
    "full": "L5_full",
    "color_only": "L5_color_only",
    "shape_only": "L5_shape_only",
    "size_only": "L5_size_only",
}
DEFAULT_FEATURES = tuple(FEATURE_DIRECTORIES)
DEFAULT_CONDITIONS = (
    "low_boundary",
    "temporal_boundary",
    "visual_boundary",
    "audio_boundary",
)
PAIR_VARIANTS = ("original", "swapped")
OUTCOME_ORDER = ("both_correct", "position_sensitive", "both_wrong")


def parse_csv(value):
    return tuple(part.strip() for part in value.split(",") if part.strip())


def jsonl_from_bytes(payload, source):
    rows = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL in {source}:{line_number}") from exc
    return rows


def load_result_rows(paths):
    rows = []
    for path_value in paths:
        path = Path(path_value)
        if path.is_file() and path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for member in sorted(archive.namelist()):
                    if member.endswith("raw_results.jsonl"):
                        rows.extend(jsonl_from_bytes(archive.read(member), f"{path}::{member}"))
            continue
        files = [path] if path.is_file() else sorted(path.rglob("raw_results.jsonl"))
        for result_file in files:
            rows.extend(read_jsonl(result_file))
    return rows


def load_annotations(annotation_root, features):
    root = Path(annotation_root)
    rows = {}
    for feature in features:
        directory = FEATURE_DIRECTORIES.get(feature)
        if directory is None:
            raise ValueError(f"Unknown feature variant: {feature}")
        path = root / directory / "annotations.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing annotation file: {path}")
        feature_rows = read_jsonl(path)
        unexpected = sorted({row.get("feature_variant") for row in feature_rows} - {feature})
        if unexpected:
            raise ValueError(f"{path} contains unexpected feature variants: {unexpected}")
        rows[feature] = feature_rows
    return rows


def freeze(value):
    if isinstance(value, dict):
        return tuple((key, freeze(value[key])) for key in sorted(value))
    if isinstance(value, list):
        return tuple(freeze(item) for item in value)
    return value


def object_motion_signature(item):
    return freeze({
        "id": item.get("id"),
        "direction": item.get("direction"),
        "motion_kind": item.get("motion_kind"),
        "motion_timing": item.get("motion_timing"),
        "from": item.get("from"),
        "to": item.get("to"),
        "start_frame": item.get("start_frame"),
        "end_frame": item.get("end_frame"),
    })


def matched_structure_signature(row):
    return freeze({
        "fps": row.get("fps"),
        "duration_sec": row.get("duration_sec"),
        "total_frames": row.get("total_frames"),
        "first_object_id": row.get("first_object_id"),
        "second_object_id": row.get("second_object_id"),
        "event_timing": row.get("event_timing"),
        "boundary_timing": row.get("boundary_timing"),
        "targets": [
            object_motion_signature(item)
            for item in sorted(row.get("target_objects") or [], key=lambda item: int(item["id"]))
        ],
        "distractors": [
            object_motion_signature(item)
            for item in sorted(row.get("distractors") or [], key=lambda item: int(item["id"]))
        ],
    })


def annotation_index(annotation_rows, features, conditions):
    index = {}
    base_ids_by_feature = {}
    for feature in features:
        feature_base_ids = set()
        for row in annotation_rows[feature]:
            if row.get("condition") not in conditions or row.get("prompt_variant") not in PAIR_VARIANTS:
                continue
            base_id = int(row["base_sample_id"])
            key = (feature, base_id, row["condition"], row["prompt_variant"])
            if key in index:
                raise ValueError(f"Duplicate annotation key: {key}")
            index[key] = row
            feature_base_ids.add(base_id)
        base_ids_by_feature[feature] = feature_base_ids
    common = set.intersection(*(base_ids_by_feature[feature] for feature in features))
    complete = []
    expected_per_base = len(features) * len(conditions) * len(PAIR_VARIANTS)
    for base_id in sorted(common):
        keys = [
            (feature, base_id, condition, variant)
            for feature in features
            for condition in conditions
            for variant in PAIR_VARIANTS
        ]
        if sum(key in index for key in keys) == expected_per_base:
            complete.append(base_id)
    return index, complete


def validate_matched_structures(index, base_ids, features, conditions):
    checked = 0
    for base_id in base_ids:
        for condition in conditions:
            signatures = {
                feature: matched_structure_signature(
                    index[(feature, base_id, condition, "original")]
                )
                for feature in features
            }
            reference_feature = features[0]
            for feature in features[1:]:
                if signatures[feature] != signatures[reference_feature]:
                    raise ValueError(
                        "Feature variants are not structurally matched for "
                        f"base_sample_id={base_id}, condition={condition}: "
                        f"{reference_feature} != {feature}"
                    )
            checked += 1
    return checked


def result_index(rows, expected_eval_ids):
    index = {}
    for row in rows:
        eval_id = row.get("eval_id")
        if eval_id not in expected_eval_ids:
            continue
        previous = index.get(eval_id)
        if previous is not None:
            comparable = (row.get("prediction"), bool(row.get("is_correct")))
            previous_comparable = (
                previous.get("prediction"),
                bool(previous.get("is_correct")),
            )
            if comparable != previous_comparable:
                raise ValueError(
                    f"Conflicting archived results found for eval_id={eval_id}. "
                    "Pass one run per feature variant."
                )
            continue
        index[eval_id] = row
    missing = sorted(expected_eval_ids - set(index))
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Archived main results are missing {len(missing)} required rows: {preview}"
        )
    return index


def first_mover_label(row):
    targets = sorted(row.get("target_objects") or [], key=lambda item: int(item["id"]))
    if len(targets) != 2:
        raise ValueError(f"Expected two targets for eval_id={row.get('eval_id')}")
    starts = {int(item["id"]): int(item.get("start_frame") or 0) for item in targets}
    first_id = min(starts, key=lambda target_id: (starts[target_id], target_id))
    return f"target_{first_id}_first"


def candidate_profile(base_id, annotation_by_key, archived_by_eval, features, conditions):
    reference = annotation_by_key[(features[0], base_id, conditions[0], "original")]
    outcomes = Counter()
    by_feature = defaultdict(Counter)
    by_condition = defaultdict(Counter)
    pair_outcomes = {}
    for feature in features:
        for condition in conditions:
            pair = []
            for variant in PAIR_VARIANTS:
                annotation = annotation_by_key[(feature, base_id, condition, variant)]
                pair.append(archived_by_eval[annotation["eval_id"]])
            outcome = pair_outcome(pair)
            if outcome is None:
                raise ValueError(
                    f"Incomplete archived mirrored pair: {feature}, {base_id}, {condition}"
                )
            outcomes[outcome] += 1
            by_feature[feature][outcome] += 1
            by_condition[condition][outcome] += 1
            pair_outcomes[f"{feature}__{condition}"] = outcome
    return {
        "base_sample_id": base_id,
        "first_mover": first_mover_label(reference),
        "outcome_counts": dict(outcomes),
        "outcomes_by_feature": {key: dict(value) for key, value in by_feature.items()},
        "outcomes_by_condition": {key: dict(value) for key, value in by_condition.items()},
        "pair_outcomes": pair_outcomes,
    }


def selection_score(profiles):
    mover_counts = Counter(profile["first_mover"] for profile in profiles)
    mover_imbalance = abs(
        mover_counts.get("target_1_first", 0) - mover_counts.get("target_2_first", 0)
    )
    global_counts = Counter()
    feature_coverage = Counter()
    condition_coverage = Counter()
    for profile in profiles:
        global_counts.update(profile["outcome_counts"])
        for feature, counts in profile["outcomes_by_feature"].items():
            for outcome, count in counts.items():
                feature_coverage[(feature, outcome)] += count
        for condition, counts in profile["outcomes_by_condition"].items():
            for outcome, count in counts.items():
                condition_coverage[(condition, outcome)] += count
    global_categories = sum(global_counts[outcome] > 0 for outcome in OUTCOME_ORDER)
    feature_categories = sum(value > 0 for value in feature_coverage.values())
    condition_categories = sum(value > 0 for value in condition_coverage.values())
    minority_outcomes = min((global_counts[outcome] for outcome in OUTCOME_ORDER), default=0)
    return (
        -mover_imbalance,
        global_categories,
        feature_categories,
        condition_categories,
        minority_outcomes,
    )


def choose_profiles(profiles, count):
    if count > len(profiles):
        raise ValueError(f"Requested {count} base samples but only {len(profiles)} are complete.")
    combination_count = math.comb(len(profiles), count)
    if combination_count > 250_000:
        raise ValueError(
            f"Selecting {count} of {len(profiles)} creates {combination_count} combinations. "
            "Use a smaller calibration run."
        )
    best = None
    best_score = None
    for candidate in itertools.combinations(profiles, count):
        score = selection_score(candidate)
        if best is None or score > best_score:
            best = candidate
            best_score = score
    return list(best), best_score, combination_count


def aggregate_profiles(profiles):
    mover_counts = Counter(profile["first_mover"] for profile in profiles)
    outcomes = Counter()
    by_feature = defaultdict(Counter)
    by_condition = defaultdict(Counter)
    for profile in profiles:
        outcomes.update(profile["outcome_counts"])
        for feature, counts in profile["outcomes_by_feature"].items():
            by_feature[feature].update(counts)
        for condition, counts in profile["outcomes_by_condition"].items():
            by_condition[condition].update(counts)
    return {
        "first_mover_counts": dict(mover_counts),
        "pair_outcome_counts": dict(outcomes),
        "pair_outcomes_by_feature": {key: dict(value) for key, value in by_feature.items()},
        "pair_outcomes_by_condition": {key: dict(value) for key, value in by_condition.items()},
    }


def build_manifest(annotation_by_key, archived_by_eval, profiles, features, conditions):
    output = []
    profile_by_id = {profile["base_sample_id"]: profile for profile in profiles}
    for base_id in sorted(profile_by_id):
        profile = profile_by_id[base_id]
        bundle_id = f"matched_feature_calibration_base_{base_id:03d}"
        for feature in features:
            for condition in conditions:
                outcome = profile["pair_outcomes"][f"{feature}__{condition}"]
                for variant in PAIR_VARIANTS:
                    annotation = annotation_by_key[(feature, base_id, condition, variant)]
                    archived = archived_by_eval[annotation["eval_id"]]
                    row = dict(annotation)
                    row.update({
                        "attention_case_label": "matched_feature_calibration",
                        "attention_case_bundle_id": bundle_id,
                        "attention_case_bundle_size": len(features) * len(conditions),
                        "attention_pairing_id": f"{feature}__{base_id:03d}__{condition}",
                        "attention_selection_first_mover": profile["first_mover"],
                        "attention_archived_pair_outcome": outcome,
                        "archived_prediction": archived.get("prediction"),
                        "archived_is_correct": archived.get("is_correct"),
                        "archived_raw_response": archived.get("raw_response"),
                    })
                    output.append(row)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_root", required=True)
    parser.add_argument("--main_results", required=True, nargs="+")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--summary_path", default=None)
    parser.add_argument("--base_samples", type=int, default=4)
    parser.add_argument("--feature_variants", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    args = parser.parse_args()
    if args.base_samples <= 0:
        parser.error("--base_samples must be positive.")
    features = parse_csv(args.feature_variants)
    conditions = parse_csv(args.conditions)
    if len(set(features)) != len(features) or len(set(conditions)) != len(conditions):
        parser.error("Feature variants and conditions must not contain duplicates.")

    annotations = load_annotations(args.annotation_root, features)
    annotation_by_key, complete_base_ids = annotation_index(annotations, features, conditions)
    structure_checks = validate_matched_structures(
        annotation_by_key, complete_base_ids, features, conditions
    )
    expected_eval_ids = {
        row["eval_id"]
        for key, row in annotation_by_key.items()
        if key[1] in complete_base_ids
    }
    archived_by_eval = result_index(load_result_rows(args.main_results), expected_eval_ids)
    profiles = [
        candidate_profile(
            base_id, annotation_by_key, archived_by_eval, features, conditions
        )
        for base_id in complete_base_ids
    ]
    selected, score, combination_count = choose_profiles(profiles, args.base_samples)
    manifest = build_manifest(
        annotation_by_key, archived_by_eval, selected, features, conditions
    )
    expected_rows = args.base_samples * len(features) * len(conditions) * 2
    if len(manifest) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} manifest rows, got {len(manifest)}.")

    output_path = Path(args.output_path)
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else output_path.with_name(f"{output_path.stem}_summary.json")
    )
    write_jsonl(output_path, manifest)
    summary = {
        "selection_schema": "matched_feature_calibration_v1",
        "annotation_root": str(args.annotation_root),
        "main_results": [str(path) for path in args.main_results],
        "feature_variants": list(features),
        "conditions": list(conditions),
        "prompt_variants": list(PAIR_VARIANTS),
        "candidate_base_samples": len(profiles),
        "combination_count_evaluated": combination_count,
        "selection_score": list(score),
        "selected_base_sample_ids": [item["base_sample_id"] for item in selected],
        "selected_base_samples": selected,
        "selection_totals": aggregate_profiles(selected),
        "structural_matches_checked": structure_checks,
        "evaluation_rows": len(manifest),
        "mirrored_video_pairs": len(manifest) // 2,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Wrote {len(manifest)} matched attention rows for base samples "
        f"{summary['selected_base_sample_ids']} to {output_path}"
    )
    print(f"Wrote selection audit to {summary_path}")


if __name__ == "__main__":
    main()
