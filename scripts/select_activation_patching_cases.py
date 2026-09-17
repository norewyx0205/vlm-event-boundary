import argparse
import json
from pathlib import Path

try:
    from .activation_patching_core import atomic_write_json, atomic_write_jsonl, pair_id
    from .common import read_jsonl
except ImportError:
    from activation_patching_core import atomic_write_json, atomic_write_jsonl, pair_id
    from common import read_jsonl


DEFAULT_RESCUE_BASES = (5, 11, 14, 15, 17, 19)
DEFAULT_CONTROL_BASES = (1, 2)
CONDITIONS = ("low_boundary", "temporal_boundary")
PROMPT_VARIANTS = ("original", "swapped")


def parse_ints(value):
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def structural_object_signature(item):
    return {
        key: item.get(key)
        for key in (
            "id",
            "shape",
            "color",
            "label",
            "reference_label",
            "radius",
            "from",
            "to",
            "direction",
            "motion_kind",
            "distractor_identity",
            "matched_target_id",
            "shared_target_attribute",
        )
    }


def validate_matched_pair(low, temporal):
    exact_fields = (
        "base_sample_id",
        "feature_variant",
        "prompt_variant",
        "question",
        "option_A",
        "option_B",
        "correct_option",
        "correct_sentence",
        "incorrect_sentence",
        "fps",
        "duration_sec",
        "total_frames",
    )
    mismatches = [field for field in exact_fields if low.get(field) != temporal.get(field)]
    if [structural_object_signature(item) for item in low.get("target_objects", [])] != [
        structural_object_signature(item) for item in temporal.get("target_objects", [])
    ]:
        mismatches.append("target_objects")
    if [structural_object_signature(item) for item in low.get("distractors", [])] != [
        structural_object_signature(item) for item in temporal.get("distractors", [])
    ]:
        mismatches.append("distractors")
    if mismatches:
        raise ValueError(
            f"Low/temporal pair {pair_id(low)} is not structurally matched: "
            + ", ".join(mismatches)
        )


def index_annotations(rows):
    index = {}
    for row in rows:
        if row.get("feature_variant") != "full":
            continue
        key = (
            int(row["base_sample_id"]),
            row.get("condition"),
            row.get("prompt_variant"),
        )
        if key in index:
            raise ValueError(f"Duplicate annotation key: {key}")
        index[key] = row
    return index


def index_results(rows):
    output = {}
    for row in rows:
        eval_id = row.get("eval_id")
        if eval_id in output:
            raise ValueError(f"Duplicate archived result eval_id: {eval_id}")
        output[eval_id] = row
    return output


def classify_original(low_result, temporal_result):
    low_correct = bool(low_result.get("is_correct"))
    temporal_correct = bool(temporal_result.get("is_correct"))
    if not low_correct and temporal_correct:
        return "temporal_rescue"
    if low_correct and temporal_correct:
        return "stable_both_correct"
    return "other"


def select_cases(annotation_rows, result_rows, rescue_bases, control_bases):
    annotations = index_annotations(annotation_rows)
    results = index_results(result_rows)
    selected = []
    audits = []
    requested = [(base, "temporal_rescue") for base in rescue_bases]
    requested += [(base, "stable_both_correct") for base in control_bases]
    for base_id, expected_category in requested:
        original_rows = {}
        for condition in CONDITIONS:
            key = (base_id, condition, "original")
            if key not in annotations:
                raise ValueError(f"Missing required L5_full annotation: {key}")
            annotation = annotations[key]
            result = results.get(annotation["eval_id"])
            if result is None:
                raise ValueError(f"Missing archived result for {annotation['eval_id']}.")
            original_rows[condition] = (annotation, result)
        observed_category = classify_original(
            original_rows["low_boundary"][1],
            original_rows["temporal_boundary"][1],
        )
        if observed_category != expected_category:
            raise ValueError(
                f"Base {base_id} was expected to be {expected_category} on the original "
                f"prompt, but archived behaviour is {observed_category}."
            )

        for variant in PROMPT_VARIANTS:
            pair_rows = []
            for condition in CONDITIONS:
                key = (base_id, condition, variant)
                if key not in annotations:
                    raise ValueError(f"Missing required L5_full annotation: {key}")
                annotation = annotations[key]
                archived = results.get(annotation["eval_id"])
                if archived is None:
                    raise ValueError(f"Missing archived result for {annotation['eval_id']}.")
                row = dict(annotation)
                row.update({
                    "phase3_pair_id": pair_id(annotation),
                    "phase3_case_category": expected_category,
                    "phase3_selection_basis": "original_prompt_archived_behaviour",
                    "archived_prediction": archived.get("prediction"),
                    "archived_is_correct": archived.get("is_correct"),
                    "archived_raw_response": archived.get("raw_response"),
                    "archived_input_metadata": archived.get("input_metadata"),
                })
                pair_rows.append(row)
            validate_matched_pair(pair_rows[0], pair_rows[1])
            selected.extend(pair_rows)
        audits.append({
            "base_sample_id": base_id,
            "expected_category": expected_category,
            "observed_original_category": observed_category,
            "original_low_prediction": original_rows["low_boundary"][1].get("prediction"),
            "original_temporal_prediction": original_rows["temporal_boundary"][1].get("prediction"),
        })
    return selected, audits


def validate_main_run_config(result_path, expected_model_name=None, expected_model_revision=None):
    config_path = Path(result_path).with_name("config.json")
    if not config_path.is_file():
        if expected_model_name or expected_model_revision:
            raise ValueError(
                "Cannot verify the archived behavioural model because its config.json "
                f"is missing beside {result_path}."
            )
        return {
            "config_path": str(config_path),
            "available": False,
            "model_name": None,
            "model_revision": None,
            "video_sampling_request": None,
        }
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_name = config.get("model_name")
    model_revision = config.get("model_revision") or (
        config.get("model_load") or {}
    ).get("model_revision")
    if expected_model_name and model_name != expected_model_name:
        raise ValueError(
            f"Archived main run model mismatch: expected {expected_model_name}, got {model_name}."
        )
    if expected_model_revision and model_revision != expected_model_revision:
        raise ValueError(
            "Archived main run revision mismatch: expected "
            f"{expected_model_revision}, got {model_revision}."
        )
    return {
        "config_path": str(config_path),
        "available": True,
        "model_name": model_name,
        "model_revision": model_revision,
        "transformers_version": (config.get("environment") or {}).get("transformers_version"),
        "video_sampling_request": config.get("video_sampling_request"),
        "decoding": config.get("decoding"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", required=True)
    parser.add_argument("--main_results", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--summary_path", default=None)
    parser.add_argument("--rescue_bases", default=",".join(map(str, DEFAULT_RESCUE_BASES)))
    parser.add_argument("--control_bases", default=",".join(map(str, DEFAULT_CONTROL_BASES)))
    parser.add_argument("--expected_model_name", default=None)
    parser.add_argument("--expected_model_revision", default=None)
    args = parser.parse_args()

    rescue_bases = parse_ints(args.rescue_bases)
    control_bases = parse_ints(args.control_bases)
    if set(rescue_bases) & set(control_bases):
        parser.error("Rescue and control base IDs must be disjoint.")
    selected, audits = select_cases(
        read_jsonl(args.annotation_path),
        read_jsonl(args.main_results),
        rescue_bases,
        control_bases,
    )
    main_run_config = validate_main_run_config(
        args.main_results,
        expected_model_name=args.expected_model_name,
        expected_model_revision=args.expected_model_revision,
    )
    expected_rows = (len(rescue_bases) + len(control_bases)) * 4
    if len(selected) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} selected rows, got {len(selected)}.")
    output_path = Path(args.output_path)
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else output_path.with_name(f"{output_path.stem}_summary.json")
    )
    atomic_write_jsonl(output_path, selected)
    atomic_write_json(summary_path, {
        "selection_schema": "phase3_temporal_rescue_v1",
        "annotation_path": str(args.annotation_path),
        "main_results": str(args.main_results),
        "rescue_bases": list(rescue_bases),
        "control_bases": list(control_bases),
        "conditions": list(CONDITIONS),
        "prompt_variants": list(PROMPT_VARIANTS),
        "evaluation_rows": len(selected),
        "matched_pairs": len(selected) // 2,
        "behavioural_audit": audits,
        "archived_main_run": main_run_config,
    })
    print(
        f"Wrote {len(selected)} Phase 3 evaluation rows "
        f"({len(selected) // 2} matched low/temporal pairs) to {output_path}"
    )


if __name__ == "__main__":
    main()
