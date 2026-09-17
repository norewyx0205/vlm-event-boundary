import argparse
import hashlib
import json
import time
import traceback
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import transformers

try:
    from .activation_patching_core import (
        DEFAULT_TOKEN_GROUPS,
        PILOT_EXCLUDED_TOKEN_GROUPS,
        POOLING_METHOD,
        RESIDUAL_STREAM_LOCATION,
        atomic_write_json,
        atomic_write_jsonl,
        divergence_metrics,
        group_manifest_rows,
        pair_id,
        prepare_example,
        run_baseline_capture,
        run_patched_forward,
        selected_capture_groups,
        validate_archived_prediction,
        validate_archived_input_metadata,
    )
    from .common import PROJECT_ROOT, read_jsonl
    from .probe_attention_roi import validate_transformers_version
    from .run_eval import configure_reproducibility, environment_metadata, load_model
except ImportError:
    from activation_patching_core import (
        DEFAULT_TOKEN_GROUPS,
        PILOT_EXCLUDED_TOKEN_GROUPS,
        POOLING_METHOD,
        RESIDUAL_STREAM_LOCATION,
        atomic_write_json,
        atomic_write_jsonl,
        divergence_metrics,
        group_manifest_rows,
        pair_id,
        prepare_example,
        run_baseline_capture,
        run_patched_forward,
        selected_capture_groups,
        validate_archived_prediction,
        validate_archived_input_metadata,
    )
    from common import PROJECT_ROOT, read_jsonl
    from probe_attention_roi import validate_transformers_version
    from run_eval import configure_reproducibility, environment_metadata, load_model


PHASE3_SCHEMA = "temporal_boundary_activation_patching_v2_methodologically_stratified"
DIRECTIONS = (
    ("temporal_to_low", "temporal_boundary", "low_boundary"),
    ("low_to_temporal", "low_boundary", "temporal_boundary"),
)


def shared_row_metadata(row):
    return {
        "phase3_pair_id": row.get("phase3_pair_id") or pair_id(row),
        "base_sample_id": row.get("base_sample_id"),
        "feature_variant": row.get("feature_variant"),
        "prompt_variant": row.get("prompt_variant"),
        "case_category": row.get("phase3_case_category"),
        "base_selection_category": row.get("phase3_base_selection_category"),
        "prompt_pair_behavior": row.get("phase3_prompt_pair_behavior"),
        "analysis_stratum": row.get("phase3_analysis_stratum"),
        "prompt_role": row.get("phase3_prompt_role"),
        "independently_satisfies_rescue": row.get(
            "phase3_independently_satisfies_rescue"
        ),
        "correct_option": row.get("correct_option"),
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fingerprint(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def load_json_list(path):
    path = Path(path)
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}.")
    return payload


def validate_pair_inputs(low, temporal):
    low_ids = low["inputs"].input_ids.detach().cpu()
    temporal_ids = temporal["inputs"].input_ids.detach().cpu()
    return {
        "same_prompt_token_count": low_ids.shape == temporal_ids.shape,
        "same_prompt_token_ids": bool(
            low_ids.shape == temporal_ids.shape and torch.equal(low_ids, temporal_ids)
        ),
        "low_prompt_tokens": int(low_ids.shape[-1]),
        "temporal_prompt_tokens": int(temporal_ids.shape[-1]),
        "low_visual_tokens": low["group_metadata"]["visual_token_count"],
        "temporal_visual_tokens": temporal["group_metadata"]["visual_token_count"],
    }


def completed_divergence_pairs(path):
    if not Path(path).is_file():
        return set(), []
    rows = read_jsonl(path)
    counts = defaultdict(int)
    for row in rows:
        counts[row["phase3_pair_id"]] += 1
    expected = len(DEFAULT_TOKEN_GROUPS) * 36
    complete = {key for key, count in counts.items() if count == expected}
    retained = [row for row in rows if row["phase3_pair_id"] in complete]
    return complete, retained


def divergence_rows_for_pair(pair_key, prepared, captures, eps):
    low = prepared["low_boundary"]
    temporal = prepared["temporal_boundary"]
    low_run = captures["low_boundary"]
    temporal_run = captures["temporal_boundary"]
    metadata = shared_row_metadata(low["row"])
    rows = []
    for layer in range(low_run["layer_count"]):
        for group in DEFAULT_TOKEN_GROUPS:
            low_capture = low_run["captures"][layer].get(
                group, {"count": 0, "mean": None, "values": None}
            )
            temporal_capture = temporal_run["captures"][layer].get(
                group, {"count": 0, "mean": None, "values": None}
            )
            if group in PILOT_EXCLUDED_TOKEN_GROUPS:
                metrics = divergence_metrics(low_capture, temporal_capture, eps)
                metrics.update({
                    "status": "excluded_unmatched_phase",
                    "cosine_distance": None,
                    "relative_l2": None,
                    "tokenwise_cosine_mean": None,
                    "tokenwise_cosine_median": None,
                    "tokenwise_relative_l2_mean": None,
                    "exclusion_reason": PILOT_EXCLUDED_TOKEN_GROUPS[group],
                })
            else:
                metrics = divergence_metrics(low_capture, temporal_capture, eps)
            rows.append({
                "schema": PHASE3_SCHEMA,
                **metadata,
                "boundary_pair": "low_boundary__temporal_boundary",
                "layer": layer,
                "token_group": group,
                "residual_stream_location": RESIDUAL_STREAM_LOCATION,
                "pooling_method": POOLING_METHOD,
                "epsilon": eps,
                "low_token_count": low_capture["count"],
                "temporal_token_count": temporal_capture["count"],
                **metrics,
                "low_prediction": low_run["decision"]["prediction"],
                "temporal_prediction": temporal_run["decision"]["prediction"],
                "low_margin": low_run["decision"]["margin"],
                "temporal_margin": temporal_run["decision"]["margin"],
                "low_is_correct": low_run["decision"]["is_correct"],
                "temporal_is_correct": temporal_run["decision"]["is_correct"],
            })
    return rows


def run_divergence(args, model, processor, pairs, runtime):
    output_path = Path(args.output_path)
    complete, outputs = completed_divergence_pairs(output_path) if args.resume else (set(), [])
    failures = load_json_list(args.errors_path) if args.resume else []
    audits = load_json_list(args.audit_path) if args.resume else []
    audited_pairs = {row.get("phase3_pair_id") for row in audits}
    complete &= audited_pairs
    outputs = [row for row in outputs if row.get("phase3_pair_id") in complete]
    failures = [
        failure
        for failure in failures
        if failure.get("phase3_pair_id") not in complete
    ]
    for index, (pair_key, pair_rows) in enumerate(pairs.items(), start=1):
        if pair_key in complete:
            print(f"Divergence {index}/{len(pairs)} resume: {pair_key}", flush=True)
            continue
        print(f"Divergence {index}/{len(pairs)}: {pair_key}", flush=True)
        try:
            prepared = {
                condition: prepare_example(model, processor, row, args)
                for condition, row in pair_rows.items()
            }
            alignment = validate_pair_inputs(
                prepared["low_boundary"], prepared["temporal_boundary"]
            )
            if not alignment["same_prompt_token_ids"]:
                raise RuntimeError(
                    "Low/temporal processor token IDs differ; positional comparison is unsafe."
                )
            captures = {
                condition: run_baseline_capture(
                    model,
                    processor,
                    example,
                    max_tokenwise_vectors=args.max_tokenwise_vectors,
                    verify_standard=args.verify_standard_generation,
                )
                for condition, example in prepared.items()
            }
            for condition in ("low_boundary", "temporal_boundary"):
                validate_archived_prediction(prepared[condition], captures[condition]["decision"])
                validate_archived_input_metadata(prepared[condition])
            if captures["low_boundary"]["layer_count"] != 36:
                raise RuntimeError(
                    "Phase 3 was designed for 36 decoder layers, found "
                    f"{captures['low_boundary']['layer_count']}."
                )
            pair_rows_out = divergence_rows_for_pair(
                pair_key, prepared, captures, args.epsilon
            )
            for output in pair_rows_out:
                output["run_fingerprint"] = args.run_fingerprint
            outputs.extend(pair_rows_out)
            atomic_write_jsonl(output_path, outputs)
            failures = [
                failure
                for failure in failures
                if failure.get("phase3_pair_id") != pair_key
            ]
            atomic_write_json(args.errors_path, failures)
            audit = {
                "phase3_pair_id": pair_key,
                "input_alignment": alignment,
                "decoder_layer_path": captures["low_boundary"]["decoder_layer_path"],
                "layer_count": captures["low_boundary"]["layer_count"],
                "low_decision": captures["low_boundary"]["decision"],
                "temporal_decision": captures["temporal_boundary"]["decision"],
                "low_standard_parity": captures["low_boundary"]["standard_parity"],
                "temporal_standard_parity": captures["temporal_boundary"]["standard_parity"],
                "low_group_metadata": prepared["low_boundary"]["group_metadata"],
                "temporal_group_metadata": prepared["temporal_boundary"]["group_metadata"],
                "low_input_metadata": prepared["low_boundary"]["input_metadata"],
                "temporal_input_metadata": prepared["temporal_boundary"]["input_metadata"],
                "low_archived_input_parity": validate_archived_input_metadata(
                    prepared["low_boundary"]
                ),
                "temporal_archived_input_parity": validate_archived_input_metadata(
                    prepared["temporal_boundary"]
                ),
                "run_fingerprint": args.run_fingerprint,
            }
            if pair_key not in audited_pairs:
                audits.append(audit)
                audited_pairs.add(pair_key)
            del prepared, captures
            if args.empty_cache_each_pair and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            failures.append({
                "phase3_pair_id": pair_key,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            })
            atomic_write_json(args.errors_path, failures)
            if index <= args.preflight_pairs or not args.continue_on_error:
                raise
            print(f"  failed: {exc}", flush=True)
    atomic_write_json(args.audit_path, audits)
    atomic_write_json(args.errors_path, failures)
    return outputs, failures, audits


def candidates_by_pair(path):
    grouped = defaultdict(list)
    for row in read_jsonl(path):
        grouped[row["phase3_pair_id"]].append(row)
    return dict(grouped)


def completed_patch_keys(path):
    if not Path(path).is_file():
        return set(), []
    rows = read_jsonl(path)
    keys = {
        (row["phase3_pair_id"], row["patch_direction"], int(row["layer"]), row["token_group"])
        for row in rows
        if row.get("status") == "ok"
    }
    return keys, rows


def validate_no_patch(model, processor, prepared, baseline, atol):
    rerun = run_baseline_capture(
        model,
        processor,
        prepared,
        layer_indices=[],
        groups_by_layer={},
        verify_standard=False,
    )
    difference = (rerun["logits"] - baseline["logits"]).abs()
    result = {
        "prediction_match": rerun["decision"]["prediction"] == baseline["decision"]["prediction"],
        "margin_abs_diff": abs(rerun["decision"]["margin"] - baseline["decision"]["margin"]),
        "logits_max_abs_diff": float(difference.max()),
        "atol": atol,
    }
    if not result["prediction_match"] or result["logits_max_abs_diff"] > atol:
        raise RuntimeError(f"No-patch control did not reproduce baseline: {result}")
    return result


def validate_identity_patch(
    model, processor, prepared, baseline, candidate, atol
):
    layer = int(candidate["layer"])
    group = candidate["token_group"]
    capture = baseline["captures"][layer][group]
    patched = run_patched_forward(
        model,
        processor,
        prepared,
        layer,
        group,
        capture,
        capture,
    )
    difference = (patched["logits"] - baseline["logits"]).abs()
    result = {
        "layer": layer,
        "token_group": group,
        "patch_method": patched["patch_method"],
        "alignment_assumption": (
            "same_sequence_positions"
            if patched["patch_method"] == "positionwise_replace"
            else "group_mean_condition_shift"
        ),
        "prediction_match": patched["decision"]["prediction"] == baseline["decision"]["prediction"],
        "margin_abs_diff": abs(patched["decision"]["margin"] - baseline["decision"]["margin"]),
        "logits_max_abs_diff": float(difference.max()),
        "atol": atol,
    }
    if not result["prediction_match"] or result["logits_max_abs_diff"] > atol:
        raise RuntimeError(f"Identity patch was not a no-op: {result}")
    return result


def patch_result_row(candidate, direction, source_condition, target_condition, baseline, patched):
    baseline_decision = baseline["decision"]
    patched_decision = patched["decision"]
    margin_change = patched_decision["margin"] - baseline_decision["margin"]
    aligned_effect = margin_change if direction == "temporal_to_low" else -margin_change
    expected_method = candidate.get("patch_method_eligibility")
    if expected_method and patched["patch_method"] != expected_method:
        raise RuntimeError(
            "Patch method changed between candidate selection and intervention: "
            f"expected={expected_method}, actual={patched['patch_method']}."
        )
    intervention_family = (
        "standard_position_aligned_activation_patching"
        if patched["patch_method"] == "positionwise_replace"
        else "exploratory_pooled_group_mean_delta"
    )
    return {
        "schema": PHASE3_SCHEMA,
        "phase3_pair_id": candidate["phase3_pair_id"],
        "base_sample_id": candidate.get("base_sample_id"),
        "feature_variant": candidate.get("feature_variant"),
        "prompt_variant": candidate.get("prompt_variant"),
        "case_category": candidate.get("case_category"),
        "base_selection_category": candidate.get("base_selection_category"),
        "prompt_pair_behavior": candidate.get("prompt_pair_behavior"),
        "analysis_stratum": candidate.get("analysis_stratum"),
        "prompt_role": candidate.get("prompt_role"),
        "independently_satisfies_rescue": candidate.get(
            "independently_satisfies_rescue"
        ),
        "source_condition": source_condition,
        "target_condition": target_condition,
        "patch_direction": direction,
        "layer": int(candidate["layer"]),
        "token_group": candidate["token_group"],
        "candidate_rank": candidate.get("candidate_rank"),
        "candidate_score": candidate.get("candidate_score"),
        "divergence_stratum": candidate.get("divergence_stratum"),
        "selection_role": candidate.get("selection_role"),
        "cosine_distance": candidate.get("cosine_distance"),
        "relative_l2": candidate.get("relative_l2"),
        "residual_stream_location": RESIDUAL_STREAM_LOCATION,
        "pooling_method": candidate.get("pooling_method", POOLING_METHOD),
        "source_token_count": (
            candidate.get("temporal_token_count")
            if direction == "temporal_to_low"
            else candidate.get("low_token_count")
        ),
        "target_token_count": (
            candidate.get("low_token_count")
            if direction == "temporal_to_low"
            else candidate.get("temporal_token_count")
        ),
        "patch_method": patched["patch_method"],
        "intervention_family": intervention_family,
        "standard_activation_patching": (
            patched["patch_method"] == "positionwise_replace"
        ),
        "primary_causal_test": (
            patched["patch_method"] == "positionwise_replace"
            and candidate.get("selection_role") == "primary_high_divergence"
            and candidate.get("analysis_stratum") == "primary_original_rescue"
        ),
        "position_alignment": candidate.get("position_alignment"),
        "positions_identical": candidate.get("positions_identical"),
        "event_relative_mapping_used": candidate.get(
            "event_relative_mapping_used", False
        ),
        "baseline_target_margin": baseline_decision["margin"],
        "baseline_source_margin": (
            candidate.get("temporal_margin")
            if direction == "temporal_to_low"
            else candidate.get("low_margin")
        ),
        "unpatched_low_margin": candidate.get("low_margin"),
        "unpatched_temporal_margin": candidate.get("temporal_margin"),
        "unpatched_low_prediction": candidate.get("low_prediction"),
        "unpatched_temporal_prediction": candidate.get("temporal_prediction"),
        "patched_margin": patched_decision["margin"],
        "margin_change": margin_change,
        "source_aligned_patch_effect": aligned_effect,
        "baseline_target_prediction": baseline_decision["prediction"],
        "patched_prediction": patched_decision["prediction"],
        "categorical_flip": baseline_decision["prediction"] != patched_decision["prediction"],
        "flip_toward_correct": (
            not baseline_decision["is_correct"] and patched_decision["is_correct"]
        ),
        "flip_away_from_correct": (
            baseline_decision["is_correct"] and not patched_decision["is_correct"]
        ),
        "status": "ok",
    }


def run_patching(args, model, processor, pairs, runtime):
    candidates = candidates_by_pair(args.candidate_path)
    unknown = sorted(set(candidates) - set(pairs))
    if unknown:
        raise ValueError(f"Candidates reference unknown Phase 3 pairs: {unknown}")
    completed, outputs = completed_patch_keys(args.output_path) if args.resume else (set(), [])
    failures = load_json_list(args.errors_path) if args.resume else []
    completed_pairs = {
        pair_key
        for pair_key, pair_candidates in candidates.items()
        if all(
            (pair_key, direction, int(candidate["layer"]), candidate["token_group"])
            in completed
            for candidate in pair_candidates
            for direction, _, _ in DIRECTIONS
        )
    }
    def remains_unresolved(failure):
        if failure.get("patch_direction") is None:
            return failure.get("phase3_pair_id") not in completed_pairs
        key = (
            failure.get("phase3_pair_id"),
            failure.get("patch_direction"),
            failure.get("layer"),
            failure.get("token_group"),
        )
        return key not in completed

    failures = [failure for failure in failures if remains_unresolved(failure)]
    validations = load_json_list(args.validation_path) if args.resume else []
    validation_pairs_done = len(validations)
    for index, (pair_key, pair_candidates) in enumerate(candidates.items(), start=1):
        pending = [
            candidate
            for candidate in pair_candidates
            if any(
                (pair_key, direction, int(candidate["layer"]), candidate["token_group"])
                not in completed
                for direction, _, _ in DIRECTIONS
            )
        ]
        if not pending:
            print(f"Patching {index}/{len(candidates)} resume: {pair_key}", flush=True)
            continue
        print(
            f"Patching {index}/{len(candidates)}: {pair_key} "
            f"({len(pending)} candidate locations)",
            flush=True,
        )
        try:
            pair_rows = pairs[pair_key]
            prepared = {
                condition: prepare_example(model, processor, row, args)
                for condition, row in pair_rows.items()
            }
            capture_groups = {
                condition: selected_capture_groups(pair_candidates, example)
                for condition, example in prepared.items()
            }
            baselines = {
                condition: run_baseline_capture(
                    model,
                    processor,
                    prepared[condition],
                    layer_indices=sorted(capture_groups[condition]),
                    groups_by_layer=capture_groups[condition],
                    max_tokenwise_vectors=10**9,
                    verify_standard=args.verify_standard_generation,
                )
                for condition in ("low_boundary", "temporal_boundary")
            }
            for condition in ("low_boundary", "temporal_boundary"):
                validate_archived_prediction(prepared[condition], baselines[condition]["decision"])
                validate_archived_input_metadata(prepared[condition])

            if validation_pairs_done < args.validation_pairs:
                candidate = pending[0]
                validation = {"phase3_pair_id": pair_key}
                for condition in ("low_boundary", "temporal_boundary"):
                    validation[f"{condition}_no_patch"] = validate_no_patch(
                        model,
                        processor,
                        prepared[condition],
                        baselines[condition],
                        args.no_op_atol,
                    )
                    validation[f"{condition}_identity_patch"] = validate_identity_patch(
                        model,
                        processor,
                        prepared[condition],
                        baselines[condition],
                        candidate,
                        args.no_op_atol,
                    )
                validations.append(validation)
                atomic_write_json(args.validation_path, validations)
                validation_pairs_done += 1

            for candidate in pending:
                layer = int(candidate["layer"])
                group = candidate["token_group"]
                for direction, source_condition, target_condition in DIRECTIONS:
                    key = (pair_key, direction, layer, group)
                    if key in completed:
                        continue
                    try:
                        source_capture = baselines[source_condition]["captures"][layer][group]
                        target_capture = baselines[target_condition]["captures"][layer][group]
                        patched = run_patched_forward(
                            model,
                            processor,
                            prepared[target_condition],
                            layer,
                            group,
                            source_capture,
                            target_capture,
                        )
                        row = patch_result_row(
                            candidate,
                            direction,
                            source_condition,
                            target_condition,
                            baselines[target_condition],
                            patched,
                        )
                        row["run_fingerprint"] = args.run_fingerprint
                        outputs.append(row)
                        completed.add(key)
                        failures = [
                            failure
                            for failure in failures
                            if not (
                                failure.get("phase3_pair_id") == pair_key
                                and failure.get("patch_direction") == direction
                                and failure.get("layer") == layer
                                and failure.get("token_group") == group
                            )
                        ]
                        atomic_write_jsonl(args.output_path, outputs)
                        atomic_write_json(args.errors_path, failures)
                    except Exception as exc:
                        failure = {
                            "phase3_pair_id": pair_key,
                            "patch_direction": direction,
                            "layer": layer,
                            "token_group": group,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                        failures.append(failure)
                        atomic_write_json(args.errors_path, failures)
                        if index <= args.preflight_pairs or not args.continue_on_error:
                            raise
            failures = [
                failure
                for failure in failures
                if not (
                    failure.get("phase3_pair_id") == pair_key
                    and failure.get("patch_direction") is None
                )
            ]
            atomic_write_json(args.errors_path, failures)
            del prepared, baselines
            if args.empty_cache_each_pair and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            failures.append({
                "phase3_pair_id": pair_key,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            })
            atomic_write_json(args.errors_path, failures)
            if index <= args.preflight_pairs or not args.continue_on_error:
                raise
            print(f"  pair failed: {exc}", flush=True)
    atomic_write_json(args.validation_path, validations)
    atomic_write_json(args.errors_path, failures)
    return outputs, failures, validations


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("divergence", "patch"))
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--candidate_path", default=None)
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--expected_transformers_version", default="5.9.0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--deterministic_warn_only", action="store_true")
    parser.add_argument("--attn_implementation", default="eager", choices=("eager",))
    parser.add_argument("--video_fps", type=float, default=None)
    parser.add_argument("--video_num_frames", type=int, default=None)
    parser.add_argument("--video_max_pixels", type=int, default=None)
    parser.add_argument("--roi_padding", type=int, default=8)
    parser.add_argument("--roi_assignment", default="overlap", choices=("overlap", "center"))
    parser.add_argument("--max_tokenwise_vectors", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--validation_pairs", type=int, default=1)
    parser.add_argument(
        "--preflight_pairs",
        type=int,
        default=1,
        help="Fail fast within the first N matched pairs before scaling the run.",
    )
    parser.add_argument("--no_op_atol", type=float, default=1e-3)
    parser.add_argument("--verify_standard_generation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue_on_error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--require_complete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Return a non-zero exit after checkpointing when unresolved items remain. "
            "Enabled by default so a rerun resumes only the missing work."
        ),
    )
    parser.add_argument("--empty_cache_each_pair", action="store_true")
    return parser


def main():
    wall_start = time.perf_counter()
    parser = build_parser()
    args = parser.parse_args()
    if args.stage == "patch" and not args.candidate_path:
        parser.error("patch stage requires --candidate_path.")
    if args.video_fps is not None and args.video_num_frames is not None:
        parser.error("Use only one of --video_fps and --video_num_frames.")
    if args.epsilon <= 0 or args.no_op_atol < 0:
        parser.error("--epsilon must be positive and --no_op_atol non-negative.")
    if args.max_pairs is not None and args.max_pairs <= 0:
        parser.error("--max_pairs must be positive.")
    if args.validation_pairs < 0 or args.preflight_pairs < 0:
        parser.error("--validation_pairs and --preflight_pairs must be non-negative.")

    validate_transformers_version(args.expected_transformers_version)
    configure_reproducibility(
        args.seed,
        deterministic=args.deterministic,
        deterministic_warn_only=args.deterministic_warn_only,
    )
    pairs = group_manifest_rows(read_jsonl(args.manifest_path))
    if args.max_pairs is not None:
        pairs = dict(list(pairs.items())[:args.max_pairs])
    if not pairs:
        raise ValueError("No complete Phase 3 matched pairs were selected.")

    output_path = Path(args.output_path)
    args.errors_path = output_path.with_name(f"{output_path.stem}_errors.json")
    args.audit_path = output_path.with_name(f"{output_path.stem}_audit.json")
    args.validation_path = output_path.with_name(f"{output_path.stem}_validation.json")
    config_path = output_path.with_name(f"{output_path.stem}_config.json")
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fingerprint_payload = {
        "schema": PHASE3_SCHEMA,
        "stage": args.stage,
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "qwen_vl_utils_version": package_version("qwen-vl-utils"),
        "seed": args.seed,
        "deterministic": args.deterministic,
        "deterministic_warn_only": args.deterministic_warn_only,
        "attention_implementation": args.attn_implementation,
        "video_fps": args.video_fps,
        "video_num_frames": args.video_num_frames,
        "video_max_pixels": args.video_max_pixels,
        "roi_padding": args.roi_padding,
        "roi_assignment": args.roi_assignment,
        "epsilon": args.epsilon,
        "preflight_pairs": args.preflight_pairs,
        "validation_pairs": args.validation_pairs,
        "no_op_atol": args.no_op_atol,
        "max_tokenwise_vectors": args.max_tokenwise_vectors,
        "pilot_excluded_token_groups": PILOT_EXCLUDED_TOKEN_GROUPS,
        "verify_standard_generation": args.verify_standard_generation,
        "manifest_sha256": file_sha256(args.manifest_path),
        "candidate_sha256": (
            file_sha256(args.candidate_path) if args.candidate_path else None
        ),
        "selected_pair_ids": list(pairs),
    }
    args.run_fingerprint = stable_fingerprint(fingerprint_payload)
    if args.resume and output_path.is_file():
        if not config_path.is_file():
            raise RuntimeError(
                f"Cannot safely resume {output_path}: matching config is missing. "
                "Preserve the output and choose a new output path or use --no-resume."
            )
        prior_config = json.loads(config_path.read_text(encoding="utf-8"))
        if prior_config.get("run_fingerprint") != args.run_fingerprint:
            raise RuntimeError(
                "Refusing to mix an existing Phase 3 checkpoint with a different "
                "manifest/runtime configuration. Preserve the old files and use a "
                "new output path."
            )

    print(
        f"Phase 3 {args.stage}: transformers={transformers.__version__}, "
        f"pairs={len(pairs)}, device="
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
        flush=True,
    )
    transformers.utils.logging.disable_progress_bar()
    load_start = time.perf_counter()
    model, processor = load_model(
        args.model_name,
        model_revision=args.model_revision,
        attn_implementation=args.attn_implementation,
    )
    load_seconds = time.perf_counter() - load_start
    runtime = {
        "schema": PHASE3_SCHEMA,
        "stage": args.stage,
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "deterministic_warn_only": args.deterministic_warn_only,
        "attention_implementation": args.attn_implementation,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "environment": environment_metadata(model),
        "model_dtype": str(next(model.parameters()).dtype),
        "residual_stream_location": RESIDUAL_STREAM_LOCATION,
        "pooling_method": POOLING_METHOD,
        "epsilon": args.epsilon,
        "preflight_pairs": args.preflight_pairs,
        "validation_pairs": args.validation_pairs,
        "no_op_atol": args.no_op_atol,
        "max_tokenwise_vectors": args.max_tokenwise_vectors,
        "pilot_excluded_token_groups": PILOT_EXCLUDED_TOKEN_GROUPS,
        "verify_standard_generation": args.verify_standard_generation,
        "roi_padding": args.roi_padding,
        "roi_assignment": args.roi_assignment,
        "video_fps": args.video_fps,
        "video_num_frames": args.video_num_frames,
        "video_max_pixels": args.video_max_pixels,
        "manifest_path": str(args.manifest_path),
        "candidate_path": str(args.candidate_path) if args.candidate_path else None,
        "pair_count": len(pairs),
        "model_load_time_sec": load_seconds,
        "run_fingerprint": args.run_fingerprint,
        "fingerprint_payload": fingerprint_payload,
    }
    atomic_write_json(config_path, runtime)

    stage_start = time.perf_counter()
    if args.stage == "divergence":
        outputs, failures, audits = run_divergence(args, model, processor, pairs, runtime)
        validations = []
    else:
        outputs, failures, validations = run_patching(args, model, processor, pairs, runtime)
        audits = []
    elapsed = time.perf_counter() - stage_start
    summary = {
        **runtime,
        "output_rows": len(outputs),
        "failed_items": len(failures),
        "failure_manifest": str(args.errors_path),
        "audit_records": len(audits),
        "validation_records": len(validations),
        "stage_runtime_sec": elapsed,
        "total_wall_time_sec": time.perf_counter() - wall_start,
    }
    atomic_write_json(summary_path, summary)
    print(
        f"Wrote {len(outputs)} {args.stage} rows to {output_path}; "
        f"failures={len(failures)}, runtime={elapsed / 60:.1f} min",
        flush=True,
    )
    if failures and args.require_complete:
        raise RuntimeError(
            f"Phase 3 {args.stage} preserved {len(failures)} unresolved failure(s). "
            "Rerun the same command to resume the missing work."
        )


if __name__ == "__main__":
    main()
