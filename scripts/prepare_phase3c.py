"""Audit frozen Phase 3B evidence and freeze a separate CPU-only Phase 3C pilot."""

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .phase3c_core import (
        CONDITIONS, MODEL, REVISION, SCHEMA, STRATA, atomic_write, baseline_failures,
        checked_positions, digest, file_hash, frozen_write, read_json, read_jsonl, select_balanced,
    )
except ImportError:
    from phase3c_core import (
        CONDITIONS, MODEL, REVISION, SCHEMA, STRATA, atomic_write, baseline_failures,
        checked_positions, digest, file_hash, frozen_write, read_json, read_jsonl, select_balanced,
    )


DEFAULT_SETTINGS = {
    "model_name": MODEL, "model_revision": REVISION,
    "expected_transformers_version": "5.9.0", "expected_torch_version": "2.11.0",
    "expected_qwen_vl_utils_version": "0.0.14", "dtype": "float16",
    "attn_implementation": "eager", "seed": 42, "roi_padding": 8,
    "video_fps": None, "video_num_frames": None, "video_max_pixels": None,
    "max_event_progress_error": 0.1, "primary_recipient_coverage": 1.0,
    "event_phase_dominance_threshold": 0.5,
    "patch_layers": [0, 4, 8, 12, 16, 20], "deepstack_timing_layers": [0, 1, 2],
    "knockout_windows": [list(range(start, start + 4)) for start in range(0, 36, 4)],
    "knockout_query_groups": ["options_all", "query_all"],
    "knockout_key_groups": ["target_1", "target_2", "both_targets"],
    "knockout_event_scope": "event_2", "knockout_heads": "all",
    "required_control": "background", "optional_control": "distractors",
    "control_edge_budget_relative_tolerance": 0.0,
    "control_matching": "exact_keys_per_temporal_bin_and_visible_causal_edges",
    "attention_mask_semantics": "pre_softmax_negative_infinity_with_renormalization",
    "selection_rule": "ascending_base_id_within_fixed_behavior_and_first_mover_quotas",
    "rescue_per_mover": 4, "stable_per_mover": 2,
    "robustness_fill_methods": [], "practical_equivalence_threshold": None,
}


def safe_output(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Phase 3C output must be separate from the Phase 3B evidence tree.")
    return output


def load_archive(source):
    source = Path(source).resolve()
    config_path = source / "vm_run_config.json"
    config = read_json(config_path)
    if config.get("artifact_type") != "real":
        raise ValueError("Phase 3C requires explicitly real Phase 3B evidence.")
    for field, expected in (("model_name", MODEL), ("model_revision", REVISION),
                            ("expected_transformers_version", "5.9.0"),
                            ("dtype", "float16"), ("attn_implementation", "eager")):
        if config.get(field) != expected:
            raise ValueError(f"Archived runtime mismatch for {field}.")
    manifest_path = source / "selection/analysis_case_manifest.jsonl"
    mapping_path = source / "selection/selected_video_mappings.jsonl"
    if file_hash(manifest_path) != config.get("manifest_sha256", {}).get("full"):
        raise ValueError("Source manifest does not match the Phase 3B configuration hash.")
    if file_hash(mapping_path) != config.get("mapping_sha256"):
        raise ValueError("Source mapping does not match the Phase 3B configuration hash.")
    mappings = {}
    for row in read_jsonl(mapping_path):
        if row["pair_id"] in mappings:
            raise ValueError("Duplicate archived mapping pair ID.")
        mappings[row["pair_id"]] = row
    grouped, eval_ids = defaultdict(dict), set()
    for row in read_jsonl(manifest_path):
        pair_id, condition = row["phase3b_pair_id"], row["condition"]
        if condition not in CONDITIONS or row["eval_id"] in eval_ids or condition in grouped[pair_id]:
            raise ValueError("Duplicate or unexpected archived evaluation row.")
        grouped[pair_id][condition] = row
        eval_ids.add(row["eval_id"])
    indices, evidence = {}, {
        "vm_run_config.json": file_hash(config_path),
        "selection/analysis_case_manifest.jsonl": file_hash(manifest_path),
        "selection/selected_video_mappings.jsonl": file_hash(mapping_path),
    }
    for path in sorted((source / "primary/checkpoints").glob("shard_*/activations/*/*/index.json")):
        index = read_json(path)
        if index["eval_id"] in indices:
            raise ValueError("Duplicate capture index for an archived eval_id.")
        shard_path = path.parents[3] / "run_config.json"
        shard = read_json(shard_path)
        if index.get("run_fingerprint") != shard.get("run_fingerprint"):
            raise ValueError("Capture index/shard fingerprint mismatch.")
        for field, expected in (("model_name", MODEL), ("model_revision", REVISION),
                                ("transformers_version", "5.9.0")):
            if shard.get(field) != expected:
                raise ValueError(f"Capture shard runtime mismatch for {field}.")
        indices[index["eval_id"]] = (index, shard)
        for item in (path, shard_path):
            evidence[str(item.relative_to(source))] = file_hash(item)
    candidates, audit = [], []
    for pair_id, rows in sorted(grouped.items()):
        if set(rows) != set(CONDITIONS):
            raise ValueError("Archived pair is missing a boundary condition.")
        low, temporal = (rows[condition] for condition in CONDITIONS)
        errors = []
        for key in ("base_sample_id", "prompt_variant", "first_object_id", "correct_option",
                    "phase3b_analysis_stratum", "phase3b_prompt_pair_behavior", "option_A", "option_B"):
            if low.get(key) != temporal.get(key):
                errors.append(f"paired_annotation_disagreement:{key}")
        stratum = STRATA.get(low["phase3b_analysis_stratum"])
        capture, videos = {}, {}
        mapping = mappings.get(pair_id)
        if mapping is None or mapping.get("eligible") is not True:
            errors.append("missing_or_ineligible_archived_mapping")
        elif (mapping.get("base_sample_id"), mapping.get("prompt_variant"), mapping.get("first_object_id")) != (
                low["base_sample_id"], low["prompt_variant"], low["first_object_id"]):
            errors.append("mapping_case_identity_disagreement")
        for condition, row in rows.items():
            if row["eval_id"] not in indices:
                errors.append(f"{condition}:missing_capture_index")
                continue
            index, shard = indices[row["eval_id"]]
            errors.extend(f"{condition}:{reason}" for reason in baseline_failures(row, index))
            capture[condition] = {key: index[key] for key in (
                "eval_id", "decision", "group_positions", "positions", "video_metadata",
                "input_metadata", "standard_parity", "archived_input_parity", "run_fingerprint",
            )}
            checked_positions(index["positions"], "captured position union")
            for name, positions in index["group_positions"].items():
                checked_positions(positions, name)
                if not set(positions) <= set(index["positions"]):
                    errors.append(f"{condition}:group_not_in_capture_union:{name}")
            path = config.get("video_paths_by_eval_id", {}).get(row["eval_id"])
            expected_hash = config.get("video_sha256", {}).get(path)
            if not expected_hash or shard.get("video_sha256_by_eval_id", {}).get(row["eval_id"]) != expected_hash:
                errors.append(f"{condition}:missing_or_conflicting_video_hash")
            videos[condition] = {"path": path, "sha256": expected_hash}
        if len(capture) == 2 and not errors and stratum:
            correct = [capture[condition]["decision"]["margin"] > 0 for condition in CONDITIONS]
            expected = [False, True] if stratum == "rescue" else [True, True]
            behavior = "temporal_rescue" if stratum == "rescue" else "stable_both_correct"
            if correct != expected or low["phase3b_prompt_pair_behavior"] != behavior:
                errors.append("prompt_pair_behavior_failed_strict_recheck")
        record = {
            "schema": SCHEMA, "pair_id": pair_id, "base_sample_id": int(low["base_sample_id"]),
            "prompt_variant": low["prompt_variant"], "first_object_id": int(low["first_object_id"]),
            "stratum": stratum, "archive_eligible": not errors and stratum is not None,
            "failures": errors, "excluded_stratum": None if stratum else low["phase3b_analysis_stratum"],
            "processor_eligibility": "pending_actual_visual_positions_and_edge_budget_audit",
        }
        audit.append(record)
        if record["archive_eligible"]:
            candidates.append({**record, "rows": rows, "capture_indices": capture,
                "archived_mapping": mapping, "video_provenance": videos})
    candidates.sort(key=lambda row: (row["base_sample_id"], row["prompt_variant"] != "original"))
    bases = [row["base_sample_id"] for row in candidates]
    if len(bases) != len(set(bases)):
        raise ValueError("Candidate evidence must contain at most one selected prompt per independent base.")
    return config, evidence, candidates, audit


def audit_archive(source, output):
    output = safe_output(source, output)
    source_config, evidence, candidates, audit = load_archive(source)
    protocol_path = Path(__file__).resolve().parents[1] / "docs/phase3c_protocol.md"
    settings = dict(DEFAULT_SETTINGS)
    for key in ("seed", "roi_padding", "video_fps", "video_num_frames", "video_max_pixels"):
        if source_config.get(key) != settings[key]:
            raise ValueError(f"Archived input settings differ from the Phase 3C plan: {key}.")
    code = {name: file_hash(Path(__file__).parent / name) for name in (
        "phase3c_core.py", "prepare_phase3c.py", "audit_phase3c_mappings.py",
        "phase3b_core.py", "probe_attention_roi.py", "run_eval.py",
        "activation_patching_core.py", "phase3b_paths.py",
    )}
    plan = {"schema": SCHEMA, "artifact_type": "real", "source_run_root": str(Path(source).resolve()),
        "source_evidence_sha256": evidence, "source_pipeline_fingerprint": source_config["pipeline_fingerprint"],
        "protocol_sha256": file_hash(protocol_path), "preparation_code_sha256": code,
        "settings": settings, "candidate_manifest_payload_sha256": digest(candidates),
        "status": "cpu_archive_audit_complete_processor_audit_pending",
        "gpu_experiment_started": False}
    plan["plan_fingerprint"] = digest(plan)
    frozen_write(output / "plan_config.json", plan)
    frozen_write(output / "archive_audit.jsonl", audit, jsonl=True)
    frozen_write(output / "candidate_manifest.jsonl", candidates, jsonl=True)
    preview, missing = select_balanced(candidates)
    summary = {"schema": SCHEMA, "plan_fingerprint": plan["plan_fingerprint"],
        "archive_pairs_audited": len(audit), "archive_eligible_independent_bases": len(candidates),
        "archive_failures": [row["pair_id"] for row in audit if row["failures"]],
        "eligible_strata": dict(Counter(row["stratum"] for row in candidates)),
        "candidate_preview_pair_ids": [row["pair_id"] for row in preview],
        "preview_missing_quotas": missing, "case_ids_frozen": False,
        "gpu_ready": False, "next_stage": "processor_only_mapping_and_control_audit"}
    frozen_write(output / "candidate_summary.json", summary)
    return summary


def freeze_selection(output):
    try:
        from .audit_phase3c_mappings import validate_plan
        from .phase3c_core import prepare_support_audit
    except ImportError:
        from audit_phase3c_mappings import validate_plan
        from phase3c_core import prepare_support_audit
    output = Path(output).resolve()
    plan = validate_plan(output)
    candidates = read_jsonl(output / "candidate_manifest.jsonl")
    records_path = output / "processor_audit/mapping_audit.jsonl"
    if not records_path.is_file() or not (output / "processor_audit/config.json").is_file():
        raise ValueError("Processor-only audit is missing; run audit_phase3c_mappings.py before freeze.")
    records = read_jsonl(records_path)
    audit_config = read_json(output / "processor_audit/config.json")
    if audit_config.get("artifact_type") != "real" or audit_config["plan_fingerprint"] != plan["plan_fingerprint"]:
        raise ValueError("Processor audit does not belong to this plan.")
    if audit_config["candidate_manifest_sha256"] != file_hash(output / "candidate_manifest.jsonl"):
        raise ValueError("Candidate manifest differs from the processor audit configuration.")
    runtime = audit_config["runtime"]
    if (runtime.get("device") != "cpu" or runtime.get("model_weights_loaded") is not False or
            runtime.get("transformers") != plan["settings"]["expected_transformers_version"] or
            runtime.get("torch", "").split("+")[0] != plan["settings"]["expected_torch_version"] or
            runtime.get("qwen_vl_utils") != plan["settings"]["expected_qwen_vl_utils_version"] or
            runtime.get("deepstack_decoder_injection_layers") != [0, 1, 2]):
        raise ValueError("CPU processor audit runtime/type is incompatible.")
    by_id = {}
    for row in records:
        if row["pair_id"] in by_id:
            raise ValueError("Duplicate processor audit pair ID.")
        if row.get("plan_fingerprint") != plan["plan_fingerprint"]:
            raise ValueError("Processor audit fingerprint mismatch.")
        by_id[row["pair_id"]] = row
    lookup = {row["pair_id"]: row for row in candidates}
    if not set(by_id) <= set(lookup):
        raise ValueError("Processor audit contains an unknown candidate.")
    for pair_id, row in by_id.items():
        if row.get("candidate_sha256") != digest(lookup[pair_id]):
            raise ValueError("Processor audit candidate hash mismatch.")
        if row.get("eligible") is True:
            for condition in CONDITIONS:
                record = row["processor_records"][condition]
                if record["video_sha256"] != lookup[pair_id]["video_provenance"][condition]["sha256"]:
                    raise ValueError("Processor video hash differs from baseline evidence.")
                if file_hash(record["video_path"]) != record["video_sha256"]:
                    raise ValueError("Processor-audited video bytes changed before freeze.")
            computed = prepare_support_audit(lookup[pair_id], lookup[pair_id]["archived_mapping"],
                row["processor_records"], plan["settings"]["max_event_progress_error"], plan["settings"]["seed"])
            if computed != row.get("support_audit"):
                raise ValueError("Stored support/control audit differs from independent reconstruction.")
    selected, missing = select_balanced(candidates, {
        pair_id for pair_id, row in by_id.items() if row.get("eligible") is True
    })
    if missing:
        raise ValueError(f"Cannot freeze 12-case pilot; processor-eligible quotas missing: {missing}.")
    last = max(candidates.index(row) for row in selected)
    if any(row["pair_id"] not in by_id for row in candidates[:last + 1]):
        raise ValueError("Earlier-ranked candidates are unaudited; deterministic selection is not established.")
    manifest, mappings = [], []
    for candidate in selected:
        pair_id = candidate["pair_id"]
        record = by_id[pair_id]
        if not record.get("support_audit") or record["support_audit"].get("primary_support_coverage") != 1.0:
            raise ValueError("Eligible audit is missing complete primary support evidence.")
        mappings.append(record)
        for condition in CONDITIONS:
            row = dict(candidate["rows"][condition])
            row.update({"phase3c_pair_id": pair_id.replace("phase3b_", "phase3c_", 1),
                "phase3c_source_pair_id": pair_id, "phase3c_analysis_stratum": candidate["stratum"],
                "phase3c_plan_fingerprint": plan["plan_fingerprint"],
                "phase3c_mapping_sha256": digest(record), "phase3c_video_sha256": candidate["video_provenance"][condition]["sha256"]})
            manifest.append(row)
    preflight = [next(row for row in selected if row["stratum"] == "rescue" and row["first_object_id"] == mover)
                 for mover in (1, 2)]
    frozen = {"schema": SCHEMA, "artifact_type": "real", "plan_fingerprint": plan["plan_fingerprint"],
        "settings": plan["settings"], "case_manifest_sha256": digest(manifest),
        "mapping_manifest_sha256": digest(mappings), "processor_runtime": audit_config["runtime"],
        "case_ids": [row["pair_id"].replace("phase3b_", "phase3c_", 1) for row in selected],
        "technical_preflight_case_ids": [row["pair_id"].replace("phase3b_", "phase3c_", 1) for row in preflight],
        "source_evidence_sha256": plan["source_evidence_sha256"],
        "source_model_placement": "recorded_in_read_only_phase3b_shard_configs",
        "execution_hardware_and_placement": "pending_execution_environment_baseline_gate",
        "status": "cohort_and_intervention_plan_frozen_gpu_preflight_pending",
        "gpu_ready": False}
    frozen["selection_fingerprint"] = digest(frozen)
    summary = {"schema": SCHEMA, "selection_fingerprint": frozen["selection_fingerprint"],
        "case_ids_frozen": True, "case_count": len(selected), "evaluation_rows": len(manifest),
        "strata": dict(Counter(row["stratum"] for row in selected)),
        "first_mover_by_stratum": {stratum: dict(Counter(str(row["first_object_id"]) for row in selected
            if row["stratum"] == stratum)) for stratum in ("rescue", "stable")},
        "selected_source_pair_ids": [row["pair_id"] for row in selected],
        "processor_ineligible_candidates": [row["pair_id"] for row in candidates[:last + 1]
            if not by_id[row["pair_id"]]["eligible"]],
        "gpu_ready": False, "next_stage": "baseline_and_technical_hook_mask_preflight"}
    artifacts = (
        ("selection/case_manifest.jsonl", manifest, True),
        ("selection/selected_mappings.jsonl", mappings, True),
        ("selection/frozen_config.json", frozen, False),
        ("selection/case_selection_summary.json", summary, False),
    )
    # Check every existing artifact before creating any missing selection file.
    for name, payload, jsonl in artifacts:
        path = output / name
        if path.exists() and (read_jsonl(path) if jsonl else read_json(path)) != payload:
            raise ValueError(f"Frozen Phase 3C artifact differs: {path}; use a new output root.")
    for name, payload, jsonl in artifacts:
        frozen_write(output / name, payload, jsonl)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "freeze"), default="audit")
    parser.add_argument("--source_run_root")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    if args.stage == "audit" and not args.source_run_root:
        parser.error("--source_run_root is required for archive audit.")
    try:
        summary = (audit_archive(args.source_run_root, args.output_dir) if args.stage == "audit"
                   else freeze_selection(args.output_dir))
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Phase 3C {args.stage} is not ready: {exc}\nGPU experiment not started.\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Phase 3C {args.stage} complete in {time.perf_counter() - started:.1f}s; "
          f"checkpoint={Path(args.output_dir).resolve()}; GPU experiment not started.", flush=True)


if __name__ == "__main__":
    main()
