"""Outcome-blind, disjoint-case selection and bindings for Phase 3C replication."""

import re
from pathlib import Path

try:
    from . import phase3c_budget as budget
    from . import phase3c_support as support
    from .audit_phase3c_mappings import load_processor, processor_condition
    from .phase3c_core import CONDITIONS, atomic_write, baseline_failures, digest, file_hash, frozen_write, prepare_support_audit, read_json, read_jsonl, select_balanced
    from .phase3c_execution import execution_hashes, load_selection
    from .prepare_phase3c import DEFAULT_SETTINGS, load_archive
    from .analyze_phase3c_footprint import geometry_audit
except ImportError:
    import phase3c_budget as budget
    import phase3c_support as support
    from audit_phase3c_mappings import load_processor, processor_condition
    from phase3c_core import CONDITIONS, atomic_write, baseline_failures, digest, file_hash, frozen_write, prepare_support_audit, read_json, read_jsonl, select_balanced
    from phase3c_execution import execution_hashes, load_selection
    from prepare_phase3c import DEFAULT_SETTINGS, load_archive
    from analyze_phase3c_footprint import geometry_audit


SCHEMA = "phase3c_independent_replication_v1"
BUDGETS = (64, 256)
ANCHORS = ("video_t1_e2", "video_t2_e2", "both_targets_event2", "whole_event2", "non_target_complement")


def code_hashes():
    return {name: file_hash(Path(__file__).parent / name) for name in
            ("phase3c_replication.py", "run_phase3c_replication.py", "analyze_phase3c_footprint.py")}


def filter_candidates(candidates, excluded):
    result = [row for row in candidates if row["base_sample_id"] not in set(excluded)]
    result.sort(key=lambda row: (row["base_sample_id"], row["prompt_variant"] != "original"))
    if len({row["base_sample_id"] for row in result}) != len(result):
        raise ValueError("Independent replication requires one prompt pair per unused base.")
    return result


def protected_output(args):
    root = Path(args.output_root).resolve()
    for source in (args.source_run, args.exclude_plan, args.reference_run):
        path = Path(source).resolve()
        if root == path or root in path.parents or path in root.parents:
            raise ValueError("Keep all previous research evidence read-only; use a separate replication root.")
    return root


def create_plan(args):
    root = protected_output(args)
    source_config, evidence, candidates, archive_audit = load_archive(args.source_run)
    excluded_config, excluded_pairs, _ = load_selection(args.exclude_plan)
    excluded = sorted({pair[CONDITIONS[0]]["base_sample_id"] for pair in excluded_pairs.values()})
    candidates = filter_candidates(candidates, excluded)
    _, missing = select_balanced(candidates)
    if missing:
        raise ValueError(f"Not enough unused archival cases for fixed mover/behavior quotas: {missing}.")
    reference = read_json(Path(args.reference_run) / "observed_runtime.json")
    if not read_json(Path(args.reference_run) / "analysis/aggregate_summary.json").get("complete"):
        raise ValueError("Reference runtime must belong to the completed fixed-budget run.")
    settings = dict(DEFAULT_SETTINGS)
    for key in ("seed", "roi_padding", "video_fps", "video_num_frames", "video_max_pixels"):
        if source_config.get(key) != settings[key]:
            raise ValueError("Source processor settings differ from the replication.")
    protocol_path = Path(__file__).resolve().parents[1] / "docs/phase3c_replication.md"
    plan = {"schema": SCHEMA, "artifact_type": "real", "source_run_root": str(Path(args.source_run).resolve()),
        "excluded_plan": str(Path(args.exclude_plan).resolve()), "excluded_selection_fingerprint": excluded_config["selection_fingerprint"],
        "excluded_base_ids": excluded, "source_evidence_sha256": evidence,
        "source_pipeline_fingerprint": source_config["pipeline_fingerprint"], "settings": settings,
        "reference_runtime": reference["runtime"], "reference_runtime_sha256": file_hash(Path(args.reference_run) / "observed_runtime.json"),
        "reference_run": str(Path(args.reference_run).resolve()), "replication_code_sha256": code_hashes(),
        "measurement_code_sha256": execution_hashes(), "budget_runtime_code_sha256": budget.code_hashes(),
        "protocol_sha256": file_hash(protocol_path), "candidate_payload_sha256": digest(candidates),
        "selection_rule": "ascending unused base ID within 4/4 rescue and 2/2 stable mover quotas; processor and live-baseline eligibility first",
        "intervention_scope": {"layers": list(budget.LAYERS), "budgets": list(BUDGETS),
            "seeds": list(budget.SEEDS), "anchors": list(ANCHORS), "knockout_enabled": False},
        "primary_contrast": "rescue temporal_to_low L16 budget256 targets_plus_context minus context_only",
        "secondary_contrasts": ["L16 context_only budget256 minus budget64", "L16 whole_event2 minus both_targets_event2",
                                "single ROI first_mover_during_event_2 versus second_mover_own_event; coverage retained"],
        "scope_limit": "Independent stimuli from the same behavioral pool, not population sampling or external-dataset replication."}
    plan["plan_fingerprint"] = digest(plan)
    frozen_write(root / "replication_plan.json", plan)
    frozen_write(root / "preparation/candidate_manifest.jsonl", candidates, jsonl=True)
    frozen_write(root / "preparation/archive_audit.jsonl", archive_audit, jsonl=True)
    return plan, candidates


def validate_plan(args):
    root = protected_output(args)
    plan = read_json(root / "replication_plan.json")
    if (plan.get("schema") != SCHEMA or plan.get("artifact_type") != "real" or
            plan["plan_fingerprint"] != digest({key: value for key, value in plan.items() if key != "plan_fingerprint"}) or
            plan["replication_code_sha256"] != code_hashes() or plan["measurement_code_sha256"] != execution_hashes() or
            plan["budget_runtime_code_sha256"] != budget.code_hashes()):
        raise ValueError("Replication code/provenance binding changed; preserve this root.")
    if plan["source_run_root"] != str(Path(args.source_run).resolve()):
        raise ValueError("Replication source root changed.")
    for relative, expected in plan["source_evidence_sha256"].items():
        if file_hash(Path(args.source_run) / relative) != expected:
            raise ValueError("Read-only Phase 3B source evidence changed.")
    if file_hash(Path(plan["reference_run"]) / "observed_runtime.json") != plan["reference_runtime_sha256"]:
        raise ValueError("Reference runtime provenance changed.")
    excluded, excluded_pairs, _ = load_selection(args.exclude_plan)
    if (excluded["selection_fingerprint"] != plan["excluded_selection_fingerprint"] or
            sorted({pair[CONDITIONS[0]]["base_sample_id"] for pair in excluded_pairs.values()}) != plan["excluded_base_ids"]):
        raise ValueError("Previously used case exclusion set changed.")
    protocol_path = Path(__file__).resolve().parents[1] / "docs/phase3c_replication.md"
    if file_hash(protocol_path) != plan["protocol_sha256"]:
        raise ValueError("Replication protocol changed after freeze.")
    candidates = read_jsonl(root / "preparation/candidate_manifest.jsonl")
    if digest(candidates) != plan["candidate_payload_sha256"] or any(row["base_sample_id"] in plan["excluded_base_ids"] for row in candidates):
        raise ValueError("Replication pool overlaps the old cohort or changed.")
    return plan, candidates


def expanded_audit(candidate, record):
    pair_id = candidate["pair_id"].replace("phase3b_", "phase3c_", 1)
    extra = budget.audit_supports(pair_id, candidate["rows"], record, budgets=BUDGETS)
    complement = support.audit_supports(pair_id, candidate["rows"], record)["supports"][support.SUPPORTS[0]]
    extras = {**record["support_audit"]["supports"], "non_target_complement": complement, **extra["supports"]}
    specs = dict(extra["support_specs"])
    for name in ANCHORS:
        specs[name] = {"family": "anchor", "anchor": name, "budget": extras[name]["mapped_token_count"], "sampling_seed": None}
    return {**extra, "supports": {name: extras[name] for name in (*ANCHORS, *extra["supports"])}, "support_specs": specs}


def processor_audit(args):
    plan, candidates = validate_plan(args)
    root = Path(args.output_root)
    output = root / "preparation/processor_audit"
    path_map = read_json(Path(args.source_run) / "vm_run_config.json").get("path_map", {})
    processor, runtime = load_processor(plan["settings"])
    frozen_write(output / "config.json", {"artifact_type": "real", "plan_fingerprint": plan["plan_fingerprint"],
        "runtime": runtime, "path_map": path_map, "project_root": args.project_root})
    records = {}
    path = output / "rows.jsonl"
    for row in read_jsonl(path) if path.exists() else []:
        if row["pair_id"] in records:
            raise ValueError("Duplicate CPU replication checkpoint.")
        records[row["pair_id"]] = row
    for number, candidate in enumerate(candidates, 1):
        pair_id = candidate["pair_id"]
        if pair_id in records:
            old = records[pair_id]
            if old["plan_fingerprint"] != plan["plan_fingerprint"] or old["candidate_sha256"] != digest(candidate):
                raise ValueError("Foreign/changed processor checkpoint.")
            continue
        print(f"Replication CPU mapping {number}/{len(candidates)}: {pair_id}", flush=True)
        row = {"pair_id": pair_id, "candidate_sha256": digest(candidate), "plan_fingerprint": plan["plan_fingerprint"], "eligible": False}
        try:
            records_by_condition = {condition: processor_condition(candidate, condition, processor, plan["settings"],
                args.project_root, path_map) for condition in CONDITIONS}
            audit = prepare_support_audit(candidate, candidate["archived_mapping"], records_by_condition,
                plan["settings"]["max_event_progress_error"], plan["settings"]["seed"])
            row.update(processor_records=records_by_condition, support_audit=audit)
            row["replication_support_audit"] = expanded_audit(candidate, row)
            row["eligible"] = True
        except (ValueError, FileNotFoundError) as exc:
            row.update(failure_type=type(exc).__name__, failure_message=str(exc))
        if row["eligible"]:
            row["geometry_audit"] = geometry_audit(candidate["rows"], row, plan["settings"]["roi_padding"])
        records[pair_id] = row
        atomic_write(path, [records[item["pair_id"]] for item in candidates if item["pair_id"] in records], jsonl=True)
    _, missing = select_balanced(candidates, {key for key, row in records.items() if row["eligible"]})
    summary = {"complete": len(records) == len(candidates), "candidate_count": len(candidates),
        "eligible_count": sum(row["eligible"] for row in records.values()), "missing_quotas": missing,
        "model_loaded": False, "plan_fingerprint": plan["plan_fingerprint"]}
    atomic_write(output / "summary.json", summary)
    if missing:
        raise ValueError(f"CPU-eligible independent cohort is unavailable: {missing}.")
    return summary


def audited_candidates(args):
    plan, candidates = validate_plan(args)
    path = Path(args.output_root) / "preparation/processor_audit/rows.jsonl"
    records = {row["pair_id"]: row for row in read_jsonl(path)}
    if len(records) != len(read_jsonl(path)) or set(records) != {row["pair_id"] for row in candidates}:
        raise ValueError("Incomplete/duplicate CPU eligibility ledger.")
    for candidate in candidates:
        record = records[candidate["pair_id"]]
        if record["plan_fingerprint"] != plan["plan_fingerprint"] or record["candidate_sha256"] != digest(candidate):
            raise ValueError("CPU evidence does not match the frozen candidate.")
        if record["eligible"]:
            for item in record["processor_records"].values():
                if file_hash(item["video_path"]) != item["video_sha256"]:
                    raise ValueError("Processor-audited video bytes changed.")
            expected = prepare_support_audit(candidate, candidate["archived_mapping"], record["processor_records"],
                plan["settings"]["max_event_progress_error"], plan["settings"]["seed"])
            if expected != record["support_audit"]:
                raise ValueError("CPU mapping/control audit no longer matches archived evidence.")
            if expanded_audit(candidate, record) != record["replication_support_audit"]:
                raise ValueError("Fixed replication support audit changed.")
    return plan, candidates, records


def storage_estimate(records):
    estimates = []
    for record in records.values():
        if not record["eligible"]:
            continue
        total = 0
        for condition in CONDITIONS:
            item = record["processor_records"][condition]
            side = "low_positions" if condition == CONDITIONS[0] else "temporal_positions"
            positions = set(record["support_audit"]["supports"]["whole_event2"][side])
            positions.update(position for group in item["group_positions"].values() for position in group)
            total += len(positions) * 4096 * 2 * 39 + 2 * 1024**2
        estimates.append(total)
    if len(estimates) < 12:
        raise ValueError("Insufficient eligible cases for conservative storage estimate.")
    upper = sum(sorted(estimates, reverse=True)[:12])
    return {"capture_upper_bytes": upper, "capture_backup_and_overhead_bytes": int(upper * 2.3),
            "safety_reserve_bytes": 12 * 1024**3, "required_free_bytes": int(upper * 2.3) + 12 * 1024**3}


def canonical_pair(candidate, plan):
    pair_id = candidate["pair_id"].replace("phase3b_", "phase3c_", 1)
    return {condition: {**candidate["rows"][condition], "phase3c_pair_id": pair_id,
        "phase3c_analysis_stratum": candidate["stratum"], "phase3c_plan_fingerprint": plan["plan_fingerprint"],
        "phase3c_video_sha256": candidate["video_provenance"][condition]["sha256"]} for condition in CONDITIONS}


def live_failures(annotation, result):
    failures = baseline_failures(annotation, result)
    if (result.get("capture_noop_parity", {}).get("exact_match") is not True or
            result.get("capture_noop_parity", {}).get("max_abs_diff") != 0):
        failures.append("live_capture_noop_failed")
    hook = result.get("hook_audit", {})
    if hook.get("block_output_sites") != 36 or hook.get("post_deepstack_layers") != [0, 1, 2]:
        failures.append("live_hook_sites_missing")
    signature = result.get("logits_signature", {})
    if not re.fullmatch(r"[0-9a-f]{64}", signature.get("sha256", "")) or signature.get("dtype") != "torch.float32":
        failures.append("live_full_logit_signature_missing")
    return failures


def freeze_cohort(args, live_records):
    plan, candidates, mappings = audited_candidates(args)
    eligible = set()
    for candidate in candidates:
        key = candidate["pair_id"]
        if key not in live_records:
            continue
        record = live_records[key]
        if record["candidate_sha256"] != digest(candidate) or record["plan_fingerprint"] != plan["plan_fingerprint"]:
            raise ValueError("Live screening fingerprint changed.")
        if record.get("eligible"):
            for condition in CONDITIONS:
                if live_failures(canonical_pair(candidate, plan)[condition], record["conditions"][condition]):
                    raise ValueError("A claimed eligible live baseline failed strict verification.")
            eligible.add(key)
    selected, missing = select_balanced(candidates, eligible)
    if missing:
        raise ValueError(f"Live-baseline eligible replication quotas missing: {missing}.")
    last = max(candidates.index(row) for row in selected)
    if any(row["pair_id"] not in live_records for row in candidates[:last + 1] if mappings[row["pair_id"]]["eligible"]):
        raise ValueError("Earlier eligible candidates were not live-audited; selection is not outcome-blind.")
    bases = [row["base_sample_id"] for row in selected]
    if len(bases) != 12 or len(set(bases)) != 12 or set(bases) & set(plan["excluded_base_ids"]):
        raise ValueError("Independent replication cohort overlaps the prior pilot or has duplicates.")
    root = Path(args.output_root)
    pairs = {candidate["pair_id"].replace("phase3b_", "phase3c_", 1): canonical_pair(candidate, plan) for candidate in selected}
    selected_maps = {key: mappings[candidate["pair_id"]] for key, candidate in zip(pairs, selected)}
    reps = [next(key for key, candidate in zip(pairs, selected) if candidate["stratum"] == "rescue" and candidate["first_object_id"] == mover)
            for mover in (1, 2)]
    selection = {"schema": SCHEMA, "artifact_type": "real", "plan_fingerprint": plan["plan_fingerprint"],
        "case_ids": list(pairs), "base_ids": bases, "preflight_case_ids": reps,
        "excluded_base_ids": plan["excluded_base_ids"], "pairs_sha256": digest(pairs), "mappings_sha256": digest(selected_maps),
        "live_audit_sha256": {row["pair_id"]: digest(live_records[row["pair_id"]]) for row in selected},
        "selection_rule": plan["selection_rule"], "new_intervention_results_used": False}
    selection["selection_fingerprint"] = digest(selection)
    frozen_write(root / "selection/pairs.json", pairs)
    frozen_write(root / "selection/mappings.json", selected_maps)
    frozen_write(root / "selection/frozen_config.json", selection)
    return selection, pairs, selected_maps


def selected(args):
    root = Path(args.output_root)
    records = {row["pair_id"]: row for row in read_jsonl(root / "live_screening/rows.jsonl")}
    return freeze_cohort(args, records)


def capture_binding(args, selection, runtime):
    value = {"selection_fingerprint": selection["selection_fingerprint"], "runtime": runtime,
             "measurement_code_sha256": execution_hashes(), "replication_code_sha256": code_hashes()}
    return digest(value)


def build_budget_context(args, freeze=False):
    import torch
    root = Path(args.output_root)
    plan, _, _ = audited_candidates(args)
    selection, pairs, mappings = selected(args)
    runtime = read_json(root / "capture_runtime.json")["runtime"]
    budget.runtime_compatibility(plan["reference_runtime"], runtime)
    baselines = {}
    capture_fp = capture_binding(args, selection, runtime)
    captures = read_jsonl(root / "capture/rows.jsonl")
    for row in captures:
        if row["eval_id"] in baselines or row.get("capture_fingerprint") != capture_fp:
            raise ValueError("Duplicate/foreign replication capture.")
        baselines[row["eval_id"]] = row
    expected_ids = {row["eval_id"] for pair in pairs.values() for row in pair.values()}
    if set(baselines) != expected_ids:
        raise ValueError("Replication captures are incomplete.")
    live = {row["pair_id"]: row for row in read_jsonl(root / "live_screening/rows.jsonl")}
    for pair_id, pair in pairs.items():
        source_id = mappings[pair_id]["pair_id"]
        for condition, annotation in pair.items():
            saved = baselines[annotation["eval_id"]]
            if baseline_failures(annotation, saved) or not saved.get("capture_noop_parity", {}).get("exact_match"):
                raise ValueError("Replication capture baseline/parity failed; do not replace a frozen case.")
            index = read_json(saved["capture_index_path"])
            if (file_hash(saved["capture_index_path"]) != saved["capture_index_sha256"] or
                    file_hash(index["vectors_path"]) != index["vectors_sha256"] or index["site_count"] != 39 or
                    index["execution_fingerprint"] != capture_fp):
                raise ValueError("Replication all-layer capture evidence changed.")
            payload = torch.load(index["vectors_path"], map_location="cpu", weights_only=True)
            screened = live[source_id]["conditions"][condition]
            if (budget.tensor_signature(payload["logits"]) != screened["logits_signature"] or
                    saved["input_tensor_sha256"] != screened["input_tensor_sha256"]):
                raise ValueError("Post-freeze capture differs from the live selection baseline.")
            del payload
    audit = {key: mappings[key]["replication_support_audit"] for key in pairs}
    config = {"schema": budget.SCHEMA, "artifact_type": "real", "experiment_kind": SCHEMA,
        "case_ids": list(pairs), "preflight_case_ids": selection["preflight_case_ids"], "layers": list(budget.LAYERS),
        "budgets": list(BUDGETS), "sampling_seeds": list(budget.SEEDS), "support_audit": audit,
        "source_runtime": runtime, "source_path_map": read_json(root / "preparation/processor_audit/config.json")["path_map"],
        "followup_code_sha256": budget.code_hashes(), "replication_code_sha256": code_hashes(),
        "selection_fingerprint": selection["selection_fingerprint"], "capture_fingerprint": capture_fp,
        "plan_fingerprint": plan["plan_fingerprint"], "primary_contrast": plan["primary_contrast"],
        "secondary_contrasts": plan["secondary_contrasts"], "new_independent_cases": True,
        "expected_replay_signatures": budget.replay_signatures(pairs, mappings, baselines),
        "capture_index_sha256": {row["capture_index_path"]: row["capture_index_sha256"] for row in baselines.values()},
        "capture_replay_count": 24, "technical_identity_count": 136, "primary_patch_count": 816,
        "seed_aggregation": "within-case means; independent bases are the resampling unit", "knockout_enabled": False}
    config["fingerprint"] = digest(config)
    path = root / "budget_config.json"
    if freeze:
        frozen_write(path, config)
    elif read_json(path) != config:
        raise ValueError("Replication intervention binding changed.")
    return config, {"settings": plan["settings"]}, pairs, mappings, baselines
