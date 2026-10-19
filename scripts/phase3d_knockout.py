"""Equal-weight KO50 planning; provenance candidates are not validated reuse."""

from pathlib import Path
from functools import lru_cache

try:
    from .phase3c_core import digest
    from .phase3c_knockout25 import WINDOWS, tasks
    from .phase3d_core import audit_pairs
except ImportError:
    from phase3c_core import digest
    from phase3c_knockout25 import WINDOWS, tasks
    from phase3d_core import audit_pairs


def intent(pairs, source_ko25_summary=None):
    cases = audit_pairs(pairs)
    result = {"schema": "phase3d_knockout50_intent_v1", "case_count": 50,
              "case_ids": [case["pair_id"] for case in cases],
              "query_groups": ["query_all", "options_all"],
              "key_groups": ["target_1", "target_2", "both_targets"],
              "whole_event2_enabled": True, "windows": [list(window) for window in WINDOWS],
              "primary_reporting": "All 50 independent bases equally; no pilot/new stratification.",
              "reuse_status": "candidate_only_requires_exact_live_input_runtime_mask_and_output_gates",
              "whole_event_control_limit": "No disjoint same-event 512-key matched background.",
              "source_ko25_summary_sha256": None}
    if source_ko25_summary is not None:
        if (source_ko25_summary.get("complete") is not True or
                source_ko25_summary.get("artifact_type") != "real" or
                source_ko25_summary.get("case_count") != 25 or
                source_ko25_summary.get("paired_rows") != 2925):
            raise ValueError("Reuse planning requires a complete real KO25 summary.")
        result["source_ko25_summary_sha256"] = digest(source_ko25_summary)
    result["paired_task_count"] = len(tasks(result, "count_only"))
    result["condition_forward_count"] = 2*result["paired_task_count"]
    result["validated_reused_tasks"] = 0
    result["fingerprint"] = digest(result)
    return result


def planned_tasks(config):
    if config["fingerprint"] != digest({key: value for key, value in config.items() if key != "fingerprint"}):
        raise ValueError("Frozen KO50 intent changed.")
    return tasks(config, config["fingerprint"])


def condition_specs(plan, pairs):
    config = intent(pairs)
    # Bind execution to this new plan without changing the historical KO manifests.
    result = []
    for task in tasks(config, plan["fingerprint"]):
        for condition in ("low_boundary", "temporal_boundary"):
            item = {**{k: v for k, v in task.items() if k != "task_id"},
                    "stage": "knockout", "view": "archived", "condition": condition}
            item["task_id"] = digest([plan["fingerprint"], item])
            result.append(item)
    return result


def semantic_key(spec, condition=None):
    """Historical IDs bind different plans; match interventions, never IDs."""
    return (spec["pair_id"], spec["query_group"], spec["key_group"], spec["control"],
            tuple(spec["window"]), condition if condition is not None else spec["condition"])


def freeze_reuse_source(source, pairs):
    """Keep an immutable, self-contained candidate ledger; summary-only rows fail closed."""
    from .phase3c_core import file_hash, read_json, read_jsonl
    from .phase3d_execution import checked_seal, code_hashes
    source = Path(source).resolve()
    plan = read_json(source / "plan.json")
    summary = read_json(source / "knockout/analysis/aggregate_summary.json")
    intent(pairs, summary)
    if (plan["fingerprint"] != digest({k: v for k, v in plan.items() if k != "fingerprint"}) or
            summary.get("fingerprint") != plan["fingerprint"]):
        raise ValueError("KO25 source plan/summary binding changed.")
    for name, sha in summary["output_sha256"].items():
        path = (source / "knockout/analysis" / name).resolve()
        if not path.is_relative_to(source / "knockout/analysis") or file_hash(path) != sha:
            raise ValueError("KO25 analysis hash/path changed.")
    source_tasks = {r["task_id"]: r for r in read_jsonl(source / "knockout/task_manifest.jsonl")}
    records = read_jsonl(source / "knockout/rows.jsonl")
    if len(records) != 2925 or len(source_tasks) != 2925 or len({r["task_id"] for r in records}) != 2925:
        raise ValueError("KO25 source task grid is incomplete/duplicated.")
    candidates, rejected = {}, []
    current_hashes = code_hashes()
    measurement = plan["measurement_code_sha256"]
    for name in ("phase3c_interventions.py", "phase3b_core.py", "activation_patching_core.py", "probe_attention_roi.py"):
        if measurement.get(name) != current_hashes.get(name):
            raise ValueError("Legacy KO mask/processor/decision implementation changed; reuse is unsafe.")
    if plan["code_sha256"].get("phase3c_knockout25.py") != current_hashes["phase3c_knockout25.py"]:
        raise ValueError("Historical KO route/outcome implementation changed.")
    source_baselines = {row["eval_id"]: checked_seal(row) for row in read_jsonl(source / "baseline/rows.jsonl")}
    runtimes = sorted((source / "runtime_observations").glob("knockout_*.json"))
    if not runtimes:
        raise ValueError("No recorded real KO25 execution runtime; cannot verify reuse.")
    from .phase3c_budget import runtime_compatibility
    runtime = read_json(runtimes[0])["runtime"]
    for path in runtimes:
        observation = read_json(path)
        if observation.get("plan_fingerprint") != plan["fingerprint"] or observation.get("stage") != "knockout":
            raise ValueError("Recorded KO25 runtime belongs to a different stage/plan.")
        runtime_compatibility(runtime, observation["runtime"])
    for row in records:
        checked_seal(row)
        if (row.get("passed") is not True or row["plan_fingerprint"] != plan["fingerprint"] or
                row["spec"] != source_tasks.get(row["task_id"])):
            raise ValueError("KO25 source checkpoint seal/spec changed.")
        if row.get("reused") or set(row.get("conditions", {})) != {"low_boundary", "temporal_boundary"}:
            rejected.append({"source_record_sha256": row["record_sha256"], "reason": "inherited_summary_without_original_condition_mask_chain"})
            continue
        for condition, result in row["conditions"].items():
            annotation = pairs[row["spec"]["pair_id"]][condition]
            baseline = source_baselines[annotation["eval_id"]]
            if (baseline.get("passed") is not True or result["baseline_record_sha256"] != baseline["record_sha256"] or
                    result["input_tensor_sha256"] != baseline["input_tensor_sha256"]):
                raise ValueError("KO25 raw condition is not bound to its sealed baseline.")
            key = digest(semantic_key(row["spec"], condition))
            if key in candidates:
                raise ValueError("Duplicated legacy semantic intervention key.")
            candidates[key] = {"source_record": row, "condition": condition, "result": result, "baseline": baseline,
                               "source_root": str(source)}
    return {"source_root": str(source), "source_runtime": runtime,
            "source_files": {name: file_hash(source / name) for name in
                ("plan.json", "baseline/rows.jsonl", "knockout/rows.jsonl", "knockout/task_manifest.jsonl", "knockout/analysis/aggregate_summary.json")},
            "candidates": candidates, "rejected": rejected,
            "candidate_condition_count": len(candidates), "validated_reused_tasks": 0}


@lru_cache(maxsize=2)
def reuse_ledger(path, expected_digest):
    from .phase3c_core import read_json
    value = read_json(path)
    if digest(value) != expected_digest:
        raise ValueError("Frozen KO reuse ledger changed.")
    return value


def reuse_condition(root, context, spec, baseline, audit, live_runtime):
    from .phase3c_core import read_json
    from .phase3d_execution import checked_seal
    from .phase3c_budget import runtime_compatibility
    path = Path(root) / "ko_reuse_source.json"
    if not path.exists():
        return None
    source = reuse_ledger(str(path), context[0]["ko_reuse_source_sha256"])
    candidate = source["candidates"].get(digest(semantic_key(spec)))
    if candidate is None:
        return None
    runtime_compatibility(source["source_runtime"], live_runtime)
    old_record = checked_seal(candidate["source_record"])
    old_baseline = checked_seal(candidate["baseline"])
    old = candidate["result"]
    mapping = context[2][spec["pair_id"]]
    label = "low" if spec["condition"] == "low_boundary" else "temporal"
    before = old_record["paired_outcome"]
    if (semantic_key(old_record["spec"], spec["condition"]) != semantic_key(spec) or
            old_record["conditions"][spec["condition"]] != old or
            old["baseline_record_sha256"] != old_baseline["record_sha256"] or
            old_baseline["decision"] != baseline["decision"] or
            old["input_tensor_sha256"] != audit["processor_record"]["input_tensor_sha256"] or
            any(before[f"{label}_base_{name}"] != baseline["decision"][name]
                for name in ("margin", "correct_logit", "incorrect_logit", "prediction")) or
            any(before[f"{label}_ko_{name}"] != old["decision"][name]
                for name in ("margin", "correct_logit", "incorrect_logit", "prediction"))):
        raise ValueError("Legacy KO semantic input/baseline/output matching failed.")
    result = {**old, "baseline_record_sha256": baseline["record_sha256"],
              "input_fingerprint": audit["input_fingerprint"], "audit_record_sha256": audit["record_sha256"],
              "reuse": {"source_record_sha256": old_record["record_sha256"],
                        "source_root": source["source_root"],
                        "semantic_key": [spec["pair_id"], spec["query_group"], spec["key_group"], spec["control"],
                                         list(spec["window"]), spec["condition"]]}}
    verify_condition(result, spec, mapping, baseline)
    return result


def execute_condition(root, context, engine, spec, live_cache):
    from .phase3d_execution import audited_live, capture_reference
    from .phase3c_knockout25 import condition_result
    from .run_phase3c_preflight import logits_parity
    key = (spec["pair_id"], spec["condition"])
    if key not in live_cache:
        if live_cache and next(iter(live_cache))[0] != key[0]:
            live_cache.clear()
            engine.torch.cuda.empty_cache()
        live, audit = audited_live(root, context, engine, spec)
        baseline, data = capture_reference(root, context, *key)
        parity = logits_parity(data["logits"], engine.forward(live), exact=True)
        if not parity["passed"] or audit["input_fingerprint"] != baseline["input_fingerprint"]:
            raise RuntimeError("KO live baseline replay failed.")
        live_cache[key] = live, audit, baseline
    live, audit, baseline = live_cache[key]
    reused = reuse_condition(root, context, spec, baseline, audit, engine.runtime)
    if reused is not None:
        return reused, None
    result = condition_result(engine, live, context[2][key[0]], spec, None, True)
    result.update(baseline_record_sha256=baseline["record_sha256"],
                  input_fingerprint=audit["input_fingerprint"], audit_record_sha256=audit["record_sha256"])
    return result, None


def verify_condition(row, spec, mapping, baseline):
    from .phase3c_knockout25 import positions
    queries, keys, budget = positions(mapping, spec["condition"], spec)
    audit = row["mask_audit"]
    if (row["baseline_record_sha256"] != baseline["record_sha256"] or
            row["input_fingerprint"] != baseline["input_fingerprint"] or
            row["audit_record_sha256"] != baseline["audit_record_sha256"] or
            row["query_positions"] != queries or row["key_positions"] != keys or row["edge_budget"] != budget or
            audit.get("enabled") is not True or audit.get("all_heads") != 32 or
            audit.get("pre_softmax_mask") != "negative_infinity" or audit.get("renormalized") is not True or
            set(audit["layers"]) != {str(layer) for layer in spec["window"]} or
            any(value["budget"] != budget or value["max_blocked_probability"] != 0 or
                not 0 <= value["max_row_sum_error"] <= .005 for value in audit["layers"].values())):
        raise ValueError("Saved KO mask/input/causal-edge budget does not match the planned intervention.")
    import math
    decision = row["decision"]
    if (any(type(decision.get(k)) not in (int, float) or not math.isfinite(decision[k])
            for k in ("margin", "correct_logit", "incorrect_logit")) or
            not math.isclose(decision["margin"], decision["correct_logit"]-decision["incorrect_logit"], abs_tol=1e-8) or
            decision["correct_option"] != baseline["decision"]["correct_option"]):
        raise ValueError("KO decision/logit evidence is inconsistent.")


def analyze_complete(root, context):
    from .phase3d_execution import capture_reference, saved_checkpoint
    from .phase3c_knockout25 import paired_row
    from .phase3c_core import CONDITIONS, atomic_write, file_hash, frozen_write
    from .analyze_phase3c import aggregate, control_contrasts
    plan, pairs, mappings = context
    specs = condition_specs(plan, pairs)
    grouped, baselines, reused_conditions = {}, {}, 0
    for count, spec in enumerate(specs, 1):
        row = saved_checkpoint(root, plan, spec)
        if row is None:
            raise ValueError("KO50 requires all 11,700 condition checkpoints, not a partial cohort.")
        key = (spec["pair_id"], spec["condition"])
        if key not in baselines:
            baseline, _ = capture_reference(root, context, *key)
            baselines[key] = baseline
        verify_condition(row, spec, mappings[spec["pair_id"]], baselines[key])
        if row.get("reuse"):
            from .phase3c_core import read_json
            from .phase3d_execution import saved_checkpoint as checked_checkpoint, feature_tasks
            audit_spec = next(s for s in feature_tasks(plan, "audit", [spec["pair_id"]])
                if s["view"] == "archived" and s["condition"] == spec["condition"])
            audit = checked_checkpoint(root, plan, audit_spec)
            rebuilt = reuse_condition(root, context, spec, baselines[key], audit, read_json(Path(root) / "runtime.json")["runtime"])
            if rebuilt is None or row["reuse"] != rebuilt["reuse"] or row["decision"] != rebuilt["decision"]:
                raise ValueError("Historical reuse provenance changed.")
            reused_conditions += 1
        pair_key = tuple([spec["pair_id"], spec["query_group"], spec["key_group"], spec["control"], tuple(spec["window"])])
        grouped.setdefault(pair_key, {})[spec["condition"]] = row
        if count % 500 == 0 or count == len(specs):
            print(f"[Phase 3D KO verification] {count}/{len(specs)} condition checkpoints checked", flush=True)
    lookup = {pairs[pair][condition]["eval_id"]: row for (pair, condition), row in baselines.items()}
    table = []
    for (pair_id, query, keys, control, window), results in grouped.items():
        if set(results) != set(CONDITIONS):
            raise ValueError("Unpaired KO conditions.")
        table.append(paired_row({"pair_id": pair_id, "query_group": query, "key_group": keys,
                                "control": control, "window": list(window)}, results, lookup, pairs))
    if len(table) != 5850:
        raise ValueError("Incomplete paired KO50 grid.")
    output = Path(root) / "knockout/analysis"
    print("[Phase 3D KO50] computing equal-weight, case-bootstrap summaries", flush=True)
    metrics = ("delta_M_temporal", "delta_M_low", "compression")
    cells = aggregate(table, ("query_group", "key_group", "control", "window_start", "window_end"), metrics)
    contrasts = control_contrasts([row for row in table if row["key_group"] != "whole_event2"])
    values = {"case_decomposition.json": table, "aggregate_cells.json": cells, "matched_control_contrasts.json": contrasts}
    for name, value in values.items():
        frozen_write(output / name, value)
    summary = {"schema": "phase3d_knockout50_results_v1", "artifact_type": "real", "complete": True,
               "plan_fingerprint": plan["fingerprint"], "case_count": 50, "paired_rows": len(table),
               "condition_rows": len(specs), "missing_count": 0,
               "verified_reused_condition_rows": reused_conditions,
               "reporting": "All 50 frozen bases equally; no pilot/new strata.",
               "control_limit": "Whole-event route has no equal-budget same-event background; do not compare its magnitude as a matched ROI contrast.",
               "interpretation": "Functional route contribution, not proof that routing is the unique mechanism.",
               "output_sha256": {name: file_hash(output / name) for name in values}}
    atomic_write(output / "aggregate_summary.json", summary)
    return summary
