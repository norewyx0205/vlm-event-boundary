"""Weight-free plan validation, task grids and completion gates for Phase 3C."""

import math
import re
from pathlib import Path

try:
    from .phase3c_core import (
        CONDITIONS, SCHEMA, atomic_write, baseline_failures, digest, edge_budget, file_hash, read_json, read_jsonl,
    )
    from .prepare_phase3c import freeze_selection
except ImportError:
    from phase3c_core import (
        CONDITIONS, SCHEMA, atomic_write, baseline_failures, digest, edge_budget, file_hash, read_json, read_jsonl,
    )
    from prepare_phase3c import freeze_selection


EXECUTION_FILES = (
    "phase3c_execution.py", "phase3c_interventions.py", "run_phase3c_preflight.py",
    "phase3c_primary.py", "run_phase3c.py",
    "phase3c_core.py", "prepare_phase3c.py", "audit_phase3c_mappings.py",
    "run_phase3b_patching.py", "phase3b_baseline.py", "phase3b_core.py", "phase3b_paths.py",
    "run_eval.py", "probe_attention_roi.py", "activation_patching_core.py", "run_phase3b_vm.py", "common.py",
)


def execution_hashes():
    return {name: file_hash(Path(__file__).parent / name) for name in EXECUTION_FILES}


def load_selection(plan_dir):
    root = Path(plan_dir).resolve()
    for name in ("case_manifest.jsonl", "selected_mappings.jsonl", "frozen_config.json", "case_selection_summary.json"):
        if not (root / "selection" / name).is_file():
            raise ValueError("Frozen Phase 3C cohort is missing; complete CPU audit and freeze first.")
    freeze_selection(root)
    config = read_json(root / "selection/frozen_config.json")
    if config["selection_fingerprint"] != digest({key: value for key, value in config.items()
                                                 if key != "selection_fingerprint"}):
        raise ValueError("Invalid frozen-selection fingerprint.")
    rows = read_jsonl(root / "selection/case_manifest.jsonl")
    maps = read_jsonl(root / "selection/selected_mappings.jsonl")
    if config["case_manifest_sha256"] != digest(rows) or config["mapping_manifest_sha256"] != digest(maps):
        raise ValueError("Frozen case/mapping payload changed.")
    mapping = {item["pair_id"].replace("phase3b_", "phase3c_", 1): item for item in maps}
    pairs, ids = {}, set()
    for row in rows:
        pair_id, condition = row["phase3c_pair_id"], row["condition"]
        if not re.fullmatch(r"phase3c_base_[0-9]+_(original|swapped)", pair_id):
            raise ValueError("Unsafe or unexpected Phase 3C pair ID.")
        pair = pairs.setdefault(pair_id, {})
        if row["eval_id"] in ids or condition in pair or condition not in CONDITIONS:
            raise ValueError("Duplicate/invalid frozen evaluation row.")
        pair[condition] = row
        ids.add(row["eval_id"])
    if (len(pairs) != 12 or set(pairs) != set(mapping) or set(pairs) != set(config["case_ids"]) or
            any(set(pair) != set(CONDITIONS) for pair in pairs.values())):
        raise ValueError("Frozen 12-case pair/mapping coverage is invalid.")
    if len(config["technical_preflight_case_ids"]) != 2:
        raise ValueError("Technical preflight requires exactly two frozen rescue cases.")
    chosen = [pairs[key][CONDITIONS[0]] for key in config["technical_preflight_case_ids"]]
    if ({row["first_object_id"] for row in chosen} != {1, 2} or
            any(row["phase3c_analysis_stratum"] != "rescue" for row in chosen)):
        raise ValueError("Technical preflight cases do not cover both rescue mover orders.")
    return config, pairs, mapping


def technical_tasks(config, mappings):
    """A fixed technical grid, not a scientific window/effect selection procedure."""
    tasks = []
    sites = [(layer, "block_output") for layer in sorted(set(
        config["settings"]["patch_layers"] + config["settings"]["deepstack_timing_layers"]))]
    sites += [(layer, "post_deepstack") for layer in config["settings"]["deepstack_timing_layers"]]
    for pair_id in config["technical_preflight_case_ids"]:
        mapping = mappings[pair_id]
        for condition in CONDITIONS:
            for layer, location in sites:
                tasks.append({"pair_id": pair_id, "condition": condition, "kind": "identity",
                    "layer": layer, "location": location, "support": "whole_event2"})
            for support in ("video_t1_e2", "video_t2_e2", "both_targets_event2", "whole_event2"):
                tasks.append({"pair_id": pair_id, "condition": condition, "kind": "transplant_smoke",
                    "layer": 12, "location": "block_output", "support": support})
            for layer, location in ((0, "block_output"), (1, "block_output"), (2, "block_output"),
                                    (0, "post_deepstack"), (1, "post_deepstack"), (2, "post_deepstack")):
                tasks.append({"pair_id": pair_id, "condition": condition, "kind": "transplant_smoke",
                    "layer": layer, "location": location, "support": "whole_event2"})
            tasks.append({"pair_id": pair_id, "condition": condition, "kind": "disabled_knockout",
                          "window": list(range(36)), "query_group": "options_all",
                          "key_group": "both_targets", "control": "target"})
            for query in config["settings"]["knockout_query_groups"]:
                for key in config["settings"]["knockout_key_groups"]:
                    for control in ("target", "background"):
                        tasks.append({"pair_id": pair_id, "condition": condition, "kind": "knockout_smoke",
                            "window": config["settings"]["knockout_windows"][0],
                            "query_group": query, "key_group": key, "control": control})
            for window in (config["settings"]["knockout_windows"][3], config["settings"]["knockout_windows"][-1]):
                for control in ("target", "background"):
                    tasks.append({"pair_id": pair_id, "condition": condition, "kind": "knockout_smoke",
                        "window": window, "query_group": "options_all", "key_group": "both_targets", "control": control})
            if condition not in mapping["support_audit"]["knockout_controls"]:
                raise ValueError("Missing frozen knockout controls.")
    for task in tasks:
        task["task_id"] = digest(task)
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("Duplicate technical intervention task.")
    return tasks


def checkpoint_rows(path, fingerprint, expected_ids):
    rows = read_jsonl(path) if Path(path).is_file() else []
    lookup = {}
    for row in rows:
        if (row["task_id"] in lookup or row["task_id"] not in expected_ids or
                row.get("execution_fingerprint") != fingerprint):
            raise ValueError("Checkpoint contains duplicate, foreign or incompatible tasks.")
        lookup[row["task_id"]] = row
    return lookup


def primary_checkpoint_rows(output, fingerprint, expected_ids):
    lookup = {}
    for path in sorted((Path(output) / "task_checkpoints").glob("*.json")):
        row = read_json(path)
        key = row["task_id"]
        if (path.stem != key or key not in expected_ids or key in lookup or
                row.get("execution_fingerprint") != fingerprint):
            raise ValueError("Primary checkpoint contains a foreign, duplicate or incompatible task.")
        lookup[key] = row
    return lookup


def baseline_reasons(pairs, rows):
    reasons = []
    for pair_id, pair in pairs.items():
        margins = []
        for condition in CONDITIONS:
            row = pair[condition]
            saved = rows.get(row["eval_id"])
            if saved is None:
                reasons.append(f"{row['eval_id']}:missing")
                continue
            failures = baseline_failures(row, saved)
            if saved.get("hook_audit", {}).get("block_output_sites") != 36 or saved.get(
                    "hook_audit", {}).get("post_deepstack_layers") != [0, 1, 2]:
                failures.append("missing_all_layer_or_deepstack_hook_audit")
            if saved.get("capture_noop_parity", {}).get("exact_match") is not True:
                failures.append("capture_hooks_changed_logits")
            if not re.fullmatch(r"[0-9a-f]{64}", saved.get("input_tensor_sha256", {}).get("input_ids", {}).get("sha256", "")):
                failures.append("missing_actual_input_tensor_hashes")
            reasons.extend(f"{row['eval_id']}:{failure}" for failure in failures)
            margins.append(saved.get("decision", {}).get("margin"))
        if len(margins) == 2:
            expected = (False, True) if pair[CONDITIONS[0]]["phase3c_analysis_stratum"] == "rescue" else (True, True)
            if any(type(value) not in (int, float) for value in margins) or tuple(value > 0 for value in margins) != expected:
                reasons.append(f"{pair_id}:behavior_changed")
    return reasons


def technical_failures(row):
    failures = []
    spec = row.get("spec", {})
    if (row.get("passed") is not True or row.get("is_primary_effect_estimate") is not False or
            row.get("task_id") != spec.get("task_id") or
            row.get("task_id") != digest({key: value for key, value in spec.items() if key != "task_id"})):
        return ["invalid_or_failed_technical_task"]
    for name in ("decision", "baseline_decision"):
        decision = row.get(name, {})
        values = [decision.get(key) for key in ("margin", "correct_logit", "incorrect_logit")]
        if (any(type(value) not in (int, float) or not math.isfinite(value) for value in values) or
                not math.isclose(values[1] - values[2], values[0], abs_tol=1e-6)):
            failures.append(f"{name}:invalid_margin")
    if spec.get("kind") in ("identity", "disabled_knockout"):
        if (row.get("noop_parity", {}).get("exact_match") is not True or
                row.get("noop_parity", {}).get("max_abs_diff") != 0 or
                row.get("decision") != row.get("baseline_decision")):
            failures.append("failed_exact_noop")
    if spec.get("kind") in ("identity", "transplant_smoke"):
        hook = row.get("hook_audit", {})
        if (hook.get("block_output_sites") != 36 or hook.get("post_deepstack_layers") != [0, 1, 2] or
                hook.get("patch_applied_count") != 1 or
                row.get("donor_capture_location") != spec.get("location") or
                row.get("recipient_patch_location") != spec.get("location")):
            failures.append("failed_residual_hook_audit")
    elif spec.get("kind") in ("disabled_knockout", "knockout_smoke"):
        audit = row.get("mask_audit", {})
        enabled = spec["kind"] == "knockout_smoke"
        try:
            if edge_budget(row["query_positions"], row["key_positions"], row["prompt_token_count"]) != row["edge_budget"]:
                failures.append("recorded_edge_budget_disagreement")
        except (KeyError, ValueError, TypeError):
            failures.append("missing_or_invalid_actual_edge_positions")
        if (audit.get("enabled") is not enabled or audit.get("all_heads") != 32 or
                audit.get("orientation") != "text_queries_to_visual_keys" or
                audit.get("renormalized") is not True or
                set(audit.get("layers", {})) != {str(layer) for layer in spec["window"]}):
            failures.append("invalid_mask_audit")
        for item in audit.get("layers", {}).values():
            if (item.get("budget") != row.get("edge_budget") or
                    type(item.get("max_row_sum_error")) not in (int, float) or
                    not math.isfinite(item["max_row_sum_error"]) or item["max_row_sum_error"] > 0.005 or
                    (enabled and item.get("max_blocked_probability") != 0)):
                failures.append("mask_probability_or_budget_failed")
    else:
        failures.append("unknown_technical_task_kind")
    return failures


def save_stage_summary(output, execution, pairs, rows, expected, stage, mappings=None, baselines=None):
    try:
        from .phase3c_primary import PRIMARY_STAGES, validate_frozen_result
    except ImportError:
        from phase3c_primary import PRIMARY_STAGES, validate_frozen_result
    output = Path(output)
    missing = sorted(set(expected) - set(rows))
    if stage == "baseline":
        failures = baseline_reasons(pairs, rows)
    else:
        failures = []
        for key, row in rows.items():
            reasons = (validate_frozen_result(row, pairs, mappings, baselines) if stage in PRIMARY_STAGES
                       else technical_failures(row))
            if reasons:
                failures.append({"task_id": key, "reasons": reasons})
    summary = {"schema": SCHEMA, "stage": stage, "artifact_type": "real",
        "execution_fingerprint": execution["execution_fingerprint"],
        "passed": not missing and not failures, "completed": len(rows), "expected": len(expected),
        "missing_task_ids": missing, "failures": failures,
        "config_sha256": file_hash(output.parent / "execution_config.json"),
        "rows_sha256": file_hash(output / "rows.jsonl") if (output / "rows.jsonl").exists() else None,
        "task_manifest_sha256": file_hash(output / "task_manifest.jsonl") if (output / "task_manifest.jsonl").exists() else None,
        "primary_intervention_grid_complete": False,
        "primary_stage_complete": stage in PRIMARY_STAGES and not missing and not failures,
        "next_stage": {"baseline": "preflight", "preflight": "patch_or_routing",
                       "patch": "routing_and_knockout", "routing": "knockout", "knockout": "cpu_analysis"}[stage]}
    atomic_write(output / "summary.json", summary)
    return summary


def require_stage(root, execution, pairs, stage, expected, mappings=None, baselines=None, expected_tasks=None):
    output = Path(root) / stage
    config_path = Path(root) / "execution_config.json"
    summary = read_json(output / "summary.json")
    if (not summary.get("passed") or summary.get("stage") != stage or
            summary.get("execution_fingerprint") != execution["execution_fingerprint"] or
            summary.get("config_sha256") != file_hash(config_path) or
            summary.get("rows_sha256") != file_hash(output / "rows.jsonl") or
            summary.get("task_manifest_sha256") != file_hash(output / "task_manifest.jsonl")):
        raise ValueError(f"{stage} completion gate is missing, stale or failed.")
    tasks = read_jsonl(output / "task_manifest.jsonl")
    if len(tasks) != len(expected) or {task["task_id"] for task in tasks} != set(expected):
        raise ValueError(f"{stage} task manifest has invalid coverage.")
    if expected_tasks is not None and tasks != expected_tasks:
        raise ValueError(f"{stage} task manifest differs from the fixed protocol grid.")
    rows = checkpoint_rows(output / "rows.jsonl", execution["execution_fingerprint"], set(expected))
    if stage in ("patch", "routing", "knockout") and rows != primary_checkpoint_rows(output, execution["execution_fingerprint"], set(expected)):
        raise ValueError("Consolidated primary rows differ from per-task checkpoints.")
    if set(rows) != set(expected) or summary.get("completed") != len(expected) or summary.get("expected") != len(expected):
        raise ValueError(f"{stage} gate has incomplete task coverage.")
    if stage == "baseline":
        if baseline_reasons(pairs, rows):
            raise ValueError("Baseline rows failed independent behavioral/parity verification.")
        for row in rows.values():
            capture = read_json(row["capture_index_path"])
            if (row["capture_index_sha256"] != file_hash(row["capture_index_path"]) or
                    capture["execution_fingerprint"] != execution["execution_fingerprint"] or
                    capture["vectors_sha256"] != file_hash(capture["vectors_path"]) or
                    capture["eval_id"] != row["eval_id"] or capture["site_count"] != 39 or
                    capture["input_tensor_sha256"] != row["input_tensor_sha256"]):
                raise ValueError("Baseline capture bytes/fingerprint changed.")
    else:
        try:
            from .phase3c_primary import PRIMARY_STAGES, validate_frozen_result
        except ImportError:
            from phase3c_primary import PRIMARY_STAGES, validate_frozen_result
        failures = [validate_frozen_result(row, pairs, mappings, baselines) if stage in PRIMARY_STAGES
                    else technical_failures(row) for row in rows.values()]
        if any(failures) or any(rows[task["task_id"]].get("spec") != task for task in tasks):
            raise ValueError(f"{stage} results failed independent intervention verification.")
    return rows
