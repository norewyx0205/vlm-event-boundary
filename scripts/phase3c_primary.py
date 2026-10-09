"""Fixed primary grids and weight-free outcome validation for Phase 3C."""

import math

try:
    from .phase3c_core import CONDITIONS, digest, edge_budget
except ImportError:
    from phase3c_core import CONDITIONS, digest, edge_budget


PRIMARY_STAGES = ("patch", "routing", "knockout")
SUPPORTS = ("video_t1_e2", "video_t2_e2", "both_targets_event2", "whole_event2")


def routes(mapping, condition):
    controls = mapping["support_audit"]["knockout_controls"][condition]
    result = []
    for query in ("options_all", "query_all"):
        for key in ("target_1", "target_2", "both_targets"):
            if not controls[query][key]["background"]["eligible"]:
                raise ValueError("A required frozen background control is ineligible; do not shrink the grid.")
            result.extend({"query_group": query, "key_group": key, "control": control}
                          for control in ("target", "background"))
    return result


def route_positions(mapping, condition, spec):
    route = mapping["support_audit"]["knockout_controls"][condition][spec["query_group"]][spec["key_group"]]["background"]
    if not route["eligible"] or route["target_budget"] != route["control_budget"]:
        raise ValueError("Frozen route has no exact background edge-budget control.")
    keys = route["target_key_positions"] if spec["control"] == "target" else route["control_key_positions"]
    return route["query_positions"], keys, route["target_budget"]


def primary_tasks(config, mappings, stage):
    if stage not in PRIMARY_STAGES:
        raise ValueError("Unknown primary Phase 3C stage.")
    settings, tasks = config["settings"], []
    for pair_id in config["case_ids"]:
        for condition in CONDITIONS:
            common = {"pair_id": pair_id, "condition": condition}
            if stage == "patch":
                sites = {(support, layer, "block_output") for support in SUPPORTS
                         for layer in settings["patch_layers"]}
                sites.update(("whole_event2", layer, location)
                             for layer in settings["deepstack_timing_layers"]
                             for location in ("block_output", "post_deepstack"))
                for support, layer, location in sorted(sites):
                    tasks.append({**common, "kind": "visual_patch", "support": support,
                                  "layer": layer, "location": location})
            elif stage == "routing":
                tasks.append({**common, "kind": "routing_baseline"})
            else:
                for route in routes(mappings[pair_id], condition):
                    for window in settings["knockout_windows"]:
                        tasks.append({**common, "kind": "attention_knockout", **route, "window": window})
    for task in tasks:
        task["task_id"] = digest(task)
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("Duplicate primary intervention task.")
    return tasks


def patch_outcomes(before, after, donor):
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in (before, after, donor)):
        raise ValueError("Patch outcomes require finite baseline, intervened and donor margins.")
    denominator = donor - before
    return {"margin_delta": after - before,
            "source_aligned_margin_delta": (after - before) * (1 if denominator > 0 else -1) if denominator != 0 else None,
            "recovery_denominator": denominator,
            "recovery": (after - before) / denominator if denominator != 0 else None,
            "strict_sign_crossing": before * after < 0,
            "strict_incorrect_to_correct": before < 0 < after,
            "zero_margin_tie": after == 0}


def finite_decision(value):
    values = [value.get(key) for key in ("margin", "correct_logit", "incorrect_logit")]
    return (all(type(item) in (int, float) and math.isfinite(item) for item in values) and
            math.isclose(values[1] - values[2], values[0], abs_tol=1e-6))


def primary_failures(row):
    try:
        from .phase3c_execution import technical_failures
    except ImportError:
        from phase3c_execution import technical_failures
    spec = row.get("spec", {})
    if (row.get("passed") is not True or
            row.get("is_primary_effect_estimate") is not (spec.get("kind") != "routing_baseline") or
            row.get("task_id") != spec.get("task_id") or
            row.get("task_id") != digest({key: value for key, value in spec.items() if key != "task_id"})):
        return ["invalid_or_failed_primary_task"]
    failures = []
    if spec.get("kind") in ("visual_patch", "attention_knockout"):
        converted = {**spec, "kind": "transplant_smoke" if spec["kind"] == "visual_patch" else "knockout_smoke"}
        converted.pop("task_id")
        converted["task_id"] = digest(converted)
        failures.extend(technical_failures({**row, "spec": converted, "task_id": converted["task_id"],
                                           "is_primary_effect_estimate": False}))
        if failures:
            return failures
        if row.get("margin_delta") != row["decision"]["margin"] - row["baseline_decision"]["margin"]:
            failures.append("invalid_margin_delta")
        if spec["kind"] == "visual_patch":
            donor = row.get("donor_baseline_decision", {})
            if not finite_decision(donor):
                failures.append("invalid_donor_baseline")
            else:
                expected = patch_outcomes(row["baseline_decision"]["margin"], row["decision"]["margin"], donor["margin"])
                if any(row.get(key) != value for key, value in expected.items()):
                    failures.append("invalid_patch_outcome_decomposition")
    elif spec.get("kind") == "routing_baseline":
        if (not finite_decision(row.get("decision", {})) or row.get("decision") != row.get("baseline_decision") or
                row.get("noop_parity", {}).get("exact_match") is not True or
                row.get("noop_parity", {}).get("max_abs_diff") != 0 or
                len(row.get("routes", [])) != 12):
            failures.append("invalid_intact_routing_diagnostic")
        route_ids = set()
        for route in row.get("routes", []):
            key = (route.get("query_group"), route.get("key_group"), route.get("control"))
            if key in route_ids:
                failures.append("duplicate_intact_route")
            route_ids.add(key)
            try:
                budget = edge_budget(route["query_positions"], route["key_positions"], row["prompt_token_count"])
                audit = route["mask_audit"]
                if (audit["enabled"] is not False or audit["all_heads"] != 32 or
                        set(audit["layers"]) != {str(layer) for layer in range(36)} or budget != route["edge_budget"]):
                    failures.append("invalid_intact_route_budget_or_layers")
                for item in audit["layers"].values():
                    mass = item["mean_selected_edge_mass_per_query_head"]
                    if (item["budget"] != budget or not math.isfinite(mass) or not 0 <= mass <= 1.005 or
                            not math.isfinite(item["max_row_sum_error"]) or item["max_row_sum_error"] > 0.005):
                        failures.append("invalid_intact_attention_mass")
            except (KeyError, ValueError, TypeError):
                failures.append("missing_intact_route_evidence")
    else:
        failures.append("unknown_primary_task_kind")
    return failures


def validate_frozen_result(row, pairs, mappings, baselines):
    """Recheck results against frozen positions and GPU baselines, not claimed passes."""
    failures = primary_failures(row)
    if failures:
        return failures
    spec = row["spec"]
    pair, condition = pairs[spec["pair_id"]], spec["condition"]
    annotation = pair[condition]
    if row["baseline_decision"] != baselines[annotation["eval_id"]]["decision"]:
        failures.append("baseline_decision_changed")
    mapping = mappings[spec["pair_id"]]
    if row.get("prompt_token_count") != len(mapping["processor_records"][condition]["input_ids"]):
        failures.append("prompt_length_changed")
    if row.get("input_tensor_sha256") != baselines[annotation["eval_id"]]["input_tensor_sha256"]:
        failures.append("actual_input_hashes_changed")
    if spec["kind"] == "visual_patch":
        donor = CONDITIONS[1] if condition == CONDITIONS[0] else CONDITIONS[0]
        side = "low_positions" if condition == CONDITIONS[0] else "temporal_positions"
        donor_side = "low_positions" if donor == CONDITIONS[0] else "temporal_positions"
        support = mapping["support_audit"]["supports"][spec["support"]]
        if (row.get("support_audit") != support or row.get("recipient_positions") != support[side] or
                row.get("donor_positions") != support[donor_side] or row.get("donor_condition") != donor or
                row.get("recipient_condition") != condition or
                row["donor_baseline_decision"] != baselines[pair[donor]["eval_id"]]["decision"]):
            failures.append("observed_donor_correspondence_changed")
        if spec["support"] not in SUPPORTS[:2] and any(
                item["recipient_coverage"] != 1 for item in support["directions"].values()):
            failures.append("expanded_support_incomplete")
    else:
        actual = row["routes"] if spec["kind"] == "routing_baseline" else [row]
        expected = routes(mapping, condition) if spec["kind"] == "routing_baseline" else [spec]
        lookup = {(item["query_group"], item["key_group"], item["control"]): item for item in expected}
        for item in actual:
            route_spec = item if spec["kind"] == "routing_baseline" else spec
            key = (route_spec["query_group"], route_spec["key_group"], route_spec["control"])
            if key not in lookup:
                failures.append("unexpected_route")
                continue
            queries, keys, budget = route_positions(mapping, condition, lookup[key])
            if (item.get("query_positions") != queries or item.get("key_positions") != keys or
                    item.get("edge_budget") != budget):
                failures.append("route_positions_or_budget_changed")
    return failures
