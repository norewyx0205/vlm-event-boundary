"""Verify and summarize the complete Phase 3C pilot without loading model weights."""

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

try:
    from .phase3c_core import CONDITIONS, SCHEMA, atomic_write, boundary_outcomes, digest, file_hash, frozen_write, read_json
    from .phase3c_execution import execution_hashes, load_selection, require_stage, technical_tasks
    from .phase3c_primary import PRIMARY_STAGES, primary_tasks
except ImportError:
    from phase3c_core import CONDITIONS, SCHEMA, atomic_write, boundary_outcomes, digest, file_hash, frozen_write, read_json
    from phase3c_execution import execution_hashes, load_selection, require_stage, technical_tasks
    from phase3c_primary import PRIMARY_STAGES, primary_tasks


def case_metadata(pair):
    row = pair[CONDITIONS[0]]
    return {"base_sample_id": row["base_sample_id"], "analysis_stratum": row["phase3c_analysis_stratum"],
            "first_object_id": row["first_object_id"], "prompt_variant": row["prompt_variant"],
            "correct_option": row["correct_option"]}


def secondary_outcomes(before, after):
    return {"categorical_flip": before["prediction"] != after["prediction"],
            "non_ab_first_token": after["prediction"] not in ("A", "B"),
            "strict_sign_crossing": before["margin"] * after["margin"] < 0,
            "strict_incorrect_to_correct": before["margin"] < 0 < after["margin"],
            "zero_margin_tie": after["margin"] == 0}


def patch_table(rows, pairs):
    result = []
    for row in rows.values():
        spec, support = row["spec"], row["support_audit"]
        direction = "temporal_to_low" if spec["condition"] == CONDITIONS[0] else "low_to_temporal"
        metadata = case_metadata(pairs[spec["pair_id"]])
        target = 1 if spec["support"] == "video_t1_e2" else 2 if spec["support"] == "video_t2_e2" else None
        role = ("first_mover_during_event_2" if target == metadata["first_object_id"] else "second_mover_own_event") if target else None
        item = {**metadata, "pair_id": spec["pair_id"], "support": spec["support"], "mover_role": role,
            "layer": spec["layer"], "location": spec["location"], "direction": direction,
            "reference_only": support["scope"] == "single_roi_reference",
            "recipient_coverage": support["directions"][direction]["recipient_coverage"],
            "donor_coverage": support["directions"]["low_to_temporal" if direction == "temporal_to_low" else "temporal_to_low"]["recipient_coverage"],
            "replaced_token_count": len(row["recipient_positions"]),
            "actual_alignment": support["method"],
            **{key: row[key] for key in ("margin_delta", "source_aligned_margin_delta", "recovery", "recovery_denominator")},
            **secondary_outcomes(row["baseline_decision"], row["decision"])}
        for label, decision in (("baseline", row["baseline_decision"]), ("patched", row["decision"]),
                                ("donor", row["donor_baseline_decision"])):
            item.update({f"{label}_{name}": decision[name] for name in ("margin", "correct_logit", "incorrect_logit", "prediction")})
        result.append(item)
    return result


def knockout_table(rows, routing_rows, pairs):
    grouped, intact = {}, {}
    for row in routing_rows.values():
        spec = row["spec"]
        intact[(spec["pair_id"], spec["condition"])] = row["routes"]
    for row in rows.values():
        spec = row["spec"]
        key = (spec["pair_id"], spec["query_group"], spec["key_group"], spec["control"], tuple(spec["window"]))
        group = grouped.setdefault(key, {})
        if spec["condition"] in group:
            raise ValueError("Duplicate condition in a matched knockout result.")
        group[spec["condition"]] = row
    result = []
    for (pair_id, query, keys, control, window), group in grouped.items():
        if set(group) != set(CONDITIONS):
            raise ValueError("Knockout analysis requires complete matched low/temporal conditions.")
        low, temporal = (group[condition] for condition in CONDITIONS)
        item = {**case_metadata(pairs[pair_id]), "pair_id": pair_id,
            "query_group": query, "key_group": keys, "control": control,
            "window_start": window[0], "window_end": window[-1],
            **boundary_outcomes(low["baseline_decision"]["margin"], temporal["baseline_decision"]["margin"],
                                low["decision"]["margin"], temporal["decision"]["margin"])}
        for condition, prefix in zip(CONDITIONS, ("low", "temporal")):
            row = group[condition]
            route = next(route for route in intact[(pair_id, condition)] if
                         (route["query_group"], route["key_group"], route["control"]) == (query, keys, control))
            if (route["edge_budget"] != row["edge_budget"] or route["query_positions"] != row["query_positions"] or
                    route["key_positions"] != row["key_positions"]):
                raise ValueError("Intact routing diagnostics do not match the actual knocked-out edges.")
            item[f"{prefix}_baseline_edge_mass_per_query_head"] = statistics.mean(
                route["mask_audit"]["layers"][str(layer)]["mean_selected_edge_mass_per_query_head"] for layer in window)
            item[f"{prefix}_causal_edges_all_heads_window"] = len(window) * row["edge_budget"]["all_head_visible_causal_edges_per_layer"]
            item[f"{prefix}_query_count"] = row["edge_budget"]["query_count"]
            item[f"{prefix}_key_count"] = row["edge_budget"]["key_count"]
            for label, decision in (("base", row["baseline_decision"]), ("ko", row["decision"])):
                item.update({f"{prefix}_{label}_{name}": decision[name]
                             for name in ("margin", "correct_logit", "incorrect_logit", "prediction")})
            item.update({f"{prefix}_{name}": value for name, value in secondary_outcomes(row["baseline_decision"], row["decision"]).items()})
        result.append(item)
    return result


def control_contrasts(rows):
    grouped = {}
    for row in rows:
        key = (row["pair_id"], row["query_group"], row["key_group"], row["window_start"], row["window_end"])
        pair = grouped.setdefault(key, {})
        if row["control"] in pair:
            raise ValueError("Duplicate knockout control row.")
        pair[row["control"]] = row
    result = []
    for group in grouped.values():
        if set(group) != {"target", "background"}:
            raise ValueError("Missing paired target/background knockout.")
        target, background = group["target"], group["background"]
        for prefix in ("low", "temporal"):
            for suffix in ("causal_edges_all_heads_window", "query_count", "key_count"):
                if target[f"{prefix}_{suffix}"] != background[f"{prefix}_{suffix}"]:
                    raise ValueError("Actual target/background causal-edge budgets differ.")
        item = {key: target[key] for key in ("pair_id", "base_sample_id", "analysis_stratum", "first_object_id",
                "prompt_variant", "correct_option", "query_group", "key_group", "window_start", "window_end")}
        for name in ("delta_M_temporal", "delta_M_low", "compression"):
            item[f"{name}_target_minus_background"] = target[name] - background[name]
        result.append(item)
    return result


def aggregate(rows, keys, metrics, seed=42, repeats=2000):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    result = []
    for cell, items in sorted(grouped.items(), key=lambda item: str(item[0])):
        if len({item["base_sample_id"] for item in items}) != len(items):
            raise ValueError("Repeated independent base in an aggregate cell; layers/conditions are not replicates.")
        output = {**dict(zip(keys, cell)), "case_count": len(items),
                  "target_1_first_count": sum(item["first_object_id"] == 1 for item in items)}
        for metric in metrics:
            values = [item[metric] for item in items if item[metric] is not None]
            if any(type(value) not in (int, float, bool) or not math.isfinite(value) for value in values):
                raise ValueError("Non-finite aggregate outcome.")
            output[f"{metric}_n"] = len(values)
            output[f"{metric}_mean"] = statistics.mean(values) if values else None
            lower = upper = None
            if len(values) > 1:
                rng = random.Random(digest([seed, cell, metric]))
                samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(repeats))
                lower, upper = samples[int((repeats - 1) * 0.025)], samples[int((repeats - 1) * 0.975)]
            output[f"{metric}_ci_low"], output[f"{metric}_ci_high"] = lower, upper
        result.append(output)
    return result


def write_csv(path, rows):
    path = Path(path)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_inputs(plan_dir, execution_dir):
    frozen, pairs, mappings = load_selection(plan_dir)
    root = Path(execution_dir)
    execution = read_json(root / "execution_config.json")
    if (execution.get("artifact_type") != "real" or execution["selection_fingerprint"] != frozen["selection_fingerprint"] or
            execution["execution_code_sha256"] != execution_hashes() or
            execution["execution_fingerprint"] != digest({key: value for key, value in execution.items() if key != "execution_fingerprint"})):
        raise ValueError("Incompatible or invalid execution provenance; no completed analysis may be reported.")
    baselines = require_stage(root, execution, pairs, "baseline", {row["eval_id"] for pair in pairs.values() for row in pair.values()})
    tasks = technical_tasks(frozen, mappings)
    require_stage(root, execution, pairs, "preflight", {task["task_id"] for task in tasks}, expected_tasks=tasks)
    stages = {}
    for stage in PRIMARY_STAGES:
        tasks = primary_tasks(frozen, mappings, stage)
        stages[stage] = require_stage(root, execution, pairs, stage, {task["task_id"] for task in tasks}, mappings, baselines, tasks)
    return frozen, pairs, execution, baselines, stages


def analyze(plan_dir, execution_dir, output_dir=None, seed=42, repeats=2000, plots=True):
    if type(repeats) is not int or repeats < 100:
        raise ValueError("At least 100 case-bootstrap repeats are required.")
    root = Path(execution_dir).resolve()
    output = Path(output_dir or root / "analysis").resolve()
    if (root not in output.parents or output.relative_to(root).parts[0] in
            ("baseline", "preflight", "patch", "routing", "knockout", "captures")):
        raise ValueError("Analysis must be in a separate directory beneath the Phase 3C execution root.")
    frozen, pairs, execution, baselines, stages = verify_inputs(plan_dir, root)
    config = {"schema": SCHEMA, "artifact_type": "real", "execution_fingerprint": execution["execution_fingerprint"],
        "selection_fingerprint": frozen["selection_fingerprint"], "analysis_code_sha256": file_hash(__file__),
        "visualization_code_sha256": file_hash(Path(__file__).with_name("visualize_phase3c.py")) if plots else None,
        "seed": seed, "case_bootstrap_repeats": repeats, "plots": plots,
        "input_sha256": {f"{stage}/{name}": file_hash(root / stage / name)
                         for stage in ("baseline", "preflight", *PRIMARY_STAGES)
                         for name in ("rows.jsonl", "summary.json", "task_manifest.jsonl")}}
    config["analysis_fingerprint"] = digest(config)
    frozen_write(output / "analysis_config.json", config)
    atomic_write(output / "analysis_status.json", {"complete": False, "analysis_fingerprint": config["analysis_fingerprint"]})
    patches = patch_table(stages["patch"], pairs)
    knockouts = knockout_table(stages["knockout"], stages["routing"], pairs)
    contrasts = control_contrasts(knockouts)
    patch_summary = aggregate(patches, ("analysis_stratum", "direction", "support", "layer", "location"),
        ("margin_delta", "source_aligned_margin_delta", "recovery", "recipient_coverage", "categorical_flip",
         "strict_sign_crossing", "strict_incorrect_to_correct", "zero_margin_tie", "non_ab_first_token"), seed, repeats)
    ko_keys = ("analysis_stratum", "query_group", "key_group", "window_start", "window_end")
    ko_summary = aggregate(knockouts, (*ko_keys, "control"),
        ("delta_M_temporal", "delta_M_low", "compression", "advantage_base", "advantage_KO",
         "low_categorical_flip", "temporal_categorical_flip", "low_strict_sign_crossing", "temporal_strict_sign_crossing",
         "low_zero_margin_tie", "temporal_zero_margin_tie", "low_non_ab_first_token", "temporal_non_ab_first_token"), seed, repeats)
    control_summary = aggregate(contrasts, ko_keys,
        ("delta_M_temporal_target_minus_background", "delta_M_low_target_minus_background", "compression_target_minus_background"), seed, repeats)
    tables = {"case_patch_effects": patches, "case_knockout_decomposition": knockouts,
        "case_matched_control_contrasts": contrasts, "patch_summary": patch_summary,
        "knockout_summary": ko_summary, "matched_control_summary": control_summary}
    for name, table in tables.items():
        atomic_write(output / f"{name}.json", table)
        write_csv(output / f"{name}.csv", table)
    figures = []
    if plots:
        try:
            from .visualize_phase3c import figures as create_figures
        except ImportError:
            from visualize_phase3c import figures as create_figures
        figures = create_figures(output, tables, frozen, pairs, baselines)
    summary = {"schema": SCHEMA, "artifact_type": "real", "complete": True, "primary_intervention_grid_complete": True,
        "analysis_fingerprint": config["analysis_fingerprint"], "execution_fingerprint": execution["execution_fingerprint"],
        "case_count": len(pairs), "rescue_cases": 8, "stable_control_cases": 4,
        "capture_conditions": len(baselines), "primary_patch_rows": len(stages["patch"]),
        "primary_knockout_rows": len(stages["knockout"]), "intact_routing_rows": len(stages["routing"]),
        "missing_task_count": 0, "failed_task_count": 0,
        "bootstrap_unit": "independent_base_case_within_stratum_and_setting", "interval_interpretation": "exploratory_pilot_diagnostics",
        "technical_preflight_case_ids": frozen["technical_preflight_case_ids"], "figures": figures,
        "output_sha256": {path.name: file_hash(path) for path in sorted(output.glob("*.csv"))},
        "interpretation_limit": "Functional route contribution is not proof that routing is the sole or primary mechanism."}
    report = ["# Phase 3C Pilot", "", "All frozen tasks and technical gates passed independent verification.",
        "", f"Cases: {len(pairs)} independent bases (8 rescue, 4 stable controls).",
        f"Primary forwards: {summary['primary_patch_rows']} visual patches and {summary['primary_knockout_rows']} knockouts.",
        f"Intact routing diagnostics: {summary['intact_routing_rows']} exact-no-op forwards.",
        "", "## Reading the Results", "",
        "Rescue and stable controls are summarized separately. No layer or repeated condition is an independent replicate.",
        "`knockout_summary.csv` reports temporal change, low change and boundary-advantage compression together.",
        "`matched_control_summary.csv` subtracts background effects within the same case, route and window.",
        "Positive compression can arise from temporal impairment, low improvement or both; inspect the components.",
        "`patch_summary.csv` preserves both directions, raw margin change and donor-aligned change; Recovery has an explicit denominator.",
        "When donor and recipient baseline margins are equal, Recovery and donor-aligned effects are undefined, not zero; raw effects remain.",
        "Single-ROI rows are references, not full-support interventions. Expanded rows use real observed donor states.",
        "Pre/post-DeepStack comparisons use location-matched donor captures, not the same vector at different locations.",
        "Categorical flips, strict sign crossings and zero-margin ties are separate secondary outcomes.",
        "", "## Limits", "",
        "Case-bootstrap intervals are exploratory pilot diagnostics, not population inference or multiplicity-adjusted grid tests.",
        "A weak/non-significant patch effect does not demonstrate equivalence or complete representational sufficiency.",
        "Knockout changes access to content and softmax competition; it cannot isolate routing from representation.",
        "These outputs support evaluating functional contributions, not declaring attention the unique or primary mechanism.",
        "Phase 3B evidence is unchanged. VM packaging does not confirm a verified local backup.", ""]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    summary["output_sha256"] = {str(path.relative_to(output)): file_hash(path)
        for path in sorted(output.rglob("*")) if path.is_file() and
        path.name not in ("aggregate_summary.json", "analysis_config.json", "analysis_status.json")}
    atomic_write(output / "aggregate_summary.json", summary)
    atomic_write(output / "analysis_status.json", {"complete": True, "analysis_fingerprint": config["analysis_fingerprint"]})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan_dir", required=True)
    parser.add_argument("--execution_dir", required=True)
    parser.add_argument("--output_dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap_repeats", type=int, default=2000)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()
    try:
        summary = analyze(args.plan_dir, args.execution_dir, args.output_dir, args.seed, args.bootstrap_repeats, not args.no_plots)
        print(f"Phase 3C complete: {summary['primary_patch_rows']} patches, {summary['primary_knockout_rows']} knockouts; "
              f"report={Path(args.output_dir or Path(args.execution_dir) / 'analysis') / 'report.md'}", flush=True)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(1, f"Phase 3C analysis blocked: {exc}\nNo completed analysis for this attempt.\n")


if __name__ == "__main__":
    main()
