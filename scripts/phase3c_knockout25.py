"""Frozen 25-rescue attention-blocking cohort and paired outcomes."""

import math
from pathlib import Path

from .analyze_phase3c import aggregate, case_metadata, control_contrasts, secondary_outcomes, write_csv
from .phase3c_core import CONDITIONS, atomic_write, boundary_outcomes, digest, edge_budget, file_hash, frozen_write, read_json
from .phase3c_interventions import AttentionKnockout
from .run_phase3c_preflight import logits_parity

WINDOWS = tuple(tuple(range(start,start+4)) for start in range(0,36,4))


def intent(plan, pairs, pilot_run, whole=True):
    retained = sorted(key for key in plan["case_ids"] if plan["cohort_groups"][key] == "original_pilot")
    if len(retained) != 8:
        raise ValueError("Retain exactly the existing 8 pilot rescues in the KO cohort.")
    selected = list(retained)
    for mover, quota in ((1,13),(2,12)):
        count = sum(pairs[key][CONDITIONS[0]]["first_object_id"] == mover for key in selected)
        for key in plan["case_ids"]:
            if count >= quota:
                break
            if key not in selected and pairs[key][CONDITIONS[0]]["first_object_id"] == mover:
                selected.append(key); count += 1
        if count != quota:
            raise ValueError("Insufficient frozen bases for the pre-specified 25-case mover quotas.")
    selected.sort(key=lambda key: pairs[key][CONDITIONS[0]]["base_sample_id"])
    source = Path(pilot_run)/"analysis/case_knockout_decomposition.json"
    return {"case_ids": selected, "case_count": 25, "retained_pilot_ids": retained, "new_ko_bases": 17,
        "selection_rule": "Retain original 8, then ascending frozen base IDs within 13 T1-first/12 T2-first quotas; no patch-effect selection.",
        "whole_event2_enabled": whole, "windows": [list(window) for window in WINDOWS],
        "original_ko_table_path": str(source.resolve()), "original_ko_table_sha256": file_hash(source),
        "query_groups": ["query_all", "options_all"], "key_groups": ["target_1", "target_2", "both_targets"],
        "whole_event_control_limit": "Whole-event route has no same-window 512-key background complement; do not claim budget/location-matched background control."}


def tasks(config, fingerprint):
    result = []
    for key in config["case_ids"]:
        routes = [(query, target, control) for query in config["query_groups"] for target in config["key_groups"]
                  for control in ("target","background")]
        if config["whole_event2_enabled"]:
            routes.append(("options_all","whole_event2","whole_event_reference"))
        for query,target,control in routes:
            for window in WINDOWS:
                spec = {"pair_id": key,"query_group":query,"key_group":target,"control":control,"window":list(window)}
                spec["task_id"] = digest([fingerprint,spec])
                result.append(spec)
    return result


def positions(mapping, condition, spec):
    record = mapping["processor_records"][condition]
    if spec["key_group"] == "whole_event2":
        queries = record["group_positions"]["options_all"]
        side = "low_positions" if condition == CONDITIONS[0] else "temporal_positions"
        keys = mapping["support_audit"]["supports"]["whole_event2"][side]
    else:
        route = mapping["support_audit"]["knockout_controls"][condition][spec["query_group"]][spec["key_group"]]["background"]
        queries = route["query_positions"]
        keys = route["target_key_positions"] if spec["control"] == "target" else route["control_key_positions"]
    return queries, keys, edge_budget(queries,keys,len(record["input_ids"]))


def old_rows(config, pairs, mappings):
    path = config["original_ko_table_path"]
    if file_hash(path) != config["original_ko_table_sha256"]:
        raise ValueError("Original KO evidence changed.")
    result = {}
    for row in read_json(path):
        if row["pair_id"] not in config["retained_pilot_ids"] or row["analysis_stratum"] != "rescue":
            continue
        key = (row["pair_id"],row["query_group"],row["key_group"],row["control"],row["window_start"])
        if key in result:
            raise ValueError("Duplicate old KO pair.")
        spec = {"query_group":row["query_group"],"key_group":row["key_group"],"control":row["control"]}
        if any(row[name] != value for name,value in case_metadata(pairs[row["pair_id"]]).items()):
            raise ValueError("Old KO stimulus metadata changed.")
        for condition, prefix in zip(CONDITIONS,("low","temporal")):
            _,_,budget = positions(mappings[row["pair_id"]],condition,spec)
            for field, expected in (("query_count",budget["query_count"]),("key_count",budget["key_count"]),
                                    ("causal_edges_all_heads_window",4*budget["all_head_visible_causal_edges_per_layer"])):
                if row[f"{prefix}_{field}"] != expected:
                    raise ValueError("Reused KO edge budgets differ from current frozen support.")
        result[key] = row
    if len(result) != 864:
        raise ValueError("Expected 864 paired old-rescue KO records (1728 forwards).")
    return result


def condition_result(engine, live, mapping, spec, before, enabled):
    queries,keys,budget = positions(mapping,live["row"]["condition"],spec)
    controller = AttentionKnockout(engine.text.layers,spec["window"],queries,keys,
                                  len(mapping["processor_records"][live["row"]["condition"]]["input_ids"]),enabled=enabled)
    with controller.installed():
        logits = engine.forward(live)
    result = {"decision":engine.decide(logits,live["row"]),"mask_audit":controller.validate(),
              "edge_budget":budget,"query_positions":queries,"key_positions":keys}
    if not enabled:
        result["noop_parity"] = logits_parity(before,logits,exact=True)
        if not result["noop_parity"]["passed"]:
            raise RuntimeError("Disabled KO hook is not an exact no-op.")
    return result


def paired_row(spec, results, baselines, pairs):
    item = {**case_metadata(pairs[spec["pair_id"]]),"pair_id":spec["pair_id"],"query_group":spec["query_group"],
        "key_group":spec["key_group"],"control":spec["control"],"window_start":spec["window"][0],"window_end":spec["window"][-1]}
    low,temporal = [results[condition]["decision"] for condition in CONDITIONS]
    before_low,before_temporal = [baselines[pairs[spec["pair_id"]][condition]["eval_id"]]["decision"] for condition in CONDITIONS]
    item.update(boundary_outcomes(before_low["margin"],before_temporal["margin"],low["margin"],temporal["margin"]))
    for condition,label in zip(CONDITIONS,("low","temporal")):
        before = baselines[pairs[spec["pair_id"]][condition]["eval_id"]]["decision"]
        after,budget = results[condition]["decision"],results[condition]["edge_budget"]
        item.update({f"{label}_{name}":value for name,value in secondary_outcomes(before,after).items()})
        correct = pairs[spec["pair_id"]][condition]["correct_option"]
        item[f"{label}_strict_incorrect_to_correct"] = (before["margin"] < 0 < after["margin"] and
            before["prediction"] != correct and after["prediction"] == correct)
        for stage,value in (("base",before),("ko",after)):
            item.update({f"{label}_{stage}_{name}":value[name] for name in ("margin","correct_logit","incorrect_logit","prediction")})
        item.update({f"{label}_query_count":budget["query_count"],f"{label}_key_count":budget["key_count"],
                     f"{label}_causal_edges_all_heads_window":4*budget["all_head_visible_causal_edges_per_layer"]})
    return item


def analyze(rows, output, fingerprint):
    output = Path(output)
    table = [r["paired_outcome"] for r in rows.values()]
    for row,record in zip(table,rows.values()):
        row["ko_cohort_group"] = record["ko_cohort_group"]
    metrics = ("delta_M_temporal","delta_M_low","compression","advantage_base","advantage_KO")
    keys = ("query_group","key_group","control","window_start","window_end")
    summary = aggregate(table,keys,metrics)
    cohort_summary = aggregate(table,("ko_cohort_group",*keys),metrics)
    for name,values in (("case_knockout_decomposition",table),("knockout_summary",summary),("ko_cohort_summary",cohort_summary)):
        atomic_write(output/f"{name}.json",values)
        write_csv(output/f"{name}.csv",values)
    contrasts = control_contrasts([row for row in table if row["control"] in ("target","background")])
    atomic_write(output/"case_matched_background_contrasts.json",contrasts)
    atomic_write(output/"matched_background_summary.json",aggregate(contrasts,
        ("query_group","key_group","window_start","window_end"),
        ("delta_M_temporal_target_minus_background","delta_M_low_target_minus_background","compression_target_minus_background")))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = [r for r in summary if r["query_group"] == "options_all"]
    fig,axes = plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    for ax,metric,title in zip(axes,metrics[:3],("Temporal margin change","Low margin change","Temporal-low gap reduction")):
        for key,color in zip(("target_1","target_2","both_targets","whole_event2"),("#127F87","#B73E64","#607831","#655199")):
            for control in ("target","background","whole_event_reference"):
                values = sorted([r for r in selected if r["key_group"] == key and r["control"] == control],key=lambda r:r["window_start"])
                if not values:continue
                label = key.replace("_"," ") + (" matched background" if control == "background" else "")
                ax.plot([r["window_start"] for r in values],[r[metric+"_mean"] for r in values],
                        marker="o",markersize=3,linestyle="--" if control == "background" else "-",label=label,color=color)
        ax.axhline(0,color="black",linewidth=.7)
        ax.set(title=title,xlabel="First layer of blocked four-layer window",ylabel="Mean correct-option logit-margin change",xticks=list(range(0,36,4)))
    axes[0].legend(fontsize=8)
    fig.suptitle("Attention blocking: option-text queries -> visual keys | 25 fixed rescue bases | paired low/temporal")
    fig.supxlabel("Whole-event and ROI routes cut different edge budgets; their effect sizes are not budget-matched comparisons.",fontsize=9)
    fig.savefig(output/"figure2_knockout25_decomposition.png",dpi=170)
    plt.close(fig)
    result = {"artifact_type":"real","complete":True,"case_count":25,"paired_rows":len(table),"fingerprint":fingerprint,
              "output_sha256":{p.name:file_hash(p) for p in output.iterdir() if p.is_file() and p.name != "aggregate_summary.json"}}
    atomic_write(output/"aggregate_summary.json",result)
    return result
