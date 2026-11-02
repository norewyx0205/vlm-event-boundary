"""Verified CPU re-analysis of a frozen Phase 3F run; never writes into the run or its sources.

References are rebuilt from raw checkpoints: the singleton KO run is replayed through its own
verify_rows (checkpoints, full logits, masks, runtime) instead of its summary table, and the
archived Phase 3D joint KO and L16 patch rows are re-checked against their payload logits.
Every cell reports both endpoints, the change, actual answer flips and strict sign crossings.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import phase3d_execution as execution
from scripts import phase3d_knockout as knockout
from scripts import phase3d_single_layer_ko as singleton
from scripts import phase3f_layout_controls as core
from scripts.phase3c_core import CONDITIONS, atomic_write, file_hash, frozen_write, read_json

SCHEMA = "phase3f_verified_analysis_v1"
ARCHIVED, PERMUTED, LOW, TEMPORAL = core.ARCHIVED, core.PERMUTED, core.LOW, core.TEMPORAL
DECISION_FIELDS = ("margin", "prediction", "correct_option", "is_correct")


def compact(decision):
    return {k: decision[k] for k in DECISION_FIELDS}


def window_label(window):
    return "" if not window else str(window[0]) if len(window) == 1 else f"{window[0]}-{window[-1]}"


def verified_singleton(plan, records):
    """Singleton L16-L19 references from raw, independently verified checkpoints."""
    root = Path(plan["singleton_root"])
    context = singleton.load_context(root)
    if context[0]["source_plan_fingerprint"] != plan["source_plan_fingerprint"]:
        raise ValueError("Singleton run is bound to a different Phase 3D source.")
    rows = singleton.verify_rows(root, context)
    own = read_json(root / "source_records.json")
    result = {}
    for row in rows:
        spec = row["spec"]
        if spec["stage"] != "knockout":
            continue
        pair_id, condition = spec["pair_id"], spec["condition"]
        if (own[pair_id][condition]["baseline"]["decision"] !=
                records[pair_id][ARCHIVED][condition]["baseline"]["decision"]):
            raise ValueError("Singleton baseline differs from the Phase 3F archived baseline.")
        key = str(spec["window"][0])
        if key in result.setdefault(pair_id, {}).setdefault(condition, {}):
            raise ValueError("Duplicate singleton reference.")
        result[pair_id][condition][key] = compact(row["decision"])
    if any(sorted(result.get(p, {}).get(c, {})) != ["16", "17", "18", "19"]
           for p in plan["case_ids"] for c in CONDITIONS):
        raise ValueError("Singleton references must cover L16-L19 for every base and condition.")
    return result, len(rows)


def verified_phase3d(plan, records):
    """Archived joint 16-19 KO and L16 patch decisions, re-checked against payload logits."""
    source = Path(plan["source_root"])
    context = execution.load_context(source)
    source_plan, pairs, mappings = context
    joint_specs = {(s["pair_id"], s["condition"]): s for s in knockout.condition_specs(source_plan, pairs)
                   if all(s[k] == v for k, v in singleton.ROUTE.items()) and s["window"] == list(singleton.LAYERS)}
    runtime = read_json(source / "runtime.json")["runtime"]
    joint, patch, evidence = {}, {}, Counter()
    for pair_id in plan["case_ids"]:
        for condition in CONDITIONS:
            baseline = records[pair_id][ARCHIVED][condition]["baseline"]
            audit = records[pair_id][ARCHIVED][condition]["audit"]
            spec = joint_specs[(pair_id, condition)]
            row = execution.saved_checkpoint(source, source_plan, spec)
            if row is None:
                raise ValueError("Missing archived joint KO checkpoint.")
            knockout.verify_condition(row, spec, mappings[pair_id], baseline)
            if row.get("reuse"):
                rebuilt = knockout.reuse_condition(source, context, spec, baseline, audit, runtime)
                if rebuilt is None or row["reuse"] != rebuilt["reuse"] or row["decision"] != rebuilt["decision"]:
                    raise ValueError("Joint KO reuse provenance does not rebuild.")
            # Phase 3D committed KO rows without logit payloads; evidence is the sealed checkpoint,
            # verify_condition (mask, budget, inputs) and, for reused rows, rebuilt provenance.
            if "payload" in row:
                singleton.check_logits(execution.payload(source, row)["logits"], row["decision"])
            evidence[("logits" if "payload" in row else "reuse_rebuilt" if row.get("reuse")
                      else "sealed_checkpoint_mask_audit")] += 1
            joint.setdefault(pair_id, {})[condition] = compact(row["decision"])
        for spec in execution.feature_tasks(source_plan, "patch", [pair_id]):
            row = execution.saved_checkpoint(source, source_plan, spec)
            baseline = records[pair_id][ARCHIVED][spec["condition"]]["baseline"]
            if row is None or row["baseline_record_sha256"] != baseline["record_sha256"]:
                raise ValueError("Archived patch is missing or bound to another baseline.")
            singleton.check_logits(execution.payload(source, row)["logits"], row["decision"])
            patch.setdefault(pair_id, {})[spec["condition"]] = compact(row["decision"])
    return joint, patch, dict(evidence)


def compare_frozen(frozen, singles, joint, patch):
    """The frozen references.json must agree with raw-derived references."""
    mismatches = []
    for pair_id, conditions in frozen["event2_ko"].items():
        for condition, windows in conditions.items():
            for window, value in windows.items():
                raw = joint[pair_id][condition] if window == "16-19" else singles[pair_id][condition][window]
                if value["margin"] != raw["margin"] or value["prediction"] != raw["prediction"]:
                    mismatches.append([pair_id, condition, window])
    for pair_id, conditions in frozen["patch"].items():
        for condition, value in conditions.items():
            if compact(value) != patch[pair_id][condition]:
                mismatches.append([pair_id, condition, "patch"])
    return mismatches


def decision_table(plan, records, rows, singles, joint, patch):
    table = {}
    def put(key, decision):
        if key in table:
            raise ValueError(f"Duplicate decision cell {key}.")
        table[key] = compact(decision)
    for pair_id in plan["case_ids"]:
        for view in execution.VIEWS:
            for condition in CONDITIONS:
                put((pair_id, view, condition, "base", None, ""), records[pair_id][view][condition]["baseline"]["decision"])
        for condition in CONDITIONS:
            put((pair_id, ARCHIVED, condition, "patch", None, ""), patch[pair_id][condition])
            put((pair_id, ARCHIVED, condition, "ko", "event2", "16-19"), joint[pair_id][condition])
            for window, decision in singles[pair_id][condition].items():
                put((pair_id, ARCHIVED, condition, "ko", "event2", window), decision)
    for row in rows:
        spec = row["spec"]
        if spec["kind"] != "gate":
            put((spec["pair_id"], spec["view"], spec["condition"], spec["kind"], spec["route"],
                 window_label(spec["window"])), row["decision"])
    return table


def transition(before, after):
    valid = before["prediction"] in ("A", "B") and after["prediction"] in ("A", "B")
    return {"answer_changed": before["prediction"] != after["prediction"],
            "correct_to_incorrect": before["is_correct"] and not after["is_correct"],
            "incorrect_to_correct": not before["is_correct"] and after["is_correct"],
            "strict_sign_crossing": before["margin"]*after["margin"] < 0 and valid,
            "zero_margin_after": after["margin"] == 0, "non_ab_after": after["prediction"] not in ("A", "B")}


def endpoint_cell(ids, before, after):
    """before/after(pair, view) -> decision. Per view and layout-symmetric endpoints and changes."""
    result = {}
    for view, name in ((ARCHIVED, "archived"), (PERMUTED, "permuted")):
        pairs = [(before(p, view), after(p, view)) for p in ids]
        moves = [transition(b, a) for b, a in pairs]
        result[name] = {"before": core.bootstrap([b["margin"] for b, _ in pairs]),
            "after": core.bootstrap([a["margin"] for _, a in pairs]),
            "change": core.bootstrap([a["margin"]-b["margin"] for b, a in pairs]),
            "correct_before": sum(b["is_correct"] for b, _ in pairs), "correct_after": sum(a["is_correct"] for _, a in pairs),
            "correct_A_prompts": sum(b["correct_option"] == "A" for b, _ in pairs),
            **{key: sum(m[key] for m in moves) for key in moves[0]}}
    sym = lambda f: core.bootstrap([(f(before(p, ARCHIVED), after(p, ARCHIVED)) + f(before(p, PERMUTED), after(p, PERMUTED)))/2
                                    for p in ids])
    result["symmetric"] = {"before": sym(lambda b, a: b["margin"]), "after": sym(lambda b, a: a["margin"]),
                           "change": sym(lambda b, a: a["margin"]-b["margin"])}
    return result


def compression_cell(ids, d, kind, route, window):
    change = lambda p, v, c: d[(p, v, c, kind, route, window)]["margin"]-d[(p, v, c, "base", None, "")]["margin"]
    values = lambda v: [change(p, v, LOW)-change(p, v, TEMPORAL) for p in ids]
    return {"archived": core.bootstrap(values(ARCHIVED)), "permuted": core.bootstrap(values(PERMUTED)),
            "symmetric": core.bootstrap([(a+b)/2 for a, b in zip(values(ARCHIVED), values(PERMUTED))])}


def analyze(run_root, output_root):
    run_root, output_root = Path(run_root).resolve(), Path(output_root).resolve()
    if output_root == run_root or run_root in output_root.parents:
        raise ValueError("Write the verified analysis outside the frozen run root.")
    context = core.load_context(run_root)
    plan = context[0]
    rows = core.verify_rows(run_root, context)
    if len(rows) != core.EXPECTED_FORWARDS + 200:
        raise ValueError("Expected 2,000 interventions and 200 gates.")
    records = read_json(run_root / "source_records.json")
    singles, singleton_rows = verified_singleton(plan, records)
    joint, patch, joint_evidence = verified_phase3d(plan, records)
    mismatches = compare_frozen(read_json(run_root / "references.json"), singles, joint, patch)
    if mismatches:
        raise ValueError(f"Frozen references disagree with raw checkpoints: {mismatches[:5]}")
    d = decision_table(plan, records, rows, singles, joint, patch)
    ids = plan["case_ids"]
    base = lambda c: lambda p, v: d[(p, v, c, "base", None, "")]
    cell = lambda c, kind, route, window: lambda p, v: d[(p, v, c, kind, route, window)]
    result = {"patch": {c: endpoint_cell(ids, base(c), cell(c, "patch", None, "")) for c in CONDITIONS},
              "event2_ko": {}, "controls": {}, "patch_x_ko": {}}
    for window in ("16", "17", "18", "19", "16-19"):
        result["event2_ko"][window] = {**{c: endpoint_cell(ids, base(c), cell(c, "ko", "event2", window)) for c in CONDITIONS},
                                       "compression": compression_cell(ids, d, "ko", "event2", window)}
    for route in ("event1", "random"):
        for window in ("18", "16-19"):
            result["controls"][f"{route}:{window}"] = {
                **{c: endpoint_cell(ids, base(c), cell(c, "ko", route, window)) for c in CONDITIONS},
                "compression": compression_cell(ids, d, "ko", route, window)}
    for window in ("17", "18", "17-18", "17-23"):
        result["patch_x_ko"][window] = {
            "ko_alone": endpoint_cell(ids, base(LOW), cell(LOW, "ko", "event2", window)),
            "patch_under_ko": endpoint_cell(ids, cell(LOW, "ko", "event2", window), cell(LOW, "patch_ko", "event2", window)),
            "patch_plus_ko_vs_base": endpoint_cell(ids, base(LOW), cell(LOW, "patch_ko", "event2", window))}
    output_root.mkdir(parents=True, exist_ok=True)
    frozen_write(output_root / "results.json", result)
    frozen_write(output_root / "decisions.json", [{"pair_id": k[0], "view": k[1], "condition": k[2], "kind": k[3],
        "route": k[4], "window": k[5], **v} for k, v in sorted(d.items(), key=lambda item: tuple(map(str, item[0])))])
    (output_root / "report.md").write_text(report(result), encoding="utf-8")
    summary = {"schema": SCHEMA, "complete": True, "run_root": str(run_root), "plan_fingerprint": plan["fingerprint"],
        "verified_phase3f_rows": len(rows), "verified_singleton_rows": singleton_rows,
        "raw_reference_mismatches": 0, "archived_joint_ko_evidence": joint_evidence, "analysis_code_sha256": file_hash(Path(__file__)),
        "output_sha256": {p.name: file_hash(p) for p in sorted(output_root.iterdir()) if p.name != "summary.json"}}
    atomic_write(output_root / "summary.json", summary)
    return summary


def ends(cell):
    return f"{cell['before']['mean']:+.3f} → {cell['after']['mean']:+.3f}"


def change(cell):
    c = cell["change"]
    return f"{c['mean']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"


def flips(cell):
    return (f"{cell['correct_before']}→{cell['correct_after']} correct; changed {cell['answer_changed']} "
            f"(+{cell['incorrect_to_correct']}/−{cell['correct_to_incorrect']}); strict {cell['strict_sign_crossing']}"
            + (f"; ties {cell['zero_margin_after']}" if cell["zero_margin_after"] else "")
            + (f"; non-A/B {cell['non_ab_after']}" if cell["non_ab_after"] else ""))


def rows_for(label, cell):
    text = []
    for name in ("archived", "permuted"):
        part = cell[name]
        text.append(f"| {label} | {name} ({part['correct_A_prompts']} correct-A) | {ends(part)} | {change(part)} | {flips(part)} |")
    text.append(f"| {label} | symmetric | {ends(cell['symmetric'])} | {change(cell['symmetric'])} | — |")
    return text


HEADER = ["| Cell | View | Margin before → after | Change [95% case bootstrap] | Answers (correct before→after; flips) |",
          "| --- | --- | --- | ---: | --- |"]


def report(result):
    text = ["# Phase 3F Verified Analysis", "",
        "Model margin = correct-option minus incorrect-option first-token logit. Endpoints are means over 50 bases.",
        "Symmetric = per-base mean of the two layouts (opposite correct letters), cancelling a constant letter preference.",
        "Answers count actual greedy A/B outputs; strict sign crossings are reported separately and are not flips.",
        "References (archived singleton, joint and patch) were rebuilt from raw verified checkpoints.", "",
        "## L16 whole-Event-2 patch", "", *HEADER]
    for condition, cell in result["patch"].items():
        text += rows_for(f"recipient {condition}", cell)
    text += ["", "## Options -> Event-2 KO", "", *HEADER]
    for window, cell in result["event2_ko"].items():
        for condition in CONDITIONS:
            text += rows_for(f"L{window} {condition}", cell[condition])
    text += ["", "Compression (Low change − Temporal change):", ""]
    text += [f"- L{w}: archived {change({'change': c['compression']['archived']})}, permuted "
             f"{change({'change': c['compression']['permuted']})}, symmetric {change({'change': c['compression']['symmetric']})}"
             for w, c in result["event2_ko"].items()]
    text += ["", "## Equal-budget route controls", "", *HEADER]
    for name, cell in result["controls"].items():
        for condition in CONDITIONS:
            text += rows_for(f"{name} {condition}", cell[condition])
    text += ["", "## Temporal->Low L16 patch x later options->Event-2 KO (Low recipient)", "", *HEADER]
    for window, cell in result["patch_x_ko"].items():
        text += rows_for(f"KO L{window} alone", cell["ko_alone"])
        text += rows_for(f"patch under KO L{window}", cell["patch_under_ko"])
    text += ["", "The no-KO patch rows are in the first table (recipient low_boundary).",
             "Blocking option->Event-2 edges does not block Event-2 information routed through other positions;",
             "a removed rescue implicates this route, a surviving rescue does not exclude middle-layer use. Not mediation.", ""]
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_root, args.output_root), indent=2))


if __name__ == "__main__":
    main()
