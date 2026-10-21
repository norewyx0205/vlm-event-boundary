"""Isolated L16--19 singleton knockout follow-up to the frozen KO50 run."""

import math
from pathlib import Path

from . import phase3d_execution as execution
from . import phase3d_knockout as knockout
from .analyze_phase3c import aggregate, case_metadata, secondary_outcomes, write_csv
from .phase3c_core import CONDITIONS, atomic_write, boundary_outcomes, digest, file_hash, frozen_write, read_json
from .phase3c_knockout25 import positions
from .phase3c_interventions import AttentionKnockout
from .run_phase3c_preflight import logits_parity, save_tensors

SCHEMA = "phase3d_single_layer_ko_v1"
ANALYSIS_STATUS = "exploratory_same_cohort_followup"
LAYER_SELECTION = ("Layers 16-19 were selected from the previously observed joint-knockout peak on the same cohort; "
                   "this is not independent confirmatory localization.")
LAYERS = (16, 17, 18, 19)
ROUTE = {"query_group": "options_all", "key_group": "whole_event2", "control": "whole_event_reference"}
METRICS = ("delta_M_temporal", "delta_M_low", "compression", "advantage_base", "advantage_KO")
SECONDARY = tuple(f"{side}_{name}" for side in ("low", "temporal") for name in
    ("categorical_flip", "non_ab_first_token", "strict_sign_crossing", "zero_margin_tie",
     "strict_incorrect_to_correct", "strict_correct_to_incorrect"))


def code_hashes():
    return {**execution.code_hashes(), **{name: file_hash(Path(__file__).with_name(name)) for name in
        ("phase3d_single_layer_ko.py", "run_phase3d_single_layer_ko.py", "analyze_phase3c.py")},
        "single_layer_protocol": file_hash(Path(__file__).resolve().parents[1] / "docs/phase3d_single_layer_ko.md")}


def specs(plan, stage="knockout", pair_ids=None):
    if stage not in ("knockout", "baseline_gate"):
        raise ValueError("Only baseline gates and four singleton knockouts are in scope.")
    ids = plan["case_ids"] if pair_ids is None else pair_ids
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(plan["case_ids"]):
        raise ValueError("Unknown, duplicated or empty case subset.")
    result = []
    for pair_id in ids:
        for condition in CONDITIONS:
            for window in ([list(LAYERS)] if stage == "baseline_gate" else [[layer] for layer in LAYERS]):
                item = {"stage": stage, "view": "archived", "pair_id": pair_id,
                        "condition": condition, "window": window, **ROUTE}
                item["task_id"] = digest([plan["fingerprint"], item])
                result.append(item)
    return result


def check_logits(logits, decision):
    import torch
    good, bad = decision["correct_token_id"], decision["incorrect_token_id"]
    if (logits.ndim != 1 or not bool(torch.isfinite(logits).all()) or
            type(good) is not int or type(bad) is not int or good == bad or
            not 0 <= min(good, bad) <= max(good, bad) < len(logits) or
            float(logits[good]) != decision["correct_logit"] or
            float(logits[bad]) != decision["incorrect_logit"] or
            not math.isclose(decision["margin"], float(logits[good])-float(logits[bad]), abs_tol=1e-8) or
            int(logits.argmax()) != decision["predicted_first_token_id"]):
        raise ValueError("Decision is not supported by the saved full-vocabulary logits.")
    token = decision["predicted_first_token_id"]
    if "predicted_first_token_text" in decision:
        from .run_eval import parse_answer
        expected = parse_answer(decision["predicted_first_token_text"])
    else:
        expected = decision["correct_option"] if token == good else (
            "B" if decision["correct_option"] == "A" else "A") if token == bad else "other"
    if (decision["prediction"] != expected or
            decision["is_correct"] != (decision["prediction"] == decision["correct_option"]) or
            (token == good and decision["prediction"] != decision["correct_option"]) or
            (token == bad and decision["prediction"] != ("B" if decision["correct_option"] == "A" else "A"))):
        raise ValueError("Prediction, correctness and measured argmax disagree.")


def source_snapshot(source):
    """Verify only the relevant archived baselines and joint route, without writing to source."""
    import torch
    source = Path(source)
    context = execution.load_context(source)
    plan, pairs, mappings = context
    full = read_json(source / "aggregate_summary.json")
    ko = read_json(source / "knockout/analysis/aggregate_summary.json")
    runtime = read_json(source / "runtime.json")
    from .phase3c_budget import runtime_compatibility
    runtime_compatibility(plan["source_runtime"], runtime["runtime"])
    if (full.get("complete") is not True or full.get("artifact_type") != "real" or
            full.get("plan_fingerprint") != plan["fingerprint"] or full.get("base_count") != 50 or
            full.get("ko_condition_rows") != 11700 or full.get("ko_paired_rows") != 5850 or
            full.get("knockout_summary_sha256") != file_hash(source / "knockout/analysis/aggregate_summary.json") or
            ko.get("complete") is not True or ko.get("artifact_type") != "real" or
            ko.get("plan_fingerprint") != plan["fingerprint"] or ko.get("case_count") != 50 or
            ko.get("condition_rows") != 11700 or ko.get("paired_rows") != 5850 or ko.get("missing_count") != 0 or
            runtime.get("plan_fingerprint") != plan["fingerprint"] or len(plan["case_ids"]) != 50):
        raise ValueError("A complete, intact Phase 3D KO50 source is required.")
    files = {name: file_hash(source / name) for name in
        ("plan.json", "pairs.json", "mappings.json", "runtime.json", "aggregate_summary.json",
         "knockout/analysis/aggregate_summary.json", "source_baselines.json")}
    for name, sha in ko["output_sha256"].items():
        path = source / "knockout/analysis" / name
        if path.resolve().parent != (source / "knockout/analysis").resolve() or file_hash(path) != sha:
            raise ValueError("Source KO analysis bytes changed.")
        files[str(path.relative_to(source))] = sha
    reported = {}
    for row in read_json(source / "knockout/analysis/case_decomposition.json"):
        if (all(row.get(k) == v for k, v in ROUTE.items()) and
                (row.get("window_start"), row.get("window_end")) == (16, 19)):
            if row["pair_id"] in reported:
                raise ValueError("Duplicated joint pair in source analysis.")
            reported[row["pair_id"]] = row
    if set(reported) != set(plan["case_ids"]):
        raise ValueError("Source analysis must contain the same 50 joint-window pairs.")
    original_refs = read_json(source / "source_baselines.json")
    joint = {(s["pair_id"], s["condition"]): s for s in knockout.condition_specs(plan, pairs)
             if all(s[k] == v for k, v in ROUTE.items()) and s["window"] == list(LAYERS)}
    records, logits = {}, {}
    for pair_id in plan["case_ids"]:
        support = mappings[pair_id]["support_audit"]["supports"]["whole_event2"]
        if (support.get("one_to_one") is not True or any(len(support[k]) != 512 or
                len(set(support[k])) != 512 for k in ("low_positions", "temporal_positions"))):
            raise ValueError("Retain the original one-to-one 512-position matched support.")
        records[pair_id], logits[pair_id] = {}, {}
        for condition in CONDITIONS:
            baseline, data = execution.capture_reference(source, context, pair_id, condition)
            audit_spec = next(s for s in execution.feature_tasks(plan, "audit", [pair_id])
                              if s["condition"] == condition and s["view"] == "archived")
            audit = execution.saved_checkpoint(source, plan, audit_spec)
            if (audit is None or baseline["audit_record_sha256"] != audit["record_sha256"] or
                    baseline["input_fingerprint"] != audit["input_fingerprint"] or
                    baseline["input_fingerprint"] != digest(audit["processor_record"]["input_tensor_sha256"]) or
                    not all(torch.equal(data[name], data["logits"]) for name in
                            ("plain_logits", "identity_logits", "disabled_ko_logits")) or
                    not logits_parity(data["logits"], data["standard_logits"])["passed"] or
                    int(data["standard_logits"].argmax()) != baseline["decision"]["predicted_first_token_id"]):
                raise ValueError("Source baseline input/no-op/standard-generation parity failed.")
            check_logits(data["logits"], baseline["decision"])
            ref = original_refs[pairs[pair_id][condition]["eval_id"]]
            old_data = execution.checked_source(ref, tensor=True)
            if (ref["decision"] != baseline["decision"] or
                    ref["input_tensor_sha256"] != audit["processor_record"]["input_tensor_sha256"] or
                    not torch.equal(old_data["logits"].float(), data["logits"])):
                raise ValueError("Source baseline differs from its original archived raw-logit/input evidence.")
            del old_data
            correct = pairs[pair_id][condition]["correct_option"]
            expected_correct = condition == CONDITIONS[1]
            if (baseline["decision"]["correct_option"] != correct or
                    baseline["decision"]["prediction"] not in ("A", "B") or
                    baseline["decision"]["is_correct"] != expected_correct or
                    baseline["decision"]["margin"] * (1 if expected_correct else -1) <= 0):
                raise ValueError("Frozen rescue behaviour changed; do not replace/drop a case.")
            spec = joint[(pair_id, condition)]
            row = execution.saved_checkpoint(source, plan, spec)
            if row is None:
                raise ValueError("Missing original joint-window checkpoint.")
            knockout.verify_condition(row, spec, mappings[pair_id], baseline)
            if row.get("reuse"):
                rebuilt = knockout.reuse_condition(source, context, spec, baseline, audit, runtime["runtime"])
                if rebuilt is None or row["reuse"] != rebuilt["reuse"] or row["decision"] != rebuilt["decision"]:
                    raise ValueError("Joint reference lacks valid original semantic reuse provenance.")
            queries, keys, _ = positions(mappings[pair_id], condition, spec)
            record = mappings[pair_id]["processor_records"][condition]
            if (not queries or not keys or not set(keys) <= set(record["visual_positions"]) or
                    max(keys) >= min(queries) or len(set(queries)) != len(queries)):
                raise ValueError("Route must contain distinct, causally earlier visual keys.")
            for checkpoint in (baseline, audit, row):
                path = execution.checkpoint_path(source, checkpoint["spec"])
                files[str(path.relative_to(source))] = file_hash(path)
            files[baseline["payload"]["path"]] = baseline["payload"]["sha256"]
            records[pair_id][condition] = {"baseline": baseline, "audit": audit, "joint": row}
            logits[pair_id][condition] = data["logits"].clone()
        rebuilt = paired_outcome({"pair_id": pair_id, "window": list(LAYERS)},
            {c: records[pair_id][c]["joint"] for c in CONDITIONS},
            {c: records[pair_id][c]["baseline"] for c in CONDITIONS}, pairs[pair_id])
        fields = [*case_metadata(pairs[pair_id]), *METRICS]
        fields += [f"{side}_{stage}_{name}" for side in ("low", "temporal") for stage in ("base", "ko")
                   for name in ("margin", "correct_logit", "incorrect_logit", "prediction")]
        fields += [f"{side}_{name}" for side in ("low", "temporal") for name in
                   ("query_count", "key_count", "causal_edges_all_heads_window")]
        if any(rebuilt[k] != reported[pair_id].get(k) for k in fields):
            raise ValueError("Joint checkpoint decisions/budgets differ from the published source analysis.")
        print(f"[single-layer KO source audit] {len(records)}/50 bases checked", flush=True)
    return context, files, records, logits


def freeze_plan(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Use a separate new output root; Phase 3D is read-only.")
    if (output / "plan.json").exists():
        context = load_context(output)
        if context[0]["source_root"] != str(source):
            raise ValueError("A frozen follow-up cannot adopt a different source.")
        return context
    if output.exists() and any(p.name != ".phase3d.lock" for p in output.iterdir()):
        raise ValueError("Preparation requires an empty output directory.")
    context, files, records, logits = source_snapshot(source)
    source_plan, pairs, mappings = context
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("source_records", records), ("pairs", pairs), ("mappings", mappings),
                        ("source_plan", source_plan), ("source_runtime", read_json(source / "runtime.json"))):
        frozen_write(output / f"{name}.json", value)
    save_tensors(output / "source_logits.pt", logits)
    plan = {"schema": SCHEMA, "artifact_type": "real", "source_root": str(source),
        "source_plan_fingerprint": source_plan["fingerprint"], "source_files": files,
        "case_ids": source_plan["case_ids"], "preflight_ids": source_plan["preflight_ids"],
        "settings": source_plan["settings"], "source_runtime": read_json(source / "runtime.json")["runtime"],
        "project_root": source_plan["project_root"], "source_path_map": source_plan["source_path_map"],
        "code_sha256": code_hashes(), "snapshot_sha256": {name: file_hash(output / name) for name in
            ("source_records.json", "pairs.json", "mappings.json", "source_plan.json", "source_runtime.json", "source_logits.pt")},
        "layers": list(LAYERS), "route": ROUTE, "new_condition_forwards": 400,
        "joint_reference_condition_rows": 100, "technical_forwards_included_in_400": False,
        "analysis_status": ANALYSIS_STATUS, "layer_selection_rationale": LAYER_SELECTION,
        "case_reporting": "All 50 original bases equally; no pilot/new stratification.",
        "interpretation": "Singleton versus joint dependence; not a unique circuit or additive decomposition."}
    plan["fingerprint"] = digest(plan)
    frozen_write(output / "plan.json", plan)
    return plan, pairs, mappings


def load_context(root):
    root = Path(root)
    plan = read_json(root / "plan.json")
    if (plan.get("schema") != SCHEMA or plan.get("artifact_type") != "real" or
            plan.get("fingerprint") != digest({k: v for k, v in plan.items() if k != "fingerprint"}) or
            plan["code_sha256"] != code_hashes() or plan["layers"] != list(LAYERS) or plan["route"] != ROUTE or
            plan.get("analysis_status") != ANALYSIS_STATUS or plan.get("layer_selection_rationale") != LAYER_SELECTION):
        raise ValueError("Follow-up plan/code/protocol changed; use a new root.")
    source_context = execution.load_context(plan["source_root"])
    if source_context[0]["fingerprint"] != plan["source_plan_fingerprint"]:
        raise ValueError("Frozen source plan changed.")
    for directory, hashes in ((root, plan["snapshot_sha256"]), (Path(plan["source_root"]), plan["source_files"])):
        for name, sha in hashes.items():
            path = directory / name
            if not path.resolve().is_relative_to(directory.resolve()) or path.is_symlink() or file_hash(path) != sha:
                raise ValueError("Frozen source/snapshot bytes changed.")
    pairs, mappings = read_json(root / "pairs.json"), read_json(root / "mappings.json")
    if pairs != source_context[1] or mappings != source_context[2] or plan["case_ids"] != source_context[0]["case_ids"]:
        raise ValueError("Cases, prompts or support changed.")
    return plan, pairs, mappings


class Executor:
    def __init__(self, root, context, engine):
        import torch
        self.root, self.context, self.engine = Path(root), context, engine
        self.records = read_json(self.root / "source_records.json")
        self.logits = torch.load(self.root / "source_logits.pt", map_location="cpu", weights_only=True)
        self.source_context = execution.load_context(context[0]["source_root"])
        self.cache = {}

    def live(self, spec):
        key = spec["pair_id"], spec["condition"]
        if key not in self.cache:
            self.cache.clear()
            self.engine.torch.cuda.empty_cache()
            live, audit = execution.audited_live(self.context[0]["source_root"], self.source_context, self.engine, spec)
            baseline = self.records[key[0]][key[1]]["baseline"]
            measured = self.engine.forward(live)
            if (audit != self.records[key[0]][key[1]]["audit"] or
                    not logits_parity(self.logits[key[0]][key[1]], measured, exact=True)["passed"] or
                    self.engine.decide(measured, live["row"]) != baseline["decision"]):
                raise RuntimeError("Exact archived input/runtime/full-logit baseline parity failed.")
            self.cache[key] = live, baseline, measured
        return self.cache[key]

    def __call__(self, spec):
        enabled = spec["stage"] == "knockout"
        gate = None
        if enabled:
            gate_spec = next(s for s in specs(self.context[0], "baseline_gate", [spec["pair_id"]])
                             if s["condition"] == spec["condition"])
            gate = execution.saved_checkpoint(self.root, self.context[0], gate_spec)
            if gate is None:
                raise ValueError("Baseline/disabled-hook gate must pass before any singleton KO.")
        print(f"[singleton KO] {spec['stage']}: {spec['pair_id']} {spec['condition']} "
              f"block(s)={spec['window']}", flush=True)
        live, baseline, before = self.live(spec)
        mapping = self.context[2][spec["pair_id"]]
        queries, keys, budget = positions(mapping, spec["condition"], spec)
        controller = AttentionKnockout(self.engine.text.layers, spec["window"], queries, keys,
            len(mapping["processor_records"][spec["condition"]]["input_ids"]), enabled=enabled)
        with controller.installed():
            logits = self.engine.forward(live)
        result = {"decision": self.engine.decide(logits, live["row"]), "mask_audit": controller.validate(),
            "query_positions": queries, "key_positions": keys, "edge_budget": budget,
            "baseline_record_sha256": baseline["record_sha256"],
            "audit_record_sha256": baseline["audit_record_sha256"], "input_fingerprint": baseline["input_fingerprint"]}
        tensors = {"logits": logits}
        if enabled:
            result["gate_record_sha256"] = gate["record_sha256"]
            knockout.verify_condition(result, spec, mapping, baseline)
        else:
            if not logits_parity(before, logits, exact=True)["passed"]:
                raise RuntimeError("Disabled-hook full-logit no-op failed.")
            result["baseline_parity"] = logits_parity(self.logits[spec["pair_id"]][spec["condition"]], before, exact=True)
            tensors["baseline_logits"] = before
        check_logits(logits, result["decision"])
        return result, tensors


def verify_rows(root, context, pair_ids=None, include_knockouts=True):
    import torch
    plan = context[0]
    from .phase3c_budget import runtime_compatibility
    runtime = read_json(Path(root) / "runtime.json")
    if runtime.get("plan_fingerprint") != plan["fingerprint"]:
        raise ValueError("Missing or foreign live-runtime proof.")
    runtime_compatibility(plan["source_runtime"], runtime["runtime"])
    records = read_json(Path(root) / "source_records.json")
    original = torch.load(Path(root) / "source_logits.pt", map_location="cpu", weights_only=True)
    gates, rows = {}, []
    requested = specs(plan, "baseline_gate", pair_ids)
    if include_knockouts:
        requested += specs(plan, "knockout", pair_ids)
    for spec in requested:
        row = execution.saved_checkpoint(root, plan, spec)
        if row is None:
            raise ValueError("Incomplete baseline/singleton grid; partial rows are not a full result.")
        data = execution.payload(root, row)
        baseline = records[spec["pair_id"]][spec["condition"]]["baseline"]
        check_logits(data["logits"], row["decision"])
        if spec["stage"] == "baseline_gate":
            audit = row["mask_audit"]
            queries, keys, budget = positions(context[2][spec["pair_id"]], spec["condition"], spec)
            if (not torch.equal(data["baseline_logits"], original[spec["pair_id"]][spec["condition"]]) or
                    not torch.equal(data["logits"], data["baseline_logits"]) or row["decision"] != baseline["decision"] or
                    row["baseline_record_sha256"] != baseline["record_sha256"] or
                    row["audit_record_sha256"] != baseline["audit_record_sha256"] or
                    row["input_fingerprint"] != baseline["input_fingerprint"] or
                    row["query_positions"] != queries or row["key_positions"] != keys or row["edge_budget"] != budget or
                    audit.get("enabled") is not False or audit.get("all_heads") != 32 or
                    audit.get("orientation") != "text_queries_to_visual_keys" or
                    audit.get("pre_softmax_mask") != "negative_infinity" or audit.get("renormalized") is not True or
                    set(audit["layers"]) != {str(layer) for layer in LAYERS} or any(
                        layer["budget"] != budget or not 0 <= layer["max_row_sum_error"] <= .005
                        for layer in audit["layers"].values())):
                raise ValueError("Baseline/disabled-hook evidence failed independent verification.")
            gates[(spec["pair_id"], spec["condition"])] = row
        else:
            knockout.verify_condition(row, spec, context[2][spec["pair_id"]], baseline)
            if (row["gate_record_sha256"] != gates[(spec["pair_id"], spec["condition"])]["record_sha256"] or
                    row["mask_audit"].get("orientation") != "text_queries_to_visual_keys" or any(
                        row["decision"][k] != baseline["decision"][k] for k in ("correct_token_id", "incorrect_token_id"))):
                raise ValueError("Singleton KO is bound to a different technical gate.")
        rows.append(row)
    return rows


def paired_outcome(spec, conditions, baselines, pair):
    item = {**case_metadata(pair), "pair_id": spec["pair_id"], **ROUTE,
            "window_start": spec["window"][0], "window_end": spec["window"][-1],
            "layer_count": len(spec["window"]),
            "measurement": "joint_reference" if len(spec["window"]) == 4 else "new_single_layer"}
    item.update(boundary_outcomes(*(baselines[c]["decision"]["margin"] for c in CONDITIONS),
                                  *(conditions[c]["decision"]["margin"] for c in CONDITIONS)))
    for condition, side in zip(CONDITIONS, ("low", "temporal")):
        before, after = baselines[condition]["decision"], conditions[condition]["decision"]
        item.update({f"{side}_{key}": value for key, value in secondary_outcomes(before, after).items()})
        for name, source_sign, dest_sign in (("strict_incorrect_to_correct", -1, 1), ("strict_correct_to_incorrect", 1, -1)):
            item[f"{side}_{name}"] = (before["margin"]*source_sign > 0 and after["margin"]*dest_sign > 0 and
                before["prediction"] in ("A", "B") and after["prediction"] in ("A", "B") and
                before["prediction"] != after["prediction"])
        for label, value in (("base", before), ("ko", after)):
            for name in ("margin", "correct_logit", "incorrect_logit", "prediction"):
                item[f"{side}_{label}_{name}"] = value[name]
        budget = conditions[condition]["edge_budget"]
        for name in ("query_count", "key_count"):
            item[f"{side}_{name}"] = budget[name]
        item[f"{side}_causal_edges_all_heads_window"] = len(spec["window"])*budget["all_head_visible_causal_edges_per_layer"]
        item[f"{side}_record_sha256"] = conditions[condition]["record_sha256"]
    return item


def analyze(root, context, plots=True):
    root = Path(root)
    rows = verify_rows(root, context)
    records = read_json(root / "source_records.json")
    grouped = {}
    for row in rows:
        spec = row["spec"]
        if spec["stage"] == "knockout":
            grouped.setdefault((spec["pair_id"], tuple(spec["window"])), {})[spec["condition"]] = row
    table = []
    for pair_id in context[0]["case_ids"]:
        baselines = {c: records[pair_id][c]["baseline"] for c in CONDITIONS}
        joint = {c: records[pair_id][c]["joint"] for c in CONDITIONS}
        for condition in CONDITIONS:
            knockout.verify_condition(joint[condition], joint[condition]["spec"], context[2][pair_id], baselines[condition])
        for window in [[layer] for layer in LAYERS] + [list(LAYERS)]:
            results = joint if len(window) == 4 else grouped[(pair_id, tuple(window))]
            if set(results) != set(CONDITIONS):
                raise ValueError("Missing a condition; paired analysis cannot proceed.")
            table.append(paired_outcome({"pair_id": pair_id, "window": window}, results, baselines, context[1][pair_id]))
    if len(table) != 250 or len(rows) != 500:
        raise ValueError("Expected exactly 400 new KOs, 100 gates and 50 read-only joint pairs.")
    cells = aggregate(table, ("measurement", "window_start", "window_end", "layer_count"), METRICS + SECONDARY)
    output = root / "analysis"
    for name, values in (("case_decomposition", table), ("aggregate_cells", cells)):
        frozen_write(output / f"{name}.json", values)
        write_csv(output / f"{name}.csv", values)
    text = ["# Single-Layer Event-2 Attention Knockout", "",
        "Exploratory same-cohort mechanistic localization follow-up.", LAYER_SELECTION,
        "A concentrated effect indicates localized causal sensitivity of the tested route, not the temporal-order computation layer.", "",
        "All 50 frozen rescue bases have equal weight. Blocks are zero-based: L16 is the 17th decoder block.",
        "400 new interventional condition forwards; technical baseline/no-op forwards are additional.",
        "The four-layer joint intervention is a verified historical reference, not a new measurement.", "",
        "| Block(s) | Temporal margin change | Low margin change | Gap compression | Temporal strict correct-to-wrong |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for cell in sorted(cells, key=lambda r: (r["layer_count"], r["window_start"])):
        label = str(cell["window_start"]) if cell["layer_count"] == 1 else "16-19 (historical joint)"
        text.append(f"| {label} | {cell['delta_M_temporal_mean']:+.3f} | {cell['delta_M_low_mean']:+.3f} | "
                    f"{cell['compression_mean']:+.3f} | {round(50*cell['temporal_strict_correct_to_incorrect_mean'])}/50 |")
    text += ["", "Margin = first-answer-token correct-option logit minus incorrect-option logit.",
        "Compression = low margin change minus temporal margin change. Positive compression need not mean temporal damage.",
        "Intervals are diagnostic 95% case-bootstrap intervals; 50 bases, not 100 independent conditions.",
        "Singleton and joint interventions remove different edge budgets; no additive or budget-matched claim is made.",
        "Weak singleton effects with a strong joint effect could reflect accumulation, redundancy, nonlinear interaction or compensation.",
        "This does not identify a unique temporal-relation circuit, mediation or an L16-only bottleneck.",
        "No leave-one-layer-out experiment is automatically triggered.", ""]
    (output / "report.md").write_text("\n".join(text), encoding="utf-8")
    if plots:
        plot(cells, output / "single_layer_vs_joint.png")
    summary = {"schema": SCHEMA, "artifact_type": "real", "complete": True,
        "plan_fingerprint": context[0]["fingerprint"], "base_count": 50, "new_condition_rows": 400,
        "baseline_gate_rows": 100, "joint_reference_condition_rows": 100, "paired_rows_with_reference": 250,
        "missing_count": 0, "source_plan_fingerprint": context[0]["source_plan_fingerprint"],
        "analysis_status": ANALYSIS_STATUS, "layer_selection_rationale": LAYER_SELECTION,
        "output_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "aggregate_summary.json"}}
    atomic_write(output / "aggregate_summary.json", summary)
    return summary


def plot(cells, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = sorted(cells, key=lambda r: (r["layer_count"], r["window_start"]))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), constrained_layout=True)
    for ax, metric, title in zip(axes, METRICS[:3], ("Temporal margin change", "Low margin change", "Boundary-gap compression")):
        for i, cell in enumerate(cells):
            mean = cell[metric+"_mean"]
            ax.errorbar(i, mean, yerr=[[max(0, mean-cell[metric+"_ci_low"])],
                                     [max(0, cell[metric+"_ci_high"]-mean)]],
                fmt="o" if i < 4 else "D", color="#127F87" if i < 4 else "#B73E64", capsize=4)
        ax.axhline(0, color="#555555", linewidth=.8)
        ax.set(title=title, xticks=range(5), xticklabels=["L16", "L17", "L18", "L19", "L16-19\njoint ref."],
               ylabel="Correct-option logit-margin change")
    fig.suptitle("Exploratory single-layer attention knockout | 50 paired rescue bases\nOptions -> whole matched Event 2")
    fig.supxlabel("Layers selected from the observed L16-19 peak on these same cases; diagnostic 95% case-bootstrap intervals.\n"
                  "Joint reference cuts four times as many layer-edges; not an additive decomposition.", fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)
