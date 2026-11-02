"""Layout-symmetric L16 patch/KO, equal-budget KO controls and L16 patch x later KO."""

import json
from pathlib import Path

from . import phase3d_execution as execution
from . import phase3d_single_layer_ko as singleton
from .phase3c_core import CONDITIONS, atomic_write, digest, edge_budget, file_hash, frozen_write, read_json
from .phase3c_interventions import AttentionKnockout
from .run_phase3c_preflight import logits_parity, save_tensors

SCHEMA = "phase3f_layout_controls_v1"
ARCHIVED, PERMUTED = execution.VIEWS
LOW, TEMPORAL = CONDITIONS
JOINT, LATE = [16, 17, 18, 19], list(range(17, 24))
# (kind, views, conditions, route, windows); 2,000 interventional forwards over 50 bases.
GRID = (
    ("patch", (PERMUTED,), CONDITIONS, None, [[]]),
    ("ko", (PERMUTED,), CONDITIONS, "event2", [[16], [17], [18], [19], JOINT]),
    ("ko", (ARCHIVED, PERMUTED), CONDITIONS, "event1", [[18], JOINT]),
    ("ko", (ARCHIVED, PERMUTED), CONDITIONS, "random", [[18], JOINT]),
    ("ko", (ARCHIVED, PERMUTED), (LOW,), "event2", [[17, 18], LATE]),
    ("patch_ko", (ARCHIVED, PERMUTED), (LOW,), "event2", [[17], [18], [17, 18], LATE]),
)
EXPECTED_FORWARDS = 2000
RANDOM_SEED = 42
PATCH = {LOW: "temporal_to_low_L16", TEMPORAL: "low_to_temporal_L16"}


def code_hashes():
    root = Path(__file__).resolve().parents[1]
    return {**singleton.code_hashes(), **{name: file_hash(Path(__file__).with_name(name)) for name in
        ("phase3f_layout_controls.py", "run_phase3f_layout_controls.py")},
        "phase3f_protocol": file_hash(root / "docs/phase3f_layout_controls.md")}


def event_positions(row, record, which):
    """Positions of sampled-frame groups dominated by one target event (inclusive end, > 0.5)."""
    timing = row["event_timing"]
    start, end = timing[f"{which}_event_start_frame"], timing[f"{which}_event_end_frame"]
    temporal, height, width = record["video_metadata"]["merged_video_grid_thw"]
    frames, positions = record["video_metadata"]["source_frame_groups"], record["visual_positions"]
    if len(frames) != temporal or len(positions) != temporal*height*width or not end > start:
        raise ValueError("Visual grid, frame groups and event interval disagree.")
    cells = []
    for index, group in enumerate(frames):
        if sum(start <= frame <= end for frame in group)/len(group) > 0.5:
            block = positions[index*height*width:(index+1)*height*width]
            cells += [(position, (index, i // width, i % width)) for i, position in enumerate(block)]
    return cells


def route_keys(pair_id, row, record, support, condition):
    """Event 2 (frozen support), Event 1 and a fixed random control of equal size."""
    event2 = support["whole_event2"]["low_positions" if condition == LOW else "temporal_positions"]
    derived2 = [p for p, _ in event_positions(row, record, "second")]
    event1_cells = event_positions(row, record, "first")
    event1 = [p for p, _ in event1_cells]
    if sorted(derived2) != sorted(event2) or len(event1) != len(event2) or set(event1) & set(event2):
        raise ValueError("Event 1 must be disjoint from and as large as the frozen Event-2 support.")
    excluded = set(event1) | set(event2)
    temporal, height, width = record["video_metadata"]["merged_video_grid_thw"]
    pool = [(p, (i // (height*width), (i % (height*width)) // width, i % width))
            for i, p in enumerate(record["visual_positions"]) if p not in excluded]
    ranked = sorted(pool, key=lambda item: digest([RANDOM_SEED, pair_id, condition, list(item[1])]))
    return {"event2": list(event2), "event1": sorted(event1), "random": sorted(p for p, _ in ranked[:len(event2)]),
            "event1_frame_groups": sorted({cell[0] for _, cell in event1_cells})}


def routes(pairs, mappings, audits):
    """Queries from each view's own prompt; all three key routes share one causal budget."""
    result = {}
    for pair_id, mapping in mappings.items():
        result[pair_id] = {}
        for view in execution.VIEWS:
            result[pair_id][view] = {}
            for condition in CONDITIONS:
                record = audits[pair_id][view][condition]["processor_record"]
                keys = route_keys(pair_id, pairs[pair_id][condition], record, mapping["support_audit"]["supports"], condition)
                queries = record["group_positions"]["options_all"]
                budgets = {name: edge_budget(queries, keys[name], len(record["input_ids"]))
                           for name in ("event2", "event1", "random")}
                if len({b["all_head_visible_causal_edges_per_layer"] for b in budgets.values()}) != 1:
                    raise ValueError("Control routes must remove exactly the Event-2 edge budget.")
                result[pair_id][view][condition] = {"queries": queries, **keys, "budgets": budgets,
                                                    "length": len(record["input_ids"])}
        if result[pair_id][ARCHIVED][LOW]["event1"] != result[pair_id][ARCHIVED][TEMPORAL]["event1"]:
            raise ValueError("Event 1 precedes the boundary and must be identical in Low and Temporal.")
    return result


def specs(plan, stage="intervention", pair_ids=None):
    if stage not in ("gate", "intervention"):
        raise ValueError("Unknown Phase 3F stage.")
    ids = plan["case_ids"] if pair_ids is None else pair_ids
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(plan["case_ids"]):
        raise ValueError("Unknown, duplicated or empty case subset.")
    result = []
    for pair_id in ids:
        for view in execution.VIEWS:
            for condition in CONDITIONS:
                items = ([{"kind": "gate", "route": None, "window": []}] if stage == "gate" else
                         [{"kind": kind, "route": route, "window": window}
                          for kind, views, conditions, route, windows in GRID
                          if view in views and condition in conditions for window in windows])
                for item in items:
                    item = {"stage": stage, "pair_id": pair_id, "view": view, "condition": condition, **item,
                            "patch": PATCH[condition] if item["kind"] in ("patch", "patch_ko") else None}
                    item["task_id"] = digest([plan["fingerprint"], item])
                    result.append(item)
    return result


def source_snapshot(source, singleton_root):
    """Archived evidence via the singleton audit; permuted captures and references by hash."""
    source, singleton_root = Path(source), Path(singleton_root)
    context, files, archived, logits = singleton.source_snapshot(source)
    plan, pairs, mappings = context
    records, tensors, references = {}, {}, {"patch": {}, "event2_ko": {}}
    summary = read_json(singleton_root / "analysis/aggregate_summary.json")
    decomposition = singleton_root / "analysis/case_decomposition.json"
    if (summary.get("complete") is not True or summary.get("source_plan_fingerprint") != plan["fingerprint"] or
            summary["output_sha256"].get("case_decomposition.json") != file_hash(decomposition)):
        raise ValueError("A complete singleton KO run over the same Phase 3D source is required.")
    files["singleton:analysis/aggregate_summary.json"] = file_hash(singleton_root / "analysis/aggregate_summary.json")
    files["singleton:analysis/case_decomposition.json"] = file_hash(decomposition)
    reported = read_json(decomposition)
    for pair_id in plan["case_ids"]:
        records[pair_id], tensors[pair_id] = {ARCHIVED: {}, PERMUTED: {}}, {ARCHIVED: {}, PERMUTED: {}}
        for condition in CONDITIONS:
            base = archived[pair_id][condition]
            records[pair_id][ARCHIVED][condition] = {"baseline": base["baseline"], "audit": base["audit"]}
            tensors[pair_id][ARCHIVED][condition] = logits[pair_id][condition]
            capture, audit = (next(execution.saved_checkpoint(source, plan, s)
                for s in execution.feature_tasks(plan, stage, [pair_id])
                if s["condition"] == condition and s["view"] == PERMUTED) for stage in ("capture", "audit"))
            if capture is None or audit is None:
                raise ValueError("Permuted capture/audit checkpoint missing in the frozen source.")
            data = execution.payload(source, capture)
            if (capture["audit_record_sha256"] != audit["record_sha256"] or
                    capture["input_fingerprint"] != audit["input_fingerprint"] or
                    not capture["gates"]["capture_noop"]["passed"] or not capture["gates"]["standard"]["passed"] or
                    "timing" not in data):
                raise ValueError("Permuted source capture lacks exact no-op/standard parity or timing.")
            singleton.check_logits(data["logits"], capture["decision"])
            if capture["decision"]["correct_option"] == base["baseline"]["decision"]["correct_option"]:
                raise ValueError("Permuted view must reverse the correct letter.")
            records[pair_id][PERMUTED][condition] = {"baseline": capture, "audit": audit}
            tensors[pair_id][PERMUTED][condition] = data["logits"].clone()
            for checkpoint in (capture, audit):
                path = execution.checkpoint_path(source, checkpoint["spec"])
                files[str(path.relative_to(source))] = file_hash(path)
            files[capture["payload"]["path"]] = capture["payload"]["sha256"]
            references["event2_ko"].setdefault(pair_id, {})[condition] = {"16-19": base["joint"]["decision"]}
            del data
        for spec in execution.feature_tasks(plan, "patch", [pair_id]):
            row = execution.saved_checkpoint(source, plan, spec)
            if row is None:
                raise ValueError("Archived L16 patch checkpoint missing.")
            singleton.check_logits(execution.payload(source, row)["logits"], row["decision"])
            references["patch"].setdefault(pair_id, {})[spec["condition"]] = row["decision"]
            path = execution.checkpoint_path(source, spec)
            files[str(path.relative_to(source))] = file_hash(path)
        rows = [r for r in reported if r["pair_id"] == pair_id and r["measurement"] == "new_single_layer"]
        if sorted(r["window_start"] for r in rows) != [16, 17, 18, 19]:
            raise ValueError("Singleton reference must contain L16-L19 for every base.")
        for row in rows:
            if (row["low_base_margin"] != archived[pair_id][LOW]["baseline"]["decision"]["margin"] or
                    row["temporal_base_margin"] != archived[pair_id][TEMPORAL]["baseline"]["decision"]["margin"]):
                raise ValueError("Singleton reference is bound to different baselines.")
            for condition, side in ((LOW, "low"), (TEMPORAL, "temporal")):
                references["event2_ko"][pair_id][condition][str(row["window_start"])] = {
                    "margin": row[f"{side}_ko_margin"], "prediction": row[f"{side}_ko_prediction"]}
    del logits
    return context, files, records, tensors, references


def freeze_plan(source, singleton_root, output):
    source, singleton_root, output = (Path(p).resolve() for p in (source, singleton_root, output))
    if any(output == p or p in output.parents or output in p.parents for p in (source, singleton_root)):
        raise ValueError("Use a separate new output root; source evidence is read-only.")
    if (output / "plan.json").exists():
        return load_context(output)
    if output.exists() and any(p.name not in {".phase3d.lock"} for p in output.iterdir()):
        raise ValueError("Preparation requires an empty output directory.")
    context, files, records, tensors, references = source_snapshot(source, singleton_root)
    source_plan, pairs, mappings = context
    audits = {p: {v: {c: records[p][v][c]["audit"] for c in CONDITIONS} for v in execution.VIEWS} for p in records}
    for pair_id in records:
        for condition in CONDITIONS:
            old, new = (audits[pair_id][v][condition]["processor_record"] for v in execution.VIEWS)
            if old["visual_positions"] != new["visual_positions"] or old["input_ids"] == new["input_ids"]:
                raise ValueError("Views must share visual positions and differ in option text.")
            if old["input_ids"][:old["visual_positions"][-1]+1] != new["input_ids"][:new["visual_positions"][-1]+1]:
                raise ValueError("Text before or inside the video differs between views.")
    frozen_routes = routes(pairs, mappings, audits)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("source_records", records), ("pairs", pairs), ("mappings", mappings),
                        ("routes", frozen_routes), ("references", references), ("source_plan", source_plan),
                        ("source_runtime", read_json(source / "runtime.json"))):
        frozen_write(output / f"{name}.json", value)
    save_tensors(output / "source_logits.pt", tensors)
    plan = {"schema": SCHEMA, "artifact_type": "real", "source_root": str(source),
        "singleton_root": str(singleton_root), "source_plan_fingerprint": source_plan["fingerprint"],
        "source_files": files, "case_ids": source_plan["case_ids"], "preflight_ids": source_plan["preflight_ids"],
        "settings": source_plan["settings"], "source_runtime": read_json(source / "runtime.json")["runtime"],
        "project_root": source_plan["project_root"], "source_path_map": source_plan["source_path_map"],
        "code_sha256": code_hashes(), "grid": json.loads(json.dumps(GRID)),
        "new_interventional_forwards": EXPECTED_FORWARDS, "gate_rows": 200, "random_seed": RANDOM_SEED,
        "snapshot_sha256": {name: file_hash(output / name) for name in
            ("source_records.json", "pairs.json", "mappings.json", "routes.json", "references.json",
             "source_plan.json", "source_runtime.json", "source_logits.pt")},
        "case_reporting": "All 50 original bases equally; no pilot/new stratification.",
        "interpretation": "Layout-symmetric effects and route controls; not a unique circuit or mediation."}
    plan["fingerprint"] = digest(plan)
    frozen_write(output / "plan.json", plan)
    return plan, pairs, mappings


def load_context(root):
    root = Path(root)
    plan = read_json(root / "plan.json")
    if (plan.get("schema") != SCHEMA or plan.get("artifact_type") != "real" or
            plan.get("fingerprint") != digest({k: v for k, v in plan.items() if k != "fingerprint"}) or
            plan["code_sha256"] != code_hashes() or plan["new_interventional_forwards"] != EXPECTED_FORWARDS):
        raise ValueError("Phase 3F plan/code/protocol changed; use a new root.")
    source_context = execution.load_context(plan["source_root"])
    if source_context[0]["fingerprint"] != plan["source_plan_fingerprint"]:
        raise ValueError("Frozen source plan changed.")
    for name, sha in plan["snapshot_sha256"].items():
        path = root / name
        if path.is_symlink() or file_hash(path) != sha:
            raise ValueError("Frozen Phase 3F snapshot changed.")
    for name, sha in plan["source_files"].items():
        base, relative = ((Path(plan["singleton_root"]), name.split(":", 1)[1]) if name.startswith("singleton:")
                          else (Path(plan["source_root"]), name))
        path = base / relative
        if not path.resolve().is_relative_to(base.resolve()) or path.is_symlink() or file_hash(path) != sha:
            raise ValueError("Frozen source evidence changed.")
    pairs, mappings = read_json(root / "pairs.json"), read_json(root / "mappings.json")
    if pairs != source_context[1] or mappings != source_context[2]:
        raise ValueError("Cases, prompts or support changed.")
    return plan, pairs, mappings


def route_positions(context_routes, spec):
    item = context_routes[spec["pair_id"]][spec["view"]][spec["condition"]]
    return item["queries"], item[spec["route"]], item["budgets"][spec["route"]], item["length"]


class Executor:
    def __init__(self, root, context, engine):
        import torch
        self.root, self.context, self.engine = Path(root), context, engine
        self.records = read_json(self.root / "source_records.json")
        self.routes = read_json(self.root / "routes.json")
        self.logits = torch.load(self.root / "source_logits.pt", map_location="cpu", weights_only=True)
        self.source = context[0]["source_root"]
        self.source_context = execution.load_context(self.source)
        self.cache, self.captures = {}, {}

    def capture(self, pair_id, condition, view):
        """Source capture payload (timing; archived also carries the L16 Event-2 donor)."""
        key = pair_id, condition, view
        if key not in self.captures:
            if len(self.captures) > 4:
                self.captures.clear()
            row = self.records[pair_id][view][condition]["baseline"]
            self.captures[key] = execution.payload(self.source, row)
        return self.captures[key]

    def live(self, spec):
        key = spec["pair_id"], spec["condition"], spec["view"]
        if key not in self.cache:
            self.cache.clear()
            self.engine.torch.cuda.empty_cache()
            live, audit = execution.audited_live(self.source, self.source_context, self.engine, spec)
            source = self.records[key[0]][key[2]][key[1]]
            measured = self.engine.forward(live)
            if (audit != source["audit"] or
                    not logits_parity(self.logits[key[0]][key[2]][key[1]], measured, exact=True)["passed"] or
                    self.engine.decide(measured, live["row"]) != source["baseline"]["decision"]):
                raise RuntimeError("Exact source input/runtime/full-logit baseline parity failed.")
            self.cache[key] = live, audit, source["baseline"], measured
        return self.cache[key]

    def knockout(self, spec, enabled=True, route=None, window=None):
        queries, keys, budget, length = route_positions(self.routes, {**spec, "route": route or spec["route"]})
        return AttentionKnockout(self.engine.text.layers, window or spec["window"], queries, keys, length,
                                 enabled=enabled), {"query_positions": queries, "key_positions": keys,
                                                    "edge_budget": budget}

    def patch(self, spec, audit):
        from .phase3d_features import mapped_l16_patch
        opposite = TEMPORAL if spec["condition"] == LOW else LOW
        donor = self.capture(spec["pair_id"], opposite, ARCHIVED)["donor16"]
        support = self.context[2][spec["pair_id"]]["support_audit"]["supports"]["whole_event2"]
        recipient = "low_positions" if spec["condition"] == LOW else "temporal_positions"
        donor_side = "temporal_positions" if spec["condition"] == LOW else "low_positions"
        return mapped_l16_patch(donor, support[donor_side], support[recipient], audit["groups"]["whole_event2"])

    def __call__(self, spec):
        from contextlib import ExitStack
        print(f"[Phase 3F] {spec['kind']} {spec['route']} {spec['window']}: {spec['pair_id']} "
              f"{spec['condition']} {spec['view']}", flush=True)
        live, audit, baseline, before = self.live(spec)
        torch = self.engine.torch
        result = {"baseline_record_sha256": baseline["record_sha256"], "audit_record_sha256": audit["record_sha256"],
                  "input_fingerprint": audit["input_fingerprint"]}
        if spec["kind"] == "gate":
            controller, extra = self.knockout(spec, enabled=False, route="event2", window=JOINT)
            with controller.installed():
                disabled = self.engine.forward(live)
            sites, observed = execution.observe(self.engine, live, audit)
            reference = self.capture(spec["pair_id"], spec["condition"], spec["view"])
            donor = self.capture(spec["pair_id"], spec["condition"], ARCHIVED)["donor16"]
            checks = {"disabled_ko": logits_parity(before, disabled, exact=True),
                      "observe_noop": logits_parity(before, observed["logits"], exact=True),
                      "timing_exact": observed["timing"] == reference["timing"],
                      "event2_l16_equals_archived": sites.donor16["positions"] == donor["positions"] and
                          torch.equal(sites.donor16["vectors"].cpu(), donor["vectors"].cpu())}
            if (not checks["disabled_ko"]["passed"] or not checks["observe_noop"]["passed"] or
                    not checks["timing_exact"] or not checks["event2_l16_equals_archived"]):
                raise RuntimeError("Replay, disabled-mask, timing or cross-view L16 state gate failed.")
            result.update(decision=self.engine.decide(before, live["row"]), gates=checks,
                          mask_audit=controller.validate(), **extra)
            tensors = {"logits": before}
        else:
            gate_spec = next(s for s in specs(self.context[0], "gate", [spec["pair_id"]])
                             if s["condition"] == spec["condition"] and s["view"] == spec["view"])
            gate = execution.saved_checkpoint(self.root, self.context[0], gate_spec)
            if gate is None:
                raise ValueError("Gate must pass before any Phase 3F intervention.")
            result["gate_record_sha256"] = gate["record_sha256"]
            with ExitStack() as stack:
                controller = None
                if spec["kind"] in ("ko", "patch_ko"):
                    controller, extra = self.knockout(spec)
                    stack.enter_context(controller.installed())
                    result.update(extra)
                if spec["kind"] == "ko":
                    logits = self.engine.forward(live)
                else:
                    patch = self.patch(spec, audit)
                    sites, observed = execution.observe(self.engine, live, audit, patch)
                    logits = observed["logits"]
            if controller is not None:
                result["mask_audit"] = controller.validate()
            if spec["kind"] != "ko":
                reference = self.capture(spec["pair_id"], spec["condition"], spec["view"])
                lookup = {p: i for i, p in enumerate(patch["recipient_positions"])}
                expected = patch["donor"]["vectors"][[lookup[p] for p in sites.donor16["positions"]]]
                if observed["timing"] != reference["timing"] or not torch.equal(expected.cpu(), sites.donor16["vectors"].cpu()):
                    raise RuntimeError("Patch changed L16 timing or did not fully replace the recipient support.")
                result.update(same_block_timing_exact=True, observed_donor_replacement_exact=True,
                              recipient_positions=patch["recipient_positions"],
                              donor_positions=patch["donor"]["positions"], hook_audit=sites.validate())
            result["decision"] = self.engine.decide(logits, live["row"])
            tensors = {"logits": logits}
        singleton.check_logits(tensors["logits"], result["decision"])
        return result, tensors


def verify_rows(root, context, pair_ids=None, include_interventions=True):
    import torch
    root, plan = Path(root), context[0]
    from .phase3c_budget import runtime_compatibility
    runtime = read_json(root / "runtime.json")
    if runtime.get("plan_fingerprint") != plan["fingerprint"]:
        raise ValueError("Missing or foreign live-runtime proof.")
    runtime_compatibility(plan["source_runtime"], runtime["runtime"])
    records, frozen = read_json(root / "source_records.json"), read_json(root / "routes.json")
    original = torch.load(root / "source_logits.pt", map_location="cpu", weights_only=True)
    requested = specs(plan, "gate", pair_ids) + (specs(plan, "intervention", pair_ids) if include_interventions else [])
    gates, rows = {}, []
    for spec in requested:
        row = execution.saved_checkpoint(root, plan, spec)
        if row is None:
            raise ValueError("Incomplete Phase 3F grid; partial rows are not a result.")
        data = execution.payload(root, row)
        baseline = records[spec["pair_id"]][spec["view"]][spec["condition"]]["baseline"]
        singleton.check_logits(data["logits"], row["decision"])
        if (row["baseline_record_sha256"] != baseline["record_sha256"] or
                row["input_fingerprint"] != baseline["input_fingerprint"] or
                any(row["decision"][k] != baseline["decision"][k] for k in
                    ("correct_option", "correct_token_id", "incorrect_token_id"))):
            raise ValueError("Row is bound to a different baseline/input.")
        if spec["kind"] == "gate":
            if (not torch.equal(data["logits"], original[spec["pair_id"]][spec["view"]][spec["condition"]]) or
                    row["decision"] != baseline["decision"] or row["mask_audit"].get("enabled") is not False or
                    not all(row["gates"][k]["passed"] for k in ("disabled_ko", "observe_noop")) or
                    row["gates"]["timing_exact"] is not True or row["gates"]["event2_l16_equals_archived"] is not True):
                raise ValueError("Gate evidence failed independent verification.")
            gates[(spec["pair_id"], spec["view"], spec["condition"])] = row
        else:
            if row["gate_record_sha256"] != gates[(spec["pair_id"], spec["view"], spec["condition"])]["record_sha256"]:
                raise ValueError("Intervention is bound to a different gate.")
            if spec["kind"] in ("ko", "patch_ko"):
                queries, keys, budget, _ = route_positions(frozen, spec)
                audit = row["mask_audit"]
                if (row["query_positions"] != queries or row["key_positions"] != keys or row["edge_budget"] != budget or
                        audit.get("enabled") is not True or audit.get("all_heads") != 32 or
                        audit.get("pre_softmax_mask") != "negative_infinity" or audit.get("renormalized") is not True or
                        set(audit["layers"]) != {str(layer) for layer in spec["window"]} or
                        any(v["budget"] != budget or v["max_blocked_probability"] != 0 or
                            not 0 <= v["max_row_sum_error"] <= .005 for v in audit["layers"].values())):
                    raise ValueError("Saved KO mask/budget does not match the frozen route.")
            if spec["kind"] in ("patch", "patch_ko") and (row.get("same_block_timing_exact") is not True or
                    row.get("observed_donor_replacement_exact") is not True):
                raise ValueError("Patch evidence is incomplete.")
        rows.append(row)
    return rows


# ---------------------------------------------------------------- analysis

def bootstrap(values, seed=0, draws=5000):
    import numpy as np
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (draws, len(values)))].mean(1)
    return {"mean": float(values.mean()), "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)), "n": int(len(values))}


def margin_table(root, context, rows):
    """margin[(pair, view, condition, kind, route, window)] from new rows, references and baselines."""
    records, refs = read_json(Path(root) / "source_records.json"), read_json(Path(root) / "references.json")
    table, correct = {}, {}
    for pair_id in context[0]["case_ids"]:
        for view in execution.VIEWS:
            for condition in CONDITIONS:
                decision = records[pair_id][view][condition]["baseline"]["decision"]
                table[(pair_id, view, condition, "base", None, "")] = decision["margin"]
                correct[(pair_id, view, condition)] = decision["correct_option"]
        for condition in CONDITIONS:
            table[(pair_id, ARCHIVED, condition, "patch", None, "")] = refs["patch"][pair_id][condition]["margin"]
            for window, decision in refs["event2_ko"][pair_id][condition].items():
                table[(pair_id, ARCHIVED, condition, "ko", "event2", window)] = decision["margin"]
    for row in rows:
        spec = row["spec"]
        if spec["kind"] == "gate":
            continue
        window = ("-".join(map(str, (spec["window"][0], spec["window"][-1]))) if len(spec["window"]) > 1 else
                  str(spec["window"][0]) if spec["window"] else "")
        key = (spec["pair_id"], spec["view"], spec["condition"], spec["kind"], spec["route"], window)
        if key in table:
            raise ValueError("Duplicate margin cell.")
        table[key] = row["decision"]["margin"]
    return table, correct


def layout_summary(ids, effect):
    """effect(pair, view) -> value; returns per-view, symmetric and B-minus-A summaries."""
    per = {view: [effect(p, view) for p in ids] for view in execution.VIEWS}
    return {"archived": bootstrap(per[ARCHIVED]), "permuted": bootstrap(per[PERMUTED]),
            "symmetric": bootstrap([(a+b)/2 for a, b in zip(per[ARCHIVED], per[PERMUTED])])}


def analyze(root, context):
    root = Path(root)
    rows = verify_rows(root, context)
    if len(rows) != EXPECTED_FORWARDS + 200:
        raise ValueError("Expected 2,000 interventions and 200 gates.")
    ids = context[0]["case_ids"]
    m, correct = margin_table(root, context, rows)
    base = lambda p, v, c: m[(p, v, c, "base", None, "")]
    result = {"patch": {}, "event2_ko": {}, "controls": {}, "patch_x_ko": {}}
    for condition in CONDITIONS:
        result["patch"][condition] = layout_summary(ids, lambda p, v: m[(p, v, condition, "patch", None, "")]-base(p, v, condition))
        result["patch"][condition]["strict_sign_crossings"] = {view: sum(
            base(p, view, condition)*m[(p, view, condition, "patch", None, "")] < 0 for p in ids) for view in execution.VIEWS}
    for window in ("16", "17", "18", "19", "16-19"):
        delta = {c: (lambda c: lambda p, v: m[(p, v, c, "ko", "event2", window)]-base(p, v, c))(c) for c in CONDITIONS}
        result["event2_ko"][window] = {"delta_low": layout_summary(ids, delta[LOW]),
            "delta_temporal": layout_summary(ids, delta[TEMPORAL]),
            "compression": layout_summary(ids, lambda p, v: delta[LOW](p, v)-delta[TEMPORAL](p, v))}
    for route in ("event1", "random"):
        for window in ("18", "16-19"):
            cell = {}
            for condition, name in ((LOW, "low"), (TEMPORAL, "temporal")):
                ko = lambda p, v, r=route, c=condition: m[(p, v, c, "ko", r, window)]-base(p, v, c)
                e2 = lambda p, v, c=condition: m[(p, v, c, "ko", "event2", window)]-base(p, v, c)
                cell[f"delta_{name}"] = layout_summary(ids, ko)
                cell[f"event2_minus_control_{name}"] = layout_summary(ids, lambda p, v, a=ko, b=e2: b(p, v)-a(p, v))
            cell["compression"] = layout_summary(ids, lambda p, v, r=route: (m[(p, v, LOW, "ko", r, window)]-base(p, v, LOW))
                                                 - (m[(p, v, TEMPORAL, "ko", r, window)]-base(p, v, TEMPORAL)))
            result["controls"][f"{route}:{window}"] = cell
    r0 = lambda p, v: m[(p, v, LOW, "patch", None, "")]-base(p, v, LOW)
    result["patch_x_ko"]["no_ko"] = layout_summary(ids, r0)
    for window in ("17", "18", "17-18", "17-23"):
        rw = lambda p, v, w=window: m[(p, v, LOW, "patch_ko", "event2", w)]-m[(p, v, LOW, "ko", "event2", w)]
        cell = {"rescue_under_ko": layout_summary(ids, rw), "ko_alone_delta": layout_summary(
            ids, lambda p, v, w=window: m[(p, v, LOW, "ko", "event2", w)]-base(p, v, LOW))}
        for view, name in ((ARCHIVED, "archived"), (PERMUTED, "permuted")):
            denominator = sum(r0(p, view) for p in ids)
            cell.setdefault("retained_fraction", {})[name] = (sum(rw(p, view) for p in ids)/denominator
                                                              if denominator else None)
        result["patch_x_ko"][window] = cell
    correct_a = {view: sum(correct[(p, view, LOW)] == "A" for p in ids) for view in execution.VIEWS}
    output = root / "analysis"
    frozen_write(output / "results.json", {**result, "correct_A_counts": correct_a})
    frozen_write(output / "margins.json", [{"pair_id": k[0], "view": k[1], "condition": k[2], "kind": k[3],
        "route": k[4], "window": k[5], "margin": v, "correct_option": correct[k[:3]]} for k, v in sorted(
            m.items(), key=lambda item: tuple(str(x) for x in item[0]))])
    (output / "report.md").write_text(report(result, correct_a), encoding="utf-8")
    summary = {"schema": SCHEMA, "artifact_type": "real", "complete": True, "plan_fingerprint": context[0]["fingerprint"],
        "base_count": 50, "interventional_rows": EXPECTED_FORWARDS, "gate_rows": 200, "missing_count": 0,
        "output_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "aggregate_summary.json"}}
    atomic_write(output / "aggregate_summary.json", summary)
    return summary


def fmt(cell):
    return f"{cell['mean']:+.3f} [{cell['ci_low']:+.3f}, {cell['ci_high']:+.3f}]"


def report(result, correct_a):
    text = ["# Phase 3F: Layout-Symmetric Patch/KO And Route Controls", "",
        f"Correct-A prompts per view (of 50): archived {correct_a[ARCHIVED]}, permuted {correct_a[PERMUTED]}.",
        "Values: mean correct-option logit-margin change, diagnostic 95% case bootstrap over 50 bases.",
        "Symmetric = mean of the two views per base, cancelling a constant A/B letter preference.", "",
        "## L16 whole-Event-2 patch", "", "| Recipient | Archived | Permuted | Symmetric | Strict crossings arch / perm |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for condition, cell in result["patch"].items():
        crossings = cell["strict_sign_crossings"]
        text.append(f"| {condition} | {fmt(cell['archived'])} | {fmt(cell['permuted'])} | {fmt(cell['symmetric'])} | "
                    f"{crossings[ARCHIVED]} / {crossings[PERMUTED]} |")
    text += ["", "## Options -> Event-2 KO", "", "| Window | Low change (sym.) | Temporal change (sym.) | Compression arch | Compression perm | Compression sym. |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for window, cell in result["event2_ko"].items():
        text.append(f"| {window} | {fmt(cell['delta_low']['symmetric'])} | {fmt(cell['delta_temporal']['symmetric'])} | "
                    f"{fmt(cell['compression']['archived'])} | {fmt(cell['compression']['permuted'])} | {fmt(cell['compression']['symmetric'])} |")
    text += ["", "## Equal-budget route controls (symmetric)", "",
             "| Route:window | Low change | Temporal change | Compression | Event2 minus control, Temporal |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, cell in result["controls"].items():
        text.append(f"| {name} | {fmt(cell['delta_low']['symmetric'])} | {fmt(cell['delta_temporal']['symmetric'])} | "
                    f"{fmt(cell['compression']['symmetric'])} | {fmt(cell['event2_minus_control_temporal']['symmetric'])} |")
    text += ["", "## Temporal->Low L16 patch x later options->Event-2 KO", "",
             f"Rescue without KO (symmetric): {fmt(result['patch_x_ko']['no_ko']['symmetric'])}", "",
             "| KO window | Rescue under KO (sym.) | KO alone on Low (sym.) | Retained arch | Retained perm |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for window in ("17", "18", "17-18", "17-23"):
        cell = result["patch_x_ko"][window]
        ret = cell["retained_fraction"]
        text.append(f"| {window} | {fmt(cell['rescue_under_ko']['symmetric'])} | {fmt(cell['ko_alone_delta']['symmetric'])} | "
                    f"{ret['archived']:.2f} | {ret['permuted']:.2f} |" if None not in ret.values() else
                    f"| {window} | {fmt(cell['rescue_under_ko']['symmetric'])} | {fmt(cell['ko_alone_delta']['symmetric'])} | n/a | n/a |")
    text += ["", "Retained = mean rescue under KO / mean rescue without KO; descriptive, not mediation.",
             "Blocking option->Event-2 edges does not block Event-2 information routed through other positions.",
             "Archived Event-2 KO singletons, the 16-19 joint row and archived patches are frozen Phase 3D references.", ""]
    return "\n".join(text)
