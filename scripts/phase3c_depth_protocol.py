"""Append-only every-four-layer expansion; keep the running two-layer prefix intact."""

import copy
from pathlib import Path
from types import SimpleNamespace

from . import phase3c_scale as prefix
from . import run_phase3c_scale as prefix_runner
from . import phase3c_knockout25 as knockout
from .phase3c_core import digest, file_hash, frozen_write, read_json

SCHEMA = "phase3c_whole_event2_depth50_v1"
CONDITIONS = prefix.CONDITIONS
GRID = tuple(range(0, 36, 4))
LAYERS = tuple(layer for layer in GRID if layer not in prefix.LAYERS)
DOC = Path(__file__).resolve().parents[1]/"docs/phase3c_depth50.md"


def code_hashes():
    root = Path(__file__).parent
    return {name: file_hash(root/name) for name in
            ("phase3c_depth_protocol.py", "depth50/run_phase3c_scale.py", "plot_phase3c_depth.py", "phase3c_knockout25.py")}


def tasks(plan, stage):
    if stage not in ("baseline", "identity", "patch"):
        raise ValueError("Unknown depth stage.")
    keys = plan["representatives"] if stage == "identity" else plan["case_ids"]
    result = []
    for key in keys:
        for condition in CONDITIONS:
            for layer in (None,) if stage == "baseline" else LAYERS:
                item = {"pair_id": key, "condition": condition, "kind": stage,
                        "support": "whole_event2", "location": "block_output", "layer": layer}
                item["task_id"] = digest([plan["fingerprint"], item])
                result.append(item)
    return result


source_replay_matches = prefix.source_replay_matches


def pending(args, freeze=False):
    root, source = Path(args.output_root).resolve(), Path(args.prefix_run).resolve()
    for path in (source, Path(args.source_run), Path(args.pilot_plan), Path(args.pilot_run), Path(args.replication_run)):
        path = path.resolve()
        if root == path or root in path.parents or path in root.parents:
            raise ValueError("Use an isolated sibling depth root; no existing artifacts may be overwritten.")
    old, pairs, _, _, _ = prefix.load(source)
    value = {"schema": SCHEMA, "artifact_type": "real", "prefix_root": str(source),
        "prefix_plan_sha256": file_hash(source/"plan.json"), "prefix_fingerprint": old["fingerprint"],
        "case_ids": old["case_ids"], "base_ids": old["base_ids"], "layers": list(GRID), "extra_layers": list(LAYERS),
        "primary_layer": 16, "patch_rows_total": 900, "new_patch_rows": 700, "reuse_patch_rows": 200,
        "scope": "Single block-output replacement every four layers, not simultaneous four-layer-window patching.",
        "layer_0_semantics": "Before recipient DeepStack reinjection; report separately, not post-DeepStack replacement.",
        "cohort_rule": "Keep the same 50 bases and prompts; no patch outcome may exclude/reselect a case.",
        "knockout": knockout.intent(old, pairs, args.pilot_run, not args.roi_only),
        "code_sha256": code_hashes(), "protocol_sha256": file_hash(DOC)}
    value["fingerprint"] = digest(value)
    path = root/"pending_plan.json"
    if freeze:
        frozen_write(path, value)
    if not path.exists() or read_json(path) != value:
        raise ValueError("Pending protocol binding changed; preserve the existing depth root.")
    return value


def storage_estimate():
    new_bytes = 68 * (len(LAYERS)*512*4096*2 + 2*1024**2)
    prefix_compact_bytes = 68 * (2*512*4096*2 + 2*1024**2)
    return {"new_capture_upper_bytes": new_bytes, "prefix_compact_backup_upper_bytes": prefix_compact_bytes,
        "safety_reserve_bytes": 12*1024**3,
        "required_free_bytes": int(new_bytes*2.3 + prefix_compact_bytes*1.2) + 12*1024**3}


def freeze_complete(args):
    intent = pending(args)
    source, root = Path(args.prefix_run), Path(args.output_root)
    context = prefix.load(source)
    prefix_runner.verify_analysis(SimpleNamespace(output_root=str(source)), context)
    old, pairs, mappings, _, _ = context
    baselines = prefix_runner.baseline_lookup(SimpleNamespace(output_root=str(source)), context)
    refs = {key: {"row": row, "index": read_json(row["capture_index_path"])} for key, row in baselines.items()}
    previous = read_json(Path(args.pilot_run)/"analysis/case_patch_effects.json")
    prior = [{"key": [row["pair_id"], row["direction"], row["layer"]], "effect": row,
              "source_table_path": str(Path(args.pilot_run)/"analysis/case_patch_effects.json"), "source_row_sha256": digest(row)}
             for row in previous if row["analysis_stratum"] == "rescue" and row["support"] == "whole_event2" and
             row["layer"] in LAYERS and row["location"] == "block_output"]
    if len(prior) != 64:
        raise ValueError("Expected 64 earlier pilot references for the additional layers.")
    evidence = {}
    for stage in ("baseline", "identity", "patch"):
        for name in ("summary.json", "rows.jsonl", "task_manifest.jsonl"):
            path = source/stage/name
            evidence[str(path.resolve())] = file_hash(path)
    output = source/"analysis"
    summary = read_json(output/"aggregate_summary.json")
    evidence[str(output/"aggregate_summary.json")] = file_hash(output/"aggregate_summary.json")
    evidence.update({str(output/name): sha for name, sha in summary["output_sha256"].items()})
    plan = {**copy.deepcopy(old), "schema": SCHEMA, "depth_intent_fingerprint": intent["fingerprint"],
        "layers": list(GRID), "extra_layers": list(LAYERS), "prefix_fingerprint": old["fingerprint"],
        "prefix_evidence_sha256": evidence, "code_sha256": code_hashes(), "depth_protocol_sha256": file_hash(DOC),
        "capture_scope": "68 additional seven-site captures plus 68 existing two-site stores; prior full-layer stores remain read-only.",
        "source_baseline_record_sha256": {key:row["record_sha256"] for key,row in baselines.items()},
        "refs_sha256": digest(refs), "prior_sha256": digest(prior), "mappings_sha256": digest(mappings)}
    plan.pop("fingerprint")
    plan["knockout"] = intent["knockout"]
    plan["pairs_sha256"] = digest(pairs)
    # Keep the stimulus annotations unchanged; the new result envelope binds the depth protocol.
    plan["fingerprint"] = digest({k: v for k, v in plan.items() if k != "fingerprint"})
    frozen_write(root/"plan.json", plan)
    for name, value in (("pairs", pairs), ("mappings", mappings), ("source_capture_refs", refs), ("prior_effects", prior)):
        frozen_write(root/f"{name}.json", value)
    return load(args)


def load(args):
    intent = pending(args)
    root = Path(args.output_root)
    plan = read_json(root/"plan.json")
    if (plan["fingerprint"] != digest({k: v for k, v in plan.items() if k != "fingerprint"}) or
            plan["depth_intent_fingerprint"] != intent["fingerprint"] or plan["code_sha256"] != code_hashes() or
            plan["depth_protocol_sha256"] != file_hash(DOC)):
        raise ValueError("Frozen depth configuration/code changed.")
    for path, sha in plan["prefix_evidence_sha256"].items():
        if file_hash(path) != sha:
            raise ValueError("Verified prefix evidence changed.")
    pairs, mappings, refs, prior = [read_json(root/f"{name}.json") for name in ("pairs", "mappings", "source_capture_refs", "prior_effects")]
    for value, name in ((pairs, "pairs_sha256"), (mappings, "mappings_sha256"), (refs, "refs_sha256"), (prior, "prior_sha256")):
        if digest(value) != plan[name]:
            raise ValueError("Depth manifest/reference bytes changed.")
    return plan, pairs, mappings, refs, {tuple(row["key"]): row for row in prior}
