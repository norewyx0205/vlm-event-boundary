"""Phase 3H gated first-token forwards for the five timing cells (fresh and rescue cohorts); no KO, patch or probe.

Rescue native Low/Temporal prompts replay the frozen archived logits exactly (Phase 3G reference machinery).
Rendered cells use the frozen processor settings; within a base and layout every cell must share prompt token
IDs and visual positions, and the Event-1/Event-2 temporal-token bins must shift exactly as scheduled.
"""

import argparse
import copy
import gc
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scripts import phase3d_execution as source
from scripts import phase3h_timing_materials as materials
from scripts import run_phase3e_order_capture as order
from scripts import run_phase3g_alllayer_capture as capture
from scripts.phase3b_core import prepare_example
from scripts.phase3c_core import atomic_write, digest, file_hash, frozen_write, read_json
from scripts.run_phase3c_preflight import Engine, save_tensors
from scripts.run_phase3d_vm import lock

SCHEMA = "phase3h_timing_behaviour_v1"
COHORTS = ("fresh", "rescue")
FRESH_IDS = tuple(range(1001, 1051))
RESERVE = 12*1024**3


def code_hashes():
    root = Path(__file__).resolve().parents[1]
    return {**capture.code_hashes(), **{n: file_hash(Path(__file__).with_name(n)) for n in
        ("phase3h_timing_materials.py", "run_phase3h_timing_behaviour.py", "run_phase3h_timing_unattended.py")},
        "phase3h_protocol": file_hash(root/"docs/phase3h_timing_dissociation.md")}


def space(root):
    free = shutil.disk_usage(root).free
    if free < RESERVE+2*1024**3:
        raise RuntimeError("Need 12-GiB reserve plus 2-GiB run/backup budget; no data deleted.")
    return free


def check_plan(plan, cohort):
    if (plan.get("schema") != materials.SCHEMA or plan.get("cohort") != cohort or plan.get("complete") is not True or
            plan["fingerprint"] != digest({k: v for k, v in plan.items() if k != "fingerprint"}) or
            plan.get("outcome_screening") is not False or
            any(r.get("decoded_pixels_exact") is not True for r in plan["reconstruction_audits"]) or
            any(a.get("passed") is not True for a in plan["timing_audits"])):
        raise ValueError(f"{cohort} timing materials are incomplete, altered or failed an audit.")
    cells = materials.CELLS if cohort == "fresh" else materials.NEW_CELLS
    expected = len(plan["base_ids"])*len(cells)*len(materials.VIEWS)
    rendered = [r for r in plan["rows"] if r["timing_cell"] in cells]
    if len(rendered) != expected or len({r["row"]["eval_id"] for r in rendered}) != expected:
        raise ValueError(f"{cohort} materials lack the full cell x layout grid.")
    return rendered


def context(root):
    root = Path(root)
    config = read_json(root/"config.json")
    if config["schema"] != SCHEMA or config["code_sha256"] != code_hashes():
        raise ValueError("Phase 3H code/protocol changed; preserve this root and use a new one.")
    for name, sha in config["input_files"].items():
        path = root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or file_hash(path) != sha:
            raise ValueError("Copied Phase 3H materials changed.")
    order_context = order.context(Path(config["order_root"]))
    if digest(order_context[0]) != config["order_config_sha256"]:
        raise ValueError("Frozen v3 order configuration changed.")
    plans = {c: read_json(root/"materials"/c/"plan.json") for c in COHORTS}
    for cohort in COHORTS:
        check_plan(plans[cohort], cohort)
    return config, order_context, plans


def prepare(root, fresh_materials, rescue_materials, order_root):
    root, order_root = Path(root).resolve(), Path(order_root).resolve()
    inputs = {"fresh": Path(fresh_materials).resolve(), "rescue": Path(rescue_materials).resolve()}
    if any(root == p or root in p.parents or p in root.parents for p in (order_root, *inputs.values())):
        raise ValueError("Use a separate new Phase 3H run root; materials and v3 evidence are read-only.")
    if root.exists() and any(p.name != ".timing.lock" for p in root.iterdir()):
        raise ValueError("Preparation requires a new empty directory.")
    plans = {c: read_json(p/"plan.json") for c, p in inputs.items()}
    check_plan(plans["fresh"], "fresh")
    check_plan(plans["rescue"], "rescue")
    order_context = order.context(order_root)
    rescue_bases = sorted(rescue_base(order_context, pair) for pair in order_context[2][0]["case_ids"])
    if tuple(plans["fresh"]["base_ids"]) != FRESH_IDS or sorted(plans["rescue"]["base_ids"]) != rescue_bases:
        raise ValueError("Phase 3H cohorts differ from the frozen fresh IDs or the rescue50 cases.")
    root.mkdir(parents=True, exist_ok=True)
    space(root)
    for cohort, path in inputs.items():
        shutil.copytree(path, root/"materials"/cohort, ignore=shutil.ignore_patterns("*.lock", "reconstructions"))
    files = {p.relative_to(root).as_posix(): file_hash(p) for p in (root/"materials").rglob("*") if p.is_file()}
    config = {"schema": SCHEMA, "order_root": str(order_root), "order_config_sha256": digest(order_context[0]),
              "input_files": files, "settings": order_context[0]["settings"],
              "gpu_weight_budget_gib": order_context[0]["gpu_weight_budget_gib"], "code_sha256": code_hashes(),
              "materials_fingerprints": {c: plans[c]["fingerprint"] for c in COHORTS},
              "cells": list(materials.CELLS), "views": list(materials.VIEWS), "new_KO_or_patch": False,
              "storage_reserve_bytes": RESERVE}
    frozen_write(root/"config.json", config)
    context(root)
    return {"prepared": True, "prompt_cells": len(cells(*context(root)[1:])), "free_GiB": space(root)/1024**3}


def rescue_base(order_context, pair_id):
    """Base ID of a frozen rescue50 pair (Phase 3D pairs: order_context[2] = (plan, pairs, mappings))."""
    return int(order_context[2][1][pair_id]["low_boundary"]["base_sample_id"])


def cells(order_context, plans):
    """Every forward: rendered cells from both plans plus the rescue native prompts (exact replay)."""
    result = []
    for cohort in COHORTS:
        for spec in check_plan(plans[cohort], cohort):
            result.append({"cell_id": spec["row"]["eval_id"], "cohort": cohort, "kind": "rendered",
                           "base_sample_id": spec["base_sample_id"], "timing_cell": spec["timing_cell"],
                           "view": spec["view"], "pair_id": None})
    for cell in capture.cells(order_context):
        if cell["kind"] == "original":
            result.append({**cell, "cohort": "rescue", "kind": "original",
                           "base_sample_id": rescue_base(order_context, cell["pair_id"]),
                           "timing_cell": materials.NATIVE[cell["condition"]]})
    if len({c["cell_id"] for c in result}) != len(result):
        raise ValueError("Duplicate Phase 3H prompt cell.")
    return result


def selected(order_context, plans, preflight):
    if not preflight:
        return cells(order_context, plans)
    rescue = {rescue_base(order_context, pair) for pair in order_context[2][0]["preflight_ids"]}
    fresh_rows = {r["base_sample_id"]: r["row"]["first_object_id"] for r in plans["fresh"]["rows"]}
    fresh = {min(b for b, first in fresh_rows.items() if first == f) for f in (1, 2)}
    return [c for c in cells(order_context, plans)
            if c["base_sample_id"] in (rescue if c["cohort"] == "rescue" else fresh)]


def paths(root, cell):
    return Path(root)/"checkpoints"/(cell["cell_id"]+".json"), Path(root)/"logits"/(cell["cell_id"]+".pt")


def material_row(root, plans, cell):
    spec = next(r for r in plans[cell["cohort"]]["rows"] if r["row"]["eval_id"] == cell["cell_id"])
    row = copy.deepcopy(spec["row"])
    video = Path(root)/"materials"/cell["cohort"]/row["video_path"]
    if (Path(row["video_path"]).is_absolute() or not video.resolve().is_relative_to((Path(root)/"materials").resolve()) or
            video.is_symlink() or file_hash(video) != spec["video_sha256"]):
        raise ValueError("Phase 3H material path or video checksum changed.")
    row["video_path"] = str(video.resolve())
    return row


def processor_summary(prepared, record):
    bins = {event: sorted({c["temporal_index"] for name, values in prepared["cells"].items()
                           if name.startswith("video_") and name.endswith("_e"+event[-1]) for c in values})
            for event in ("event_1", "event_2")}
    return {"prompt_input_ids_sha256": record["prompt_input_ids_sha256"],
            "visual_positions_sha256": digest(record["visual_positions"]), "visual_token_count": len(record["visual_positions"]),
            "video_metadata": record["video_metadata"], "event_token_bins": bins,
            "input_tensor_sha256": record["input_tensor_sha256"]}


def run(root, preflight=False):
    from scripts.phase3c_budget import runtime_compatibility
    root = Path(root)
    config, order_context, plans = context(root)
    if not preflight and read_json(root/"preflight_summary.json") != verify(root, True):
        raise ValueError("Full Phase 3H run requires the same passed preflight.")
    engine = Engine(config["settings"], "model_parallel", config["gpu_weight_budget_gib"])
    reference_config = {"order_root": config["order_root"], "settings": config["settings"]}
    try:
        runtime_compatibility(order_context[1]["source_runtime"], engine.runtime)
        if (root/"runtime.json").exists():
            runtime_compatibility(read_json(root/"runtime.json"), engine.runtime)
        else:
            frozen_write(root/"runtime.json", engine.runtime)
        chosen = selected(order_context, plans, preflight)
        settings = config["settings"]
        for index, cell in enumerate(chosen):
            space(root)
            path, tensor = paths(root, cell)
            if path.exists():
                continue
            if tensor.exists():
                raise ValueError("Uncommitted logits exist; inspect before resuming.")
            if cell["kind"] == "original":
                reference = capture.references(reference_config, order_context, cell)
                live, row, record = capture.live_input(engine, reference_config, order_context, cell, reference)
            else:
                row = material_row(root, plans, cell)
                live = prepare_example(row, engine.processor, str(root), settings["video_fps"],
                    settings["video_num_frames"], settings["video_max_pixels"], settings["roi_padding"])
                record = source.processor_record(live)
                live["inputs"] = live["inputs"].to(engine.model.device)
            logits = engine.forward(live)
            exact = None
            if cell["kind"] == "original":
                exact = bool(torch.equal(logits, reference["logits"]))
                if not exact:
                    raise ValueError("Rescue native prompt fails exact archived-logit replay.")
            save_tensors(tensor, {"logits": logits})
            atomic_write(path, source.seal({"config_sha256": digest(config), "cell": cell,
                "logits_sha256": file_hash(tensor), "decision": engine.decide(logits, row),
                "processor": processor_summary(live, record), "archived_logits_exact": exact}))
            atomic_write(root/"progress_status.json", {"stage": "preflight" if preflight else "full",
                "completed": index+1, "expected": len(chosen), "cell_id": cell["cell_id"]})
            print("TIMING_FORWARD", index+1, len(chosen), cell["cell_id"], flush=True)
            del live, logits
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    proof = verify(root, preflight)
    atomic_write(root/("preflight_summary.json" if preflight else "run_summary.json"), proof)
    return proof


def alignment(records, plans):
    """Within base x layout: identical prompt tokens/visual positions; event bins shift exactly as scheduled."""
    onsets = {cell: (onset, onset+30+gap) for cell, (_, onset, gap) in materials.CELLS.items()}
    groups = defaultdict(dict)
    for record in records:
        cell = record["cell"]
        groups[cell["cohort"], cell["base_sample_id"], cell["view"]][cell["timing_cell"]] = record["processor"]
    for key, by_cell in groups.items():
        if len({(p["prompt_input_ids_sha256"], p["visual_positions_sha256"]) for p in by_cell.values()}) != 1:
            raise ValueError(f"Prompt tokens or visual positions differ across timing cells for {key}.")
        reference = "low" if "low" in by_cell else next(iter(by_cell))
        for cell, processor in by_cell.items():
            for event, column in (("event_1", 0), ("event_2", 1)):
                shift, observed = (onsets[cell][column]-onsets[reference][column])//materials.TOKEN_FRAMES, processor["event_token_bins"][event]
                base = by_cell[reference]["event_token_bins"][event]
                if not observed or [b+shift for b in base] != observed:
                    raise ValueError(f"{event} token bins of {key} {cell} do not shift by the scheduled {shift}.")
    return len(groups)


def verify(root, preflight=False):
    root = Path(root)
    config, order_context, plans = context(root)
    reference_config = {"order_root": config["order_root"], "settings": config["settings"]}
    records, by_cohort = [], defaultdict(int)
    for cell in selected(order_context, plans, preflight):
        path, tensor = paths(root, cell)
        record = source.checked_seal(read_json(path))
        if record["config_sha256"] != digest(config) or record["cell"] != cell or record["logits_sha256"] != file_hash(tensor):
            raise ValueError("Phase 3H checkpoint configuration/cell/logits changed.")
        if cell["kind"] == "original":
            logits = torch.load(tensor, map_location="cpu", weights_only=True)["logits"]
            if not torch.equal(logits, capture.references(reference_config, order_context, cell)["logits"]):
                raise ValueError("Rescue native logits no longer equal the archived logits.")
        records.append(record)
        by_cohort[cell["cohort"]] += 1
    groups = alignment(records, plans)
    return {"schema": SCHEMA, "complete": not preflight, "prompt_cells": len(records), "by_cohort": dict(by_cohort),
            "config_sha256": digest(config), "record_sha256": [r["record_sha256"] for r in records],
            "rescue_native_logits_exact": True, "prompt_tokens_identical_within_base_layout": True,
            "event_bins_shift_as_scheduled": True, "base_layout_groups": groups, "new_KO_or_patch": False}


def export(root):
    root = Path(root)
    config, order_context, plans = context(root)
    proof = verify(root)
    if read_json(root/"run_summary.json") != proof:
        raise ValueError("Full run summary does not replay.")
    rows = []
    for cell in cells(order_context, plans):
        record = read_json(paths(root, cell)[0])
        decision = record["decision"]
        rows.append({k: cell[k] for k in ("cohort", "base_sample_id", "timing_cell", "view", "cell_id")} | {
            k: decision[k] for k in ("correct_option", "prediction", "is_correct", "margin", "correct_logit", "incorrect_logit")})
    result = {"schema": SCHEMA+"_results", "complete": True, "rows": rows, "run_summary_sha256": file_hash(root/"run_summary.json"),
              "config_sha256": digest(config)}
    result["fingerprint"] = digest(result)
    if (root/"results.json").exists():
        if read_json(root/"results.json") != result:
            raise ValueError("Existing results differ from the verified checkpoints.")
        return result
    atomic_write(root/"results.json", result)
    return result


def public_files(root, reports_only=False):
    root = Path(root)
    allowed = {"checkpoints", "logits", "materials", "code_snapshot", "logs"}
    names = {"config.json", "runtime.json", "progress_status.json", "preflight_summary.json", "run_summary.json",
             "results.json", "job_status.json", "unattended_job.log"}
    result = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Backup refuses symlinks.")
        if not path.is_file() or path.name.startswith(".") and path.name.endswith(".lock"):
            continue
        relative = path.relative_to(root)
        if relative.parts[0] not in allowed and relative.as_posix() not in names:
            raise ValueError("Unexpected Phase 3H output; never package private configuration.")
        if not reports_only or path.suffix not in (".pt", ".mp4"):
            result["run/"+relative.as_posix()] = path
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "run", "verify", "export"), required=True)
    parser.add_argument("--stage", choices=("preflight", "full"), default="preflight")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--fresh_materials")
    parser.add_argument("--rescue_materials")
    parser.add_argument("--order_root")
    args = parser.parse_args()
    if args.action == "prepare":
        if not (args.fresh_materials and args.rescue_materials and args.order_root):
            parser.error("prepare needs --fresh_materials, --rescue_materials and --order_root.")
        result = prepare(args.output_root, args.fresh_materials, args.rescue_materials, args.order_root)
    else:
        with lock(args.output_root, ".timing.lock"):
            result = (run(args.output_root, args.stage == "preflight") if args.action == "run" else
                      verify(args.output_root, args.stage == "preflight") if args.action == "verify" else
                      export(args.output_root))
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "record_sha256")}, indent=2))
