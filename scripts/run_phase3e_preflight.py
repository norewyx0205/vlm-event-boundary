"""Execute only the frozen two-scene Phase 3E technical preflight."""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import phase3e_preflight as core
from scripts.phase3c_core import atomic_write, digest, file_hash, read_json


def code_hashes():
    return {path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))}


def storage_gate(root):
    free = shutil.disk_usage(Path(root).parent).free
    if free < 14*1024**3:
        raise RuntimeError("Need 14 GiB free including 12 GiB untouched reserve; no data deleted.")
    return {"free_bytes": free, "reserve_bytes": 12*1024**3}


def mapping(prepared, processor):
    from scripts.phase3c_core import event_layout
    from scripts.phase3d_execution import processor_record
    from scripts.phase3d_features import capture_groups
    record = processor_record(prepared)
    row = prepared["row"]
    layout = event_layout(row, record["video_metadata"], record["visual_positions"], len(record["input_ids"]))
    whole = sorted(layout["lookup"])
    groups = capture_groups(row, record, whole, processor.tokenizer)
    groups["options_all"] = record["group_positions"]["options_all"]
    groups["subject_mentions"] = sorted(groups[f"option_A_t{row['subject_id']}"] +
                                        groups[f"option_B_t{row['subject_id']}"])
    groups["reference_mentions"] = sorted(groups[f"option_A_t{3-row['subject_id']}"] +
                                          groups[f"option_B_t{3-row['subject_id']}"])
    return {"processor_record": record, "capture_groups": groups,
            "event2_bins": [{k: b[k] for k in ("temporal_index", "source_frames", "event_fraction", "progress")}
                            for b in layout["bins"]]}


def require_same_video_inputs(records, plan):
    for video in plan["videos"]:
        items = [records[r["eval_id"]] for r in plan["rows"] if r["video_id"] == video]
        signatures = []
        for item in items:
            values = item["processor_record"]["input_tensor_sha256"]
            signatures.append({k: v for k, v in values.items() if "video" in k})
        if not signatures[0] or any(s != signatures[0] for s in signatures):
            raise ValueError("Text-only variants changed actual video tensor bytes/grid.")
    budgets = {len(r["processor_record"]["visual_positions"]) for r in records.values()}
    supports = {len(r["capture_groups"]["whole_event2"]) for r in records.values()}
    if len(budgets) != 1 or len(supports) != 1:
        raise ValueError("Visual token/Event-2 support counts differ; inspect sampling, do not silently crop.")
    return {"visual_tokens_per_prompt": next(iter(budgets)),
            "whole_event2_positions_per_prompt": next(iter(supports)), "video_tensor_parity_across_text_views": True}


def run(root, source_plan):
    from transformers import AutoProcessor
    import torch
    from scripts.phase3b_core import prepare_example
    from scripts.probe_attention_roi import standard_first_token
    from scripts.run_phase3c_preflight import Engine, logits_parity, save_tensors
    root = Path(root).resolve()
    storage = storage_gate(root)
    plan = core.load_plan(root)
    settings = read_json(source_plan)["settings"]
    config = {"plan_fingerprint": plan["fingerprint"], "settings": settings,
              "source_plan_sha256": file_hash(source_plan), "code_sha256": code_hashes(),
              "execution_mode": "model_parallel", "gpu_weight_budget_gib": 14,
              "capture_sites": "L0/16/17/18/20 block outputs; L16/17/18 input; L17/18 attention contribution and residual"}
    config_path = root / "execution_config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError("Code/config changed since this run was frozen; use a new run root.")
    atomic_write(config_path, config)
    processor = AutoProcessor.from_pretrained(settings["model_name"], revision=settings["model_revision"],
                                             local_files_only=True)
    records = {}
    for row in plan["rows"]:
        prepared = prepare_example(row, processor, root, padding=settings["roi_padding"])
        records[row["eval_id"]] = mapping(prepared, processor)
        print("TOKEN_AUDIT", row["eval_id"], flush=True)
    budget = require_same_video_inputs(records, plan)
    atomic_write(root / "processor_audit.json", {"records": records, "budget": budget})
    del processor, prepared
    gc.collect()
    engine = Engine(settings, "model_parallel", 14)
    atomic_write(root / "runtime.json", engine.runtime)
    started = time.perf_counter()
    for index, row in enumerate(plan["rows"]):
        storage_gate(root)
        checkpoint = root / "checkpoints" / (row["eval_id"] + ".json")
        if checkpoint.exists():
            verify_row(root, row, config)
            continue
        prepared = prepare_example(row, engine.processor, root, padding=settings["roi_padding"])
        if mapping(prepared, engine.processor) != records[row["eval_id"]]:
            raise ValueError("Processor replay differs from pre-model CPU audit.")
        prepared["inputs"] = prepared["inputs"].to(engine.model.device)
        baseline = engine.forward(prepared)
        groups = records[row["eval_id"]]["capture_groups"]
        controller = core.AttentionSites(engine.text, len(prepared["inputs"].input_ids[0]), groups)
        with controller.installed():
            captured = engine.forward(prepared)
        timing_audit = controller.validate()
        parity = logits_parity(baseline, captured, exact=True)
        standard_token, standard_logits = standard_first_token(engine.model, prepared["inputs"])
        standard = logits_parity(baseline, standard_logits)
        if not parity["passed"] or not standard["passed"] or standard_token != int(baseline.argmax()):
            raise ValueError("Read-only hook/no-op or independent generation parity failed.")
        decision = engine.decide(baseline, row)
        tensor = root / "captures" / (row["eval_id"] + ".pt")
        save_tensors(tensor, {"eval_id": row["eval_id"], "plan_fingerprint": plan["fingerprint"],
            "baseline_logits": baseline, "capture_logits": captured, "standard_logits": standard_logits,
            **controller.payload()})
        item = {"eval_id": row["eval_id"], "scene_seed": row["scene_seed"],
            "plan_fingerprint": plan["fingerprint"], "execution_config_sha256": digest(config),
            "tensor_path": str(tensor.relative_to(root)), "tensor_sha256": file_hash(tensor),
            "hook_logits_parity": parity, "standard_logits_parity": standard,
            "standard_token_matches": True, "timing_audit": timing_audit, "decision": decision}
        item["record_sha256"] = digest(item)
        atomic_write(checkpoint, item)
        print(f"CAPTURE {index+1}/32 {row['eval_id']} {decision}", flush=True)
        del prepared, baseline, captured, standard_logits, controller
        gc.collect()
        torch.cuda.empty_cache()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    summary = verify(root)
    summary.update(elapsed_capture_sec=time.perf_counter()-started, initial_storage=storage)
    atomic_write(root / "summary.json", summary)
    return summary


def verify_row(root, row, config):
    import torch
    from scripts.run_phase3c_preflight import logits_parity
    root = Path(root)
    record = read_json(root / "checkpoints" / (row["eval_id"] + ".json"))
    if (record.get("record_sha256") != digest({k: v for k, v in record.items() if k != "record_sha256"}) or
            record["plan_fingerprint"] != config["plan_fingerprint"] or
            record["execution_config_sha256"] != digest(config)):
        raise ValueError("Altered checkpoint provenance.")
    tensor = root / record["tensor_path"]
    if tensor.parent != root / "captures" or file_hash(tensor) != record["tensor_sha256"]:
        raise ValueError("Capture tensor changed or escaped expected directory.")
    payload = torch.load(tensor, map_location="cpu", weights_only=True)
    if (payload["eval_id"] != row["eval_id"] or payload["plan_fingerprint"] != config["plan_fingerprint"] or
            not logits_parity(payload["baseline_logits"], payload["capture_logits"], exact=True)["passed"] or
            not logits_parity(payload["baseline_logits"], payload["standard_logits"])["passed"] or
            int(payload["baseline_logits"].argmax()) != int(payload["standard_logits"].argmax())):
        raise ValueError("Saved logits fail independent replay validation.")
    controller = core.AttentionSites.__new__(core.AttentionSites)
    from collections import Counter
    controller.groups, controller.union = payload["groups"], payload["union_positions"]
    controller.raw, controller.means = payload["timing_raw"], payload["means"]
    controller.counts = Counter({k: 1 for k in controller.means})
    controller.validate()
    logits, decision = payload["baseline_logits"], record["decision"]
    correct = float(logits[decision["correct_token_id"]])
    incorrect = float(logits[decision["incorrect_token_id"]])
    if (decision["correct_option"] != row["correct_option"] or decision["correct_logit"] != correct or
            decision["incorrect_logit"] != incorrect or decision["margin"] != correct-incorrect or
            decision["predicted_first_token_id"] != int(logits.argmax())):
        raise ValueError("Saved behavioural decision does not match raw logits/ground truth.")
    return record


def verify(root):
    root = Path(root)
    plan = core.load_plan(root)
    config = read_json(root / "execution_config.json")
    audit = read_json(root / "processor_audit.json")
    budget = require_same_video_inputs(audit["records"], plan)
    if budget != audit["budget"]:
        raise ValueError("Processor budget audit changed.")
    records = [verify_row(root, row, config) for row in plan["rows"]]
    return {"schema": core.SCHEMA, "status": "technical_preflight_passed",
        "plan_fingerprint": plan["fingerprint"], "scene_seeds": list(core.SEEDS),
        "independent_scene_count": 2, "prompt_cells": len(records), "videos": len(plan["videos"]),
        "budget": budget, "read_only_hook_exact_logit_replays": len(records),
        "standard_generation_parity_passes": len(records), "attention_timing_passes": len(records),
        "verified_record_hashes": [r["record_sha256"] for r in records],
        "probe_fitting_performed": False, "new_causal_interventions_performed": False,
        "full_run_started": False, "event2_mover_shortcut_eliminated": False}


def public_files(root, reports_only=False):
    root = Path(root).resolve()
    allowed = {"videos", "captures", "checkpoints", "code_snapshot", "logs"}
    names = {"plan.json", "execution_config.json", "processor_audit.json", "runtime.json",
             "summary.json", "verification.json", "report.md", "failure.json"}
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Backup refuses symlinks.")
        if not path.is_file() or path.name in {".preflight.lock", ".pipeline.lock"}:
            continue
        relative = path.relative_to(root)
        if relative.parts[0] not in allowed and str(relative) not in names:
            raise ValueError(f"Unexpected public output: {relative}; do not package credentials.")
        result["run/"+relative.as_posix()] = path
    return result


def package(root, backup_dir):
    from scripts.backup_phase3b import create_backup, verify_backup
    summary = verify(root)
    atomic_write(Path(root) / "verification.json", summary)
    bundle = create_backup(root, backup_dir, file_provider=public_files, bundle_prefix="phase3c_backup",
        metadata={"schema": "phase3e_preflight_backup_v1", "preflight_plan_fingerprint": summary["plan_fingerprint"],
                  "scope": "Two-scene technical preflight, not the full Phase 3E experiment"})
    verify_backup(bundle)
    return {"backup_directory": str(bundle), "verified_on_vm": True}


def finish(root, backup_dir):
    from scripts.backup_phase3b import verify_backup
    from scripts.phase3b_email import EmailNotifier, load_private_config as email_config, DEFAULT_CONFIG as EMAIL
    from scripts.surf_workspace import SurfClient, load_private_config as surf_config, DEFAULT_CONFIG as SURF
    manifest = verify_backup(backup_dir)
    if manifest.get("schema") != "phase3e_preflight_backup_v1" or manifest["source_run_root"] != str(Path(root).resolve()):
        raise ValueError("Finish requires this run's verified full backup.")
    summary = verify(root)
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True)
    if gpu.stdout.strip():
        raise RuntimeError("GPU compute still active; refusing whole-VM Pause.")
    body = ("Phase 3E small technical preflight: PASSED\n"
            f"Independent scene seeds: {summary['independent_scene_count']}\n"
            f"Videos: {summary['videos']}; prompt cells: {summary['prompt_cells']}\n"
            "All read-only hook, residual timing and standard-generation checks passed.\n"
            "No probes fitted; no patch/KO; full 40-scene experiment NOT started.\n"
            f"Run: {root}\nVerified full VM backup: {backup_dir}\n"
            "The next action is a SURF Pause request after this email attempt.\n")
    receipt = {"preflight_passed": True, "gpu_quiescent": True}
    try:
        receipt["email"] = EmailNotifier(email_config(EMAIL, (root, backup_dir))).send(
            "[Phase 3E] Two-scene technical preflight PASSED", body)
    except Exception as exc:
        receipt["email_error_type"] = type(exc).__name__
    receipt_path = Path(root).parent / (Path(root).name+"_completion.json")
    atomic_write(receipt_path, receipt)
    # An accepted Pause may close SSH. Persist the receipt before returning.
    receipt["pause"] = SurfClient(surf_config(SURF, (root, backup_dir))).request_pause()
    atomic_write(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "capture", "verify", "package", "finish"), required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--source_plan")
    parser.add_argument("--backup_dir")
    args = parser.parse_args()
    if args.stage == "prepare":
        result = core.prepare(args.output_root)["factorial_audit"]
    elif args.stage == "capture":
        if not args.source_plan:
            parser.error("capture requires --source_plan for the frozen runtime settings")
        result = run(args.output_root, args.source_plan)
    elif args.stage == "verify":
        result = verify(args.output_root)
    elif args.stage == "package":
        if not args.backup_dir:
            parser.error("package requires --backup_dir")
        result = package(args.output_root, args.backup_dir)
    else:
        if not args.backup_dir:
            parser.error("finish requires --backup_dir pointing to the verified package")
        result = finish(args.output_root, args.backup_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
