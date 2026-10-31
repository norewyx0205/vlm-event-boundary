"""Isolated compact Phase 3E-B capture; every GPU stage requires an explicit command."""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import phase3e_paired_design as design
from scripts.phase3c_core import MODEL, REVISION, atomic_write, digest, file_hash, frozen_write, read_json
from scripts.phase3e_option_features import (CompactOptionSites, FEATURE_NAMES, audit_paired_readouts,
    option_readouts, statement_features)
from scripts.phase3e_option_store import write_store
from scripts.run_phase3d_vm import lock
from scripts.run_phase3e_preflight import code_hashes, mapping as old_mapping, require_same_video_inputs

RESERVE = 12*1024**3
SCHEMA = "phase3e_paired_option_capture_v1"


def load_plan(root):
    plan = read_json(Path(root) / "plan.json")
    design.validate_plan(plan)
    if tuple(plan["scene_seeds"]) != design.SCENE_SEEDS:
        raise ValueError("Capture requires the fixed 40 new scene families.")
    return plan


def space(root, required=RESERVE+2*1024**3):
    free = shutil.disk_usage(root).free
    if free < required:
        raise RuntimeError(f"Insufficient storage: need {required} free bytes, have {free}; no files deleted.")
    return free


def prepare(root):
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a new empty run root; historical artifacts are never adopted.")
    root.mkdir(parents=True, exist_ok=True)
    space(root)
    plan = design.build_plan(root)
    atomic_write(root / "plan.json", plan)
    rendered = design.render_examples(root, plan, count=40)
    atomic_write(root / "videos.json", rendered)
    return {"status": "cpu_materials_rendered", "videos": len(rendered["videos"]), "GPU_forwards": 0}


def check_videos(root, plan, rows):
    inventory = {v["video_id"]: v for v in read_json(Path(root) / "videos.json")["videos"]}
    for identity in {r["video_id"] for r in rows}:
        value = inventory[identity]
        path = Path(value["path"])
        if path.parent != Path(root) / "videos" or file_hash(path) != value["sha256"]:
            raise ValueError("Frozen paired video bytes/path changed.")
    return inventory


def config_for(root, plan, source_plan=None):
    path = Path(root) / "capture_config.json"
    if source_plan is None:
        config = read_json(path)
        if config["code_sha256"] != code_hashes() or config["plan_fingerprint"] != plan["fingerprint"]:
            raise ValueError("Capture code or plan changed; preserve this run and use a new root.")
        return config
    source_plan = Path(source_plan).resolve()
    settings = read_json(source_plan)["settings"]
    if settings["model_name"] != MODEL or settings["model_revision"] != REVISION or settings["dtype"] != "float16":
        raise ValueError("Source settings differ from the frozen Qwen checkpoint/precision.")
    config = {"schema": SCHEMA, "plan_fingerprint": plan["fingerprint"],
        "settings": settings, "source_settings_plan_sha256": file_hash(source_plan),
        "code_sha256": code_hashes(), "execution_mode": "model_parallel", "gpu_weight_budget_gib": 14,
        "feature_names": list(FEATURE_NAMES), "per_prompt_forwards": 3,
        "new_KO_forwards": 0, "new_patch_forwards": 0,
        "timing_raw_scope": "reference-last and option-final positions only"}
    frozen_write(path, config)
    return config


def mapping(prepared, processor):
    result = old_mapping(prepared, processor)
    groups, audit = option_readouts(prepared["row"], result["processor_record"], processor.tokenizer)
    for name in ("source_t1", "source_t2", "whole_event2"):
        groups[name] = result["capture_groups"][name]
    result.update(capture_groups=groups, option_readout_audit=audit)
    return result


def selected_rows(plan, preflight):
    seeds = set(plan["scene_seeds"][:2]) if preflight else set(plan["scene_seeds"])
    return [r for r in plan["rows"] if r["scene_seed"] in seeds]


def budget_contract(budget):
    if (budget["visual_tokens_per_prompt"] != 4608 or
            budget["whole_event2_positions_per_prompt"] != 512 or
            budget["video_tensor_parity_across_text_views"] is not True):
        raise ValueError("Visual sampling differs from the prior 4,608-token/512-Event-2 contract.")
    return budget


def audit_inputs(root, rows, settings):
    from transformers import AutoProcessor
    from scripts.phase3b_core import prepare_example
    processor = AutoProcessor.from_pretrained(settings["model_name"], revision=settings["model_revision"],
                                             local_files_only=True)
    records = {}
    for row in rows:
        prepared = prepare_example(row, processor, root, settings["video_fps"], settings["video_num_frames"],
                                   settings["video_max_pixels"], settings["roi_padding"])
        records[row["eval_id"]] = mapping(prepared, processor)
        print("TOKEN_AUDIT", row["eval_id"], flush=True)
    budget = budget_contract(require_same_video_inputs(records, {"rows": rows, "videos": {r["video_id"] for r in rows}}))
    alignment = audit_paired_readouts(rows, records)
    return {"records": records, "budget": budget, "alignment": alignment}


def verify_row(root, row, config, mapping_):
    import torch
    from scripts.run_phase3c_preflight import logits_parity
    root = Path(root)
    record = read_json(root / "checkpoints" / (row["eval_id"]+".json"))
    if (record.get("record_sha256") != digest({k: v for k, v in record.items() if k != "record_sha256"}) or
            record["eval_id"] != row["eval_id"] or record["config_sha256"] != digest(config) or
            record["mapping_sha256"] != digest(mapping_)):
        raise ValueError("Checkpoint input/configuration provenance changed.")
    path = root / record["tensor_path"]
    if path.parent != root / "captures" or file_hash(path) != record["tensor_sha256"]:
        raise ValueError("Capture path/hash changed.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    parity = logits_parity(payload["baseline_logits"], payload["capture_logits"], exact=True)
    standard = logits_parity(payload["baseline_logits"], payload["standard_logits"])
    if (payload["eval_id"] != row["eval_id"] or payload["config_sha256"] != digest(config) or
            payload["groups"] != mapping_["capture_groups"] or not parity["passed"] or not standard["passed"] or
            int(payload["baseline_logits"].argmax()) != int(payload["standard_logits"].argmax())):
        raise ValueError("Raw capture failed full-logit/input replay.")
    controller = CompactOptionSites.__new__(CompactOptionSites)
    controller.groups, controller.union = payload["groups"], payload["union_positions"]
    controller.raw, controller.means = payload["timing_raw"], payload["means"]
    controller.counts = Counter({site: 1 for site in controller.means})
    timing = controller.validate()
    if timing != payload["audit"] or timing != record["timing_audit"]:
        raise ValueError("Capture timing audit changed.")
    for option in ("A", "B"):
        statement_features(payload, option)
    logits, decision = payload["baseline_logits"], record["decision"]
    correct, incorrect = (float(logits[decision[k]]) for k in ("correct_token_id", "incorrect_token_id"))
    if (decision["correct_option"] != row["correct_option"] or decision["correct_logit"] != correct or
            decision["incorrect_logit"] != incorrect or decision["margin"] != correct-incorrect or
            decision["predicted_first_token_id"] != int(logits.argmax()) or
            decision["is_correct"] != (decision["prediction"] == row["correct_option"])):
        raise ValueError("Behavioural record differs from raw first-token logits.")
    return record, payload


def verify(root, preflight=False):
    root = Path(root).resolve()
    plan = load_plan(root)
    config = config_for(root, plan)
    rows = selected_rows(plan, preflight)
    check_videos(root, plan, rows)
    audit = read_json(root / ("processor_audit_preflight.json" if preflight else "processor_audit.json"))
    if set(audit["records"]) != {r["eval_id"] for r in rows}:
        raise ValueError("Missing/unexpected processor audit cell.")
    budget = budget_contract(require_same_video_inputs(audit["records"], {"rows": rows, "videos": {r["video_id"] for r in rows}}))
    alignment = audit_paired_readouts(rows, audit["records"])
    if audit["budget"] != budget or audit["alignment"] != alignment:
        raise ValueError("Processor pairing/budget audit changed.")
    records = []
    for row in rows:
        record, _ = verify_row(root, row, config, audit["records"][row["eval_id"]])
        records.append(record["record_sha256"])
    return {"schema": SCHEMA, "status": "technical_preflight_passed" if preflight else "full_capture_verified",
        "complete": not preflight, "prompt_cells": len(rows), "scene_families": 2 if preflight else 40,
        "plan_fingerprint": plan["fingerprint"], "config_sha256": digest(config),
        "record_hashes": records, "paired_token_alignment": True, "capture_exact_noop": True,
        "standard_generation_parity": True, "residual_timing": True, "video_tensor_parity": True,
        "all_statement_features_present": True, "budget": budget,
        "probe_fitting_performed": False, "new_KO_or_patch": False}


def storage_projection(root):
    root = Path(root)
    tensors = list((root / "captures").glob("*.pt"))
    if len(tensors) < 32:
        raise ValueError("Measure all 32 technical preflight cells before projecting storage.")
    payload_bytes = max(p.stat().st_size for p in tensors)
    metadata_bytes = (root / "processor_audit_preflight.json").stat().st_size*20
    video_bytes = sum(p.stat().st_size for p in (root / "videos").glob("*.mp4"))
    # FP32 feature-store upper bound; assume 4096-d hidden states, verify on export.
    feature_bytes = 1280*len(FEATURE_NAMES)*4096*4
    full_bytes = int((payload_bytes*640+metadata_bytes+video_bytes+feature_bytes)*1.25)
    current_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    remaining_plus_backup = max(0, full_bytes-current_bytes)+full_bytes
    return {"max_preflight_payload_bytes": payload_bytes, "projected_run_bytes": full_bytes,
        "projected_remaining_and_backup_bytes": remaining_plus_backup,
        "reserve_bytes": RESERVE, "required_free_bytes": remaining_plus_backup+RESERVE,
        "free_bytes": shutil.disk_usage(root).free, "assumed_hidden_dimension": 4096}


def capture(root, source_plan, preflight=False):
    import torch
    from scripts.phase3b_core import prepare_example
    from scripts.probe_attention_roi import standard_first_token
    from scripts.run_phase3c_preflight import Engine, logits_parity, save_tensors
    root = Path(root).resolve()
    space(root)
    plan = load_plan(root)
    config = config_for(root, plan, source_plan)
    settings, rows = config["settings"], selected_rows(plan, preflight)
    check_videos(root, plan, rows)
    if not preflight:
        expected = verify(root, preflight=True)
        if read_json(root / "preflight_summary.json") != expected:
            raise ValueError("Full capture requires the same verified two-scene technical preflight.")
        projection = storage_projection(root)
        space(root, projection["required_free_bytes"])
        atomic_write(root / "storage_projection.json", projection)
    audit = audit_inputs(root, rows, settings)
    frozen_write(root / ("processor_audit_preflight.json" if preflight else "processor_audit.json"), audit)
    if not preflight:
        initial = read_json(root / "processor_audit_preflight.json")["records"]
        if any(audit["records"][identity] != value for identity, value in initial.items()):
            raise ValueError("Full-run processor differs from the technical preflight.")
    engine = Engine(settings, "model_parallel", 14)
    frozen_write(root / "runtime.json", engine.runtime)
    try:
        for index, row in enumerate(rows):
            space(root)
            identity = row["eval_id"]
            mapping_ = audit["records"][identity]
            checkpoint = root / "checkpoints" / (identity+".json")
            if checkpoint.exists():
                verify_row(root, row, config, mapping_)
                continue
            tensor = root / "captures" / (identity+".pt")
            if tensor.exists():
                raise ValueError("Uncommitted capture tensor exists; inspect it before resuming.")
            prepared = prepare_example(row, engine.processor, root, settings["video_fps"], settings["video_num_frames"],
                settings["video_max_pixels"], settings["roi_padding"])
            if mapping(prepared, engine.processor) != mapping_:
                raise ValueError("GPU preparation differs from actual contextual token audit.")
            prepared["inputs"] = prepared["inputs"].to(engine.model.device)
            baseline = engine.forward(prepared)
            collector = CompactOptionSites(engine.text, len(mapping_["processor_record"]["input_ids"]), mapping_["capture_groups"])
            with collector.installed():
                observed = engine.forward(prepared)
            payload = collector.payload()
            token, standard = standard_first_token(engine.model, prepared["inputs"])
            if (not logits_parity(baseline, observed, exact=True)["passed"] or
                    not logits_parity(baseline, standard)["passed"] or token != int(baseline.argmax())):
                raise ValueError("Read-only capture/standard generation parity failed.")
            save_tensors(tensor, {**payload, "eval_id": identity, "config_sha256": digest(config),
                "baseline_logits": baseline, "capture_logits": observed, "standard_logits": standard})
            record = {"eval_id": identity, "config_sha256": digest(config), "mapping_sha256": digest(mapping_),
                "tensor_path": tensor.relative_to(root).as_posix(), "tensor_sha256": file_hash(tensor),
                "timing_audit": payload["audit"], "decision": engine.decide(baseline, row)}
            record["record_sha256"] = digest(record)
            atomic_write(checkpoint, record)
            print(f"CAPTURE {index+1}/{len(rows)} {identity}", flush=True)
            del prepared, baseline, observed, standard, collector, payload
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    summary = verify(root, preflight)
    atomic_write(root / ("preflight_summary.json" if preflight else "capture_summary.json"), summary)
    return summary


def export(root):
    import numpy as np
    root = Path(root).resolve()
    space(root, RESERVE+1280*len(FEATURE_NAMES)*4096*4)
    summary = verify(root)
    plan = load_plan(root)
    config = config_for(root, plan)
    audit = read_json(root / "processor_audit.json")["records"]
    vectors, decisions = {}, {}
    for row in plan["rows"]:
        record, payload = verify_row(root, row, config, audit[row["eval_id"]])
        for option in ("A", "B"):
            vectors[row["eval_id"]+"__statement_"+option] = statement_features(payload, option)
        decisions[row["eval_id"]] = record["decision"]
    examples = plan["statement_examples"]
    values = np.stack([vectors[e["example_id"]] for e in examples])
    if values.shape[2] != 4096:
        raise ValueError("Actual hidden dimension differs from the measured resource contract.")
    return write_store(root / "feature_store", plan, examples, values, decisions, summary,
        {"capture_config": config, "runtime": read_json(root / "runtime.json"), "verification": summary,
         "readout_metadata": {identity: {
             "input_ids_sha256": value["processor_record"]["prompt_input_ids_sha256"],
             "input_tensor_sha256": value["processor_record"]["input_tensor_sha256"],
             "logical_identity_spans": {o: f"option_{o}_t{value['option_readout_audit'][o]['reference_id']}" for o in ("A", "B")},
             "capture_groups": value["capture_groups"], "option_readout_audit": value["option_readout_audit"]}
             for identity, value in audit.items()}})


def persistent_volume(root, storage_root):
    storage, root = Path(storage_root).resolve(), Path(root).resolve()
    if (not storage.is_dir() or not root.is_relative_to(storage) or root == storage or
            (sys.platform.startswith("linux") and not os.path.ismount(storage))):
        raise ValueError("Run root must be inside the mounted persistent volume.")


def public_files(root, reports_only=False):
    root = Path(root).resolve()
    allowed = {"videos", "captures", "checkpoints", "feature_store", "logs", "code_snapshot"}
    names = {"plan.json", "videos.json", "capture_config.json", "processor_audit_preflight.json",
        "processor_audit.json", "runtime.json", "preflight_summary.json", "capture_summary.json",
        "storage_projection.json", "verification.json", "failure.json"}
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Backup refuses symlinks.")
        if not path.is_file() or path.name in {".phase3e_option.lock", ".pipeline.lock"}:
            continue
        relative = path.relative_to(root)
        if relative.parts[0] not in allowed and relative.as_posix() not in names:
            raise ValueError(f"Unexpected public artifact: {relative}; never package credentials.")
        result["run/"+relative.as_posix()] = path
    return result


def package(root, backup_dir):
    from scripts.backup_phase3b import create_backup, verify_backup
    from scripts.phase3e_option_store import load_store
    summary = verify(root)
    load_store(Path(root) / "feature_store/manifest.json")
    atomic_write(Path(root) / "verification.json", summary)
    space(root, RESERVE+int(1.1*sum(p.stat().st_size for p in public_files(root).values())))
    bundle = create_backup(root, backup_dir, file_provider=public_files, bundle_prefix="phase3e_backup",
        metadata={"schema": "phase3e_paired_option_backup_v1", "capture_plan_fingerprint": summary["plan_fingerprint"]})
    verify_backup(bundle)
    return {"backup_directory": str(bundle), "verified_on_VM": True}


def finish(root, backup_dir):
    from scripts.backup_phase3b import verify_backup
    from scripts.phase3b_email import EmailNotifier, load_private_config as email_config, DEFAULT_CONFIG as EMAIL
    from scripts.surf_workspace import SurfClient, load_private_config as surf_config, DEFAULT_CONFIG as SURF
    manifest = verify_backup(backup_dir)
    if manifest.get("schema") != "phase3e_paired_option_backup_v1" or manifest["source_run_root"] != str(Path(root).resolve()):
        raise ValueError("Finish requires this run's full verified backup.")
    summary = verify(root)
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True)
    if gpu.stdout.strip():
        raise RuntimeError("GPU compute remains active; refusing whole-VM Pause.")
    receipt = {"full_capture_verified": True, "verified_backup": str(backup_dir), "gpu_quiescent": True}
    body = (f"Phase 3E-B compact option capture VERIFIED\nScene families: 40; prompts: 640\nRun: {root}\n"
            f"Verified full VM backup: {backup_dir}\nNo new KO/patch. CPU probe fitting is a separate step.\n"
            "The next action is a SURF Pause request; this email does not confirm Pause/billing stop.\n")
    try:
        receipt["email"] = EmailNotifier(email_config(EMAIL, (root, backup_dir))).send(
            "[Phase 3E-B] Paired-video option capture VERIFIED", body)
    except Exception as exc:
        receipt["email_error_type"] = type(exc).__name__
    path = Path(root).parent / (Path(root).name+"_completion.json")
    atomic_write(path, receipt)
    receipt["pause"] = SurfClient(surf_config(SURF, (root, backup_dir))).request_pause()
    atomic_write(path, receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "preflight", "capture", "verify", "export", "package", "finish"), required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--source_plan")
    parser.add_argument("--backup_dir")
    parser.add_argument("--storage_root")
    args = parser.parse_args()
    if args.stage in ("preflight", "capture") and not args.source_plan:
        parser.error("GPU stages require --source_plan for the frozen model settings")
    if args.stage in ("package", "finish") and not args.backup_dir:
        parser.error("package/finish require --backup_dir")
    if args.stage in ("preflight", "capture", "package", "finish") or (args.stage == "prepare" and sys.platform.startswith("linux")):
        if not args.storage_root:
            parser.error("VM stages require --storage_root for the persistent-volume gate")
        persistent_volume(args.output_root, args.storage_root)
        if args.backup_dir:
            backup, root = Path(args.backup_dir).resolve(), Path(args.output_root).resolve()
            if (backup == root or backup in root.parents or root in backup.parents or
                    not backup.is_relative_to(Path(args.storage_root).resolve())):
                parser.error("Use a separate backup directory on the persistent volume")
    if args.stage == "prepare":
        result = prepare(args.output_root)
    else:
        with lock(Path(args.output_root), ".phase3e_option.lock"):
            if args.stage in ("preflight", "capture"):
                result = capture(args.output_root, args.source_plan, args.stage == "preflight")
            elif args.stage == "verify":
                result = verify(args.output_root)
            elif args.stage == "export":
                result = export(args.output_root)
            elif args.stage == "package":
                result = package(args.output_root, args.backup_dir)
            else:
                result = finish(args.output_root, args.backup_dir)
    print(json.dumps(result, indent=2))
