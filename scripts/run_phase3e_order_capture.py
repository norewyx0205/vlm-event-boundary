"""Capture only order-reversed endpoints of the frozen rescue50 direct-D study."""

import argparse
import copy
import gc
import json
import shutil
import sys
from contextlib import ExitStack
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts import phase3d_execution as source
from scripts import phase3e_order_materials as materials
from scripts import phase3e_order_probes as probes
from scripts.phase3b_core import prepare_example
from scripts.phase3c_core import atomic_write, digest, file_hash, frozen_write, read_json
from scripts.phase3d_core import audit_row
from scripts.phase3d_single_layer_ko import check_logits
from scripts.phase3e_option_features import option_readouts
from scripts.phase3e_rescue50_probes import load_store as original_store
from scripts.run_phase3c_preflight import Engine, logits_parity, save_tensors
from scripts.run_phase3d_probes import save_npz
from scripts.run_phase3d_vm import lock
from scripts.run_phase3e_rescue50_capture import OptionBlocks, source_audit

SCHEMA = "phase3e_rescue50_order_capture_v1"
RESERVE = 12*1024**3


def code_hashes():
    names = ("run_phase3e_order_capture.py", "run_phase3e_order_unattended.py", "phase3e_order_materials.py",
        "phase3e_order_probes.py", "phase3e_rescue50_probes.py", "run_phase3e_rescue50_capture.py",
        "phase3d_execution.py", "phase3e_option_features.py", "run_phase3c_preflight.py", "phase3c_core.py")
    return {n: file_hash(Path(__file__).with_name(n)) for n in names}


def space(root):
    free = shutil.disk_usage(root).free
    if free < RESERVE+3*1024**3:
        raise RuntimeError("Need 12-GiB reserve plus 3-GiB capture/backup budget; no data deleted.")
    return free


def safe_video(root, spec):
    path = Path(spec["row"]["video_path"])
    if path.is_absolute():
        path = Path("videos")/path.name
    candidate = Path(root)/"materials"/path
    if (not candidate.resolve().is_relative_to((Path(root)/"materials/videos").resolve()) or
            candidate.is_symlink() or file_hash(candidate) != spec["video_sha256"]):
        raise ValueError("Reversed material path or video checksum changed.")
    return candidate.resolve()


def context(root):
    root = Path(root)
    config = read_json(root/"config.json")
    if config["schema"] != SCHEMA or config["code_sha256"] != code_hashes():
        raise ValueError("Order-capture code changed; preserve this attempt and use a new run root.")
    for name, sha in config["input_files"].items():
        path = root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or file_hash(path) != sha:
            raise ValueError("Copied direct-D input snapshot changed.")
    old_store, preparation, examples, values, decisions = original_store(root/"original_store/manifest.json")
    plan = read_json(root/"materials/plan.json")
    historical = source.load_context(config["source_root"])
    if (plan.get("schema") != materials.SCHEMA or plan.get("complete") is not True or
            plan["fingerprint"] != digest({k: v for k, v in plan.items() if k != "fingerprint"}) or
            plan["source_store_sha256"] != file_hash(root/"original_store/manifest.json") or
            plan["source_preparation_fingerprint"] != preparation["fingerprint"] or
            plan["source_plan_fingerprint"] != historical[0]["fingerprint"] or
            plan["case_ids"] != preparation["case_ids"] or plan["outer_folds"] != preparation["outer_folds"] or
            plan["settings"] != preparation["settings"] or plan["settings"] != historical[0]["settings"] or
            len(plan["reconstruction_audits"]) != 100 or
            any(r.get("decoded_pixels_exact") is not True for r in plan["reconstruction_audits"])):
        raise ValueError("Order materials do not reconstruct the exact original rescue50 source.")
    expected = {(t["pair_id"], t["condition"], t["view"]): t for t in preparation["tasks"]}
    actual = [(r["pair_id"], r["condition"], r["view"]) for r in plan["rows"]]
    if len(actual) != 200 or set(actual) != set(expected):
        raise ValueError("Exactly 200 reversed-order prompt cells are required.")
    for spec in plan["rows"]:
        key = spec["pair_id"], spec["condition"], spec["view"]
        old = historical[1][spec["pair_id"]][spec["condition"]]
        reverse = materials.reversed_row(old, spec["row"]["video_path"])
        if spec["view"] == "option_order_permuted":
            reverse = source.permuted_row(reverse)
        if spec["row"] != reverse or spec["original_eval_id"] != expected[key]["eval_id"]:
            raise ValueError("Order reversal changed original identities, paths, text or timing semantics.")
    return config, plan, historical, (old_store, preparation, examples, values, decisions)


def prepare(root, material_root, source_root, store_path):
    root, material_root, source_root, store_path = map(lambda p: Path(p).resolve(),
        (root, material_root, source_root, store_path))
    if any(root == p or root in p.parents or p in root.parents for p in (material_root, source_root, store_path.parent)):
        raise ValueError("Use an isolated new order-capture directory.")
    if root.exists() and any(root.iterdir()):
        raise ValueError("Do not overwrite a previous order-capture attempt.")
    root.mkdir(parents=True)
    space(root)
    shutil.copytree(material_root, root/"materials", ignore=shutil.ignore_patterns("*.lock"))
    shutil.copytree(store_path.parent, root/"original_store")
    plan = read_json(root/"materials/plan.json")
    files = {p.relative_to(root).as_posix(): file_hash(p) for directory in (root/"materials", root/"original_store")
             for p in directory.rglob("*") if p.is_file()}
    config = {"schema": SCHEMA, "source_root": str(source_root), "input_files": files,
        "code_sha256": code_hashes(), "settings": plan["settings"],
        "gpu_weight_budget_gib": plan["source_runtime"]["gpu_weight_budget_gib"],
        "materials_fingerprint": plan["fingerprint"], "new_prompt_cells": 200, "new_KO_or_patch": False,
        "storage_reserve_bytes": RESERVE, "protocol_sha256": file_hash(Path(__file__).resolve().parents[1]/
            "docs/phase3e_paired_difference_training.md")}
    frozen_write(root/"config.json", config)
    context(root)
    return {"prepared": True, "new_prompt_cells": 200, "preflight_cells": 8, "free_GiB": space(root)/1024**3}


def selected(plan, historical, preflight):
    return [s for s in plan["rows"] if not preflight or s["pair_id"] in historical[0]["preflight_ids"]]


def checkpoint_paths(root, spec):
    uid = spec["row"]["eval_id"]
    return Path(root)/"checkpoints"/(uid+".json"), Path(root)/"captures"/(uid+".pt")


def original_spec(spec):
    return {"pair_id": spec["pair_id"], "condition": spec["condition"], "view": spec["view"],
            "stage": "capture", "intervention": "baseline"}


def old_audit(config, historical, spec):
    candidates = source.feature_tasks(historical[0], "capture", [spec["pair_id"]])
    task = next(s for s in candidates if s["condition"] == spec["condition"] and s["view"] == spec["view"])
    return source_audit(config["source_root"], historical, task)


def reversed_input_audit(original, observed, original_readouts, observed_readouts):
    if original_readouts != observed_readouts or any(
            original[key] != observed[key] for key in ("input_ids", "attention_mask", "visual_positions")):
        raise ValueError("Order reversal changed contextual option tokens or input positions/mask.")
    left, right = original["video_metadata"], observed["video_metadata"]
    # This ROI-assignment diagnostic depends on motion, not video sampling.
    diagnostic = "ambiguous_visual_cells"
    if (set(left) != set(right) or diagnostic not in left or
            any(left[key] != right[key] for key in left if key != diagnostic) or
            any(type(m[diagnostic]) is not int or not 0 <= m[diagnostic] <= m["visual_token_count"]
                for m in (left, right))):
        raise ValueError("Order reversal changed video sampling/grid or invalidated its ROI diagnostic.")
    text = lambda record: {k: v for k, v in record["group_positions"].items() if not k.startswith("video_")}
    if text(original) != text(observed):
        raise ValueError("Order reversal changed the archived text groups.")
    a, b = original["input_tensor_sha256"], observed["input_tensor_sha256"]
    pixel = "pixel_values_videos"
    if (set(a) != set(b) or pixel not in a or
            any(a[name] != b[name] for name in a if name != pixel) or
            {k: v for k, v in a[pixel].items() if k != "sha256"} !=
            {k: v for k, v in b[pixel].items() if k != "sha256"} or
            a[pixel]["sha256"] == b[pixel]["sha256"]):
        raise ValueError("Order reversal must change pixels, retaining their shape/dtype and all other input tensors.")
    return {"same_contextual_option_readouts": True, "same_text_groups_positions_and_mask": True,
            "same_sampling_and_grid": True, "same_nonpixel_input_tensors": True,
            "visual_pixels_changed_shape_and_dtype_retained": True,
            "roi_assignment_diagnostic": {"original_ambiguous_visual_cells": left[diagnostic],
                                          "reversed_ambiguous_visual_cells": right[diagnostic]}}


def original_replay(root, config, historical, original, spec, engine=None):
    uid = spec["original_eval_id"]
    path, tensor = Path(root)/"original_replay"/(uid+".json"), Path(root)/"original_replay"/(uid+".pt")
    audit, baseline, _, tokens = old_audit(config, historical, spec)
    _, _, examples, features, _ = original
    indices = [next(i for i, e in enumerate(examples) if e["eval_id"] == uid and e["option"] == o) for o in ("A", "B")]
    expected = torch.from_numpy(features[indices])
    if not path.exists():
        if engine is None or tensor.exists():
            raise ValueError("Missing or uncommitted original replay gate.")
        task = next(t for t in source.feature_tasks(historical[0], "capture", [spec["pair_id"]])
                    if t["condition"] == spec["condition"] and t["view"] == spec["view"])
        live, processor = source.prepare_live(engine.processor, historical, task)
        if processor != audit["processor_record"]:
            raise ValueError("Original preflight inputs differ from the historical processor audit.")
        positions = [tokens[o]["reference_last_position"] for o in ("A", "B")]
        collector = OptionBlocks(engine.text, len(processor["input_ids"]), positions)
        live["inputs"] = live["inputs"].to(engine.model.device)
        with collector.installed():
            logits = engine.forward(live)
        saved = source.payload(config["source_root"], baseline)
        if not torch.equal(logits, saved["logits"]) or not torch.equal(collector.features(), expected):
            raise ValueError("Original model logits or reference-token states do not replay exactly.")
        save_tensors(tensor, {"logits": logits, "features": collector.features()})
        atomic_write(path, source.seal({"config_sha256": digest(config), "original_eval_id": uid,
            "source_audit_sha256": audit["record_sha256"], "source_baseline_sha256": baseline["record_sha256"],
            "tensor_sha256": file_hash(tensor)}))
        del live, collector, logits, saved
        gc.collect()
        torch.cuda.empty_cache()
    record = source.checked_seal(read_json(path))
    payload = torch.load(tensor, map_location="cpu", weights_only=True)
    saved = source.payload(config["source_root"], baseline)
    if (record["config_sha256"] != digest(config) or record["original_eval_id"] != uid or
            record["source_audit_sha256"] != audit["record_sha256"] or
            record["source_baseline_sha256"] != baseline["record_sha256"] or
            record["tensor_sha256"] != file_hash(tensor) or not torch.equal(payload["logits"], saved["logits"]) or
            not torch.equal(payload["features"], expected)):
        raise ValueError("Original baseline/token replay checkpoint changed.")
    return record


def verify_one(root, config, historical, spec):
    path, tensor = checkpoint_paths(root, spec)
    record = source.checked_seal(read_json(path))
    if (record["config_sha256"] != digest(config) or record["spec"] != spec or
            record["tensor_sha256"] != file_hash(tensor)):
        raise ValueError("Order checkpoint configuration/spec/tensor changed.")
    safe_video(root, spec)
    old, _, _, token_audit = old_audit(config, historical, spec)
    parity = reversed_input_audit(old["processor_record"], record["processor_record"], token_audit, record["readout_audit"])
    if (record["original_audit_sha256"] != old["record_sha256"] or record["input_parity_audit"] != parity or
            record["processor_record"]["video_sha256"] != spec["video_sha256"]):
        raise ValueError("Same-statement original/reversed token alignment changed.")
    payload = torch.load(tensor, map_location="cpu", weights_only=True)
    check_logits(payload["logits"], record["decision"])
    if (record["decision"]["correct_option"] != spec["row"]["correct_option"] or
            payload["features"].shape != (2, 4, 4096) or not bool(torch.isfinite(payload["features"]).all()) or
            payload["positions"] != [token_audit[o]["reference_last_position"] for o in ("A", "B")] or
            not torch.equal(payload["logits"], payload["capture_logits"]) or
            not logits_parity(payload["logits"], payload["standard_logits"])["passed"] or
            int(payload["logits"].argmax()) != int(payload["standard_logits"].argmax())):
        raise ValueError("Order capture no-op, first-token parity, feature or truth gate failed.")
    return record, payload


def verify(root, preflight=False):
    config, plan, historical, original = context(root)
    gates = [original_replay(root, config, historical, original, s) for s in selected(plan, historical, True)]
    specs = selected(plan, historical, preflight)
    records = [verify_one(root, config, historical, s)[0] for s in specs]
    return {"schema": SCHEMA, "complete": not preflight, "new_prompt_cells": len(records),
        "config_sha256": digest(config), "record_sha256": [r["record_sha256"] for r in records],
        "capture_noop_exact": True, "standard_generation_parity": True, "same_contextual_reference_positions": True,
        "all_original_frames_reconstructed_exactly": True, "new_KO_or_patch": False,
        "original_replay_gate": {"prompt_cells": len(gates), "logits_and_reference_tokens_exact": True,
            "record_sha256": [r["record_sha256"] for r in gates]}}


def run(root, preflight=False):
    from scripts.phase3c_budget import runtime_compatibility
    from scripts.probe_attention_roi import standard_first_token
    root = Path(root)
    config, plan, historical, original = context(root)
    if not preflight and read_json(root/"preflight_summary.json") != verify(root, True):
        raise ValueError("Full order capture requires the same passed eight-prompt preflight.")
    engine = Engine(config["settings"], "model_parallel", config["gpu_weight_budget_gib"])
    try:
        runtime_compatibility(plan["source_runtime"], engine.runtime)
        frozen_write(root/"runtime.json", engine.runtime)
        specs = selected(plan, historical, preflight)
        for index, spec in enumerate(specs):
            space(root)
            if preflight:
                original_replay(root, config, historical, original, spec, engine)
            path, tensor = checkpoint_paths(root, spec)
            if path.exists():
                verify_one(root, config, historical, spec)
                continue
            if tensor.exists():
                raise ValueError("Uncommitted tensor exists; inspect before resuming.")
            row = {**copy.deepcopy(spec["row"]), "video_path": str(safe_video(root, spec))}
            settings = config["settings"]
            live = prepare_example(row, engine.processor, root, settings["video_fps"], settings["video_num_frames"],
                settings["video_max_pixels"], settings["roi_padding"])
            processor = source.processor_record(live)
            _, tokens = option_readouts(row, processor, engine.processor.tokenizer)
            old, _, _, old_tokens = old_audit(config, historical, spec)
            parity = reversed_input_audit(old["processor_record"], processor, old_tokens, tokens)
            live["inputs"] = live["inputs"].to(engine.model.device)
            plain = engine.forward(live)
            positions = [tokens[o]["reference_last_position"] for o in ("A", "B")]
            collector = OptionBlocks(engine.text, len(processor["input_ids"]), positions)
            with collector.installed():
                observed = engine.forward(live)
            token, standard = standard_first_token(engine.model, live["inputs"])
            if not torch.equal(plain, observed) or not logits_parity(plain, standard)["passed"] or token != int(plain.argmax()):
                raise ValueError("Order capture changes model output or fails standard decoding parity.")
            save_tensors(tensor, {"logits": plain, "capture_logits": observed, "standard_logits": standard,
                "features": collector.features(), "positions": positions})
            record = source.seal({"config_sha256": digest(config), "spec": spec, "tensor_sha256": file_hash(tensor),
                "original_audit_sha256": old["record_sha256"], "readout_audit": tokens,
                "processor_record": processor, "input_parity_audit": parity, "decision": engine.decide(plain, row)})
            atomic_write(path, record)
            atomic_write(root/"progress_status.json", {"stage": "preflight" if preflight else "full",
                "completed": index+1, "expected": len(specs), "eval_id": row["eval_id"]})
            print("ORDER_CAPTURE", index+1, len(specs), row["eval_id"], flush=True)
            del live, plain, observed, standard, collector
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    proof = verify(root, preflight)
    atomic_write(root/("preflight_summary.json" if preflight else "capture_summary.json"), proof)
    return proof


def export(root):
    root = Path(root)
    config, plan, historical, original = context(root)
    _, preparation, examples, features, decisions = original
    proof = verify(root)
    if read_json(root/"capture_summary.json") != proof:
        raise ValueError("Full capture summary does not replay.")
    endpoints = [{**e, "order_variant": "original", "parent_original_example_id": e["example_id"]} for e in examples]
    arrays = [features]
    old_lookup = {e["example_id"]: e for e in examples}
    decisions = copy.deepcopy(decisions)
    for spec in plan["rows"]:
        record, payload = verify_one(root, config, historical, spec)
        for index, option in enumerate(("A", "B")):
            old = old_lookup[spec["original_eval_id"]+"__statement_"+option]
            endpoints.append({**old, "example_id": spec["row"]["eval_id"]+"__statement_"+option,
                "eval_id": spec["row"]["eval_id"], "parent_original_example_id": old["example_id"],
                "order_variant": "order_reversed", "y_statement_true": 1-old["y_statement_true"]})
        arrays.append(payload["features"].numpy())
        decisions[spec["row"]["eval_id"]] = record["decision"]
    values = np.concatenate(arrays)
    probes.paired_dataset(preparation, endpoints, values)
    destination = root/"feature_store"
    contents = {"preparation.json": preparation, "endpoints.json": endpoints, "decisions.json": decisions,
                "capture_verification.json": proof, "materials_plan.json": plan}
    if (destination/"manifest.json").exists():
        for name, value in contents.items():
            if read_json(destination/name) != value:
                raise ValueError("Existing order store differs from actual captures.")
        with np.load(destination/"features.npz", allow_pickle=False) as payload:
            if not np.array_equal(payload["features"], values):
                raise ValueError("Existing order vectors differ from actual endpoint tensors.")
        return probes.load_store(destination/"manifest.json")[0]
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Partial order feature store exists; inspect before exporting.")
    destination.mkdir(exist_ok=True)
    for name, value in contents.items():
        atomic_write(destination/name, value)
    save_npz(destination/"features.npz", features=values)
    manifest = {"schema": probes.STORE_SCHEMA, "artifact_type": "real", "complete": True,
        "base_count": 50, "prompt_cells": 400, "shape": list(values.shape), "feature_names": list(probes.READOUTS),
        "materials_fingerprint": plan["fingerprint"], "preparation_fingerprint": preparation["fingerprint"],
        "files": {p.name: file_hash(p) for p in destination.iterdir() if p.is_file()}}
    manifest["fingerprint"] = digest(manifest)
    atomic_write(destination/"manifest.json", manifest)
    probes.load_store(destination/"manifest.json")
    return manifest


def public_files(root, reports_only=False):
    root = Path(root)
    allowed = {"materials", "original_store", "original_replay", "checkpoints", "captures", "logs", "feature_store", "code_snapshot"}
    names = {"config.json", "runtime.json", "progress_status.json", "preflight_summary.json", "capture_summary.json",
             "job_status.json", "unattended_job.log", "failure.json"}
    result = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Order backup refuses symlinks.")
        if not path.is_file() or path.name.startswith(".") and path.name.endswith(".lock"):
            continue
        relative = path.relative_to(root)
        if relative.parts[0] not in allowed and relative.as_posix() not in names:
            raise ValueError("Unexpected order output; never package private configuration.")
        result["run/"+relative.as_posix()] = path
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "run", "verify", "export"), required=True)
    parser.add_argument("--stage", choices=("preflight", "full"), default="preflight")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--materials_root")
    parser.add_argument("--source_root")
    parser.add_argument("--source_store")
    args = parser.parse_args()
    with lock(args.output_root, ".order_capture.lock") if args.action != "prepare" else ExitStack():
        if args.action == "prepare":
            result = prepare(args.output_root, args.materials_root, args.source_root, args.source_store)
        elif args.action == "run":
            result = run(args.output_root, args.stage == "preflight")
        elif args.action == "verify":
            result = verify(args.output_root, args.stage == "preflight")
        else:
            result = export(args.output_root)
        print(json.dumps(result, indent=2))
