"""Capture only missing pure-A/B rescue50 reference tokens, never new cases."""

import argparse
import gc
import json
import shutil
import sys
from collections import Counter
from contextlib import ExitStack, contextmanager
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts import phase3d_execution as source
from scripts.phase3c_core import atomic_write, digest, file_hash, frozen_write, read_json
from scripts.phase3c_interventions import assert_hidden, hidden_output
from scripts.phase3d_single_layer_ko import check_logits
from scripts.phase3e_option_features import archived_readouts, option_readouts
from scripts.prepare_phase3e_rescue50 import READOUTS, SCHEMA as PLAN_SCHEMA
from scripts.run_phase3c_preflight import Engine, logits_parity, save_tensors
from scripts.run_phase3d_probes import save_npz
from scripts.run_phase3d_vm import lock
from scripts.verify_phase3e_rescue50_recovery import load_partial

SCHEMA = "phase3e_rescue50_missing_option_capture_v1"
LAYERS = (16, 17, 18, 20)
RESERVE = 12*1024**3


def code_hashes():
    names = ("run_phase3e_rescue50_capture.py", "run_phase3e_rescue50_unattended.py",
             "phase3d_execution.py", "phase3e_option_features.py", "verify_phase3e_rescue50_recovery.py",
             "recover_phase3e_rescue50.py", "phase3c_interventions.py", "run_phase3c_preflight.py")
    return {name: file_hash(Path(__file__).with_name(name)) for name in names}


def space(root, extra=0):
    free = shutil.disk_usage(root).free
    if free < RESERVE+extra:
        raise RuntimeError("Insufficient persistent space; no files will be deleted.")
    return free


def input_context(root):
    root = Path(root)
    config = read_json(root/"config.json")
    if config["schema"] != SCHEMA or config["code_sha256"] != code_hashes():
        raise ValueError("Capture implementation changed; preserve this run and use a new root.")
    for name, sha in config["input_files"].items():
        path = root/"inputs"/name
        if path.is_symlink() or not path.resolve().is_relative_to((root/"inputs").resolve()) or file_hash(path) != sha:
            raise ValueError("Portable recovery inputs changed.")
    plan = read_json(root/"inputs/plan.json")
    if (plan.get("schema") != PLAN_SCHEMA or plan.get("fingerprint") != digest(
            {k: v for k, v in plan.items() if k != "fingerprint"}) or plan["readouts"] != READOUTS or
            plan["fingerprint"] != config["preparation_fingerprint"] or len(plan["case_ids"]) != 50):
        raise ValueError("Require the unchanged frozen rescue50 preparation.")
    manifest, values = load_partial(root/"inputs/recovery/recovery_manifest.json")
    proof = read_json(root/"inputs/verification.json")
    if (proof.get("status") != "verified" or proof.get("vectors_replayed_exactly") != 800 or
            proof["recovery_manifest_sha256"] != file_hash(root/"inputs/recovery/recovery_manifest.json") or
            manifest["preparation_sha256"] != file_hash(root/"inputs/plan.json") or
            proof["preparation_sha256"] != manifest["preparation_sha256"]):
        raise ValueError("Recovered original-token verification does not bind these inputs.")
    context = source.load_context(config["source_root"])
    old = context[0]
    if (old["fingerprint"] != plan["source_plan_fingerprint"] or old["case_ids"] != plan["case_ids"] or
            old["settings"] != plan["settings"] or
            read_json(Path(config["source_root"])/"outer_folds.json") != plan["outer_folds"]):
        raise ValueError("Live historical source changed cohort, model/input settings or folds.")
    return config, plan, context, values


def tasks(context, plan):
    specs = [s for s in source.feature_tasks(context[0], "capture") if s["view"] == "option_order_permuted"]
    selected = {(t["pair_id"], t["condition"]): t for t in plan["tasks"] if t["view"] == "option_order_permuted"}
    if len(specs) != 100 or len(selected) != 100:
        raise ValueError("Exactly 100 missing pure A/B prompt views are required.")
    for spec in specs:
        row = source.original_row(context, spec)
        expected = selected[(spec["pair_id"], spec["condition"])]
        if any(row[k] != expected[k] for k in ("eval_id", "option_A", "option_B", "correct_option")):
            raise ValueError("Missing prompt changed original sentence content or answer permutation.")
    return specs


def select(specs, context, preflight):
    ids = set(context[0]["preflight_ids"])
    result = [s for s in specs if s["pair_id"] in ids] if preflight else specs
    if preflight and (len(result) != 4 or len(ids) != 2):
        raise ValueError("Preflight must include both conditions of two original first-mover cases.")
    return result


def source_audit(root, context, spec):
    audit_spec = next(s for s in source.feature_tasks(context[0], "audit", [spec["pair_id"]])
                      if s["condition"] == spec["condition"] and s["view"] == spec["view"])
    audit = source.saved_checkpoint(root, context[0], audit_spec)
    baseline = source.saved_checkpoint(root, context[0], spec)
    if audit is None or baseline is None or baseline["audit_record_sha256"] != audit["record_sha256"]:
        raise ValueError("Missing sealed original permuted-view processor/baseline reference.")
    row = source.original_row(context, spec)
    groups, token_audit = archived_readouts(row, {"processor_record": audit["processor_record"],
                                                 "capture_groups": audit["groups"]})
    return audit, baseline, groups, token_audit


class OptionBlocks:
    """Read two contextual reference tokens at four outputs without modifying them."""

    def __init__(self, text, length, positions):
        if len(text.layers) != 36 or len(positions) != 2 or len(set(positions)) != 2 or any(
                type(p) is not int or p < 0 or p >= length for p in positions):
            raise ValueError("Invalid fixed option readout positions.")
        self.text, self.length, self.positions = text, length, positions
        self.values, self.visits = {}, Counter()

    @contextmanager
    def installed(self):
        def hook(layer):
            def observed(_module, _args, output):
                hidden = hidden_output(output)
                assert_hidden(hidden, self.length)
                self.visits[layer] += 1
                if self.visits[layer] != 1:
                    raise RuntimeError("Repeated decoder visit; one uncached prefill is required.")
                self.values[layer] = hidden[0, self.positions].detach().float().cpu().clone()
            return observed
        with ExitStack() as stack:
            for layer in LAYERS:
                stack.callback(self.text.layers[layer].register_forward_hook(hook(layer)).remove)
            yield self

    def features(self):
        if set(self.values) != set(LAYERS) or any(v != 1 for v in self.visits.values()):
            raise ValueError("Incomplete option block-output capture.")
        result = torch.stack([self.values[layer] for layer in LAYERS]).permute(1, 0, 2).contiguous()
        if result.ndim != 3 or result.shape[:2] != (2, 4) or not bool(torch.isfinite(result).all()):
            raise ValueError("Invalid/nonfinite reference-token capture.")
        return result


def prepare(root, source_root, inputs):
    root, source_root, inputs = (Path(p).resolve() for p in (root, source_root, inputs))
    if any(root == p or root in p.parents or p in root.parents for p in (source_root, inputs)):
        raise ValueError("Use a separate new capture root.")
    if root.exists() and any(root.iterdir()):
        raise ValueError("Capture preparation requires a new empty directory.")
    context = source.load_context(source_root)
    plan = read_json(inputs/"plan.json")
    if plan["source_plan_fingerprint"] != context[0]["fingerprint"]:
        raise ValueError("Wrong rescue50 source.")
    root.mkdir(parents=True, exist_ok=True)
    space(root, 1024**3)
    shutil.copytree(inputs, root/"inputs")
    files = {p.relative_to(root/"inputs").as_posix(): file_hash(p) for p in (root/"inputs").rglob("*") if p.is_file()}
    config = {"schema": SCHEMA, "source_root": str(source_root), "source_plan_sha256": file_hash(source_root/"plan.json"),
        "preparation_fingerprint": plan["fingerprint"], "input_files": files, "code_sha256": code_hashes(),
        "settings": plan["settings"], "execution_mode": "model_parallel",
        "gpu_weight_budget_gib": context[0]["source_runtime"]["gpu_weight_budget_gib"],
        "readouts": READOUTS, "new_KO_or_patch_forwards": 0, "missing_prompt_cells": 100}
    frozen_write(root/"config.json", config)
    _, plan, context, _ = input_context(root)
    specs = tasks(context, plan)
    frozen_write(root/"task_manifest.json", specs)
    return {"status": "prepared", "prompt_cells": len(specs), "preflight_prompt_cells": 4,
            "free_GiB": space(root)/1024**3, "GPU_started": False}


def paths(root, spec):
    return Path(root)/"checkpoints"/(spec["task_id"]+".json"), Path(root)/"captures"/(spec["task_id"]+".pt")


def verify_one(root, config, context, spec):
    checkpoint, tensor = paths(root, spec)
    row = source.checked_seal(read_json(checkpoint))
    audit, baseline, groups, token_audit = source_audit(config["source_root"], context, spec)
    positions = [token_audit[o]["reference_last_position"] for o in ("A", "B")]
    if (row["spec"] != spec or row["config_sha256"] != digest(config) or row["tensor_sha256"] != file_hash(tensor) or
            row["source_audit_sha256"] != audit["record_sha256"] or row["source_baseline_sha256"] != baseline["record_sha256"] or
            row["readout_audit"] != token_audit or row["decision"] != baseline["decision"]):
        raise ValueError("New capture checkpoint binding changed.")
    data = torch.load(tensor, map_location="cpu", weights_only=True)
    original = source.payload(config["source_root"], baseline)
    check_logits(data["logits"], row["decision"])
    if (data["positions"] != positions or data["features"].shape != (2, 4, 4096) or
            data["features"].dtype != torch.float32 or not bool(torch.isfinite(data["features"]).all()) or
            not torch.equal(data["logits"], original["logits"]) or
            not torch.equal(data["logits"], data["capture_logits"]) or
            not logits_parity(data["logits"], data["standard_logits"])["passed"] or
            int(data["standard_logits"].argmax()) != row["decision"]["predicted_first_token_id"]):
        raise ValueError("Exact original baseline/no-op or standard-generation parity failed.")
    return row, data


def verify(root, preflight=False):
    config, plan, context, _ = input_context(root)
    specs = select(tasks(context, plan), context, preflight)
    rows = [verify_one(root, config, context, spec)[0] for spec in specs]
    return {"schema": SCHEMA, "stage": "preflight" if preflight else "full", "complete": not preflight,
        "stage_verified": True, "prompt_cells": len(rows), "base_count": 2 if preflight else 50,
        "config_sha256": digest(config), "record_sha256": [r["record_sha256"] for r in rows],
        "original_baseline_exact": True, "capture_noop_exact": True, "option_alignment": True,
        "new_KO_or_patch": False, "readouts": READOUTS}


def run(root, preflight=False):
    from scripts.phase3c_budget import runtime_compatibility
    from scripts.probe_attention_roi import standard_first_token
    root = Path(root)
    config, plan, context, _ = input_context(root)
    specs = select(tasks(context, plan), context, preflight)
    if not preflight:
        if read_json(root/"preflight_summary.json") != verify(root, True):
            raise ValueError("Full supplementation requires the same verified four-prompt preflight.")
    space(root, 1024**3)
    engine = Engine(config["settings"], "model_parallel", config["gpu_weight_budget_gib"])
    try:
        runtime_compatibility(context[0]["source_runtime"], engine.runtime)
        frozen_write(root/"runtime.json", engine.runtime)
        for index, spec in enumerate(specs):
            space(root)
            checkpoint, tensor = paths(root, spec)
            if checkpoint.exists():
                verify_one(root, config, context, spec)
                continue
            if tensor.exists():
                raise ValueError("Uncommitted tensor exists; inspect it before resuming.")
            audit, baseline, _, old_tokens = source_audit(config["source_root"], context, spec)
            live, record = source.prepare_live(engine.processor, context, spec)
            groups, token_audit = option_readouts(live["row"], record, engine.processor.tokenizer)
            if record != audit["processor_record"] or token_audit != old_tokens:
                raise ValueError("Live tokenization/tensor inputs differ from the original A/B control.")
            old = source.payload(config["source_root"], baseline)
            live["inputs"] = live["inputs"].to(engine.model.device)
            plain = engine.forward(live)
            if not torch.equal(plain, old["logits"]) or engine.decide(plain, live["row"]) != baseline["decision"]:
                raise ValueError("Frozen permuted baseline did not replay exactly; do not replace/drop the case.")
            positions = [token_audit[o]["reference_last_position"] for o in ("A", "B")]
            collector = OptionBlocks(engine.text, len(record["input_ids"]), positions)
            with collector.installed():
                observed = engine.forward(live)
            token, standard = standard_first_token(engine.model, live["inputs"])
            if (not torch.equal(plain, observed) or not logits_parity(plain, standard)["passed"] or
                    token != int(plain.argmax())):
                raise ValueError("Read-only option capture changed model output or generation parity.")
            features = collector.features()
            if features.shape != (2, 4, 4096):
                raise ValueError("Wrong Qwen hidden dimension.")
            save_tensors(tensor, {"features": features, "positions": positions,
                "logits": plain, "capture_logits": observed, "standard_logits": standard})
            row = source.seal({"spec": spec, "config_sha256": digest(config), "tensor_sha256": file_hash(tensor),
                "source_audit_sha256": audit["record_sha256"], "source_baseline_sha256": baseline["record_sha256"],
                "readout_audit": token_audit, "decision": baseline["decision"]})
            atomic_write(checkpoint, row)
            atomic_write(root/"progress_status.json", {"stage": "preflight" if preflight else "full",
                "completed": index+1, "expected": len(specs), "eval_id": live["row"]["eval_id"]})
            print(f"RESCUE50_CAPTURE {index+1}/{len(specs)} {live['row']['eval_id']}", flush=True)
            del old, live, plain, observed, standard, features, collector
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    summary = verify(root, preflight)
    atomic_write(root/("preflight_summary.json" if preflight else "capture_summary.json"), summary)
    return summary


def export(root):
    root = Path(root)
    proof = verify(root)
    config, plan, context, original = input_context(root)
    examples = read_json(root/"inputs/recovery/examples.json")
    vectors = {e["example_id"]: original[i] for i, e in enumerate(examples)}
    decisions = read_json(root/"inputs/recovery/decisions.json")
    for spec in tasks(context, plan):
        record, data = verify_one(root, config, context, spec)
        row = source.original_row(context, spec)
        for i, option in enumerate(("A", "B")):
            vectors[row["eval_id"]+"__statement_"+option] = data["features"][i].numpy()
        decisions[row["eval_id"]] = record["decision"]
    ordered = plan["statement_examples"]
    if set(vectors) != {e["example_id"] for e in ordered} or len(decisions) != 200:
        raise ValueError("Combined feature export is missing an unchanged statement/view.")
    values = np.stack([vectors[e["example_id"]] for e in ordered])
    if values.shape != (400, 4, 4096) or not np.isfinite(values).all():
        raise ValueError("Invalid complete rescue50 feature array.")
    destination = root/"feature_store"
    if destination.exists():
        existing = read_json(destination/"manifest.json")
        if (existing.get("schema") != "phase3e_rescue50_option_store_v1" or
                existing.get("complete") is not True or existing.get("config_sha256") != digest(config) or
                existing.get("fingerprint") != digest({k: v for k, v in existing.items() if k != "fingerprint"}) or
                set(existing["files"]) != {"plan.json", "examples.json", "decisions.json", "capture_verification.json", "features.npz"} or
                any(file_hash(destination/n) != sha for n, sha in existing["files"].items())):
            raise ValueError("Existing completed store changed.")
        with np.load(destination/"features.npz", allow_pickle=False) as stored:
            if not np.array_equal(stored["features"], values):
                raise ValueError("Completed store does not replay the actual recovered/captured vectors.")
        return existing
    destination.mkdir()
    for name, item in (("plan", plan), ("examples", ordered), ("decisions", decisions), ("capture_verification", proof)):
        atomic_write(destination/f"{name}.json", item)
    save_npz(destination/"features.npz", features=values)
    manifest = {"schema": "phase3e_rescue50_option_store_v1", "artifact_type": "real", "complete": True,
        "base_count": 50, "prompt_cells": 200, "statement_rows": 400, "shape": list(values.shape),
        "feature_names": READOUTS, "preparation_fingerprint": plan["fingerprint"], "config_sha256": digest(config),
        "files": {p.name: file_hash(p) for p in destination.iterdir() if p.is_file()}}
    manifest["fingerprint"] = digest(manifest)
    atomic_write(destination/"manifest.json", manifest)
    return manifest


def public_files(root, reports_only=False):
    root = Path(root)
    allowed = {"inputs", "captures", "checkpoints", "logs", "code_snapshot", "feature_store"}
    names = {"config.json", "task_manifest.json", "runtime.json", "progress_status.json", "preflight_summary.json",
             "capture_summary.json", "verification.json", "job_status.json", "unattended_job.log", "failure.json"}
    result = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Backup refuses symlinks.")
        if not path.is_file() or path.name.startswith(".") and path.name.endswith(".lock"):
            continue
        relative = path.relative_to(root)
        if relative.parts[0] not in allowed and relative.as_posix() not in names:
            raise ValueError("Unexpected public output; never package private configuration.")
        result["run/"+relative.as_posix()] = path
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "run", "verify", "export"), required=True)
    parser.add_argument("--stage", choices=("preflight", "full"), default="preflight")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--source_root")
    parser.add_argument("--input_root")
    args = parser.parse_args()
    with lock(args.output_root, ".rescue50_capture.lock") if args.action != "prepare" else ExitStack():
        if args.action == "prepare":
            result = prepare(args.output_root, args.source_root, args.input_root)
        elif args.action == "run":
            result = run(args.output_root, args.stage == "preflight")
        elif args.action == "verify":
            result = verify(args.output_root, args.stage == "preflight")
        else:
            result = export(args.output_root)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
