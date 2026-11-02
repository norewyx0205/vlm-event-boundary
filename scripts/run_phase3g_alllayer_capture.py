"""Capture all 36 block outputs at the frozen direct-D reference tokens (400 prompt cells)."""

import argparse
import copy
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
from scripts import phase3e_order_probes as probes
from scripts import run_phase3e_order_capture as order
from scripts.phase3b_core import prepare_example
from scripts.phase3c_core import atomic_write, digest, file_hash, frozen_write, read_json
from scripts.phase3c_interventions import assert_hidden, hidden_output
from scripts.run_phase3c_preflight import Engine, save_tensors
from scripts.run_phase3d_probes import save_npz
from scripts.run_phase3d_vm import lock

SCHEMA = "phase3g_alllayer_reference_capture_v1"
STORE_SCHEMA = "phase3g_alllayer_reference_store_v1"
LAYERS = tuple(range(36))
OLD_LAYERS = (16, 17, 18, 20)
RESERVE = 12*1024**3


def code_hashes():
    root = Path(__file__).resolve().parents[1]
    return {**order.code_hashes(), **{n: file_hash(Path(__file__).with_name(n)) for n in
        ("run_phase3g_alllayer_capture.py", "run_phase3g_alllayer_unattended.py")},
        "phase3g_protocol": file_hash(root/"docs/phase3g_alllayer_probes.md")}


def space(root):
    free = shutil.disk_usage(root).free
    if free < RESERVE+3*1024**3:
        raise RuntimeError("Need 12-GiB reserve plus 3-GiB capture/backup budget; no data deleted.")
    return free


class AllBlocks:
    """Read the two reference tokens at every decoder block output without modifying them."""

    def __init__(self, text, length, positions):
        if len(text.layers) != len(LAYERS) or len(positions) != 2 or len(set(positions)) != 2 or any(
                type(p) is not int or p < 0 or p >= length for p in positions):
            raise ValueError("Invalid fixed reference-token positions.")
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
            raise ValueError("Incomplete all-layer capture.")
        result = torch.stack([self.values[layer] for layer in LAYERS]).permute(1, 0, 2).contiguous()
        if result.shape[:2] != (2, len(LAYERS)) or not bool(torch.isfinite(result).all()):
            raise ValueError("Invalid/nonfinite all-layer capture.")
        return result


def cells(order_context):
    """Reversed cells from the v3 plan plus their original-video parents, grouped by base."""
    _, plan, _, _ = order_context
    result = []
    for spec in plan["rows"]:
        for kind in ("original", "reversed"):
            uid = spec["original_eval_id"] if kind == "original" else spec["row"]["eval_id"]
            result.append({"cell_id": uid, "kind": kind, "pair_id": spec["pair_id"], "condition": spec["condition"],
                           "view": spec["view"], "reversed_eval_id": spec["row"]["eval_id"]})
    if len(result) != 400 or len({c["cell_id"] for c in result}) != 400:
        raise ValueError("Exactly 400 distinct original/reversed prompt cells are required.")
    return result


def context(root):
    root = Path(root)
    config = read_json(root/"config.json")
    if config["schema"] != SCHEMA or config["code_sha256"] != code_hashes():
        raise ValueError("All-layer capture code/protocol changed; preserve this root and use a new one.")
    order_root = Path(config["order_root"])
    order_context = order.context(order_root)
    for name, sha in config["order_files"].items():
        if file_hash(order_root/name) != sha:
            raise ValueError("Frozen v3 order evidence changed.")
    if digest(order_context[0]) != config["order_config_sha256"]:
        raise ValueError("Frozen v3 order configuration changed.")
    return config, order_context


def prepare(root, order_root):
    root, order_root = Path(root).resolve(), Path(order_root).resolve()
    if root == order_root or root in order_root.parents or order_root in root.parents:
        raise ValueError("Use a separate new capture root; v3 evidence is read-only.")
    if root.exists() and any(p.name != ".alllayer.lock" for p in root.iterdir()):
        raise ValueError("Preparation requires a new empty directory.")
    order_context = order.context(order_root)
    summary = read_json(order_root/"capture_summary.json")
    if summary != order.verify(order_root):
        raise ValueError("v3 order capture summary does not replay.")
    probes.load_store(order_root/"feature_store/manifest.json")
    files = {name: file_hash(order_root/name) for name in
             ("config.json", "capture_summary.json", "feature_store/manifest.json", "feature_store/endpoints.json",
              "feature_store/features.npz", "materials/plan.json")}
    root.mkdir(parents=True, exist_ok=True)
    space(root)
    config = {"schema": SCHEMA, "order_root": str(order_root), "order_files": files,
        "order_config_sha256": digest(order_context[0]), "settings": order_context[0]["settings"],
        "gpu_weight_budget_gib": order_context[0]["gpu_weight_budget_gib"], "code_sha256": code_hashes(),
        "layers": list(LAYERS), "replayed_layers": list(OLD_LAYERS), "prompt_cells": 400,
        "new_KO_or_patch": False, "storage_reserve_bytes": RESERVE}
    frozen_write(root/"config.json", config)
    context(root)
    return {"prepared": True, "prompt_cells": 400, "free_GiB": space(root)/1024**3}


def selected(order_context, preflight):
    ids = order_context[2][0]["preflight_ids"]
    return [c for c in cells(order_context) if not preflight or c["pair_id"] in ids]


def paths(root, cell):
    return Path(root)/"checkpoints"/(cell["cell_id"]+".json"), Path(root)/"captures"/(cell["cell_id"]+".pt")


def references(config, order_context, cell):
    """Frozen inputs, reference positions, full logits and four-layer vectors for one cell."""
    order_config, plan, historical, original = order_context
    order_root = Path(config["order_root"])
    spec = next(s for s in plan["rows"] if s["row"]["eval_id"] == cell["reversed_eval_id"])
    if cell["kind"] == "original":
        audit, baseline, _, tokens = order.old_audit(order_config, historical, spec)
        _, _, examples, features, _ = original
        index = [next(i for i, e in enumerate(examples) if e["eval_id"] == cell["cell_id"] and e["option"] == o)
                 for o in ("A", "B")]
        task = next(t for t in source.feature_tasks(historical[0], "capture", [cell["pair_id"]])
                    if t["condition"] == cell["condition"] and t["view"] == cell["view"])
        return {"processor": audit["processor_record"], "task": task, "spec": spec,
                "positions": [tokens[o]["reference_last_position"] for o in ("A", "B")],
                "logits": source.payload(order_config["source_root"], baseline)["logits"],
                "four": torch.from_numpy(features[index])}
    record, payload = order.verify_one(order_root, order_config, historical, spec)
    return {"processor": record["processor_record"], "spec": spec, "positions": payload["positions"],
            "logits": payload["logits"], "four": payload["features"]}


def live_input(engine, config, order_context, cell, reference):
    historical = order_context[2]
    if cell["kind"] == "original":
        live, processor = source.prepare_live(engine.processor, historical, reference["task"])
        row = live["row"]
    else:
        spec, settings = reference["spec"], config["settings"]
        row = {**copy.deepcopy(spec["row"]), "video_path": str(order.safe_video(config["order_root"], spec))}
        live = prepare_example(row, engine.processor, config["order_root"], settings["video_fps"],
            settings["video_num_frames"], settings["video_max_pixels"], settings["roi_padding"])
        processor = source.processor_record(live)
    if processor != reference["processor"]:
        raise ValueError("Live processor inputs differ from the frozen record.")
    live["inputs"] = live["inputs"].to(engine.model.device)
    return live, row, processor


def check_payload(payload, reference):
    old = [LAYERS.index(layer) for layer in OLD_LAYERS]
    if (payload["features"].shape != (2, len(LAYERS), 4096) or not bool(torch.isfinite(payload["features"]).all()) or
            payload["positions"] != reference["positions"] or
            not torch.equal(payload["logits"], reference["logits"]) or
            not torch.equal(payload["features"][:, old], reference["four"])):
        raise ValueError("All-layer capture fails exact logit or four-layer replay.")


def verify(root, preflight=False):
    root = Path(root)
    config, order_context = context(root)
    records = []
    for cell in selected(order_context, preflight):
        path, tensor = paths(root, cell)
        record = source.checked_seal(read_json(path))
        if record["config_sha256"] != digest(config) or record["cell"] != cell or record["tensor_sha256"] != file_hash(tensor):
            raise ValueError("All-layer checkpoint configuration/cell/tensor changed.")
        reference = references(config, order_context, cell)
        payload = torch.load(tensor, map_location="cpu", weights_only=True)
        check_payload(payload, reference)
        records.append(record["record_sha256"])
    return {"schema": SCHEMA, "complete": not preflight, "prompt_cells": len(records), "config_sha256": digest(config),
            "record_sha256": records, "logits_exact": True, "four_layer_replay_exact": True,
            "layers": len(LAYERS), "new_KO_or_patch": False}


def run(root, preflight=False):
    from scripts.phase3c_budget import runtime_compatibility
    root = Path(root)
    config, order_context = context(root)
    if not preflight and read_json(root/"preflight_summary.json") != verify(root, True):
        raise ValueError("Full all-layer capture requires the same passed preflight.")
    engine = Engine(config["settings"], "model_parallel", config["gpu_weight_budget_gib"])
    try:
        runtime_compatibility(order_context[1]["source_runtime"], engine.runtime)
        if (root/"runtime.json").exists():
            runtime_compatibility(read_json(root/"runtime.json"), engine.runtime)
        else:
            frozen_write(root/"runtime.json", engine.runtime)
        chosen = selected(order_context, preflight)
        for index, cell in enumerate(chosen):
            space(root)
            path, tensor = paths(root, cell)
            if path.exists():
                continue
            if tensor.exists():
                raise ValueError("Uncommitted tensor exists; inspect before resuming.")
            reference = references(config, order_context, cell)
            live, row, processor = live_input(engine, config, order_context, cell, reference)
            collector = AllBlocks(engine.text, len(processor["input_ids"]), reference["positions"])
            with collector.installed():
                logits = engine.forward(live)
            payload = {"features": collector.features(), "logits": logits, "positions": reference["positions"]}
            check_payload(payload, reference)
            save_tensors(tensor, payload)
            atomic_write(path, source.seal({"config_sha256": digest(config), "cell": cell,
                "tensor_sha256": file_hash(tensor), "input_fingerprint": digest(processor["input_tensor_sha256"]),
                "decision": engine.decide(logits, row), "logits_exact": True, "four_layer_replay_exact": True}))
            atomic_write(root/"progress_status.json", {"stage": "preflight" if preflight else "full",
                "completed": index+1, "expected": len(chosen), "cell_id": cell["cell_id"]})
            print("ALLLAYER_CAPTURE", index+1, len(chosen), cell["cell_id"], flush=True)
            del live, collector, logits, payload
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
    config, order_context = context(root)
    proof = verify(root)
    if read_json(root/"capture_summary.json") != proof:
        raise ValueError("Full capture summary does not replay.")
    order_root = Path(config["order_root"])
    old_store = order_root/"feature_store"
    endpoints = read_json(old_store/"endpoints.json")
    with np.load(old_store/"features.npz", allow_pickle=False) as payload:
        four = payload["features"]
    captured = {c["cell_id"]: torch.load(paths(root, c)[1], map_location="cpu", weights_only=True)["features"]
                for c in cells(order_context)}
    values = np.stack([captured[e["eval_id"]][{"A": 0, "B": 1}[e["option"]]].numpy() for e in endpoints])
    if values.shape != (800, len(LAYERS), 4096) or not np.array_equal(values[:, list(OLD_LAYERS)], four):
        raise ValueError("All-layer store does not reproduce the frozen v3 four-layer store.")
    destination = root/"feature_store"
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Feature store already exists; inspect before exporting.")
    destination.mkdir()
    for name in ("endpoints.json", "preparation.json", "decisions.json"):
        shutil.copyfile(old_store/name, destination/name)
    save_npz(destination/"features.npz", features=values)
    manifest = {"schema": STORE_SCHEMA, "artifact_type": "real", "complete": True, "base_count": 50,
        "prompt_cells": 400, "shape": list(values.shape), "layers": list(LAYERS),
        "source_v3_store_manifest_sha256": file_hash(old_store/"manifest.json"),
        "four_layer_equal_to_v3": list(OLD_LAYERS),
        "files": {p.name: file_hash(p) for p in destination.iterdir() if p.is_file()}}
    manifest["fingerprint"] = digest(manifest)
    atomic_write(destination/"manifest.json", manifest)
    return manifest


def load_store(path):
    """Validated all-layer store plus the frozen v3 preparation and endpoint order."""
    path = Path(path).resolve()
    manifest = read_json(path)
    if (manifest.get("schema") != STORE_SCHEMA or manifest.get("complete") is not True or
            manifest.get("fingerprint") != digest({k: v for k, v in manifest.items() if k != "fingerprint"}) or
            manifest.get("shape") != [800, len(LAYERS), 4096]):
        raise ValueError("Need the complete all-layer reference-token store.")
    for name, sha in manifest["files"].items():
        if (path.parent/name).is_symlink() or file_hash(path.parent/name) != sha:
            raise ValueError("All-layer store checksum changed.")
    with np.load(path.parent/"features.npz", allow_pickle=False) as payload:
        values = payload["features"]
    return manifest, read_json(path.parent/"preparation.json"), read_json(path.parent/"endpoints.json"), values


def public_files(root, reports_only=False):
    root = Path(root)
    allowed = {"checkpoints", "captures", "feature_store", "code_snapshot", "logs"}
    names = {"config.json", "runtime.json", "progress_status.json", "preflight_summary.json", "capture_summary.json",
             "job_status.json", "unattended_job.log"}
    result = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Backup refuses symlinks.")
        if not path.is_file() or path.name.startswith(".") and path.name.endswith(".lock"):
            continue
        relative = path.relative_to(root)
        if relative.parts[0] not in allowed and relative.as_posix() not in names:
            raise ValueError("Unexpected all-layer output; never package private configuration.")
        if not reports_only or path.suffix not in (".pt", ".npz"):
            result["run/"+relative.as_posix()] = path
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "run", "verify", "export"), required=True)
    parser.add_argument("--stage", choices=("preflight", "full"), default="preflight")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--order_root")
    args = parser.parse_args()
    if args.action == "prepare":
        if not args.order_root:
            parser.error("--order_root must name the completed v3 direct-D capture.")
        result = prepare(args.output_root, args.order_root)
    else:
        with lock(args.output_root, ".alllayer.lock"):
            result = (run(args.output_root, args.stage == "preflight") if args.action == "run" else
                      verify(args.output_root, args.stage == "preflight") if args.action == "verify" else
                      export(args.output_root))
    print(json.dumps(result, indent=2))
