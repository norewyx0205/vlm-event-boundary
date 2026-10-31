"""Portable, checksum-bound option feature stores; partial smoke data cannot train."""

from pathlib import Path

import numpy as np

from .phase3c_core import atomic_write, digest, file_hash, read_json
from .phase3e_option_features import FEATURE_NAMES, SCHEMA
from .phase3e_option_probes import validate_dataset
from .run_phase3d_probes import resource, save_npz

GATES = ("paired_token_alignment", "capture_exact_noop", "standard_generation_parity",
         "residual_timing", "video_tensor_parity", "all_statement_features_present")


def validate_decisions(decisions, rows):
    if set(decisions) != {row["eval_id"] for row in rows}:
        raise ValueError("Missing behavioural decisions.")
    for row in rows:
        decision = decisions[row["eval_id"]]
        levels = [decision[k] for k in ("correct_logit", "incorrect_logit", "margin")]
        if (not all(type(value) in (int, float) and np.isfinite(value) for value in levels) or
                decision["correct_option"] != row["correct_option"] or
                decision["margin"] != decision["correct_logit"]-decision["incorrect_logit"] or
                decision["is_correct"] != (decision["prediction"] == row["correct_option"])):
            raise ValueError("Behavioural decision has inconsistent raw levels, margin or annotation.")


def write_store(root, plan, examples, values, decisions, gates, source):
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a new immutable feature-store directory.")
    validate_dataset(plan, examples, FEATURE_NAMES, values)
    if any(gates.get(name) is not True for name in GATES) or gates.get("prompt_cells") != 640:
        raise ValueError("Formal feature export requires every real capture gate.")
    validate_decisions(decisions, plan["rows"])
    root.mkdir(parents=True, exist_ok=True)
    for name, value in (("plan", plan), ("examples", examples), ("decisions", decisions),
                        ("gates", gates), ("source", source)):
        atomic_write(root / f"{name}.json", value)
    save_npz(root / "features.npz", features=values.astype(np.float32))
    manifest = {"schema": SCHEMA, "artifact_type": "real", "complete": True,
        "plan_fingerprint": plan["fingerprint"], "feature_names": list(FEATURE_NAMES),
        "shape": list(values.shape), "prompt_cells": 640, "statement_rows": 1280,
        "independent_scene_families": 40, "files": {name: {"path": name, "sha256": file_hash(root/name)}
            for name in ("plan.json", "examples.json", "decisions.json", "gates.json", "source.json", "features.npz")}}
    manifest["fingerprint"] = digest(manifest)
    atomic_write(root / "manifest.json", manifest)
    return manifest


def load_store(path):
    path = Path(path).resolve()
    root, manifest = path.parent, read_json(path)
    if (manifest.get("schema") != SCHEMA or manifest.get("artifact_type") != "real" or
            manifest.get("complete") is not True or
            manifest.get("fingerprint") != digest({k: v for k, v in manifest.items() if k != "fingerprint"})):
        raise ValueError("Training requires an intact, complete real Phase 3E-B store, not smoke data.")
    files = manifest["files"]
    paths = {name: resource(root, spec) for name, spec in files.items()}
    plan, examples, decisions, gates = (read_json(paths[f"{name}.json"])
                                      for name in ("plan", "examples", "decisions", "gates"))
    if (plan["fingerprint"] != manifest["plan_fingerprint"] or
            any(gates.get(name) is not True for name in GATES) or gates.get("prompt_cells") != 640):
        raise ValueError("Feature-store gates or design provenance changed.")
    with np.load(paths["features.npz"], allow_pickle=False) as payload:
        values = payload["features"]
    validate_dataset(plan, examples, manifest["feature_names"], values)
    validate_decisions(decisions, plan["rows"])
    if list(values.shape) != manifest["shape"] or set(decisions) != {r["eval_id"] for r in plan["rows"]}:
        raise ValueError("Feature-store shape/decisions changed.")
    return manifest, plan, examples, values, decisions
