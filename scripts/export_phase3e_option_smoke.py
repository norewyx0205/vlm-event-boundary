"""CPU-only readout smoke test on the frozen two-scene real preflight backup."""

import argparse
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts.backup_phase3b import verify_backup
from scripts.phase3c_core import atomic_write, digest, file_hash
from scripts.phase3e_option_features import (READOUTS, archived_readouts, audit_paired_readouts,
    from_preflight_payload, statement_features)
from scripts.phase3e_paired_design import labels_and_pairs, validate_materials
from scripts.phase3e_preflight import AttentionSites
from scripts.run_phase3c_preflight import logits_parity
from scripts.run_phase3d_probes import save_npz


def run(backup_dir, output_root):
    backup, root = Path(backup_dir).resolve(), Path(output_root).resolve()
    if root == backup or root in backup.parents or backup in root.parents:
        raise ValueError("Use a new smoke output root outside the frozen backup.")
    if root.exists() and any(root.iterdir()):
        raise ValueError("Never overwrite a prior export.")
    manifest = verify_backup(backup)
    if manifest.get("schema") != "phase3e_preflight_backup_v1":
        raise ValueError("Need the original real two-scene technical preflight package.")
    with ExitStack() as stack:
        archives = [stack.enter_context(zipfile.ZipFile(p)) for p in sorted(backup.glob("part_*.zip"))]
        members = {}
        for archive in archives:
            for name in archive.namelist():
                if name in members:
                    raise ValueError("Duplicate archive member.")
                members[name] = archive

        def read(name):
            return members["run/"+name].read("run/"+name)

        def json_(name):
            return json.loads(read(name))

        plan, config = json_("plan.json"), json_("execution_config.json")
        if (plan["fingerprint"] != digest({k: v for k, v in plan.items() if k != "fingerprint"}) or
                config["plan_fingerprint"] != plan["fingerprint"]):
            raise ValueError("Archived preflight plan/config changed.")
        seeds = sorted({r["scene_seed"] for r in plan["rows"]})
        validate_materials(plan["rows"], seeds)
        original = json_("processor_audit.json")["records"]
        mappings, captured = {}, {}
        for row in plan["rows"]:
            identity = row["eval_id"]
            checkpoint = json_(f"checkpoints/{identity}.json")
            if (checkpoint["record_sha256"] != digest({k: v for k, v in checkpoint.items() if k != "record_sha256"}) or
                    checkpoint["plan_fingerprint"] != plan["fingerprint"] or
                    checkpoint["execution_config_sha256"] != digest(config)):
                raise ValueError("Archived capture checkpoint changed.")
            data = read(checkpoint["tensor_path"])
            if hashlib.sha256(data).hexdigest() != checkpoint["tensor_sha256"]:
                raise ValueError("Archived capture tensor checksum mismatch.")
            payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
            if (payload["eval_id"] != identity or payload["plan_fingerprint"] != plan["fingerprint"] or
                    not logits_parity(payload["baseline_logits"], payload["capture_logits"], exact=True)["passed"] or
                    not logits_parity(payload["baseline_logits"], payload["standard_logits"])["passed"] or
                    payload["groups"] != original[identity]["capture_groups"]):
                raise ValueError("Archived logits/token groups failed replay.")
            controller = AttentionSites.__new__(AttentionSites)
            controller.groups, controller.union = payload["groups"], payload["union_positions"]
            controller.raw, controller.means = payload["timing_raw"], payload["means"]
            controller.counts = Counter({site: 1 for site in controller.means})
            controller.validate()
            groups, audit = archived_readouts(row, original[identity])
            mappings[identity] = {"option_readout_audit": audit}
            captured[identity] = from_preflight_payload(payload, groups)
            del data, payload, controller
        alignment = audit_paired_readouts(plan["rows"], mappings)
        sites = sorted(next(iter(captured.values()))["means"])
        names = [f"{kind}:{site}" for kind in READOUTS for site in sites]
        examples, pairs, boundaries = labels_and_pairs(plan["rows"])
        features = np.stack([statement_features(captured[r["eval_id"]], r["option"], names) for r in examples])
    root.mkdir(parents=True, exist_ok=True)
    save_npz(root / "smoke_features.npz", features=features)
    for name, value in (("examples", examples), ("order_pairs", pairs), ("boundary_pairs", boundaries),
                        ("readout_alignment", mappings)):
        atomic_write(root / f"{name}.json", value)
    summary = {"schema": "phase3e_option_export_smoke_v1", "status": "real_capture_export_smoke_passed",
        "independent_scene_families": len(seeds), "scene_seeds": seeds,
        "prompt_cells": len(plan["rows"]), "statement_rows": len(examples),
        "feature_names": names, "shape": list(features.shape), "alignment": alignment,
        "source_backup_manifest_sha256": file_hash(backup / "backup_manifest.json"),
        "missing_formal_sites": ["L16_reference_last", "L20_reference_last"],
        "new_GPU_inference": False, "probe_fitting_performed": False,
        "eligible_for_formal_training": False,
        "code_sha256": {name: file_hash(Path(__file__).with_name(name)) for name in (
            "export_phase3e_option_smoke.py", "phase3e_option_features.py", "phase3e_paired_design.py")},
        "files": {p.name: file_hash(p) for p in root.iterdir() if p.is_file()}}
    atomic_write(root / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup_dir", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.backup_dir, args.output_root), indent=2))
