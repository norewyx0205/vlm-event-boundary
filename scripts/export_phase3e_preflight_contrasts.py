"""Export paired representation contrasts from a verified technical preflight."""

import sys
from pathlib import Path

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.phase3c_core import atomic_write, file_hash
from scripts.run_phase3c_preflight import save_tensors
from scripts import phase3e_preflight as core
from scripts.run_phase3e_preflight import verify


def paired_contrasts(plan, means):
    result = []
    for low in plan["rows"]:
        if low["condition"] != "low_boundary":
            continue
        fields = ("scene_seed", "first_object_id", "subject_id", "layout")
        temporal = [r for r in plan["rows"] if r["condition"] == "temporal_boundary" and
                    all(r[k] == low[k] for k in fields)]
        if len(temporal) != 1:
            raise ValueError("Require exactly one matched temporal prompt per low prompt.")
        temporal = temporal[0]
        left, right = means[low["eval_id"]], means[temporal["eval_id"]]
        if set(left) != set(right):
            raise ValueError("Paired capture sites differ.")
        sites = {}
        for site in left:
            mu = lambda item, target: item[site][f"source_t{target}"]
            d_low, d_temporal = mu(left, 1)-mu(left, 2), mu(right, 1)-mu(right, 2)
            b_low, b_temporal = left[site]["whole_event2"], right[site]["whole_event2"]
            sites[site] = {"D_low": d_low, "D_temporal": d_temporal,
                "delta_D": d_temporal-d_low, "B_low": b_low, "B_temporal": b_temporal,
                "delta_B": b_temporal-b_low}
        result.append({**{k: low[k] for k in fields}, "low_eval_id": low["eval_id"],
                       "temporal_eval_id": temporal["eval_id"], "sites": sites})
    return result


def export(root):
    root = Path(root).resolve()
    verified = verify(root)
    plan = core.load_plan(root)
    means = {}
    for row in plan["rows"]:
        payload = torch.load(root / "captures" / (row["eval_id"]+".pt"), map_location="cpu", weights_only=True)
        means[row["eval_id"]] = payload["means"]
        del payload
    pairs = paired_contrasts(plan, means)
    output = root / "logs/paired_representation_contrasts.pt"
    save_tensors(output, {"plan_fingerprint": plan["fingerprint"], "pairs": pairs})
    receipt = {"plan_fingerprint": plan["fingerprint"], "independent_scene_count": 2,
        "paired_prompt_cells": len(pairs), "capture_sites_per_pair": len(pairs[0]["sites"]),
        "path": str(output.relative_to(root)), "sha256": file_hash(output),
        "source_verified_record_hashes": verified["verified_record_hashes"],
        "exporter_sha256": file_hash(__file__), "probe_fitting_performed": False,
        "interpretation": "Vector contrasts retain mixed identity, spatial, motion and timing content; not isolated temporal directions."}
    atomic_write(root / "logs/paired_representation_contrasts.json", receipt)
    return receipt


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", required=True)
    print(json.dumps(export(parser.parse_args().output_root), indent=2))
