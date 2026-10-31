"""Add L17 to the completed balanced CPU analysis without refitting old sites."""

import argparse
from pathlib import Path
import platform
import sys

import numpy as np
import sklearn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import reanalyze_phase3d_balanced_probes as base
from scripts.phase3c_core import atomic_write, file_hash, read_json


PRIMARY = ["source_pair:L17_output", "option_pair:L17_output"]
SECONDARY = [f"option_{option}_{kind}:L17_output" for option in ("A", "B") for kind in ("pair", "relation")]


def runtime_info():
    return {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__}


def verify_reference(reference, source):
    root = Path(reference).resolve()
    summary, config = read_json(root / "summary.json"), read_json(root / "run_config.json")
    if (summary.get("schema") != "phase3e_balanced_probe_reanalysis_v1" or
            summary.get("status") != "complete" or summary.get("base_count") != 50 or
            summary.get("baseline_rows") != 200 or summary.get("source_bytes_unchanged") is not True):
        raise ValueError("Need the completed, unchanged 50-base balanced reference run")
    if config["source_manifest_sha256"] != file_hash(Path(source)):
        raise ValueError("Reference and extension must use the exact same feature manifest")
    if config["settings"] != {**base.SETTINGS, "label_permutations": 0}:
        raise ValueError("Reference probe settings differ from the fixed extension settings")
    if config["runtime"] != runtime_info():
        raise ValueError("Extension and reference need the same recorded numeric runtime")
    for name, digest in config["code_sha256"].items():
        if file_hash(Path(base.__file__).with_name(name)) != digest:
            raise ValueError("The reference implementation changed; do not mix readout versions")
    manifest = read_json(root / "output_manifest.json")
    seen = set()
    for item in manifest["files"]:
        path = (root / item["path"]).resolve()
        if (not path.is_relative_to(root) or item["path"] in seen or
                file_hash(path) != item["sha256"]):
            raise ValueError("Changed, duplicate or unsafe reference output")
        seen.add(item["path"])
    if not {"summary.json", "run_config.json", "input_audit.json"} <= seen:
        raise ValueError("Incomplete reference hash inventory")
    audit = read_json(root / "input_audit.json")
    if not audit["complete"]:
        raise ValueError("Reference input audit is incomplete")
    reference_bytes = {item.get("resource", "manifest"): item["sha256"] for item in audit["resources"]}
    current = base.input_audit(source)
    if (not current["complete"] or reference_bytes != {
            item.get("resource", "manifest"): item["sha256"] for item in current["resources"]}):
        raise ValueError("Reference and restored input resources do not match")
    return config, {"path": str(root), "summary_sha256": file_hash(root / "summary.json"),
                    "output_manifest_sha256": file_hash(root / "output_manifest.json"),
                    "config_sha256": file_hash(root / "run_config.json")}


def run(manifest_path, reference_run, output):
    source, root = base.isolated_output(manifest_path, output)
    reference = Path(reference_run).resolve()
    if root == reference or root in reference.parents or reference in root.parents:
        raise ValueError("Extension output must be outside the frozen reference run")
    reference_config, reference_record = verify_reference(reference, source)
    audit = base.input_audit(source)
    _, rows, cases, folds, names, values = base.load_dataset(source)
    balance = base.balanced_grid(rows, cases, folds)
    if not set(PRIMARY + SECONDARY) <= set(names):
        raise ValueError("Missing declared L17 readouts; no silent omissions")
    root.mkdir(parents=True, exist_ok=False)
    atomic_write(root / "input_audit.json", audit)
    settings = reference_config["settings"]
    config = {"schema": "phase3e_l17_balanced_extension_v1", "reference_run": reference_record,
              "source_manifest_sha256": file_hash(source), "settings": settings,
              "primary": PRIMARY, "secondary": SECONDARY,
              "code_sha256": {"extend_phase3e_l17_probes.py": file_hash(Path(__file__)),
                              **reference_config["code_sha256"]},
              "layer_selection": "Author-requested intervention-informed L17 follow-up, not independent confirmation.",
              "training": reference_config["training"], "runtime": runtime_info()}
    atomic_write(root / "run_config.json", config)
    outputs = {}
    for name in PRIMARY + SECONDARY:
        print("Fitting L17 readout:", name, flush=True)
        fitted = base.crossfit_balanced(rows, values[:, names.index(name), :], cases, folds, settings)
        outputs[name] = base.export_fit(root / name.replace(":", "__"), fitted, settings, paired=True)
    name = "delta_source_pair:L17"
    drows, deltas = base.delta_features(rows, values[:, names.index("source_pair:L17_output"), :])
    print("Fitting L17 boundary-modulation diagnostic", flush=True)
    fitted = base.crossfit_balanced(drows, deltas, cases, folds, settings)
    destination = root / name.replace(":", "__")
    outputs[name] = base.export_fit(destination, fitted, settings, paired=False)
    base.save_npz(destination / "paired_deltas.npz", features=deltas,
                  base_sample_ids=np.array([row["base_sample_id"] for row in drows]),
                  views=np.array([row["view"] for row in drows]))
    outputs[name].update(mean_delta_norm=float(np.linalg.norm(deltas, axis=1).mean()),
                         interpretation="Boundary modulation of object contrast; not a pure temporal direction.")
    atomic_write(destination / "summary.json", outputs[name])
    if base.input_audit(source) != audit or verify_reference(reference, source)[1] != reference_record:
        raise ValueError("Source or reference run changed during the extension")
    summary = {"schema": config["schema"], "status": "complete", "base_count": 50,
               "baseline_rows": 200, "correct_option_by_condition": balance, "results": outputs,
               "source_bytes_unchanged": True, "reference_run_unchanged": True,
               "reference_run": reference_record, "GPU_inference": False,
               "controls": "Reuse hash-verified lexical/static reference results under identical settings/folds; no new spatial/motion control.",
               "interpretation": "Exploratory L16/L17/L18 comparison, not an identified causal chain or independent replication."}
    atomic_write(root / "summary.json", summary)
    atomic_write(root / "output_manifest.json", {"files": [
        {"path": path.relative_to(root).as_posix(), "sha256": file_hash(path)}
        for path in sorted(root.rglob("*")) if path.is_file()]})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(run(args.manifest, args.reference_run, args.out)["status"])


if __name__ == "__main__":
    main()
