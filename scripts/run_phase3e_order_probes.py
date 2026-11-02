"""Fit/replay direct-D rescue50 probes, keeping all single-video results frozen."""

import argparse
import json
import platform
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

from scripts import phase3e_order_probes as core
from scripts.phase3c_core import atomic_write, digest, file_hash, frozen_write, read_json
from scripts.phase3e_option_probes import SETTINGS
from scripts.run_phase3d_probes import probe_lock, save_npz
from scripts.run_phase3e_option_probes import verify_job

DEPENDENCIES = ("phase3e_order_probes.py", "run_phase3e_order_probes.py", "phase3e_rescue50_probes.py",
    "phase3e_option_probes.py", "phase3d_probes.py", "run_phase3d_probes.py", "run_phase3e_option_probes.py",
    "phase3d_core.py", "phase3c_core.py", "phase3e_condition_probes.py")


def write_job(job, fitted, measured):
    job.mkdir(parents=True, exist_ok=True)
    records = []
    for record in fitted["models"]:
        classifier, scale = (record["model"].named_steps[n] for n in ("probe", "scale"))
        width = classifier.coef_.shape[1]
        save_npz(job/(record["model_id"]+".npz"), coefficient=classifier.coef_, intercept=np.zeros(1),
            scaler_mean=np.zeros(width), scaler_scale=scale.scale_, feature_columns=np.asarray([str(i) for i in range(width)]))
        records.append({k: v for k, v in record.items() if k != "model"})
    for name, payload in (("models", records), ("predictions", fitted["predictions"]), ("evaluation", measured)):
        atomic_write(job/(name+".json"), payload)
    atomic_write(job/"result.json", {"schema": core.SCHEMA, "complete": True,
        "files": {p.name: file_hash(p) for p in job.iterdir() if p.name != "result.json"}})


def verify_saved(job, rows, values, left, right, folds, selection, regime, plan, settings):
    result = read_json(job/"result.json")
    names = {f"outer_{f['outer_fold']}.npz" for f in folds} | {"models.json", "predictions.json", "evaluation.json"}
    if (result.get("schema") != core.SCHEMA or result.get("complete") is not True or
            set(result["files"]) != names or {p.name for p in job.iterdir()} != names|{"result.json"} or
            any((job/p).is_symlink() for p in names)):
        raise ValueError("Incomplete direct-D output contract.")
    predictions = verify_job(job, rows, values, folds, False)
    expected_ids = set(selection["selected_example_ids"][regime])
    lookup = {r["example_id"]: r for r in predictions}
    records = {r["model_id"]: r for r in read_json(job/"models.json")}
    for fold in folds:
        model_id = f"outer_{fold['outer_fold']}"
        record = records[model_id]
        indices = [i for i, r in enumerate(rows) if r["example_id"] in expected_ids and
                   r["base_sample_id"] in fold["train_scene_seeds"]]
        expected = [rows[i]["example_id"] for i in indices]
        if (record["train_example_ids"] != expected or record["train_directed_rows"] != len(expected) or
                record.get("fit_intercept") is not False or record.get("center_D") is not False):
            raise ValueError("Direct difference training source changed.")
        candidates = record["inner_scores"]
        if (len(candidates) != len(settings["C_grid"]) or {r["C"] for r in candidates} != set(settings["C_grid"]) or
                any(not 0 <= r["inner_balanced_accuracy"] <= 1 for r in candidates) or
                record["chosen_C"] != max(candidates, key=lambda r: (r["inner_balanced_accuracy"], -r["C"]))["C"]):
            raise ValueError("Direct-D regularization choice changed.")
        scale = StandardScaler(with_mean=False).fit(values[indices])
        with np.load(job/(model_id+".npz"), allow_pickle=False) as payload:
            if (not np.array_equal(payload["intercept"], np.zeros(1)) or
                    not np.array_equal(payload["scaler_mean"], np.zeros(values.shape[1])) or
                    not np.allclose(payload["scaler_scale"], scale.scale_, rtol=1e-12, atol=1e-12)):
                raise ValueError("Direct-D probe has a nonzero intercept/centering or wrong source-only scale.")
            direction = payload["coefficient"][0]/payload["scaler_scale"]
        for i, row in enumerate(rows):
            if row["base_sample_id"] not in fold["test_scene_seeds"]:
                continue
            saved = lookup[row["example_id"]]
            a, b = float(left[i]@direction), float(right[i]@direction)
            if (not np.isclose(a, saved["left_endpoint_projection"], rtol=1e-12, atol=1e-12) or
                    not np.isclose(b, saved["right_endpoint_projection"], rtol=1e-12, atol=1e-12) or
                    not np.isclose(a-b, saved["probe_logit"], rtol=1e-10, atol=1e-10) or
                    saved["zero_margin_tie"] != (saved["probe_logit"] == 0) or
                    saved.get("endpoint_projections_are_single_video_predictions") is not False):
                raise ValueError("Endpoint levels do not replay the direct-D score.")
    for i in range(0, len(rows), 2):
        a, b = (lookup[r["example_id"]] for r in rows[i:i+2])
        if (a["model_id"] != b["model_id"] or
                not np.isclose(a["probe_logit"], -b["probe_logit"], rtol=1e-12, atol=1e-12) or
                a["y_statement_true"] != 1-b["y_statement_true"]):
            raise ValueError("Mirrored direction antisymmetry failed.")
    measured = core.evaluation(predictions, plan, settings)
    if read_json(job/"evaluation.json") != measured:
        raise ValueError("Direct-D statistics do not replay.")
    return measured


def report(summary):
    lines = ["# Rescue50: Direct Paired-Video Difference Probes", "",
        "Training input is D = h(original order) - h(reversed order), not a single h.",
        "Normal-forward option reference-last tokens; original 50 bases and historical folds.",
        "Both D/-D are trained together with opposite statement-truth labels, zero intercept and no D centering.",
        "Primary metrics use original-layout, canonical-direction pairs. Reverse directions are not independent samples.", "",
        "| Readout | Training | Low pair accuracy | Temporal pair accuracy | T - L (pp) | Low evidence | Temporal evidence | T - L evidence |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, regimes in summary["results"].items():
        for regime, item in regimes.items():
            values = item["archived"]
            accuracy = {r["condition"]: r["accuracy"] for r in values["metrics"] if r["group_by"] is None}
            paired = values["paired_summary"]
            a, b = (accuracy[c] for c in core.CONDITIONS)
            x, y = paired["evidence_low"]["mean"], paired["evidence_temporal"]["mean"]
            lines.append(f"| {name} | {regime} | {a:.1%} | {b:.1%} | {100*(b-a):+.1f} | {x:.3f} | {y:.3f} | {y-x:+.3f} |")
    lines += ["", "Saved predictions include left/right endpoint projection levels, their difference and explicit identities.",
        "Endpoint projections are algebraic components of a pair score, not single-video probabilities.",
        "This experiment measures pairwise discrimination; its accuracy is not the earlier single-video probe accuracy.",
        "All boundaries, views and directions of each base stay in the same held-out group. No new KO or patch.", ""]
    return "\n".join(lines)


def run(manifest_path, output_root, verify_only=False, settings=None):
    path, root = Path(manifest_path).resolve(), Path(output_root).resolve()
    if root == path.parent or root in path.parents or path.parent in root.parents:
        raise ValueError("Keep direct-D fitting separate from the immutable source store.")
    manifest, plan, rows, values, left, right = core.load_store(path)
    folds, selection = core.previous.folds_for(plan), core.selections(rows, plan)
    settings = {**SETTINGS, **(settings or {})}
    repo = Path(__file__).resolve().parents[1]
    config = {"schema": core.SCHEMA, "manifest_sha256": file_hash(path), "settings": settings,
        "selection_sha256": digest(selection), "folds_sha256": digest(folds), "input": "D_order",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
        "code_sha256": {n: file_hash(repo/"scripts"/n) for n in DEPENDENCIES},
        "protocol_sha256": file_hash(repo/"docs/phase3e_paired_difference_training.md")}
    with probe_lock(root):
        for name, payload in (("config", config), ("selection", selection), ("folds", folds)):
            if verify_only:
                if read_json(root/(name+".json")) != payload:
                    raise ValueError("Direct-D frozen configuration changed.")
            else:
                frozen_write(root/(name+".json"), payload)
        results = {}
        for index, name in enumerate(core.READOUTS):
            results[name] = {}
            for regime in core.REGIMES:
                job = root/name.replace(":", "__")/regime
                x, a, b = values[:, index, :], left[:, index, :], right[:, index, :]
                if not (job/"result.json").exists():
                    if verify_only or (job.exists() and any(job.iterdir())):
                        raise ValueError("Missing/partial direct-D job; inspect and use a new root.")
                    fitted = core.crossfit(rows, x, a, b, folds, selection, regime, settings)
                    write_job(job, fitted, core.evaluation(fitted["predictions"], plan, settings))
                results[name][regime] = verify_saved(job, rows, x, a, b, folds, selection, regime, plan, settings)
                print("VERIFIED_DIRECT_D", name, regime, flush=True)
        summary = {"schema": core.SCHEMA, "complete": True, "base_count": 50, "directed_rows": len(rows),
            "saved_outer_models": 80, "results": results, "new_GPU_inference_in_this_CPU_step": False}
        if verify_only:
            if read_json(root/"summary.json") != summary or (root/"report.md").read_text() != report(summary):
                raise ValueError("Direct-D report does not replay.")
        else:
            atomic_write(root/"summary.json", summary)
            (root/"report.md").write_text(report(summary), encoding="utf-8")
        if core.load_store(path)[0] != manifest:
            raise ValueError("Order-reversal source store changed during fitting.")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--verify_only", action="store_true")
    args = parser.parse_args()
    result = run(args.manifest_path, args.output_root, args.verify_only)
    print(json.dumps({k: result[k] for k in ("complete", "base_count", "directed_rows", "saved_outer_models")}))
