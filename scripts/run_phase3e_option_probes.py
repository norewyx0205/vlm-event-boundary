"""Fit/verify CPU statement-truth probes from a complete paired-video feature store."""

import argparse
import json
import platform
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import sklearn
from scipy.special import expit
from sklearn.feature_extraction import DictVectorizer

from scripts.phase3c_core import atomic_write, digest, file_hash, frozen_write, read_json
from scripts.phase3e_option_features import SITES
from scripts.phase3e_option_probes import SETTINGS, control_features, crossfit, metrics, paired_analysis
from scripts.phase3e_option_store import load_store
from scripts.run_phase3d_probes import frozen_numeric_scores, probe_lock, save_npz

DEFAULT_NAMES = [f"reference_last:{site}" for site in SITES if site != "L0_block_output"]


def control_matrix(path, values):
    with np.load(path, allow_pickle=False) as payload:
        columns = payload["feature_columns"].tolist()
    vocabulary = DictVectorizer(sparse=False)
    vocabulary.vocabulary_ = {name: i for i, name in enumerate(columns)}
    vocabulary.feature_names_ = columns
    return vocabulary.transform(values)


def verify_job(root, examples, values, folds, categorical):
    result = read_json(root / "result.json")
    for relative, sha in result["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != sha:
            raise ValueError("Probe output checksum changed.")
    predictions = read_json(root / "predictions.json")
    lookup = {r["example_id"]: r for r in predictions}
    if len(lookup) != len(examples) or len(predictions) != len(examples):
        raise ValueError("Incomplete held-out predictions.")
    records = read_json(root / "models.json")
    if len(records) != len(folds):
        raise ValueError("Missing frozen fold model.")
    fold_lookup = {f"outer_{f['outer_fold']}": f for f in folds}
    if {r["model_id"] for r in records} != set(fold_lookup):
        raise ValueError("Repeated/missing frozen fold model.")
    for record in records:
        fold = fold_lookup[record["model_id"]]
        if any(record[k] != fold[k] for k in ("train_scene_seeds", "test_scene_seeds")):
            raise ValueError("Saved model fold membership changed.")
        indices = [i for i, r in enumerate(examples) if r["scene_seed"] in fold["test_scene_seeds"]]
        path = root / (record["model_id"]+".npz")
        selected = [values[i] for i in indices] if categorical else values[indices]
        matrix = control_matrix(path, selected) if categorical else selected
        scores = frozen_numeric_scores(path, matrix)
        for i, score in zip(indices, scores):
            row, prediction = examples[i], lookup[examples[i]["example_id"]]
            if (any(prediction[k] != v for k, v in row.items()) or
                    prediction["model_id"] != record["model_id"] or
                    not np.isclose(score, prediction["probe_logit"], rtol=1e-12, atol=1e-12) or
                    not np.isclose(expit(score), prediction["probe_probability"], rtol=1e-12, atol=1e-12) or
                    prediction["probe_prediction"] != int(score > 0) or
                    not np.isclose(prediction["true_statement_evidence"], (2*row["y_statement_true"]-1)*score)):
                raise ValueError("Frozen probe replay differs from held-out predictions.")
    return predictions


def write_job(root, fitted, examples, rows):
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for item in fitted["models"]:
        pipe = item["model"]
        classifier, scaler = pipe.named_steps["probe"], pipe.named_steps["scale"]
        columns = (pipe.named_steps["vocabulary"].get_feature_names_out() if "vocabulary" in pipe.named_steps
                   else np.asarray([str(i) for i in range(classifier.coef_.shape[1])]))
        save_npz(root / (item["model_id"]+".npz"), coefficient=classifier.coef_, intercept=classifier.intercept_,
            scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, feature_columns=np.asarray(columns, dtype=str))
        records.append({k: v for k, v in item.items() if k != "model"})
    atomic_write(root / "models.json", records)
    atomic_write(root / "predictions.json", fitted["predictions"])
    atomic_write(root / "paired_analysis.json", paired_analysis(fitted["predictions"], rows))
    result = {"complete": True, "held_out_metrics": metrics(fitted["predictions"]),
              "files": {p.name: file_hash(p) for p in sorted(root.iterdir()) if p.is_file() and p.name != "result.json"}}
    atomic_write(root / "result.json", result)
    return result


def verify_contrast_projection(root, examples, values, predictions, pairs):
    """A contrast uses scaling only: no scaler centering or model intercept."""
    indices = {r["example_id"]: i for i, r in enumerate(examples)}
    lookup = {r["example_id"]: r for r in predictions}
    parameters = {}
    errors = []
    for pair in pairs["order_pairs"]+pairs["boundary_pairs"]:
        is_order = "left_example_id" in pair
        left, right = ((pair["left_example_id"], pair["right_example_id"]) if is_order else
                       (pair["temporal_example_id"], pair["low_example_id"]))
        model_id = lookup[left]["model_id"]
        if model_id != lookup[right]["model_id"]:
            raise ValueError("Cannot project a contrast through two different models.")
        if model_id not in parameters:
            with np.load(root / (model_id+".npz"), allow_pickle=False) as payload:
                parameters[model_id] = payload["coefficient"][0]/payload["scaler_scale"]
        difference = np.asarray(values[indices[left]], dtype=float)-np.asarray(values[indices[right]], dtype=float)
        projection = float(difference @ parameters[model_id])
        score_difference = lookup[left]["probe_logit"]-lookup[right]["probe_logit"]
        if not np.isclose(projection, score_difference, rtol=1e-10, atol=1e-10):
            raise ValueError("Direct D projection differs from the paired frozen score difference.")
        errors.append(abs(projection-score_difference))
    return {"order_and_boundary_contrasts_checked": len(errors),
            "maximum_absolute_error": max(errors), "centering_or_intercept_applied_to_D": False}


def behaviour(decisions, rows):
    result = []
    for condition in ("low_boundary", "temporal_boundary"):
        selected = [r for r in rows if r["condition"] == condition]
        for field in (None, "subject_id", "correct_option"):
            for value in ([None] if field is None else sorted({r[field] for r in selected})):
                cells = [decisions[r["eval_id"]] for r in selected if field is None or r[field] == value]
                result.append({"condition": condition, "group_by": field, "group": value,
                    "prompt_cells": len(cells), "correct": sum(d["is_correct"] for d in cells),
                    "accuracy": float(np.mean([d["is_correct"] for d in cells])),
                    "mean_answer_margin": float(np.mean([d["margin"] for d in cells])),
                    "zero_margin_ties": sum(d["margin"] == 0 for d in cells),
                    "non_AB_outputs": sum(d["prediction"] not in ("A", "B") for d in cells)})
    return result


def report_text(summary):
    lines = ["# Phase 3E-B: Paired-Video Option Probing", "", "## Material Passport", "",
        "- Complete real captures; 40 independent scene families, 640 prompts, 1,280 statements.",
        "- Single-video statement-truth training; grouped nested cross-validation.",
        "- Paired contrasts use the same frozen held-out probe on both sides.", "",
        "## Statement Truth", "",
        "| Readout | Low accuracy | Temporal accuracy | Low evidence | Temporal evidence | T - L |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, result in summary["results"].items():
        overall = {r["condition"]: r for r in result["metrics"] if r["group_by"] is None}
        boundary = result["boundary_summary"]
        lines.append(f"| {name} | {overall['low_boundary']['accuracy']:.1%} | "
            f"{overall['temporal_boundary']['accuracy']:.1%} | {boundary['evidence_low']['mean']:.3f} | "
            f"{boundary['evidence_temporal']['mean']:.3f} | {boundary['evidence_temporal_minus_low']['mean']:+.3f} |")
    lines += ["", "Scores are site-specific. Compare paired conditions within a readout, not score magnitudes across fitted probes.",
        "", "## Order Reversal", "", "Scores below refer to the same option text in order-reversed videos.", "",
        "| Readout | Boundary | Score when true | Score when false | True - false |",
        "| --- | --- | ---: | ---: | ---: |"]
    for name, result in summary["results"].items():
        for condition, order in result["order_summary"].items():
            lines.append(f"| {name} | {condition} | {order['left_score']['mean']:.3f} | "
                f"{order['right_score']['mean']:.3f} | {order['order_consistent_contrast']['mean']:+.3f} |")
    lines += ["", "## Model Behaviour", "", "| Boundary | Correct / prompts | Mean margin | Zero ties | Non-A/B |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for row in summary["behaviour"]:
        if row["group_by"] is None:
            lines.append(f"| {row['condition']} | {row['correct']}/{row['prompt_cells']} | "
                f"{row['mean_answer_margin']:.3f} | {row['zero_margin_ties']} | {row['non_AB_outputs']} |")
    lines += ["", "Family-level intervals, difference-in-differences, subject/option breakdowns and raw paired rows are in the JSON artifacts.",
        "No new KO or patch was run. This normal-forward analysis measures decodability and paired representational change.", ""]
    return "\n".join(lines)


def run(manifest_path, output_root, include_secondary=False, verify_only=False):
    manifest_path, root = Path(manifest_path).resolve(), Path(output_root).resolve()
    if root == manifest_path.parent or root in manifest_path.parents or manifest_path.parent in root.parents:
        raise ValueError("Keep probe outputs separate from the immutable feature store.")
    manifest, plan, examples, features, decisions = load_store(manifest_path)
    names = manifest["feature_names"]
    selected = [name.replace("reference_last:", kind+":") for kind in
        (("reference_last", "reference_mean", "option_final") if include_secondary else ("reference_last",))
        for name in DEFAULT_NAMES]
    config = {"schema": "phase3e_option_probe_run_v1", "manifest_sha256": file_hash(manifest_path),
        "settings": SETTINGS, "readouts": selected, "controls": ["lexical_only", "static_identity"],
        "code_sha256": {name: file_hash(Path(__file__).with_name(name)) for name in (
            "phase3e_option_features.py", "phase3e_option_probes.py", "phase3e_option_store.py",
            "run_phase3e_option_probes.py", "phase3d_probes.py", "run_phase3d_probes.py",
            "phase3e_paired_design.py", "phase3e_preflight.py", "phase3d_core.py", "phase3c_core.py")},
        "protocol_sha256": file_hash(Path(__file__).resolve().parents[1] / "docs/phase3e_b_paired_option_protocol.md"),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__}}
    with probe_lock(root):
        if verify_only:
            if read_json(root / "config.json") != config:
                raise ValueError("Probe configuration/source/runtime changed.")
        else:
            frozen_write(root / "config.json", config)
        summaries = {}
        for name in selected+config["controls"]:
            categorical = name in config["controls"]
            values = control_features(examples, plan["rows"], name) if categorical else features[:, names.index(name), :]
            job = root / name.replace(":", "__")
            if not (job / "result.json").exists():
                if verify_only:
                    raise ValueError("Missing completed probe job.")
                if job.exists() and any(job.iterdir()):
                    raise ValueError("Partial probe job needs inspection; use a new output root.")
                fitted = crossfit(examples, values, plan["folds"], categorical=categorical)
                write_job(job, fitted, examples, plan["rows"])
            predictions = verify_job(job, examples, values, plan["folds"], categorical)
            pairs = paired_analysis(predictions, plan["rows"])
            if read_json(job / "paired_analysis.json") != pairs:
                raise ValueError("Paired analysis differs from frozen held-out scores.")
            summaries[name] = {"metrics": metrics(predictions), "order_summary": pairs["order_summary"],
                "boundary_summary": pairs["boundary_summary"], "difference_in_differences": pairs["difference_in_differences"]}
            if not categorical:
                summaries[name]["direct_D_projection_replay"] = verify_contrast_projection(
                    job, examples, values, predictions, pairs)
            print("VERIFIED_PROBE", name, flush=True)
        summary = {"schema": config["schema"], "complete": True, "scene_families": 40,
            "statement_rows": len(examples), "fitted_fold_models": 5*len(summaries),
            "results": summaries, "behaviour": behaviour(decisions, plan["rows"]),
            "new_GPU_inference": False, "new_KO": False}
        if verify_only:
            if read_json(root / "summary.json") != summary:
                raise ValueError("Aggregate probe report changed.")
            if (root / "report.md").read_text(encoding="utf-8") != report_text(summary):
                raise ValueError("Readable report differs from verified raw levels and differences.")
        else:
            atomic_write(root / "summary.json", summary)
            (root / "report.md").write_text(report_text(summary), encoding="utf-8")
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--include_secondary", action="store_true")
    parser.add_argument("--verify_only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.manifest_path, args.output_root, args.include_secondary, args.verify_only), indent=2))
