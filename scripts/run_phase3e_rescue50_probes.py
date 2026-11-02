"""Fit and independently replay rescue50 condition-generalization truth probes."""

import argparse
import json
import platform
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

from scripts import phase3e_rescue50_probes as core
from scripts.phase3c_core import atomic_write, digest, file_hash, frozen_write, read_json
from scripts.phase3d_probes import subset
from scripts.phase3e_option_probes import SETTINGS
from scripts.run_phase3d_probes import probe_lock, save_npz
from scripts.run_phase3e_option_probes import control_matrix, verify_job

DEPENDENCIES = ("phase3e_rescue50_probes.py", "run_phase3e_rescue50_probes.py", "phase3e_option_probes.py",
    "run_phase3e_option_probes.py", "phase3e_condition_probes.py", "phase3d_probes.py",
    "run_phase3d_probes.py", "phase3d_core.py", "phase3c_core.py")


def projection_check(job, examples, values, predictions, plan):
    lookup = {r["example_id"]: r for r in predictions}
    indices = {r["example_id"]: i for i, r in enumerate(examples)}
    directions, errors = {}, []
    for pair in plan["boundary_pairs"]:
        low, temporal = pair["low_example_id"], pair["temporal_example_id"]
        model_id = lookup[low]["model_id"]
        if model_id != lookup[temporal]["model_id"]:
            raise ValueError("Boundary contrast uses two different frozen models.")
        if model_id not in directions:
            with np.load(job/(model_id+".npz"), allow_pickle=False) as payload:
                directions[model_id] = payload["coefficient"][0]/payload["scaler_scale"]
        delta = np.asarray(values[indices[temporal]], dtype=float)-np.asarray(values[indices[low]], dtype=float)
        direct = float(delta @ directions[model_id])
        observed = lookup[temporal]["probe_logit"]-lookup[low]["probe_logit"]
        if not np.isclose(direct, observed, rtol=1e-10, atol=1e-10):
            raise ValueError("Direct boundary-D projection does not replay.")
        errors.append(abs(direct-observed))
    return {"boundary_contrasts": len(errors), "maximum_absolute_error": max(errors),
            "centering_or_intercept_applied_to_D": False}


def write_job(job, fitted, audit, evaluated):
    job.mkdir(parents=True, exist_ok=True)
    records = []
    for record in fitted["models"]:
        pipe = record["model"]
        classifier, scaler = pipe.named_steps["probe"], pipe.named_steps["scale"]
        columns = (pipe.named_steps["vocabulary"].get_feature_names_out() if "vocabulary" in pipe.named_steps
                   else np.asarray([str(i) for i in range(classifier.coef_.shape[1])]))
        save_npz(job/(record["model_id"]+".npz"), coefficient=classifier.coef_, intercept=classifier.intercept_,
            scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, feature_columns=np.asarray(columns, dtype=str))
        records.append({k: v for k, v in record.items() if k != "model"})
    for name, content in (("models", records), ("predictions", fitted["predictions"]),
                          ("training_audit", audit), ("evaluation", evaluated)):
        atomic_write(job/(name+".json"), content)
    atomic_write(job/"result.json", {"schema": core.SCHEMA, "complete": True,
        "files": {p.name: file_hash(p) for p in sorted(job.iterdir()) if p.name != "result.json"}})


def verify_saved(job, examples, values, folds, plan, selection, regime, categorical, settings):
    result = read_json(job/"result.json")
    expected = {f"outer_{fold['outer_fold']}.npz" for fold in folds} | {
        "models.json", "predictions.json", "training_audit.json", "evaluation.json"}
    if (result.get("schema") != core.SCHEMA or result.get("complete") is not True or
            set(result["files"]) != expected or {p.name for p in job.iterdir()} != expected | {"result.json"} or
            any((job/name).is_symlink() for name in expected)):
        raise ValueError("Incomplete or unexpected rescue50 probe resources.")
    predictions = verify_job(job, examples, values, folds, categorical)
    audit = core.training_audit(examples, folds, selection, regime, plan)
    evaluated = core.evaluation(predictions, plan, settings)
    if read_json(job/"training_audit.json") != audit or read_json(job/"evaluation.json") != evaluated:
        raise ValueError("Training budget or paired metrics do not replay.")
    selected = set(selection["selected_example_ids"][regime])
    for record in read_json(job/"models.json"):
        candidates = record["inner_scores"]
        if (len(candidates) != len(settings["C_grid"]) or
                {r["C"] for r in candidates} != set(settings["C_grid"]) or
                any(not 0 <= r["inner_balanced_accuracy"] <= 1 for r in candidates) or
                record["chosen_C"] != max(candidates, key=lambda r: (r["inner_balanced_accuracy"], -r["C"]))["C"]):
            raise ValueError("Regularization selection changed.")
        indices = [i for i, r in enumerate(examples) if r["example_id"] in selected and
                   r["base_sample_id"] in record["train_scene_seeds"]]
        train = subset(values, indices)
        if categorical:
            vocabulary = DictVectorizer(sparse=False).fit(train)
            with np.load(job/(record["model_id"]+".npz"), allow_pickle=False) as payload:
                if payload["feature_columns"].tolist() != vocabulary.get_feature_names_out().tolist():
                    raise ValueError("Vocabulary was not fitted on source-only training rows.")
            train = control_matrix(job/(record["model_id"]+".npz"), train)
        scaler = StandardScaler().fit(np.asarray(train, dtype=float))
        with np.load(job/(record["model_id"]+".npz"), allow_pickle=False) as payload:
            if (not np.allclose(scaler.mean_, payload["scaler_mean"], rtol=1e-12, atol=1e-12) or
                    not np.allclose(scaler.scale_, payload["scaler_scale"], rtol=1e-12, atol=1e-12)):
                raise ValueError("Scaler was not fitted on source-only training rows.")
    projection = None if categorical else projection_check(job, examples, values, predictions, plan)
    return {"evaluation": evaluated, "training_audit": audit, "direct_D_projection_replay": projection}


def report_text(summary):
    lines = ["# Original Rescue50: Condition-Generalization Truth Probes", "",
        "## Materials", "", "Same 50 frozen rescue bases as KO50 and visual patching; 200 prompts, 400 statements.",
        "Normal-forward last reference-object token; L16/L17/L18 block outputs and L20 reference.",
        "Original historical five outer folds; 40 training and 10 held-out bases per fit.",
        "Statement truth is the label. Pure A/B swaps enter training; all views of a base stay together.",
        "Low-only, Temporal-only and matched mixed use 160 training statements per outer fit; full mixed uses 320.",
        "Every base has one total classifier-weight unit. All four regimes are newly fitted on this readout/cohort.", "",
        "## Primary: Original Prompts", "",
        "| Readout | Training | Low accuracy | Temporal accuracy | T - L (pp) | Low AUC | Temporal AUC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, regimes in summary["results"].items():
        for regime, result in regimes.items():
            rows = {r["condition"]: r for r in result["evaluation"]["archived"]["metrics"] if r["group_by"] is None}
            low, temporal = (rows[c] for c in core.CONDITIONS)
            lines.append(f"| {name} | {regime} | {low['accuracy']:.1%} | {temporal['accuracy']:.1%} | "
                f"{100*(temporal['accuracy']-low['accuracy']):+.1f} | {low['mean_within_fold_roc_auc']:.3f} | "
                f"{temporal['mean_within_fold_roc_auc']:.3f} |")
    lines += ["", "Accuracy is statement-truth decoding, not the model's A/B answer accuracy.",
        "AUC is averaged within frozen outer models. Separate permuted-view and all-view tables are in summary.json.",
        "", "## Primary: Paired Evidence", "",
        "| Readout | Training | Low evidence | Temporal evidence | T - L | Difference 95% diagnostic interval |",
        "| --- | --- | ---: | ---: | ---: | --- |"]
    for name, regimes in summary["results"].items():
        for regime, result in regimes.items():
            paired = result["evaluation"]["archived"]["paired_summary"]
            low, temporal, delta = (paired["evidence_"+s] for s in ("low", "temporal", "temporal_minus_low"))
            lo, hi = delta["diagnostic_95_interval"]
            lines.append(f"| {name} | {regime} | {low['mean']:.3f} | {temporal['mean']:.3f} | "
                         f"{delta['mean']:+.3f} | [{lo:.3f}, {hi:.3f}] |")
    lines += ["", "Paired differences apply one held-out frozen probe to the same statement under both conditions.",
        "Direct temporal-minus-low feature projections replay without centering/intercept on the difference.",
        "Intervals resample 50 bases conditional on saved probes. Independently trained logit scales are not comparable across rows.",
        "", "## Lexical Control", "",
        "| Training | View | Low accuracy | Temporal accuracy | T - L (pp) |",
        "| --- | --- | ---: | ---: | ---: |"]
    for regime, result in summary["lexical_control"].items():
        for view in core.VIEWS:
            rows = {r["condition"]: r for r in result["evaluation"][view]["metrics"] if r["group_by"] is None}
            low, temporal = (rows[c]["accuracy"] for c in core.CONDITIONS)
            lines.append(f"| {regime} | {view} | {low:.1%} | {temporal:.1%} | {100*(temporal-low):+.1f} |")
    lines += ["", "Pure A/B views balance answer letters, not all lexical associations in a rescue-selected cohort.",
        "The selected original prompts have Low wrong and Temporal correct by construction; new permuted prompts need not.",
        "Per-condition model logits, margins and accuracy, including permuted behavior, are in summary.json.",
        "", "## Scope", "",
        "No new GPU inference, KO, patching or case selection. Original and 40-family supplementary results are unchanged.",
        "Cross-condition transfer tests accessibility of a frozen linear readout, not a new causal pathway.",
        "Fixed candidate layers/regimes and source-only scaling/selection are retained, rather than choosing a best result after fitting.", ""]
    return "\n".join(lines)


def run(manifest_path, output_root, verify_only=False, settings=None):
    path, root = Path(manifest_path).resolve(), Path(output_root).resolve()
    if root == path.parent or root in path.parent.parents or path.parent in root.parents:
        raise ValueError("Keep new analysis separate from the immutable source store.")
    manifest, plan, examples, features, decisions = core.load_store(path)
    folds, selection = core.folds_for(plan), core.selections(examples, plan)
    settings = {**SETTINGS, **(settings or {})}
    repo = Path(__file__).resolve().parents[1]
    config = {"schema": core.SCHEMA, "manifest_sha256": file_hash(path), "settings": settings,
        "selection_sha256": digest(selection), "folds_sha256": digest(folds),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "code_sha256": {name: file_hash(repo/"scripts"/name) for name in DEPENDENCIES},
        "protocol_sha256": file_hash(repo/"docs/phase3e_rescue50_cpu_probes.md")}
    with probe_lock(root):
        for name, payload in (("config", config), ("selection_manifest", selection), ("folds", folds)):
            if verify_only:
                if read_json(root/(name+".json")) != payload:
                    raise ValueError("Frozen probe configuration changed.")
            else:
                frozen_write(root/(name+".json"), payload)
        results, controls = {}, {}
        jobs = [(name, features[:, index, :], False) for index, name in enumerate(core.READOUTS)]
        jobs.append(("lexical_only", core.lexical_features(examples, plan), True))
        for name, values, categorical in jobs:
            destination = controls if categorical else results.setdefault(name, {})
            for regime in core.REGIMES:
                job = root/name.replace(":", "__")/regime
                if not (job/"result.json").exists():
                    if verify_only or (job.exists() and any(job.iterdir())):
                        raise ValueError("Missing or partial probe job; inspect and use a new output root.")
                    audit = core.training_audit(examples, folds, selection, regime, plan)
                    fitted = core.fit_source(examples, values, folds, selection, regime, categorical, settings)
                    write_job(job, fitted, audit, core.evaluation(fitted["predictions"], plan, settings))
                destination[regime] = verify_saved(job, examples, values, folds, plan, selection, regime, categorical, settings)
                print("VERIFIED_RESCUE50_PROBE", name, regime, flush=True)
        summary = {"schema": core.SCHEMA, "complete": True, "base_count": 50,
            "statement_rows_per_job": len(examples), "new_outer_models": 100,
            "results": results, "lexical_control": controls, "model_behavior": core.behaviour(decisions, plan),
            "new_GPU_inference": False, "new_KO": False, "source_manifest_sha256": file_hash(path)}
        if verify_only:
            if read_json(root/"summary.json") != summary or (root/"report.md").read_text() != report_text(summary):
                raise ValueError("Aggregate rescue50 report does not replay.")
        else:
            atomic_write(root/"summary.json", summary)
            (root/"report.md").write_text(report_text(summary), encoding="utf-8")
        if core.load_store(path)[0] != manifest:
            raise ValueError("Source store changed during analysis.")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--verify_only", action="store_true")
    args = parser.parse_args()
    run(args.manifest_path, args.output_root, args.verify_only)
