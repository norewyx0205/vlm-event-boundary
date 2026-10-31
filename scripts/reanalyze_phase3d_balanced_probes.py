"""Isolated Phase 3E-A: grouped, balanced-view CPU reanalysis of frozen features."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import platform
import sys
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.phase3c_core import atomic_write, file_hash, read_json
from scripts.phase3d_core import base_folds
from scripts.phase3d_probes import SETTINGS, categorical_controls, model, subset, validate_rows
from scripts.run_phase3d_probes import DATASET_SCHEMA, load_dataset, save_npz


CONDITIONS = ("low_boundary", "temporal_boundary")
VIEWS = ("archived", "option_order_permuted")
PRIMARY = [f"source_pair:L{layer}_output" for layer in (0, 16, 18)]
PRIMARY += [f"option_pair:L{layer}_output" for layer in (16, 18, 20)]
SECONDARY = [f"option_{option}_{kind}:L{layer}_output" for layer in (16, 18, 20)
             for option in ("A", "B") for kind in ("pair", "relation")]
RESOURCES = ("case_metadata", "outer_folds", "row_metadata", "gate_summary", "features")


def isolated_output(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if (source.parent == destination or source.parent in destination.parents or
            destination in source.parents):
        raise ValueError("Output must be outside and not an ancestor of the source store")
    if destination.exists():
        raise ValueError("Use a NEW output directory; existing outputs are never overwritten")
    return source, destination


def input_audit(source):
    source = Path(source).resolve()
    manifest = read_json(source)
    if (manifest.get("schema") != DATASET_SCHEMA or manifest.get("artifact_type") != "real"
            or manifest.get("complete") is not True):
        raise ValueError("Need an immutable complete real Phase 3D manifest")
    resources = [{"path": str(source), "sha256": file_hash(source), "status": "verified"}]
    for key in RESOURCES:
        spec = manifest[key]
        path = (source.parent / spec["path"]).resolve()
        if not path.is_relative_to(source.parent):
            raise ValueError("Source resource escaped the feature store")
        status = "missing" if not path.is_file() else (
            "verified" if file_hash(path) == spec["sha256"] else "hash_mismatch")
        resources.append({"resource": key, "path": str(path), "sha256": spec["sha256"], "status": status})
    return {"schema": "phase3e_input_audit_v1", "resources": resources,
            "complete": all(item["status"] == "verified" for item in resources),
            "GPU_inference": False,
            "scope": "Resource bytes only; real dataset gates and row checks run before fitting."}


def balanced_grid(rows, cases, folds):
    validate_rows(rows, cases, folds, require_complete=True)
    labels = {case["base_sample_id"]: case["y"] for case in cases}
    baseline = [row for row in rows if row["intervention"] == "baseline"]
    if len(cases) != 50 or len(baseline) != 200:
        raise ValueError("Require exactly 50 frozen bases and 200 baseline views")
    for base in labels:
        group = [row for row in baseline if row["base_sample_id"] == base]
        if {(row["view"], row["condition"]) for row in group} != {
                (view, condition) for view in VIEWS for condition in CONDITIONS}:
            raise ValueError("Incomplete per-base view/condition grid")
        for condition in CONDITIONS:
            a, b = (next(row for row in group if row["view"] == view and
                         row["condition"] == condition) for view in VIEWS)
            if (a["correct_option"] not in ("A", "B") or
                    b["correct_option"] != ("B" if a["correct_option"] == "A" else "A") or
                    a["input_fingerprint"] == b["input_fingerprint"]):
                raise ValueError("Invalid pure option permutation")
    balance = {condition: dict(Counter(row["correct_option"] for row in baseline
                                     if row["condition"] == condition)) for condition in CONDITIONS}
    if any(counts != {"A": 50, "B": 50} for counts in balance.values()):
        raise ValueError("Correct option positions must be balanced within each condition")
    return balance


def fit(values, targets, C, settings, categorical, weights):
    estimator = model(C, settings, categorical)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(values, targets, probe__sample_weight=weights)
    if not np.array_equal(estimator.named_steps["probe"].classes_, [0, 1]):
        raise ValueError("Unexpected probe class/sign orientation")
    return estimator


def crossfit_balanced(rows, values, cases, folds, settings=None, categorical=False):
    """All baseline views fit; patched views and all held-out bases only evaluate."""
    settings = {**SETTINGS, **(settings or {})}
    labels = {case["base_sample_id"]: case["y"] for case in cases}
    if not categorical:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Need finite row-by-dimension numeric features")
    if len(rows) != len(values) or len(labels) != len(cases):
        raise ValueError("Incompatible rows/features/cases")
    targets = np.array([labels[row["base_sample_id"]] for row in rows])
    if any(row["y"] != labels[row["base_sample_id"]] for row in rows):
        raise ValueError("Row label differs from its base")
    trainable = [i for i, row in enumerate(rows) if row["intervention"] == "baseline"]
    counts = Counter(rows[i]["base_sample_id"] for i in trainable)
    if set(counts) != set(labels) or len(set(counts.values())) != 1:
        raise ValueError("Each base must have the same complete baseline grid")
    held_out = Counter(base for fold in folds for base in fold["test_base_ids"])
    if (held_out != Counter({base: 1 for base in labels}) or
            len({fold["outer_fold"] for fold in folds}) != len(folds)):
        raise ValueError("Each base must be held out exactly once")
    predictions, models = [], []
    for fold in folds:
        train_b, test_b = set(fold["train_base_ids"]), set(fold["test_base_ids"])
        if train_b & test_b or train_b | test_b != set(labels):
            raise ValueError("Outer folds leak or omit bases")
        inner_cases = [{"base_sample_id": base, "y": labels[base],
                        "canonical_relation": "before" if labels[base] else "after"}
                       for base in sorted(train_b)]
        inner = base_folds(inner_cases, settings["inner_folds"], settings["seed"] + fold["outer_fold"])
        def indices(bases):
            return [i for i in trainable if rows[i]["base_sample_id"] in bases]
        def fitted(ids, C):
            return fit(subset(values, ids), targets[ids], C, settings, categorical,
                       np.array([1 / counts[rows[i]["base_sample_id"]] for i in ids]))
        scores = []
        for C in sorted(settings["C_grid"]):
            validation = []
            for split in inner:
                training, testing = indices(split["train_base_ids"]), indices(split["test_base_ids"])
                estimator = fitted(training, C)
                validation.append(balanced_accuracy_score(targets[testing],
                                  estimator.predict(subset(values, testing))))
            scores.append({"C": C, "inner_balanced_accuracy": float(np.mean(validation))})
        C = max(scores, key=lambda item: (item["inner_balanced_accuracy"], -item["C"]))["C"]
        training = indices(train_b)
        testing = [i for i, row in enumerate(rows) if row["base_sample_id"] in test_b]
        estimator = fitted(training, C)
        score = estimator.decision_function(subset(values, testing))
        predicted = estimator.predict(subset(values, testing))
        model_id = f"outer_{fold['outer_fold']}"
        models.append({"model_id": model_id, "model": estimator, "chosen_C": C,
                       "train_base_ids": sorted(train_b), "test_base_ids": sorted(test_b),
                       "inner_scores": scores, "baseline_weight_per_row": 1 / next(iter(counts.values()))})
        for i, z, prediction in zip(testing, score, predicted):
            predictions.append({**rows[i], "model_id": model_id, "probe_logit": float(z),
                                "prediction": int(prediction), "zero_logit_tie": bool(z == 0),
                                "signed_evidence": float((2 * targets[i] - 1) * z)})
    if len(predictions) != len(rows):
        raise ValueError("Incomplete held-out predictions")
    return {"predictions": predictions, "models": models}


def cluster_metrics(predictions, settings):
    """Case bootstrap conditional on fitted outer-fold probes, not refit uncertainty."""
    groups = defaultdict(list)
    for row in predictions:
        if row["intervention"] != "baseline":
            continue
        groups[row["view"], row["condition"]].append(row)
        groups["both_views", row["condition"]].append(row)
    output = []
    for (view, condition), rows in sorted(groups.items()):
        bases = sorted({row["base_sample_id"] for row in rows})
        labels = np.array([row["y"] for row in rows])
        correct = np.array([row["prediction"] == row["y"] for row in rows])
        base_labels = np.array([next(row["y"] for row in rows if row["base_sample_id"] == base)
                                for base in bases])
        base_accuracy = np.array([correct[[row["base_sample_id"] == base for row in rows]].mean()
                                  for base in bases])
        rng = np.random.default_rng(settings["seed"])
        bootstrap = []
        for _ in range(settings["bootstrap_replicates"]):
            draw = rng.integers(0, len(bases), len(bases))
            if len(set(base_labels[draw])) == 2:
                bootstrap.append([float(base_accuracy[draw].mean()), float(np.mean([
                    base_accuracy[draw][base_labels[draw] == y].mean() for y in (0, 1)]))])
        intervals = np.quantile(bootstrap, [.025, .975], axis=0).T.tolist() if bootstrap else [None, None]
        output.append({"view": view, "condition": condition, "n_bases": len(bases), "n_rows": len(rows),
                       "accuracy": float(correct.mean()),
                       "balanced_accuracy": float(balanced_accuracy_score(labels, [r["prediction"] for r in rows])),
                       "accuracy_diagnostic_95_interval": intervals[0],
                       "balanced_accuracy_diagnostic_95_interval": intervals[1],
                       "zero_logit_ties": sum(row["zero_logit_tie"] for row in rows)})
    return output


def paired_evidence(predictions):
    bins = defaultdict(dict)
    for row in predictions:
        key = (row["condition"], row["intervention"])
        group = bins[row["base_sample_id"], row["view"]]
        if key in group:
            raise ValueError("Duplicate paired prediction")
        group[key] = row
    result = []
    for (base, view), group in sorted(bins.items()):
        low, temporal = [group[condition, "baseline"] for condition in CONDITIONS]
        if low["model_id"] != temporal["model_id"]:
            raise ValueError("Boundary contrast used different frozen probes")
        item = {"base_sample_id": base, "view": view, "y": low["y"],
                "correct_option": low["correct_option"], "model_id": low["model_id"],
                "delta_signed_evidence_temporal_minus_low": temporal["signed_evidence"] - low["signed_evidence"]}
        for condition, intervention, original in (
                (CONDITIONS[0], "temporal_to_low_L16", low),
                (CONDITIONS[1], "low_to_temporal_L16", temporal)):
            if (condition, intervention) in group:
                patched = group[condition, intervention]
                if patched["model_id"] != original["model_id"]:
                    raise ValueError("Patched contrast used a different frozen probe")
                item["delta_evidence_" + intervention] = patched["signed_evidence"] - original["signed_evidence"]
                item["delta_margin_" + intervention] = patched["answer_margin"] - original["answer_margin"]
        result.append(item)
    return result


def paired_summary(evidence, settings):
    groups = {"all": evidence}
    groups.update({f"canonical_y={y}": [row for row in evidence if row["y"] == y] for y in (0, 1)})
    groups.update({f"correct_option={option}": [row for row in evidence if row["correct_option"] == option]
                   for option in ("A", "B")})
    for view in VIEWS:
        selected = [row for row in evidence if row["view"] == view]
        groups["view=" + view] = selected
        for option in ("A", "B"):
            groups[f"view={view}:correct_option={option}"] = [
                row for row in selected if row["correct_option"] == option]
    results = []
    for name, rows in groups.items():
        bases = sorted({row["base_sample_id"] for row in rows})
        if not bases:
            continue
        values = np.array([np.mean([r["delta_signed_evidence_temporal_minus_low"] for r in rows
                                   if r["base_sample_id"] == base]) for base in bases])
        rng = np.random.default_rng(settings["seed"])
        means = values[rng.integers(0, len(bases), (settings["bootstrap_replicates"], len(bases)))].mean(1)
        results.append({"stratum": name, "n_bases": len(bases), "mean": float(values.mean()),
                        "diagnostic_95_interval": np.quantile(means, [.025, .975]).tolist()})
    return results


def delta_features(rows, values):
    lookup = {(row["base_sample_id"], row["view"], row["condition"]): (row, values[i])
              for i, row in enumerate(rows) if row["intervention"] == "baseline"}
    result, differences = [], []
    for base, view in sorted({(base, view) for base, view, _ in lookup}):
        low, temporal = [lookup[base, view, condition] for condition in CONDITIONS]
        result.append({**low[0], "condition": "paired_delta"})
        differences.append(temporal[1].astype(np.float64) - low[1].astype(np.float64))
    return result, np.stack(differences)


def option_a_heuristic(case, view):
    option = "A" if view == "archived" else "B"
    semantics = case["option_semantics"][option]
    return int((semantics["relation"] == "before") == (semantics["subject_id"] == 1))


def export_fit(root, fitted, settings, paired):
    root.mkdir(parents=True, exist_ok=False)
    models = []
    for record in fitted["models"]:
        pipeline = record["model"]
        classifier, scaler = pipeline.named_steps["probe"], pipeline.named_steps["scale"]
        columns = (pipeline.named_steps["vocabulary"].get_feature_names_out() if "vocabulary" in pipeline.named_steps
                   else np.array([str(i) for i in range(classifier.coef_.shape[1])]))
        save_npz(root / (record["model_id"] + ".npz"), coefficient=classifier.coef_, intercept=classifier.intercept_,
                 scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, feature_columns=np.asarray(columns, dtype=str))
        models.append({key: value for key, value in record.items() if key != "model"})
    atomic_write(root / "models.json", models)
    atomic_write(root / "held_out_predictions.json", fitted["predictions"])
    summary = {"cells": cluster_metrics(fitted["predictions"], settings),
               "uncertainty": "Base bootstrap conditional on frozen cross-fits; not full-refit uncertainty."}
    if paired:
        evidence = paired_evidence(fitted["predictions"])
        atomic_write(root / "paired_evidence.json", evidence)
        summary["boundary_contrast"] = paired_summary(evidence, settings)
    atomic_write(root / "summary.json", summary)
    return summary


def run(manifest_path, output, audit_only=False):
    source, root = isolated_output(manifest_path, output)
    audit = input_audit(source)
    root.mkdir(parents=True, exist_ok=False)
    atomic_write(root / "input_audit.json", audit)
    if not audit["complete"] or audit_only:
        return {"status": "inputs_verified_not_fit" if audit["complete"] else "needs_input",
                "analysis_performed": False, "input_audit": audit}
    _, rows, cases, folds, names, features = load_dataset(source)
    balance = balanced_grid(rows, cases, folds)
    settings = {**SETTINGS, "label_permutations": 0}
    if not set(PRIMARY + SECONDARY) <= set(names):
        raise ValueError("Missing predeclared readouts; do not silently omit a feature")
    config = {"schema": "phase3e_balanced_probe_reanalysis_v1", "settings": settings,
              "source_manifest_sha256": file_hash(source), "primary": PRIMARY, "secondary": SECONDARY,
              "code_sha256": {name: file_hash(Path(__file__).with_name(name)) for name in
                              (Path(__file__).name, "phase3d_probes.py", "phase3d_core.py", "run_phase3d_probes.py")},
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
              "training": "All 200 baseline views only; weight 1/4 per row; grouped outer and inner CV.",
              "provenance_boundary": "Pure-permutation token/video equality inherits the hash-bound Phase 3D capture gates; raw inputs are not replayed here."}
    atomic_write(root / "run_config.json", config)
    outputs = {}
    for name in PRIMARY + SECONDARY:
        print("Fitting balanced readout:", name, flush=True)
        result = crossfit_balanced(rows, features[:, names.index(name), :], cases, folds, settings)
        outputs[name] = export_fit(root / name.replace(":", "__"), result, settings, paired=True)
    for kind in ("option_lexical", "static_attributes"):
        print("Fitting control:", kind, flush=True)
        result = crossfit_balanced(rows, categorical_controls(rows, cases, kind), cases, folds, settings, categorical=True)
        outputs[kind] = export_fit(root / kind, result, settings, paired=True)
    for layer in (16, 18):
        name = f"delta_source_pair:L{layer}"
        drows, deltas = delta_features(rows, features[:, names.index(f"source_pair:L{layer}_output"), :])
        result = crossfit_balanced(drows, deltas, cases, folds, settings)
        outputs[name] = export_fit(root / name.replace(":", "__"), result, settings, paired=False)
        save_npz(root / name.replace(":", "__") / "paired_deltas.npz", features=deltas,
                 base_sample_ids=np.array([row["base_sample_id"] for row in drows]),
                 views=np.array([row["view"] for row in drows]))
        outputs[name]["mean_delta_norm"] = float(np.linalg.norm(deltas, axis=1).mean())
        outputs[name]["interpretation"] = "Separately trained boundary-modulation diagnostic, not a semantic direction or the frozen D probe's evidence gain."
        atomic_write(root / name.replace(":", "__") / "summary.json", outputs[name])
    # Hash every original resource again before claiming a completed reanalysis.
    if input_audit(source) != audit:
        raise ValueError("Original feature-store bytes changed during reanalysis")
    heuristic = {view: float(np.mean([option_a_heuristic(case, view) == case["y"] for case in cases]))
                 for view in VIEWS}
    heuristic["both_views"] = float(np.mean(list(heuristic.values())))
    summary = {"schema": config["schema"], "status": "complete", "analysis_performed": True,
               "base_count": 50, "baseline_rows": 200, "correct_option_by_condition": balance,
               "always_A_heuristic": heuristic, "results": outputs, "source_bytes_unchanged": True,
               "missing_readouts": ["whole_Event2_mean:L18", "individual_visual_T1_T2_means",
                                    "L18_pre_post_attention_option_states", "new_subject_role_reversals",
                                    "spatial_trajectory_controls_not_in_compact_case_metadata"],
               "new_label_permutation_fits": False,
               "interpretation": "Exploratory rescue-cohort diagnostics, not independent relation-specific or mediation evidence.",
               "GPU_inference": False}
    atomic_write(root / "summary.json", summary)
    files = [path for path in root.rglob("*") if path.is_file()]
    atomic_write(root / "output_manifest.json", {"files": [
        {"path": path.relative_to(root).as_posix(), "sha256": file_hash(path)} for path in sorted(files)]})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    result = run(args.manifest, args.out, args.audit_only)
    print(result["status"], flush=True)
    if result["status"] == "needs_input":
        for item in result["input_audit"]["resources"]:
            if item["status"] != "verified":
                print(item["status"] + ": " + item["path"])
        raise SystemExit(2)


if __name__ == "__main__":
    main()
