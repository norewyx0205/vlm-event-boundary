"""Grouped, cross-fitted linear readouts; no fitting on patched or mirrored rows."""

import warnings
from collections import Counter

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .phase3d_core import base_folds, relation_evidence
except ImportError:
    from phase3d_core import base_folds, relation_evidence

SETTINGS = {"C_grid": [0.001, 0.01, 0.1, 1.0], "inner_folds": 4, "seed": 42,
            "solver": "liblinear", "dual": True, "class_weight": "balanced",
            "max_iter": 10000, "tol": 1e-7, "bootstrap_replicates": 2000,
            "label_permutations": 20, "primary_feature": "option_pair:L20_output",
            "numeric_dtype": "float64"}
INTERVENTIONS = {"baseline", "temporal_to_low_L16", "low_to_temporal_L16"}


def validate_rows(rows, cases, folds, require_complete=False):
    labels = {case["base_sample_id"]: case["y"] for case in cases}
    if len(labels) != len(cases) or set(labels.values()) != {0, 1}:
        raise ValueError("Need unique bases with both canonical relations.")
    expected, seen, primary = set(labels), set(), Counter()
    for row in rows:
        base = row["base_sample_id"]
        key = (base, row["condition"], row["view"], row["intervention"])
        if (base not in labels or row["y"] != labels[base] or key in seen or
                row["condition"] not in ("low_boundary", "temporal_boundary") or
                row["view"] not in ("archived", "option_order_permuted") or
                row["intervention"] not in INTERVENTIONS):
            raise ValueError("Invalid/duplicated condition, view, intervention or canonical label.")
        seen.add(key)
        if row["intervention"] != "baseline":
            expected_condition = "low_boundary" if row["intervention"] == "temporal_to_low_L16" else "temporal_boundary"
            if row["view"] != "archived" or row["condition"] != expected_condition:
                raise ValueError("A patch must retain its actual recipient condition and archived prompt.")
        if row["view"] == "archived" and row["intervention"] == "baseline":
            primary[base] += 1
    if set(primary) != expected or any(value != 2 for value in primary.values()):
        raise ValueError("Each base needs both unmodified primary conditions, exactly once.")
    test_bases = []
    for fold in folds:
        train, test = set(fold["train_base_ids"]), set(fold["test_base_ids"])
        if train & test or train | test != expected or not train or not test:
            raise ValueError("Outer folds leak bases or omit a member of the frozen cohort.")
        test_bases.extend(test)
    if Counter(test_bases) != Counter({base: 1 for base in expected}):
        raise ValueError("Each base must be held out exactly once.")
    if len({fold["outer_fold"] for fold in folds}) != len(folds):
        raise ValueError("Duplicate outer fold identifiers.")
    if require_complete:
        required = {(base, condition, view, "baseline") for base in expected
                    for condition in ("low_boundary", "temporal_boundary")
                    for view in ("archived", "option_order_permuted")}
        required |= {(base, condition, "archived", intervention) for base in expected
                     for condition, intervention in (("low_boundary", "temporal_to_low_L16"),
                                                      ("temporal_boundary", "low_to_temporal_L16"))}
        if seen != required:
            raise ValueError("Formal probing needs all baseline/permuted views and both fixed patch directions.")
    return labels


def model(C, settings, categorical=False):
    stages = [("vocabulary", DictVectorizer(sparse=False))] if categorical else []
    stages += [("scale", StandardScaler()), ("probe", LogisticRegression(
        C=C, penalty="l2", solver=settings["solver"], dual=settings["dual"],
        class_weight=settings["class_weight"], random_state=settings["seed"],
        max_iter=settings["max_iter"], tol=settings["tol"]))]
    return Pipeline(stages)


def subset(values, indices):
    return values[indices] if isinstance(values, np.ndarray) else [values[index] for index in indices]


def fit_model(values, targets, C, settings, categorical):
    fitted = model(C, settings, categorical)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        # Two paired rows contribute one unit of total base weight.
        fitted.fit(values, targets, probe__sample_weight=np.full(len(targets), .5))
    if not np.array_equal(fitted.named_steps["probe"].classes_, [0, 1]):
        raise RuntimeError("Probe class/sign orientation differs from the canonical binary task.")
    return fitted


def crossfit(rows, values, cases, folds, settings=None, target="relation", categorical=False,
             permuted_labels=None, progress=None):
    settings = {**SETTINGS, **(settings or {})}
    labels = validate_rows(rows, cases, folds)
    if len(values) != len(rows) or target not in ("relation", "condition"):
        raise ValueError("Invalid feature rows or probe target.")
    if not categorical:
        values = np.asarray(values, dtype=np.float64)
    if not categorical and (np.asarray(values).ndim != 2 or not np.isfinite(values).all()):
        raise ValueError("Numeric probe features must be a finite row-by-dimension matrix.")
    if permuted_labels is not None:
        if target != "relation" or set(permuted_labels) != set(labels) or Counter(permuted_labels.values()) != Counter(labels.values()):
            raise ValueError("Label permutation must retain a single paired label for every base.")
        labels = permuted_labels
    if not settings["C_grid"] or any(not np.isfinite(C) or C <= 0 for C in settings["C_grid"]):
        raise ValueError("Invalid fixed regularization grid.")
    targets = np.asarray([labels[row["base_sample_id"]] if target == "relation" else
                          int(row["condition"] == "temporal_boundary") for row in rows])
    primary = [i for i, row in enumerate(rows) if row["view"] == "archived" and row["intervention"] == "baseline"]
    predictions, models = [], []
    for fold in folds:
        train_bases, test_bases = set(fold["train_base_ids"]), set(fold["test_base_ids"])
        inner_cases = [{"base_sample_id": base, "y": labels[base],
                        "canonical_relation": "before" if labels[base] else "after"}
                       for base in sorted(train_bases)]
        inner = base_folds(inner_cases, settings["inner_folds"], settings["seed"]+fold["outer_fold"])
        scores = []
        for C in sorted(settings["C_grid"]):
            validations = []
            for split in inner:
                training = [i for i in primary if rows[i]["base_sample_id"] in split["train_base_ids"]]
                validation = [i for i in primary if rows[i]["base_sample_id"] in split["test_base_ids"]]
                fitted = fit_model(subset(values, training), targets[training], C, settings, categorical)
                predicted = fitted.predict(subset(values, validation))
                validations.append(float(balanced_accuracy_score(targets[validation], predicted)))
            scores.append({"C": C, "inner_balanced_accuracy": float(np.mean(validations))})
        # Stable ties favor the stronger regularizer, never outer-test performance.
        chosen = max(scores, key=lambda item: (item["inner_balanced_accuracy"], -item["C"]))["C"]
        training = [i for i in primary if rows[i]["base_sample_id"] in train_bases]
        testing = [i for i, row in enumerate(rows) if row["base_sample_id"] in test_bases]
        fitted = fit_model(subset(values, training), targets[training], chosen, settings, categorical)
        z = fitted.decision_function(subset(values, testing))
        probability = fitted.predict_proba(subset(values, testing))[:, 1]
        model_id = f"outer_{fold['outer_fold']}"
        models.append({"model_id": model_id, "model": fitted, "train_base_ids": sorted(train_bases),
                       "test_base_ids": sorted(test_bases), "chosen_C": chosen, "inner_scores": scores})
        for index, logit, p in zip(testing, z, probability):
            predictions.append({**rows[index], "model_id": model_id, "probe_target": target,
                                "probe_label": int(targets[index]), "probe_logit": float(logit),
                                "probe_probability": float(p), "probe_prediction": int(logit > 0),
                                "true_relation_evidence": relation_evidence(int(targets[index]), float(logit))
                                if target == "relation" and permuted_labels is None else None})
        if progress is not None:
            progress(f"{model_id}: train={len(train_bases)} bases; test={len(test_bases)} bases; C={chosen}")
    if len(predictions) != len(rows):
        raise RuntimeError("Missing/duplicated held-out predictions.")
    return {"predictions": predictions, "models": models, "settings": settings,
            "target": target, "label_permutation": permuted_labels is not None}


def prediction_metrics(predictions):
    result = []
    condition_target = predictions[0]["probe_target"] == "condition"
    groups = sorted({(row["view"], "paired_conditions" if condition_target else row["condition"])
                     for row in predictions if row["intervention"] == "baseline"})
    for view, condition in groups:
        rows = [row for row in predictions if row["view"] == view and
                (condition_target or row["condition"] == condition) and row["intervention"] == "baseline"]
        labels = [row["probe_label"] for row in rows]
        result.append({"view": view, "condition": condition,
                       "base_count": len({row["base_sample_id"] for row in rows}), "condition_rows": len(rows),
                       "class_counts": {str(key): value for key, value in Counter(labels).items()},
                       "majority_accuracy": max(Counter(labels).values())/len(labels),
                       "balanced_accuracy": (float(balanced_accuracy_score(labels, [row["probe_prediction"] for row in rows]))
                                             if len(set(labels)) == 2 else None),
                       "accuracy": float(np.mean(np.asarray(labels) == [row["probe_prediction"] for row in rows])),
                       "log_loss": float(log_loss(labels, [row["probe_probability"] for row in rows], labels=[0, 1]))})
    return result


def statement_truth_predictions(predictions, cases, option):
    """Express held-out canonical-relation predictions as actual statement truth.

    This is a diagnostic re-expression of the same frozen probe, not another
    independent accuracy result or a probe fitted on the imbalanced A/B label.
    """
    lookup = {case["base_sample_id"]: case for case in cases}
    result = []
    for row in predictions:
        semantics = lookup[row["base_sample_id"]]["option_semantics"]
        key = ("B" if option == "A" else "A") if row["view"] == "option_order_permuted" else option
        statement = semantics[key]
        before = (statement["relation"] == "before") == (statement["subject_id"] == 1)
        score = row["probe_logit"] * (1 if before else -1)
        probability = row["probe_probability"] if before else 1-row["probe_probability"]
        result.append({**row, "probe_target": "statement_truth", "option": option,
                       "canonical_probe_label": row["probe_label"], "canonical_probe_logit": row["probe_logit"],
                       "probe_label": int(row["y"] == int(before)), "probe_logit": score,
                       "probe_prediction": int(score > 0), "probe_probability": probability,
                       "true_relation_evidence": None})
    return result


def case_evidence(predictions):
    grouped = {}
    for row in predictions:
        key = (row["base_sample_id"], row["view"])
        grouped.setdefault(key, {})[(row["condition"], row["intervention"])] = row
    result = []
    for (base, view), rows in sorted(grouped.items()):
        low, temporal = (rows[(condition, "baseline")] for condition in ("low_boundary", "temporal_boundary"))
        if low["model_id"] != temporal["model_id"] or low["probe_target"] != "relation":
            raise ValueError("Paired evidence must use the same frozen relation readout.")
        item = {"base_sample_id": base, "view": view, "model_id": low["model_id"], "y": low["y"],
                "S_low": low["true_relation_evidence"], "S_temporal": temporal["true_relation_evidence"]}
        item["delta_S_boundary"] = item["S_temporal"]-item["S_low"]
        for condition, intervention, baseline in (("low_boundary", "temporal_to_low_L16", low),
                                                 ("temporal_boundary", "low_to_temporal_L16", temporal)):
            if (condition, intervention) in rows:
                patched = rows[(condition, intervention)]
                if patched["model_id"] != baseline["model_id"]:
                    raise ValueError("Patched readout uses a different probe from its recipient baseline.")
                item[f"delta_S_{intervention}"] = patched["true_relation_evidence"]-baseline["true_relation_evidence"]
                item[f"delta_M_{intervention}"] = patched["answer_margin"]-baseline["answer_margin"]
        result.append(item)
    return result


def categorical_controls(rows, cases, kind):
    lookup = {case["base_sample_id"]: case for case in cases}
    values = []
    for row in rows:
        case = lookup[row["base_sample_id"]]
        if kind == "static_attributes":
            features = {f"T{identity}.{name}": "<missing>" if value is None else str(value)
                        for identity, attributes in case["static_attributes"].items()
                        for name, value in attributes.items() if name in ("label", "color", "shape", "size")}
        elif kind == "option_lexical":
            semantics = case["option_semantics"]
            if row["view"] == "option_order_permuted":
                semantics = {"A": semantics["B"], "B": semantics["A"]}
            features = {f"option_{key}.{name}": str(value) for key, info in semantics.items()
                        for name, value in info.items() if name in ("relation", "subject_id", "reference_id")}
            features.update({f"option_{key}.subject_relation": f"{info['subject_id']}:{info['relation']}"
                             for key, info in semantics.items()})
        else:
            raise ValueError("Unknown shortcut control.")
        values.append(features)
    return values


def bootstrap_mean(values, seed=42, replicates=2000):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires one finite effect per independent base.")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(1)
    return {"base_count": len(values), "mean": float(values.mean()),
            "diagnostic_95_interval": np.quantile(means, [.025, .975]).tolist(),
            "uncertainty": "Case bootstrap conditional on saved cross-fitted probes; no full refitting."}
