"""Family-cross-fitted statement-truth probes and frozen paired score contrasts."""

import warnings
from collections import Counter, defaultdict

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score

from .phase3d_probes import bootstrap_mean, model, subset
from .phase3e_paired_design import labels_and_pairs, validate_materials

SETTINGS = {"C_grid": [0.001, 0.01, 0.1, 1.0], "seed": 42,
    "solver": "liblinear", "dual": True, "class_weight": "balanced",
    "max_iter": 10000, "tol": 1e-7, "bootstrap_replicates": 2000,
    "numeric_dtype": "float64", "training_unit": "single_video_single_statement",
    "family_total_weight": 1.0,
    "primary_features": [f"reference_last:L{layer}_block_output" for layer in (16, 17, 18)]}


def validate_folds(examples, folds):
    families = {r["scene_seed"] for r in examples}
    ids = [r["example_id"] for r in examples]
    if (not families or len(ids) != len(set(ids)) or
            any(r["intervention"] != "normal_forward" or r["y_statement_true"] not in (0, 1)
                for r in examples)):
        raise ValueError("Need unique, normal single-video statement rows.")
    tested = Counter()
    for fold in folds:
        train, test = set(fold["train_scene_seeds"]), set(fold["test_scene_seeds"])
        if train & test or train | test != families or not train or not test:
            raise ValueError("Outer folds leak or omit scene families.")
        tested.update(test)
        validated = Counter()
        for inner in fold["inner_folds"]:
            fit, val = set(inner["train_scene_seeds"]), set(inner["validation_scene_seeds"])
            if fit & val or fit | val != train or not fit or not val:
                raise ValueError("Inner folds leak outer test families.")
            validated.update(val)
        if validated != Counter({s: 1 for s in train}):
            raise ValueError("Each outer training family must be validated exactly once.")
    if tested != Counter({s: 1 for s in families}):
        raise ValueError("Each scene family must be held out exactly once.")
    if len({f["outer_fold"] for f in folds}) != len(folds):
        raise ValueError("Duplicate fold identifiers.")


def crossfit(examples, values, folds, settings=None, categorical=False):
    settings = {**SETTINGS, **(settings or {})}
    validate_folds(examples, folds)
    if len(values) != len(examples):
        raise ValueError("Feature count differs from the statement manifest.")
    if not categorical:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Probe input must be a finite row-by-feature matrix.")
    grid = sorted(settings["C_grid"])
    if not grid or any(not np.isfinite(c) or c <= 0 for c in grid):
        raise ValueError("Invalid regularization grid.")
    labels = np.asarray([r["y_statement_true"] for r in examples])

    def indices(seeds):
        return [i for i, row in enumerate(examples) if row["scene_seed"] in seeds]

    def fit(indices_, C):
        counts = Counter(examples[i]["scene_seed"] for i in indices_)
        weights = np.asarray([1/counts[examples[i]["scene_seed"]] for i in indices_])
        if set(labels[indices_]) != {0, 1}:
            raise ValueError("A training fold is missing a statement-truth class.")
        fitted = model(C, settings, categorical)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            fitted.fit(subset(values, indices_), labels[indices_], probe__sample_weight=weights)
        return fitted

    predictions, models = [], []
    for fold in folds:
        candidates = []
        for C in grid:
            scores = []
            for inner in fold["inner_folds"]:
                training = indices(set(inner["train_scene_seeds"]))
                validation = indices(set(inner["validation_scene_seeds"]))
                fitted = fit(training, C)
                scores.append(float(balanced_accuracy_score(labels[validation],
                    fitted.predict(subset(values, validation)))))
            candidates.append({"C": C, "inner_balanced_accuracy": float(np.mean(scores))})
        chosen = max(candidates, key=lambda r: (r["inner_balanced_accuracy"], -r["C"]))["C"]
        training = indices(set(fold["train_scene_seeds"]))
        testing = indices(set(fold["test_scene_seeds"]))
        fitted = fit(training, chosen)
        logits = fitted.decision_function(subset(values, testing))
        probabilities = fitted.predict_proba(subset(values, testing))[:, 1]
        model_id = f"outer_{fold['outer_fold']}"
        for i, z, p in zip(testing, logits, probabilities):
            predictions.append({**examples[i], "model_id": model_id, "probe_logit": float(z),
                "probe_probability": float(p), "probe_prediction": int(z > 0),
                "true_statement_evidence": float((2*labels[i]-1)*z)})
        models.append({"model_id": model_id, "model": fitted, "chosen_C": chosen,
            "inner_scores": candidates, "train_scene_seeds": fold["train_scene_seeds"],
            "test_scene_seeds": fold["test_scene_seeds"]})
    return {"predictions": predictions, "models": models, "settings": settings}


def control_features(examples, rows, kind):
    lookup = {r["eval_id"]: r for r in rows}
    result = []
    for example in examples:
        row = lookup[example["eval_id"]]
        if kind == "lexical_only":
            item = {"statement": example["option_text"], "option_position": example["option"],
                "option_A": row["option_A"], "option_B": row["option_B"],
                "subject_id": str(example["subject_id"]), "reference_id": str(example["reference_id"])}
        elif kind == "static_identity":
            # No event times, mover IDs, truth labels or scene identifiers enter controls.
            item = {f"object_{o['id']}_{key}": str(o[key]) for o in row["target_objects"]
                    for key in ("label", "shape", "color", "from", "to") if key in o}
            item.update(relation=example["relation_word"], subject=str(example["subject_id"]),
                        reference=str(example["reference_id"]), option=example["option"])
        else:
            raise ValueError("Unknown fixed control.")
        result.append(item)
    return result


def metrics(predictions):
    result = []
    for condition in ("all", "low_boundary", "temporal_boundary"):
        selected = [r for r in predictions if condition == "all" or r["condition"] == condition]
        for field in (None, "subject_id", "option"):
            groups = [None] if field is None else sorted({r[field] for r in selected})
            for group in groups:
                rows = [r for r in selected if field is None or r[field] == group]
                labels = np.asarray([r["y_statement_true"] for r in rows])
                guessed = [r["probe_prediction"] for r in rows]
                result.append({"condition": condition, "group_by": field, "group": group,
                    "independent_scene_families": len({r["scene_seed"] for r in rows}),
                    "statement_rows": len(rows), "label_counts": {str(k): v for k, v in Counter(map(int, labels)).items()},
                    "accuracy": float(np.mean(labels == guessed)),
                    "balanced_accuracy": float(balanced_accuracy_score(labels, guessed)),
                    "mean_true_statement_evidence": float(np.mean([r["true_statement_evidence"] for r in rows]))})
    return result


def paired_analysis(predictions, rows, settings=None):
    """Scores on both sides always come from the same held-out frozen probe."""
    settings = {**SETTINGS, **(settings or {})}
    examples, pairs, boundaries = labels_and_pairs(rows)
    expected = {r["example_id"]: r for r in examples}
    lookup = {r["example_id"]: r for r in predictions}
    if len(lookup) != len(predictions) or set(lookup) != set(expected):
        raise ValueError("Incomplete/duplicated paired predictions.")
    for identity, row in lookup.items():
        if any(row[k] != v for k, v in expected[identity].items()):
            raise ValueError("Prediction statement metadata changed.")
    orders, boundary_rows = [], []
    for pair in pairs:
        left, right = (lookup[pair[k]] for k in ("left_example_id", "right_example_id"))
        if left["model_id"] != right["model_id"]:
            raise ValueError("Order pair uses different fitted probes.")
        delta = left["probe_logit"]-right["probe_logit"]
        orders.append({**pair, "left_score": left["probe_logit"], "right_score": right["probe_logit"],
            "score_left_minus_right": delta,
            "order_consistent_contrast": (2*left["y_statement_true"]-1)*delta})
    for pair in boundaries:
        low, temporal = (lookup[pair[k]] for k in ("low_example_id", "temporal_example_id"))
        if low["model_id"] != temporal["model_id"]:
            raise ValueError("Boundary pair uses different fitted probes.")
        a, b = low["true_statement_evidence"], temporal["true_statement_evidence"]
        boundary_rows.append({**pair, "score_low": low["probe_logit"], "score_temporal": temporal["probe_logit"],
            "evidence_low": a, "evidence_temporal": b, "evidence_temporal_minus_low": b-a})

    def aggregate(items, fields):
        families = defaultdict(list)
        for item in items:
            families[item["scene_seed"]].append(item)
        per_scene = [{"scene_seed": seed, **{field: float(np.mean([r[field] for r in values]))
                                            for field in fields}} for seed, values in sorted(families.items())]
        return {"scene_rows": per_scene, **{field: bootstrap_mean([r[field] for r in per_scene],
                settings["seed"], settings["bootstrap_replicates"]) for field in fields}}

    order_summary = {}
    for condition in ("low_boundary", "temporal_boundary"):
        # Keep all directed rows in the artifact, use true-left once per unordered pair here.
        unique = [r for r in orders if r["condition"] == condition and r["y_left_statement_true"] == 1]
        order_summary[condition] = aggregate(unique, ("left_score", "right_score", "order_consistent_contrast"))
    low = {r["scene_seed"]: r for r in order_summary["low_boundary"]["scene_rows"]}
    temporal = {r["scene_seed"]: r for r in order_summary["temporal_boundary"]["scene_rows"]}
    differences = [{"scene_seed": s, "order_contrast_low": low[s]["order_consistent_contrast"],
        "order_contrast_temporal": temporal[s]["order_consistent_contrast"],
        "difference_in_differences": temporal[s]["order_consistent_contrast"]-low[s]["order_consistent_contrast"]}
        for s in sorted(low)]
    return {"order_pairs": orders, "boundary_pairs": boundary_rows, "order_summary": order_summary,
        "boundary_summary": aggregate(boundary_rows, ("evidence_low", "evidence_temporal", "evidence_temporal_minus_low")),
        "difference_in_differences": aggregate(differences, ("order_contrast_low", "order_contrast_temporal", "difference_in_differences")),
        "inference_unit": "scene family; bootstrap conditional on frozen cross-fitted probes"}


def validate_dataset(plan, examples, names, values, formal=True):
    from .phase3e_option_features import FEATURE_NAMES
    from .phase3e_paired_design import SCENE_SEEDS, validate_plan
    if formal:
        validate_plan(plan)
        if tuple(plan["scene_seeds"]) != SCENE_SEEDS:
            raise ValueError("Formal probing requires the fixed 40 scene families.")
    else:
        validate_materials(plan["rows"], plan["scene_seeds"])
    if examples != labels_and_pairs(plan["rows"])[0]:
        raise ValueError("Feature rows changed statement labels or their order.")
    validate_folds(examples, plan["folds"])
    if (tuple(names) != FEATURE_NAMES or values.shape[:2] != (len(examples), len(names)) or
            values.ndim != 3 or values.shape[2] < 1 or not np.isfinite(values).all()):
        raise ValueError("Incomplete/nonfinite declared option features.")
