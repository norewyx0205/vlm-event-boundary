"""Source-condition-only truth probes with family-held-out cross-condition tests."""

import random
import warnings
from collections import Counter, defaultdict

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .phase3c_core import digest
from .phase3d_probes import bootstrap_mean, model
from .phase3e_option_probes import SETTINGS, metrics, paired_analysis, validate_folds
from .phase3e_paired_design import labels_and_pairs

SCHEMA = "phase3e_condition_generalization_v1"
CONDITIONS = ("low_boundary", "temporal_boundary")
REGIMES = ("low_only", "temporal_only", "mixed_matched", "mixed_full")
READOUTS = tuple(f"reference_last:L{layer}_block_output" for layer in (16, 17, 18, 20))


def selections(examples, rows, seed=42):
    if examples != labels_and_pairs(rows)[0]:
        raise ValueError("Training selections need the original statement labels and order.")
    lookup = {row["eval_id"]: row for row in rows}
    families = sorted({row["scene_seed"] for row in examples})
    shuffled = families.copy()
    random.Random(seed).shuffle(shuffled)
    phase = {family: i % 2 for i, family in enumerate(shuffled)}
    selected = {name: [] for name in REGIMES}
    for example in examples:
        row = lookup[example["eval_id"]]
        condition = CONDITIONS.index(example["condition"])
        selected["mixed_full"].append(example["example_id"])
        selected[("low_only", "temporal_only")[condition]].append(example["example_id"])
        # Keep both event orders and both statements of each selected prompt.
        assigned = (row["subject_id"]-1) ^ int(row["layout"] == "after_first") ^ phase[row["scene_seed"]]
        if condition == assigned:
            selected["mixed_matched"].append(example["example_id"])
    for regime, identities in selected.items():
        chosen = set(identities)
        for family in families:
            subset = [row for row in examples if row["scene_seed"] == family and row["example_id"] in chosen]
            expected = 32 if regime == "mixed_full" else 16
            if (len(subset) != expected or
                    Counter(row["y_statement_true"] for row in subset) != Counter({0: expected//2, 1: expected//2}) or
                    Counter(row["option"] for row in subset) != Counter({"A": expected//2, "B": expected//2}) or
                    Counter(row["subject_id"] for row in subset) != Counter({1: expected//2, 2: expected//2})):
                raise ValueError("Unbalanced scene-level training budget, labels or token positions.")
            for condition in {row["condition"] for row in subset}:
                cells = {row["eval_id"]: lookup[row["eval_id"]] for row in subset if row["condition"] == condition}
                count = len(cells)
                for key, levels in (("first_object_id", (1, 2)), ("subject_id", (1, 2)),
                                    ("layout", ("before_first", "after_first")), ("correct_option", ("A", "B"))):
                    if Counter(row[key] for row in cells.values()) != Counter({level: count//2 for level in levels}):
                        raise ValueError("Training source loses event-order, subject or answer-layout balance.")
                text_labels = defaultdict(Counter)
                for row in subset:
                    if row["condition"] == condition:
                        text_labels[(row["option_text"], row["option"])][row["y_statement_true"]] += 1
                if any(counts[0] != counts[1] for counts in text_labels.values()):
                    raise ValueError("Selected source creates a lexical truth shortcut.")
        selected[regime] = sorted(identities)
    return {"selection_seed": seed, "family_phase": {str(k): v for k, v in sorted(phase.items())},
            "selected_example_ids": selected,
            "matched_rule": "complementary subject/layout halves across conditions; both orders and statements retained",
            "selection_uses_features_or_model_outcomes": False}


def training_audit(examples, folds, selection, regime):
    validate_folds(examples, folds)
    if regime not in REGIMES:
        raise ValueError("Unknown source-condition regime.")
    selected = set(selection["selected_example_ids"][regime])
    lookup = {row["example_id"]: row for row in examples}
    if not selected or not selected <= set(lookup):
        raise ValueError("Selected training IDs are missing or unknown.")
    expected = {"low_only": {CONDITIONS[0]: 16}, "temporal_only": {CONDITIONS[1]: 16},
        "mixed_matched": {CONDITIONS[0]: 8, CONDITIONS[1]: 8},
        "mixed_full": {CONDITIONS[0]: 16, CONDITIONS[1]: 16}}[regime]
    for family in {row["scene_seed"] for row in examples}:
        counts = Counter(lookup[key]["condition"] for key in selected if lookup[key]["scene_seed"] == family)
        if counts != Counter(expected):
            raise ValueError("Source-condition selection has the wrong per-family training budget.")

    def describe(seeds):
        identities = sorted(key for key in selected if lookup[key]["scene_seed"] in set(seeds))
        return {"example_ids_sha256": digest(identities), "statement_rows": len(identities),
                "condition_counts": dict(sorted(Counter(lookup[key]["condition"] for key in identities).items())),
                "scene_seeds": sorted(seeds)}

    records = []
    for fold in folds:
        train = describe(fold["train_scene_seeds"])
        inner = [{"inner_fold": item["inner_fold"], "train": describe(item["train_scene_seeds"]),
                  "validation": describe(item["validation_scene_seeds"])} for item in fold["inner_folds"]]
        records.append({"model_id": f"outer_{fold['outer_fold']}", "train": train,
            "test_scene_seeds": fold["test_scene_seeds"], "inner_folds": inner,
            "test_conditions": list(CONDITIONS), "family_total_training_weight": 1.0})
    return {"regime": regime, "selection_sha256": digest(selection), "outer_models": records,
            "test_rows": len(examples), "test_scene_families": len({row["scene_seed"] for row in examples})}


def crossfit_condition(examples, values, folds, selection, regime, settings=None):
    settings = {**SETTINGS, **(settings or {})}
    audit = training_audit(examples, folds, selection, regime)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(examples) or not np.isfinite(values).all():
        raise ValueError("Expected finite row-by-feature values.")
    chosen = set(selection["selected_example_ids"][regime])
    labels = np.asarray([row["y_statement_true"] for row in examples])
    grid = sorted(settings["C_grid"])
    if not grid or any(not np.isfinite(value) or value <= 0 for value in grid):
        raise ValueError("Invalid regularization grid.")

    def indices(seeds, source=True):
        seeds = set(seeds)
        return [i for i, row in enumerate(examples) if row["scene_seed"] in seeds and
                (not source or row["example_id"] in chosen)]

    def fit(indices_, C):
        counts = Counter(examples[i]["scene_seed"] for i in indices_)
        if set(labels[indices_]) != {0, 1}:
            raise ValueError("A source training fold is missing a truth class.")
        weights = np.asarray([1/counts[examples[i]["scene_seed"]] for i in indices_])
        fitted = model(C, settings)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            fitted.fit(values[indices_], labels[indices_], probe__sample_weight=weights)
        return fitted

    predictions, models = [], []
    for fold in folds:
        candidates = []
        for C in grid:
            scores = []
            for inner in fold["inner_folds"]:
                train, validation = indices(inner["train_scene_seeds"]), indices(inner["validation_scene_seeds"])
                fitted = fit(train, C)
                scores.append(float(balanced_accuracy_score(labels[validation], fitted.predict(values[validation]))))
            candidates.append({"C": C, "inner_balanced_accuracy": float(np.mean(scores))})
        C = max(candidates, key=lambda row: (row["inner_balanced_accuracy"], -row["C"]))["C"]
        fitted = fit(indices(fold["train_scene_seeds"]), C)
        test = indices(fold["test_scene_seeds"], source=False)
        logits, probabilities = fitted.decision_function(values[test]), fitted.predict_proba(values[test])[:, 1]
        model_id = f"outer_{fold['outer_fold']}"
        for i, score, probability in zip(test, logits, probabilities):
            predictions.append({**examples[i], "model_id": model_id, "probe_logit": float(score),
                "probe_probability": float(probability), "probe_prediction": int(score > 0),
                "true_statement_evidence": float((2*labels[i]-1)*score)})
        models.append({"model_id": model_id, "model": fitted, "chosen_C": C, "inner_scores": candidates,
            "train_scene_seeds": fold["train_scene_seeds"], "test_scene_seeds": fold["test_scene_seeds"]})
    return {"models": models, "predictions": predictions, "training_audit": audit}


def evaluation(predictions, rows, settings=None):
    settings = {**SETTINGS, **(settings or {})}
    paired = paired_analysis(predictions, rows, settings)
    measured = metrics(predictions)
    for item in measured:
        subset = [row for row in predictions if
            (item["condition"] == "all" or row["condition"] == item["condition"]) and
            (item["group_by"] is None or row[item["group_by"]] == item["group"])]
        aucs = []
        for model_id in sorted({row["model_id"] for row in subset}):
            fold = [row for row in subset if row["model_id"] == model_id]
            aucs.append({"model_id": model_id, "roc_auc": float(roc_auc_score(
                [row["y_statement_true"] for row in fold], [row["probe_logit"] for row in fold]))})
        item["mean_within_fold_roc_auc"] = float(np.mean([row["roc_auc"] for row in aucs]))
        item["fold_roc_auc"] = aucs
    by_family = defaultdict(lambda: defaultdict(list))
    for row in predictions:
        by_family[row["scene_seed"]][row["condition"]].append(int(row["probe_prediction"] == row["y_statement_true"]))
    family_rows = [{"scene_seed": seed, "accuracy_low": float(np.mean(item[CONDITIONS[0]])),
                   "accuracy_temporal": float(np.mean(item[CONDITIONS[1]]))} for seed, item in sorted(by_family.items())]
    for item in family_rows:
        item["accuracy_temporal_minus_low"] = item["accuracy_temporal"]-item["accuracy_low"]
    accuracy = {field: bootstrap_mean([row[field] for row in family_rows], settings["seed"],
        settings["bootstrap_replicates"]) for field in ("accuracy_low", "accuracy_temporal", "accuracy_temporal_minus_low")}
    return {"metrics": measured, "accuracy_summary": {**accuracy, "scene_rows": family_rows},
        "boundary_summary": paired["boundary_summary"], "order_summary": paired["order_summary"],
        "difference_in_differences": paired["difference_in_differences"],
        "auc_scope": "mean of within-outer-fold AUCs; never pool scores from separately fitted models",
        "inference_unit": "40 scene families for formal data; intervals conditional on frozen probes"}
