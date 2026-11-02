"""Train on same-statement, same-object order-reversal differences, not h."""

import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts import phase3e_rescue50_probes as previous
from scripts.phase3c_core import digest, file_hash, read_json
from scripts.phase3e_option_probes import SETTINGS, validate_folds

SCHEMA = "phase3e_rescue50_direct_order_probes_v1"
STORE_SCHEMA = "phase3e_rescue50_order_store_v1"
READOUTS, REGIMES, CONDITIONS = previous.READOUTS, previous.REGIMES, previous.CONDITIONS


def paired_dataset(preparation, endpoints, features):
    values = np.asarray(features, dtype=np.float64)
    if (values.ndim != 3 or values.shape[:2] != (len(endpoints), 4) or
            not np.isfinite(values).all()):
        raise ValueError("Need finite per-endpoint, per-layer vectors.")
    lookup = {r["example_id"]: (i, r) for i, r in enumerate(endpoints)}
    if len(lookup) != len(endpoints) or len(endpoints) != 2*len(preparation["statement_examples"]):
        raise ValueError("Missing/duplicate original or order-reversed endpoints.")
    rows, left, right = [], [], []
    for original in preparation["statement_examples"]:
        case = next(c for c in preparation["case_metadata"] if c["pair_id"] == original["pair_id"])
        option = original["option"] if original["view"] == "archived" else {"A": "B", "B": "A"}[original["option"]]
        semantics = case["option_semantics"][option]
        candidates = [(i, r) for i, r in enumerate(endpoints)
                      if r.get("parent_original_example_id") == original["example_id"]]
        by_order = {r["order_variant"]: (i, r) for i, r in candidates}
        if len(candidates) != 2 or set(by_order) != {"original", "order_reversed"}:
            raise ValueError("An order-reversal partner is missing; Low/Temporal is not an order pair.")
        a, b = (by_order[k] for k in ("original", "order_reversed"))
        for i, r in (a, b):
            if any(r[k] != original[k] for k in
                    ("base_sample_id", "pair_id", "condition", "view", "option", "option_text")):
                raise ValueError("Order pair changes object statement, role, view or condition.")
            if r["intervention"] != "normal_forward":
                raise ValueError("Order probes require unpatched, non-KO endpoints.")
            if any(r[k] != v for k, v in semantics.items()):
                raise ValueError("Paired endpoints do not refer to the same object and sentence role.")
        if a[1]["y_statement_true"] != original["y_statement_true"] or b[1]["y_statement_true"] != 1-original["y_statement_true"]:
            raise ValueError("Order reversal must reverse the same statement's truth.")
        for name, first, second in (("original_minus_reversed", a, b), ("reversed_minus_original", b, a)):
            rows.append({**original, **semantics, "relation_word": semantics["relation"],
                "example_id": original["example_id"]+"__"+name,
                "original_example_id": original["example_id"], "scene_seed": original["base_sample_id"],
                "direction": name, "left_example_id": first[1]["example_id"],
                "right_example_id": second[1]["example_id"], "y_statement_true": first[1]["y_statement_true"]})
            left.append(first[0])
            right.append(second[0])
    differences = values[left]-values[right]
    if not np.array_equal(differences[::2], -differences[1::2]):
        raise ValueError("D/-D construction is not antisymmetric.")
    return rows, differences, values[left], values[right]


def load_store(path):
    path = Path(path).resolve()
    manifest = read_json(path)
    if (manifest.get("schema") != STORE_SCHEMA or manifest.get("complete") is not True or
            manifest.get("artifact_type") != "real" or manifest.get("shape") != [800, 4, 4096] or
            manifest.get("base_count") != 50 or manifest.get("feature_names") != list(READOUTS) or
            manifest.get("fingerprint") != digest({k: v for k, v in manifest.items() if k != "fingerprint"}) or
            set(manifest["files"]) != {"preparation.json", "endpoints.json", "features.npz", "decisions.json",
                                      "capture_verification.json", "materials_plan.json"}):
        raise ValueError("Need a complete real rescue50 order-reversal store, not the old single-video store.")
    for name, sha in manifest["files"].items():
        if (path.parent/name).is_symlink() or file_hash(path.parent/name) != sha:
            raise ValueError("Order store checksum changed.")
    plan = read_json(path.parent/"preparation.json")
    if (plan["fingerprint"] != manifest["preparation_fingerprint"] or
            plan["fingerprint"] != digest({k: v for k, v in plan.items() if k != "fingerprint"}) or
            len(plan["case_ids"]) != 50 or plan["cohort"] != "original_frozen_rescue50"):
        raise ValueError("Original cohort preparation changed.")
    proof = read_json(path.parent/"capture_verification.json")
    if (any(proof.get(k) is not True for k in ("complete", "capture_noop_exact", "standard_generation_parity",
            "same_contextual_reference_positions", "all_original_frames_reconstructed_exactly")) or
            proof.get("new_prompt_cells") != 200):
        raise ValueError("Order capture gates are incomplete.")
    material = read_json(path.parent/"materials_plan.json")
    if material["fingerprint"] != manifest["materials_fingerprint"] or material["fingerprint"] != digest(
            {k: v for k, v in material.items() if k != "fingerprint"}) or material["case_ids"] != plan["case_ids"]:
        raise ValueError("Wrong order-reversal material cohort.")
    endpoints = read_json(path.parent/"endpoints.json")
    with np.load(path.parent/"features.npz", allow_pickle=False) as payload:
        values = payload["features"]
    if list(values.shape) != manifest["shape"]:
        raise ValueError("Incomplete paired endpoint features.")
    rows, differences, left, right = paired_dataset(plan, endpoints, values)
    return manifest, plan, rows, differences, left, right


def selections(rows, plan):
    original = previous.selections(plan["statement_examples"], plan)
    result = {regime: [r["example_id"] for r in rows if r["original_example_id"] in
              set(original["selected_example_ids"][regime])] for regime in REGIMES}
    return {"original_prompt_selection": original, "selected_example_ids": result,
            "training_unit": "direct same-object order-reversal difference", "directions_are_augmentation": True}


def model(C, settings):
    return Pipeline([("scale", StandardScaler(with_mean=False)), ("probe", LogisticRegression(
        C=C, penalty="l2", solver=settings["solver"], dual=settings["dual"], fit_intercept=False,
        class_weight="balanced", random_state=settings["seed"], max_iter=settings["max_iter"], tol=settings["tol"]))])


def crossfit(rows, values, left, right, folds, selection, regime, settings=None):
    settings = {**SETTINGS, **(settings or {})}
    validate_folds(rows, folds)
    values, left, right = (np.asarray(v, dtype=np.float64) for v in (values, left, right))
    if values.shape != left.shape or values.shape != right.shape or not np.array_equal(values, left-right):
        raise ValueError("Training input must be the actual endpoint difference, not a single h vector.")
    if any(not np.isfinite(v).all() for v in (values, left, right)):
        raise ValueError("Nonfinite difference or endpoint.")
    selected = set(selection["selected_example_ids"][regime])
    labels = np.asarray([r["y_statement_true"] for r in rows])
    grid = sorted(settings["C_grid"])
    if not grid or any(not np.isfinite(c) or c <= 0 for c in grid):
        raise ValueError("Invalid regularization candidates.")

    def indices(bases, source=True):
        return [i for i, r in enumerate(rows) if r["base_sample_id"] in bases and
                (not source or r["example_id"] in selected)]

    def fit(ids, C):
        counts = Counter(rows[i]["base_sample_id"] for i in ids)
        if set(labels[ids]) != {0, 1}:
            raise ValueError("Source fold lacks a truth class.")
        pipe = model(C, settings)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            pipe.fit(values[ids], labels[ids], probe__sample_weight=[1/counts[rows[i]["base_sample_id"]] for i in ids])
        return pipe

    predictions, models = [], []
    for fold in folds:
        candidates = []
        for C in grid:
            scores = []
            for inner in fold["inner_folds"]:
                pipe = fit(indices(inner["train_scene_seeds"]), C)
                ids = indices(inner["validation_scene_seeds"])
                scores.append(float(balanced_accuracy_score(labels[ids], pipe.predict(values[ids]))))
            candidates.append({"C": C, "inner_balanced_accuracy": float(np.mean(scores))})
        C = max(candidates, key=lambda r: (r["inner_balanced_accuracy"], -r["C"]))["C"]
        train, test = indices(fold["train_scene_seeds"]), indices(fold["test_scene_seeds"], False)
        pipe = fit(train, C)
        model_id = f"outer_{fold['outer_fold']}"
        logits, probabilities = pipe.decision_function(values[test]), pipe.predict_proba(values[test])[:, 1]
        direction = pipe.named_steps["probe"].coef_[0]/pipe.named_steps["scale"].scale_
        for i, z, p in zip(test, logits, probabilities):
            a, b = float(left[i]@direction), float(right[i]@direction)
            if not np.isclose(a-b, z, rtol=1e-10, atol=1e-10):
                raise ValueError("Endpoint projection levels do not reconstruct the paired score.")
            predictions.append({**rows[i], "model_id": model_id, "probe_logit": float(z),
                "probe_probability": float(p), "probe_prediction": int(z > 0), "zero_margin_tie": bool(z == 0),
                "true_statement_evidence": float((2*labels[i]-1)*z),
                "left_endpoint_projection": a, "right_endpoint_projection": b,
                "endpoint_projections_are_single_video_predictions": False})
        models.append({"model_id": model_id, "model": pipe, "chosen_C": C, "inner_scores": candidates,
            "train_scene_seeds": fold["train_scene_seeds"], "test_scene_seeds": fold["test_scene_seeds"],
            "train_example_ids": [rows[i]["example_id"] for i in train], "train_directed_rows": len(train),
            "base_total_classifier_weight": 1.0, "fit_intercept": False, "center_D": False})
    return {"predictions": predictions, "models": models, "settings": settings}


def evaluation(predictions, plan, settings=None):
    rows = [r for r in predictions if r["direction"] == "original_minus_reversed"]
    lookup = {r["original_example_id"]: r for r in rows}
    pairs = [{**p, "low_example_id": lookup[p["low_example_id"]]["example_id"],
              "temporal_example_id": lookup[p["temporal_example_id"]]["example_id"]} for p in plan["boundary_pairs"]]
    result = previous.evaluation(rows, {"boundary_pairs": pairs}, settings)
    result["scope"] = "held-out pairwise discrimination; canonical direction only, base-clustered"
    result["directed_rows_verified"] = len(predictions)
    result["independent_bases"] = len({r["base_sample_id"] for r in rows})
    return result
