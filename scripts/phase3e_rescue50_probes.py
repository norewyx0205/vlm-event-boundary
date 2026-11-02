"""Original-rescue-cohort truth probes; no video generation or model inference."""

import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .phase3c_core import digest, file_hash, read_json
from .phase3d_core import base_folds
from .phase3d_probes import bootstrap_mean, subset
from .phase3e_condition_probes import CONDITIONS, READOUTS, REGIMES
from .phase3e_option_probes import SETTINGS, crossfit, validate_folds

SCHEMA = "phase3e_rescue50_condition_probes_v1"
VIEWS = ("archived", "option_order_permuted", "all")


def load_store(path):
    path = Path(path).resolve()
    manifest = read_json(path)
    if (manifest.get("schema") != "phase3e_rescue50_option_store_v1" or
            manifest.get("artifact_type") != "real" or manifest.get("complete") is not True or
            manifest.get("fingerprint") != digest({k: v for k, v in manifest.items() if k != "fingerprint"}) or
            manifest.get("base_count") != 50 or manifest.get("prompt_cells") != 200 or
            manifest.get("statement_rows") != 400 or manifest.get("shape") != [400, 4, 4096] or
            manifest.get("feature_names") != list(READOUTS) or set(manifest["files"]) != {
                "capture_verification.json", "decisions.json", "examples.json", "features.npz", "plan.json"}):
        raise ValueError("Need the complete real rescue50 reference-token store, not the 40-family store.")
    for name, sha in manifest["files"].items():
        resource = path.parent/name
        if resource.is_symlink() or file_hash(resource) != sha:
            raise ValueError("Rescue50 source resource checksum changed.")
    plan = read_json(path.parent/"plan.json")
    cases = plan["case_metadata"]
    if (plan.get("schema") != "phase3e_rescue50_option_preparation_v1" or
            plan.get("fingerprint") != digest({k: v for k, v in plan.items() if k != "fingerprint"}) or
            plan["fingerprint"] != manifest["preparation_fingerprint"] or
            plan["cohort"] != "original_frozen_rescue50" or len(cases) != 50 or
            len({c["base_sample_id"] for c in cases}) != 50 or
            len({c["pair_id"] for c in cases}) != 50 or
            plan["outer_folds"] != base_folds(cases) or plan["readouts"] != list(READOUTS)):
        raise ValueError("Frozen rescue50 preparation or historical folds changed.")
    proof = read_json(path.parent/"capture_verification.json")
    if (any(proof.get(k) is not True for k in ("complete", "stage_verified", "original_baseline_exact",
            "capture_noop_exact", "option_alignment")) or proof.get("new_KO_or_patch") is not False or
            proof.get("prompt_cells") != 100 or proof.get("config_sha256") != manifest["config_sha256"]):
        raise ValueError("Supplementary capture gates did not pass.")
    examples, decisions = read_json(path.parent/"examples.json"), read_json(path.parent/"decisions.json")
    if examples != plan["statement_examples"] or len(examples) != 400:
        raise ValueError("Statement rows differ from frozen preparation.")
    tasks = {r["eval_id"]: r for r in plan["tasks"]}
    lookup = {c["pair_id"]: c for c in cases}
    if len(tasks) != 200 or set(decisions) != set(tasks):
        raise ValueError("Incomplete prompt decisions.")
    seen = Counter()
    enriched = []
    for row in examples:
        task, case = tasks[row["eval_id"]], lookup[row["pair_id"]]
        if row["view"] not in VIEWS[:2] or row["condition"] not in CONDITIONS or row["option"] not in ("A", "B"):
            raise ValueError("Unknown prompt cell.")
        archived = row["view"] == "archived"
        correct = case["correct_option"] if archived else ({"A": "B", "B": "A"}[case["correct_option"]])
        expected_eval = case["condition_eval_ids"][row["condition"]] + ("" if archived else "__pure_option_permutation")
        semantic_option = row["option"] if archived else {"A": "B", "B": "A"}[row["option"]]
        semantics = case["option_semantics"][semantic_option]
        subject_before = bool(case["y"]) if semantics["subject_id"] == 1 else not bool(case["y"])
        truth = int(subject_before == (semantics["relation"] == "before"))
        if (row["eval_id"] != expected_eval or row["example_id"] != expected_eval+"__statement_"+row["option"] or
                row["base_sample_id"] != case["base_sample_id"] or
                any(row[k] != task[k] for k in ("pair_id", "condition", "view", "intervention", "base_sample_id")) or
                row["intervention"] != "normal_forward" or task["correct_option"] != correct or
                row["option_text"] != task["option_"+row["option"]] or
                row["y_statement_true"] != truth or truth != int(row["option"] == correct)):
            raise ValueError("Statement identity, semantics or truth changed.")
        seen[(case["base_sample_id"], row["condition"], row["view"], row["option"])] += 1
        enriched.append({**row, "scene_seed": row["base_sample_id"], **semantics,
                         "relation_word": semantics["relation"]})
    expected = Counter({(c["base_sample_id"], condition, view, option): 1 for c in cases
                        for condition in CONDITIONS for view in VIEWS[:2] for option in ("A", "B")})
    if seen != expected:
        raise ValueError("Missing or duplicate base/condition/view/statement.")
    for uid, task in tasks.items():
        decision = decisions[uid]
        if (decision["correct_option"] != task["correct_option"] or
                not np.isclose(decision["margin"], decision["correct_logit"]-decision["incorrect_logit"], rtol=0, atol=1e-12) or
                decision["prediction"] not in ("A", "B") or
                decision["is_correct"] != (decision["prediction"] == task["correct_option"]) or
                (decision["margin"] != 0 and decision["is_correct"] != (decision["margin"] > 0)) or
                (task["view"] == "archived" and decision["is_correct"] != (task["condition"] == CONDITIONS[1]))):
            raise ValueError("Historical baseline decisions or rescue behavior changed.")
    with np.load(path.parent/"features.npz", allow_pickle=False) as payload:
        if set(payload.files) != {"features"}:
            raise ValueError("Unexpected feature arrays.")
        values = payload["features"]
    if list(values.shape) != manifest["shape"] or not np.isfinite(values).all():
        raise ValueError("Nonfinite or incomplete reference-token features.")
    # These are original base IDs, not new scene seeds. The alias is only for
    # the existing grouped cross-fitting primitive.
    return manifest, plan, enriched, values, decisions


def folds_for(plan):
    result = []
    for fold in plan["outer_folds"]:
        cases = [c for c in plan["case_metadata"] if c["base_sample_id"] in fold["train_base_ids"]]
        inner = base_folds(cases, 4, SETTINGS["seed"]+fold["outer_fold"])
        result.append({"outer_fold": fold["outer_fold"], "train_scene_seeds": fold["train_base_ids"],
            "test_scene_seeds": fold["test_base_ids"], "inner_folds": [
                {"inner_fold": f["outer_fold"], "train_scene_seeds": f["train_base_ids"],
                 "validation_scene_seeds": f["test_base_ids"]} for f in inner]})
    return result


def selections(examples, plan):
    rng, phases = random.Random(SETTINGS["seed"]), {}
    for correct in ("A", "B"):
        bases = sorted(c["base_sample_id"] for c in plan["case_metadata"] if c["correct_option"] == correct)
        rng.shuffle(bases)
        phases.update({base: i % 2 for i, base in enumerate(bases)})
    selected = {regime: [] for regime in REGIMES}
    for row in examples:
        uid = row["example_id"]
        condition = CONDITIONS.index(row["condition"])
        selected["mixed_full"].append(uid)
        selected[("low_only", "temporal_only")[condition]].append(uid)
        view = int(row["view"] != "archived")
        if view == condition ^ phases[row["base_sample_id"]]:
            selected["mixed_matched"].append(uid)
    return {"seed": SETTINGS["seed"], "base_phase": {str(k): v for k, v in sorted(phases.items())},
        "rule": "complementary archived/permuted prompts across conditions; both statements retained",
        "selected_example_ids": {k: sorted(v) for k, v in selected.items()},
        "new_features_or_outcomes_used": False}


def describe(rows, plan):
    task_lookup = {t["eval_id"]: t for t in plan["tasks"]}
    prompts = {r["eval_id"]: task_lookup[r["eval_id"]] for r in rows}
    return {"base_ids": sorted({r["base_sample_id"] for r in rows}), "statement_rows": len(rows),
        "condition_counts": dict(Counter(r["condition"] for r in rows)),
        "truth_counts": dict(Counter(str(r["y_statement_true"]) for r in rows)),
        "correct_answer_counts_by_condition": {condition: dict(Counter(t["correct_option"] for t in prompts.values()
            if t["condition"] == condition)) for condition in CONDITIONS},
        "example_ids_sha256": digest(sorted(r["example_id"] for r in rows))}


def training_audit(examples, folds, selection, regime, plan):
    validate_folds(examples, folds)
    chosen = set(selection["selected_example_ids"][regime])
    selected = [r for r in examples if r["example_id"] in chosen]
    if len(selected) != len(chosen) or regime not in REGIMES:
        raise ValueError("Unknown training IDs or regime.")
    expected = {"low_only": {CONDITIONS[0]: 4}, "temporal_only": {CONDITIONS[1]: 4},
        "mixed_matched": {CONDITIONS[0]: 2, CONDITIONS[1]: 2},
        "mixed_full": {CONDITIONS[0]: 4, CONDITIONS[1]: 4}}[regime]
    for base in {r["base_sample_id"] for r in examples}:
        if Counter(r["condition"] for r in selected if r["base_sample_id"] == base) != Counter(expected):
            raise ValueError("Invalid per-base source-condition budget.")
    records = []
    for fold in folds:
        train = [r for r in selected if r["base_sample_id"] in fold["train_scene_seeds"]]
        records.append({"model_id": f"outer_{fold['outer_fold']}", "train": describe(train, plan),
            "test_base_ids": fold["test_scene_seeds"], "base_total_classifier_weight": 1.0,
            "inner_folds": [{"inner_fold": inner["inner_fold"],
                "train": describe([r for r in train if r["base_sample_id"] in inner["train_scene_seeds"]], plan),
                "validation": describe([r for r in train if r["base_sample_id"] in inner["validation_scene_seeds"]], plan)}
                for inner in fold["inner_folds"]]})
    return {"regime": regime, "outer_models": records, "selection_sha256": digest(selection)}


def fit_source(examples, values, folds, selection, regime, categorical=False, settings=None):
    if not categorical:
        # Use the same precision for source fitting, held-out prediction and
        # independent exported-coefficient replay.
        values = np.asarray(values, dtype=np.float64)
    chosen = set(selection["selected_example_ids"][regime])
    indices = [i for i, r in enumerate(examples) if r["example_id"] in chosen]
    fitted = crossfit([examples[i] for i in indices], subset(values, indices), folds, settings, categorical)
    predictions = []
    for record in fitted["models"]:
        test = [i for i, r in enumerate(examples) if r["base_sample_id"] in record["test_scene_seeds"]]
        scores = record["model"].decision_function(subset(values, test))
        probabilities = record["model"].predict_proba(subset(values, test))[:, 1]
        for i, score, probability in zip(test, scores, probabilities):
            row = examples[i]
            predictions.append({**row, "model_id": record["model_id"], "probe_logit": float(score),
                "probe_probability": float(probability), "probe_prediction": int(score > 0),
                "true_statement_evidence": float((2*row["y_statement_true"]-1)*score)})
    return {**fitted, "predictions": predictions}


def lexical_features(examples, plan):
    tasks = {r["eval_id"]: r for r in plan["tasks"]}
    return [{"statement": r["option_text"], "option_position": r["option"],
        "option_A": tasks[r["eval_id"]]["option_A"], "option_B": tasks[r["eval_id"]]["option_B"],
        "relation": r["relation_word"], "subject": str(r["subject_id"]), "reference": str(r["reference_id"])}
        for r in examples]


def evaluation(predictions, plan, settings=None):
    settings = {**SETTINGS, **(settings or {})}
    result = {}
    for view in VIEWS:
        rows = [r for r in predictions if view == "all" or r["view"] == view]
        metrics = []
        for condition in CONDITIONS:
            selected = [r for r in rows if r["condition"] == condition]
            for field in (None, "option", "subject_id"):
                for group in ([None] if field is None else sorted({r[field] for r in selected})):
                    items = [r for r in selected if field is None or r[field] == group]
                    labels = [r["y_statement_true"] for r in items]
                    aucs = []
                    for model_id in sorted({r["model_id"] for r in items}):
                        part = [r for r in items if r["model_id"] == model_id]
                        if len({r["y_statement_true"] for r in part}) == 2:
                            aucs.append(float(roc_auc_score([r["y_statement_true"] for r in part],
                                                          [r["probe_logit"] for r in part])))
                    metrics.append({"condition": condition, "group_by": field, "group": group,
                        "base_count": len({r["base_sample_id"] for r in items}), "statement_rows": len(items),
                        "accuracy": float(np.mean([r["probe_prediction"] == r["y_statement_true"] for r in items])),
                        "balanced_accuracy": (float(balanced_accuracy_score(labels, [r["probe_prediction"] for r in items]))
                                              if len(set(labels)) == 2 else None),
                        "mean_within_fold_roc_auc": float(np.mean(aucs)) if aucs else None,
                        "eligible_auc_folds": len(aucs), "truth_counts": dict(Counter(map(str, labels))),
                        "mean_true_statement_evidence": float(np.mean([r["true_statement_evidence"] for r in items]))})
        lookup = {r["example_id"]: r for r in rows}
        grouped = defaultdict(list)
        for pair in plan["boundary_pairs"]:
            if view != "all" and pair["view"] != view:
                continue
            low, temporal = (lookup[pair[side+"_example_id"]] for side in ("low", "temporal"))
            if low["model_id"] != temporal["model_id"] or any(low[k] != temporal[k] for k in
                    ("base_sample_id", "option_text", "y_statement_true", "view", "option")):
                raise ValueError("Boundary pair is not the same held-out statement/model.")
            grouped[low["base_sample_id"]].append((low, temporal))
        base_rows = []
        for base, pairs in sorted(grouped.items()):
            item = {"base_sample_id": base}
            for side, index in (("low", 0), ("temporal", 1)):
                item["evidence_"+side] = float(np.mean([p[index]["true_statement_evidence"] for p in pairs]))
                item["accuracy_"+side] = float(np.mean([p[index]["probe_prediction"] == p[index]["y_statement_true"] for p in pairs]))
            for name in ("evidence", "accuracy"):
                item[name+"_temporal_minus_low"] = item[name+"_temporal"]-item[name+"_low"]
            base_rows.append(item)
        summaries = {name: bootstrap_mean([r[name] for r in base_rows], settings["seed"], settings["bootstrap_replicates"])
            for name in ("evidence_low", "evidence_temporal", "evidence_temporal_minus_low",
                         "accuracy_low", "accuracy_temporal", "accuracy_temporal_minus_low")}
        result[view] = {"metrics": metrics, "paired_summary": summaries, "base_rows": base_rows}
    return result


def behaviour(decisions, plan):
    result = {}
    for view in VIEWS:
        result[view] = {}
        for condition in CONDITIONS:
            tasks = [r for r in plan["tasks"] if r["condition"] == condition and (view == "all" or r["view"] == view)]
            rows = [decisions[r["eval_id"]] for r in tasks]
            result[view][condition] = {"prompt_cells": len(rows), "correct": sum(r["is_correct"] for r in rows),
                "accuracy": float(np.mean([r["is_correct"] for r in rows])),
                "mean_correct_logit": float(np.mean([r["correct_logit"] for r in rows])),
                "mean_incorrect_logit": float(np.mean([r["incorrect_logit"] for r in rows])),
                "mean_margin": float(np.mean([r["margin"] for r in rows])),
                "correct_answer_counts": dict(Counter(r["correct_option"] for r in tasks))}
    return result
