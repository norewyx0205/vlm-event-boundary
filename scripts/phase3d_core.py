"""CPU-only canonical labels, paired evidence and leakage-safe base folds."""

import math
import random
from collections import Counter

CONDITIONS = ("low_boundary", "temporal_boundary")
SCHEMA = "phase3d_metadata_preparation_v1"


def canonical_relation(row):
    if any(type(item["id"]) is not int for item in row["target_objects"]):
        raise ValueError("Literal target identities must be integer IDs.")
    objects = {item["id"]: item for item in row["target_objects"]}
    if len(row["target_objects"]) != 2 or set(objects) != {1, 2}:
        raise ValueError("Require exactly the two literal identities T1 and T2.")
    for obj in objects.values():
        start, end = obj["start_frame"], obj["end_frame"]
        if any(type(value) not in (int, float) or not math.isfinite(value)
               for value in (start, end)) or not 0 <= start < end:
            raise ValueError("Target events require finite, positive-duration timing.")
    first = min(objects, key=lambda key: objects[key]["start_frame"])
    second = 3 - first
    if objects[first]["end_frame"] > objects[second]["start_frame"]:
        raise ValueError("Overlapping/tied events do not define this before/after task.")
    if row["first_object_id"] != first or row.get("second_object_id", second) != second:
        raise ValueError("Event timing disagrees with first/second-object metadata.")
    return "before" if first == 1 else "after"


def statement(row, text):
    labels = {obj["id"]: obj["label"] for obj in row["target_objects"]}
    if len({label.casefold() for label in labels.values()}) != 2:
        raise ValueError("Target labels cannot be identical.")
    for subject in (1, 2):
        for relation in ("before", "after"):
            expected = f"{labels[subject]} moves {relation} {labels[3-subject]}."
            if text.casefold() == expected.casefold():
                return {"subject_id": subject, "reference_id": 3-subject, "relation": relation}
    raise ValueError("Unsupported option template; do not guess semantic labels.")


def audit_row(row):
    relation = canonical_relation(row)
    options = {key: statement(row, row[f"option_{key}"]) for key in ("A", "B")}
    if (options["A"]["subject_id"] != options["B"]["subject_id"] or
            options["A"]["relation"] == options["B"]["relation"]):
        raise ValueError("Require contradictory options with the same ordered object pair.")
    correct = row["correct_option"]
    if correct not in options:
        raise ValueError("Correct option must be A or B.")
    expected = relation if options[correct]["subject_id"] == 1 else (
        "after" if relation == "before" else "before")
    if options[correct]["relation"] != expected:
        raise ValueError("Correct option contradicts the actual event order.")
    if row.get("correct_relation", expected) != expected:
        raise ValueError("Sentence-relative annotation contradicts the correct option.")
    if row.get("correct_sentence", row[f"option_{correct}"]) != row[f"option_{correct}"]:
        raise ValueError("Correct sentence differs from the selected answer.")
    return {"canonical_relation": relation, "y": int(relation == "before"),
            "sentence_relation": expected, "option_semantics": options}


def option_order_control(row):
    """A pure permutation, not an archived subject/reference reversal."""
    label = audit_row(row)
    return {"base_sample_id": row["base_sample_id"], "view": "option_order_permuted",
            "option_A": row["option_B"], "option_B": row["option_A"],
            "correct_option": "B" if row["correct_option"] == "A" else "A",
            "canonical_relation": label["canonical_relation"], "y": label["y"],
            "same_video_and_sentence_contents": True,
            "requires_fresh_tokenization_and_baseline": True,
            "require_rescue_behavior": False}


def audit_pairs(pairs, expected_cases=50):
    if not isinstance(pairs, dict) or len(pairs) != expected_cases:
        raise ValueError("Do not drop/add cases from the frozen rescue cohort.")
    result, bases, evaluations = [], set(), set()
    for pair_id, pair in pairs.items():
        if set(pair) != set(CONDITIONS):
            raise ValueError("Each base needs exactly low and temporal rows.")
        low, temporal = (pair[name] for name in CONDITIONS)
        labels = [audit_row(row) for row in (low, temporal)]
        if labels[0] != labels[1]:
            raise ValueError("Paired conditions have different relation/option semantics.")
        fields = ("base_sample_id", "prompt_variant", "correct_option", "option_A", "option_B")
        if any(low[name] != temporal[name] for name in fields):
            raise ValueError("Condition pairs must retain the same base and prompt.")
        if low.get("question") != temporal.get("question"):
            raise ValueError("Question text changed between paired conditions.")
        attributes = lambda row: sorted((obj["id"], obj["label"], obj.get("color"),
                                        obj.get("shape"), obj.get("size")) for obj in row["target_objects"])
        if attributes(low) != attributes(temporal):
            raise ValueError("Static target attributes changed between conditions.")
        base = low["base_sample_id"]
        if type(base) is not int or base in bases:
            raise ValueError("Every independent base must occur exactly once.")
        bases.add(base)
        for condition, row in zip(CONDITIONS, (low, temporal)):
            if row["condition"] != condition or row["eval_id"] in evaluations:
                raise ValueError("Condition/evaluation ID is inconsistent or duplicated.")
            if row.get("phase3c_analysis_stratum") != "rescue":
                raise ValueError("Use the unchanged frozen rescue cohort.")
            evaluations.add(row["eval_id"])
        result.append({"pair_id": pair_id, "base_sample_id": base, **labels[0],
                       "first_object_id": low["first_object_id"],
                       "prompt_variant": low["prompt_variant"],
                       "correct_option": low["correct_option"],
                       "condition_eval_ids": {name: pair[name]["eval_id"] for name in CONDITIONS},
                       "static_attributes": {str(obj["id"]): {name: obj.get(name)
                           for name in ("label", "color", "shape", "size")}
                           for obj in low["target_objects"]},
                       "option_order_control": option_order_control(low)})
    return sorted(result, key=lambda row: row["base_sample_id"])


def base_folds(cases, n_folds=5, seed=42):
    if type(n_folds) is not int or n_folds < 2:
        raise ValueError("Need at least two outer folds.")
    bases = [row["base_sample_id"] for row in cases]
    if len(set(bases)) != len(bases) or any(row["y"] not in (0, 1) for row in cases):
        raise ValueError("Folds require one binary relation label per unique base.")
    if min(Counter(row["y"] for row in cases).get(y, 0) for y in (0, 1)) < n_folds:
        raise ValueError("Insufficient bases of either relation for stratified folds.")
    rng, assignments, offset = random.Random(seed), {}, 0
    for label in (0, 1):
        group = sorted(row["base_sample_id"] for row in cases if row["y"] == label)
        rng.shuffle(group)
        for index, base in enumerate(group):
            assignments[base] = (offset + index) % n_folds
        offset = (offset + len(group)) % n_folds
    return [{"outer_fold": fold,
             "train_base_ids": sorted(base for base in bases if assignments[base] != fold),
             "test_base_ids": sorted(base for base in bases if assignments[base] == fold),
             "test_class_counts": dict(Counter(row["canonical_relation"] for row in cases
                 if assignments[row["base_sample_id"]] == fold))}
            for fold in range(n_folds)]


def relation_evidence(y, logit):
    if type(y) is not int or y not in (0, 1) or type(logit) not in (int, float) or not math.isfinite(logit):
        raise ValueError("Require a canonical binary label and finite signed probe logit.")
    return (2*y - 1) * logit


def evidence_changes(y, low_logit, temporal_logit, patched_low_logit):
    low, temporal, patched = (relation_evidence(y, value)
                              for value in (low_logit, temporal_logit, patched_low_logit))
    return {"S_low": low, "S_temporal": temporal, "S_low_patched": patched,
            "delta_S_boundary": temporal-low, "delta_S_patch": patched-low}


def declared_saved_sites(index):
    """Observed hooks are not necessarily persisted tensors."""
    audit = index.get("hook_audit", {})
    saved = audit.get("saved_sites")
    if saved is not None:
        sites = {tuple(value) for value in saved}
        if len(sites) != len(saved) or len(sites) != index.get("site_count"):
            raise ValueError("Saved-site declaration disagrees with site count.")
        return sites
    if (index.get("site_count") == 39 and audit.get("block_output_sites") == 36 and
            audit.get("post_deepstack_layers") == [0, 1, 2]):
        return {(layer, "block_output") for layer in range(36)} | {
            (layer, "post_deepstack") for layer in (0, 1, 2)}
    return None
