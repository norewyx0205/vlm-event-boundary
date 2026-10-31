"""CPU-only scene/label manifests for the revised option-side Phase 3E-B design."""

import copy
import random
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

from . import phase3e_preflight as fixture
from .phase3c_core import digest
from .phase3d_core import audit_row, statement

SCHEMA = "phase3e_paired_option_design_v1"
SCENE_SEEDS = tuple(range(2026103101, 2026103141))
PRIMARY_LAYERS = (16, 17, 18)
REFERENCE_LAYERS = (0, 20)
PRIMARY_READOUT = "last_token_of_option_reference_object_mention"


def statement_truth(row, option):
    semantics = statement(row, row[f"option_{option}"])
    objects = {obj["id"]: obj for obj in row["target_objects"]}
    subject_first = objects[semantics["subject_id"]]["start_frame"] < objects[semantics["reference_id"]]["start_frame"]
    return int(subject_first == (semantics["relation"] == "before"))


def group_folds(seeds, outer_count=5, inner_count=4, seed=42):
    if (len(set(seeds)) != len(seeds) or outer_count < 2 or inner_count < 2 or
            len(seeds)-((len(seeds)+outer_count-1)//outer_count) < inner_count):
        raise ValueError("Need distinct scene families and enough groups for nested folds.")
    shuffled = sorted(seeds)
    random.Random(seed).shuffle(shuffled)
    folds = []
    for outer in range(outer_count):
        test = sorted(shuffled[outer::outer_count])
        train = sorted(set(seeds)-set(test))
        inner_order = train.copy()
        random.Random(seed+outer+1).shuffle(inner_order)
        inner = []
        for index in range(inner_count):
            validation = sorted(inner_order[index::inner_count])
            inner.append({"inner_fold": index, "train_scene_seeds": sorted(set(train)-set(validation)),
                          "validation_scene_seeds": validation})
        folds.append({"outer_fold": outer, "train_scene_seeds": train, "test_scene_seeds": test,
                      "inner_folds": inner})
    return folds


def labels_and_pairs(rows):
    lookup = {(r["scene_seed"], r["first_object_id"], r["condition"], r["subject_id"], r["layout"]): r
              for r in rows}
    examples, order_pairs, boundary_pairs = [], [], []
    for row in rows:
        order_key = (row["scene_seed"], 3-row["first_object_id"], row["condition"], row["subject_id"], row["layout"])
        other_condition = next(c for c in fixture.CONDITIONS if c != row["condition"])
        boundary_key = (row["scene_seed"], row["first_object_id"], other_condition, row["subject_id"], row["layout"])
        if order_key not in lookup or boundary_key not in lookup:
            raise ValueError("Missing order-reversal or boundary partner.")
        reverse, boundary = lookup[order_key], lookup[boundary_key]
        for option in ("A", "B"):
            semantics = statement(row, row[f"option_{option}"])
            label = statement_truth(row, option)
            group = f"option_{option}_t{semantics['reference_id']}"
            example_id = row["eval_id"]+"__statement_"+option
            reference = next(o for o in row["target_objects"] if o["id"] == semantics["reference_id"])
            examples.append({"example_id": example_id, "eval_id": row["eval_id"],
                "scene_seed": row["scene_seed"], "condition": row["condition"], "option": option,
                "option_text": row[f"option_{option}"], "subject_id": semantics["subject_id"],
                "reference_id": semantics["reference_id"], "reference_label": reference["label"],
                "relation_word": semantics["relation"], "y_statement_true": label,
                "group_name": group, "primary_selection": "last_actual_contextual_token",
                "family_weight": 1/32, "intervention": "normal_forward"})
            order_pairs.append({"left_example_id": example_id,
                "right_example_id": reverse["eval_id"]+"__statement_"+option,
                "scene_seed": row["scene_seed"], "condition": row["condition"],
                "group_name": group, "same_reference_label": reference["label"],
                "y_left_statement_true": label,
                "y_right_statement_true": statement_truth(reverse, option),
                "subtraction": "left_minus_right", "feature": "same_object_same_option_position",
                "unordered_pair_id": digest(sorted((example_id, reverse["eval_id"]+"__statement_"+option)))})
            if row["condition"] == "low_boundary":
                boundary_pairs.append({"low_example_id": example_id,
                    "temporal_example_id": boundary["eval_id"]+"__statement_"+option,
                    "scene_seed": row["scene_seed"], "first_object_id": row["first_object_id"],
                    "group_name": group, "y_statement_true": label, "subtraction": "temporal_minus_low"})
    return examples, order_pairs, boundary_pairs


def validate_materials(rows, seeds):
    original = fixture.audit_factorial(rows, seeds)
    lexical = defaultdict(Counter)
    lookup = {(r["scene_seed"], r["first_object_id"], r["condition"], r["subject_id"], r["layout"]): r
              for r in rows}
    for row in rows:
        audit_row(row)
        truth = {o: statement_truth(row, o) for o in ("A", "B")}
        if sum(truth.values()) != 1 or truth[row["correct_option"]] != 1:
            raise ValueError("Named statement truth disagrees with the answer annotation.")
        reverse = lookup[row["scene_seed"], 3-row["first_object_id"], row["condition"], row["subject_id"], row["layout"]]
        boundary = lookup[row["scene_seed"], row["first_object_id"],
            next(c for c in fixture.CONDITIONS if c != row["condition"]), row["subject_id"], row["layout"]]
        if (any(row[k] != reverse[k] or row[k] != boundary[k]
                for k in ("option_A", "option_B", "subject_id", "layout")) or
                row.get("question") != reverse.get("question") or row.get("question") != boundary.get("question")):
            raise ValueError("Order-reversal videos must have identical question/option contents.")
        if row["event_timing"] != fixture.timing(row["condition"]):
            raise ValueError("Boundary condition disagrees with the fixed low/temporal timing schedule.")
        for obj in row["target_objects"]:
            event = "first" if obj["id"] == row["first_object_id"] else "second"
            expected = tuple(row["event_timing"][f"{event}_event_{k}_frame"] for k in ("start", "end"))
            if (obj["start_frame"], obj["end_frame"]) != expected:
                raise ValueError("Actual object timing does not match the event schedule.")
        for option in ("A", "B"):
            if statement_truth(row, option) == statement_truth(reverse, option):
                raise ValueError("Order reversal failed to reverse the same statement's truth.")
            key = row["scene_seed"], row["condition"], row[f"option_{option}"]
            lexical[key][truth[option]] += 1
    if any(counts[0] != counts[1] for counts in lexical.values()):
        raise ValueError("Identical option words have an unbalanced truth label.")
    return {k: v for k, v in original.items() if k != "event2_mover_shortcut_eliminated"} | {
        "same_object_appearance_and_path_preserved": True,
        "order_change_is_motion_retiming_not_appearance_swap": True,
        "identical_statement_label_strata": len(lexical), "identical_statement_labels_balanced": True,
        "second_event_mover_is_valid_task_evidence": True}


def validate_plan(plan):
    rows, seeds = plan["rows"], plan["scene_seeds"]
    audit = validate_materials(rows, seeds)
    examples, pairs, boundaries = labels_and_pairs(rows)
    if (plan["material_audit"] != audit or plan["statement_examples"] != examples or
            plan["order_pairs"] != pairs or plan["boundary_pairs"] != boundaries):
        raise ValueError("Changed labels, identity correspondence or pairing manifest.")
    folds = plan["folds"]
    tested = Counter()
    for fold in folds:
        train, test = set(fold["train_scene_seeds"]), set(fold["test_scene_seeds"])
        if train & test or train | test != set(seeds) or not train or not test:
            raise ValueError("Outer scene folds overlap or omit families.")
        tested.update(test)
        validation = Counter()
        for inner in fold["inner_folds"]:
            fit, val = set(inner["train_scene_seeds"]), set(inner["validation_scene_seeds"])
            if fit & val or fit | val != train or not fit or not val:
                raise ValueError("Inner folds leak an outer test family or omit training families.")
            validation.update(val)
        if validation != Counter({s: 1 for s in train}):
            raise ValueError("Inner validation must hold out each training family exactly once.")
    if tested != Counter({s: 1 for s in seeds}):
        raise ValueError("Outer folds must hold out each scene family exactly once.")
    if plan["fingerprint"] != digest({k: v for k, v in plan.items() if k != "fingerprint"}):
        raise ValueError("Design fingerprint changed.")
    return audit


def build_plan(root, seeds=SCENE_SEEDS):
    if len(seeds) != 40:
        raise ValueError("The revised planning cohort is exactly 40 scene families.")
    scenes, rows = fixture.factorial_rows(root, seeds)
    rows = copy.deepcopy(rows)
    for row in rows:
        row.pop("render_objects")
        row.pop("render_distractors")
        row["dataset_version"] = SCHEMA
    examples, order_pairs, boundary_pairs = labels_and_pairs(rows)
    plan = {"schema": SCHEMA, "status": "cpu_design_only", "scene_seeds": list(seeds),
        "scene_families": scenes, "rows": rows, "statement_examples": examples,
        "order_pairs": order_pairs, "boundary_pairs": boundary_pairs,
        "material_audit": validate_materials(rows, seeds), "folds": group_folds(seeds),
        "primary_readout": PRIMARY_READOUT, "primary_layers": list(PRIMARY_LAYERS),
        "reference_layers": list(REFERENCE_LAYERS),
        "primary_training": "normal single-video statement truth; no cross-object subtraction",
        "paired_analysis": "same identity and same option occurrence; both explicit order-pair directions",
        "secondary_pairwise_training": "normal ordered-pair differences; label is truth in the left video",
        "alignment_with_prior_ko": {"endpoint_class": "option tokens", "prior_query_group": "options_all",
            "prior_key_group": "whole_event2", "new_probe_attention_blocked": False,
            "requires_new_ko_forwards": False, "ko_conditioned_probing": "optional_not_scheduled"},
        "counts": {"independent_scene_families": 40, "videos": 160, "baseline_prompt_cells": 640,
            "single_video_statement_examples": len(examples), "directed_order_statement_pairs": len(order_pairs),
            "low_temporal_statement_pairs": len(boundary_pairs), "new_ko_forwards": 0,
            "new_patch_forwards": 0, "technical_parity_replays": "additional_execution_gate_not_yet_frozen"},
        "gpu_launch_authorized_by_this_plan": False,
        "live_capture_and_probe_fit_executed": False}
    plan["fingerprint"] = digest(plan)
    validate_plan(plan)
    return plan


def render_examples(root, plan, count=2):
    """Render the first fixed families, never replace them based on behaviour."""
    seeds = tuple(plan["scene_seeds"][:count])
    _, render_rows = fixture.factorial_rows(root, seeds)
    seen, audits = set(), []
    for row in render_rows:
        if row["video_id"] in seen:
            continue
        seen.add(row["video_id"])
        path = Path(row["video_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        fixture.generator.render_video(path, row["condition"], row["fps"], row["duration_sec"],
            row["render_objects"], row["render_distractors"], row["event_timing"])
        audits.append({"video_id": row["video_id"], "scene_seed": row["scene_seed"],
                       "path": str(path), **fixture.video_audit(path)})
    return {"status": "cpu_render_only", "scene_seeds": list(seeds), "videos": audits,
            "model_forward_count": 0}
