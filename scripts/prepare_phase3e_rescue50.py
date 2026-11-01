"""Freeze the original rescue50 cohort for option-side condition probing.

This CPU preparation audits archived metadata/tables, not raw GPU checkpoints.
It never generates videos, fits probes, runs KO or changes historical sources.
"""

import argparse
import copy
import math
import sys
from collections import Counter
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.phase3c_core import atomic_write, digest, file_hash, read_json
from scripts.phase3d_core import CONDITIONS, audit_pairs, base_folds, option_order_control

SCHEMA = "phase3e_rescue50_option_preparation_v1"
READOUTS = [f"reference_last:L{layer}_block_output" for layer in (16, 17, 18, 20)]
ROUTE = {"query_group": "options_all", "key_group": "whole_event2",
         "control": "whole_event_reference"}
VIEWS = ("archived", "option_order_permuted")


def intact_plan(plan):
    if (plan.get("artifact_type") != "real" or
            plan.get("fingerprint") != digest({k: v for k, v in plan.items() if k != "fingerprint"})):
        raise ValueError("Require an intact historical real plan.")


def close(a, b):
    return all(type(x) in (int, float) and math.isfinite(x) for x in (a, b)) and math.isclose(
        a, b, rel_tol=0, abs_tol=1e-9)


def checked_ko_row(row, case):
    if (any(row.get(k) != case[k] for k in
            ("pair_id", "base_sample_id", "first_object_id", "prompt_variant", "correct_option")) or
            any(row.get(k) != v for k, v in ROUTE.items())):
        raise ValueError("KO table does not describe the original pair and option-to-video route.")
    for side, expected in (("low", False), ("temporal", True)):
        for stage in ("base", "ko"):
            prefix = f"{side}_{stage}_"
            if not close(row[prefix+"margin"], row[prefix+"correct_logit"]-row[prefix+"incorrect_logit"]):
                raise ValueError("KO table margin disagrees with its original logit levels.")
        prediction = row[f"{side}_base_prediction"]
        margin = row[f"{side}_base_margin"]
        if (prediction not in ("A", "B") or (prediction == case["correct_option"]) != expected or
                (margin > 0) != expected or margin == 0 or row[f"{side}_key_count"] != 512):
            raise ValueError("Archived rescue behavior or 512-position support changed.")
        if not close(row[f"delta_M_{side}"], row[f"{side}_ko_margin"]-margin):
            raise ValueError("KO change disagrees with its raw baseline and intervention levels.")
    if (not close(row["advantage_base"], row["temporal_base_margin"]-row["low_base_margin"]) or
            not close(row["advantage_KO"], row["temporal_ko_margin"]-row["low_ko_margin"]) or
            not close(row["compression"], row["advantage_base"]-row["advantage_KO"])):
        raise ValueError("Boundary decomposition disagrees with original levels.")


def task_manifest(pairs, cases):
    tasks, examples, contrasts = [], [], []
    for case in cases:
        pair_id = case["pair_id"]
        for condition in CONDITIONS:
            original = pairs[pair_id][condition]
            for view in VIEWS:
                control = option_order_control(original)
                options = {key: original[f"option_{key}"] if view == "archived" else control[f"option_{key}"]
                           for key in ("A", "B")}
                correct = original["correct_option"] if view == "archived" else control["correct_option"]
                identity = original["eval_id"] + ("" if view == "archived" else "__pure_option_permutation")
                task = {"eval_id": identity, "archived_eval_id": original["eval_id"],
                    "pair_id": pair_id, "base_sample_id": case["base_sample_id"], "condition": condition,
                    "view": view, "video_path": original["video_path"],
                    "video_sha256": original["phase3c_video_sha256"],
                    "question": original.get("question"), "option_A": options["A"], "option_B": options["B"],
                    "correct_option": correct, "require_rescue_behavior": view == "archived",
                    "intervention": "normal_forward"}
                tasks.append(task)
                for option in ("A", "B"):
                    examples.append({"example_id": identity+"__statement_"+option, "eval_id": identity,
                        "pair_id": pair_id, "base_sample_id": case["base_sample_id"],
                        "condition": condition, "view": view, "option": option,
                        "option_text": options[option], "y_statement_true": int(option == correct),
                        "intervention": "normal_forward"})
        for view in VIEWS:
            for option in ("A", "B"):
                matched = [e for e in examples if e["pair_id"] == pair_id and
                           e["view"] == view and e["option"] == option]
                low, temporal = (next(e for e in matched if e["condition"] == condition) for condition in CONDITIONS)
                if any(low[k] != temporal[k] for k in ("option_text", "y_statement_true")):
                    raise ValueError("Boundary partners changed statement text or truth.")
                contrasts.append({"pair_id": pair_id, "view": view, "option": option,
                    "low_example_id": low["example_id"], "temporal_example_id": temporal["example_id"]})
    return tasks, examples, contrasts


def prepare(source_root, pairs_path, mappings_path, singleton_root, output_root):
    source, singleton, output = (Path(p).resolve() for p in (source_root, singleton_root, output_root))
    pairs_path, mappings_path = Path(pairs_path).resolve(), Path(mappings_path).resolve()
    inputs = [source, singleton, pairs_path, mappings_path]
    if any(output == p or output in p.parents or p in output.parents for p in inputs):
        raise ValueError("Use a separate new output directory; historical sources remain read-only.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory; do not overwrite an earlier preparation.")
    source_hashes = {}

    def read(path):
        source_hashes[str(path)] = file_hash(path)
        return read_json(path)

    old = read(source / "plan.json")
    intact_plan(old)
    pairs, mappings = read(pairs_path), read(mappings_path)
    if any(file_hash(path) != old["source_files"][name]
           for path, name in ((pairs_path, "pairs.json"), (mappings_path, "mappings.json"))):
        raise ValueError("Pairs/mappings differ from the frozen Phase 3D source bytes.")
    cases = audit_pairs(pairs)
    metadata, folds = read(source / "case_metadata.json"), read(source / "outer_folds.json")
    if (metadata != cases or digest(metadata) != old["case_metadata_sha256"] or
            old["case_ids"] != [c["pair_id"] for c in cases] or
            folds != base_folds(cases) or digest(folds) != old["folds_sha256"] or
            set(mappings) != set(old["case_ids"])):
        raise ValueError("Do not add, replace, reorder or re-split the original 50 bases.")
    for case in cases:
        mapping = mappings[case["pair_id"]]
        support = mapping["support_audit"]["supports"]["whole_event2"]
        if (support.get("one_to_one") is not True or any(len(support[k]) != 512 or
                len(set(support[k])) != 512 for k in ("low_positions", "temporal_positions"))):
            raise ValueError("Keep the historical one-to-one whole Event-2 grid.")
        for condition in CONDITIONS:
            record = mapping["processor_records"][condition]
            if record["video_sha256"] != pairs[case["pair_id"]][condition]["phase3c_video_sha256"]:
                raise ValueError("Video checksum differs between frozen row and processor audit.")

    single_plan = read(singleton / "plan.json")
    intact_plan(single_plan)
    if (single_plan["source_plan_fingerprint"] != old["fingerprint"] or
            single_plan["case_ids"] != old["case_ids"] or single_plan["settings"] != old["settings"]):
        raise ValueError("Singleton KO is not from the same frozen cohort and runtime settings.")

    def ko_table(root, single):
        summary = read(root / "aggregate_summary.json")
        path = root / "case_decomposition.json"
        if (summary.get("complete") is not True or summary.get("artifact_type") != "real" or
                summary.get("missing_count") != 0 or
                summary["plan_fingerprint"] != (single_plan if single else old)["fingerprint"] or
                file_hash(path) != summary["output_sha256"][path.name]):
            raise ValueError("KO metadata/table checksum or completion check failed.")
        return read(path)

    joint = [r for r in ko_table(source / "knockout/analysis", False)
             if all(r.get(k) == v for k, v in ROUTE.items()) and
             (r["window_start"], r["window_end"]) == (16, 19)]
    single = ko_table(singleton / "analysis", True)
    if len(joint) != 50 or len(single) != 250:
        raise ValueError("Need all 50 joint references and all 200 singleton pairs plus 50 references.")
    cases_by_id, joint_by_id = {c["pair_id"]: c for c in cases}, {r["pair_id"]: r for r in joint}
    if set(joint_by_id) != set(cases_by_id):
        raise ValueError("Joint KO table changed the original base IDs.")
    for row in joint:
        checked_ko_row(row, cases_by_id[row["pair_id"]])
    seen = set()
    for row in single:
        key = (row["pair_id"], row["window_start"], row["window_end"])
        if (row["pair_id"] not in cases_by_id or key in seen or
                key[1:] not in ((16, 16), (17, 17), (18, 18), (19, 19), (16, 19))):
            raise ValueError("Missing/foreign/duplicated singleton KO cell.")
        seen.add(key)
        checked_ko_row(row, cases_by_id[row["pair_id"]])
        reference = joint_by_id[row["pair_id"]]
        fields = [f"{side}_base_{name}" for side in ("low", "temporal")
                  for name in ("correct_logit", "incorrect_logit", "margin", "prediction")]
        if key[1:] == (16, 19):
            fields += [f"{side}_ko_{name}" for side in ("low", "temporal")
                       for name in ("correct_logit", "incorrect_logit", "margin", "prediction")]
        if any(row[k] != reference[k] for k in fields):
            raise ValueError("Singleton baselines/joint reference disagree with KO50.")

    tasks, examples, contrasts = task_manifest(pairs, cases)
    manifest = read(source / "feature_store/manifest.json")
    if (manifest.get("complete") is not True or manifest.get("artifact_type") != "real" or
            manifest["plan_fingerprint"] != old["fingerprint"]):
        raise ValueError("Existing feature inventory belongs to a different experiment.")
    reusable = [name for name in READOUTS if name in manifest["feature_names"]]
    audit = {"status": "cpu_cohort_alignment_complete", "scope": "metadata_and_checksum_bound_analysis_tables",
        "raw_checkpoints_or_video_bytes_replayed": False, "independent_bases": 50,
        "original_low_temporal_pairs": 50, "original_prompt_cells": 100,
        "capture_prompt_cells_including_AB_control": len(tasks), "statement_rows": len(examples),
        "boundary_statement_pairs": len(contrasts), "joint_KO_pairs": len(joint),
        "singleton_KO_pairs": len(single)-50, "new_KO_forwards_required_for_cohort_alignment": 0,
        "archived_correct_answer_positions": dict(Counter(c["correct_option"] for c in cases)),
        "correct_answer_positions_with_permutation_per_condition": dict(Counter(
            t["correct_option"] for t in tasks if t["condition"] == CONDITIONS[0])),
        "existing_exact_readouts": reusable, "missing_readouts": [n for n in READOUTS if n not in reusable],
        "next_gate": "restore_original_sources_and_audit_raw_option_token_coverage",
        "GPU_capture_budget": "not_yet_determined; reuse verified raw per-token states before recapture",
        "no_gpu_or_probe_fitting_performed": True}
    levels = {}
    for side in ("low", "temporal"):
        levels[side] = sum(r[f"{side}_base_margin"] for r in joint)/50
    audit["archived_model_margin"] = {**levels, "temporal_minus_low": levels["temporal"]-levels["low"]}
    plan = {"schema": SCHEMA, "status": "awaiting_option_capture", "cohort": "original_frozen_rescue50",
        "source_plan_fingerprint": old["fingerprint"], "case_ids": old["case_ids"],
        "settings": copy.deepcopy(old["settings"]), "source_runtime": old["source_runtime"],
        "outer_folds": folds, "case_metadata": cases, "tasks": tasks, "statement_examples": examples,
        "boundary_pairs": contrasts, "readouts": READOUTS,
        "training_regimes": ["low_only", "temporal_only", "mixed_full", "mixed_matched"],
        "primary_evaluation_view": "archived", "AB_control_view": "option_order_permuted",
        "training_label": "statement_truth", "training_feature": "single_statement_reference_last_hidden_state",
        "contrast": "same_statement_temporal_minus_low; no T1-minus-T2 training feature",
        "new_order_reversed_videos": False, "regenerate_original_videos": False,
        "other_40_family_data": "preserve_as_separate_supplementary_experiment",
        "source_files_sha256": source_hashes, "preparation_code_sha256": file_hash(Path(__file__)),
        "protocol_sha256": file_hash(Path(__file__).resolve().parents[1]/"docs/phase3e_rescue50_option_protocol.md")}
    plan["fingerprint"] = digest(plan)
    if any(file_hash(Path(path)) != sha for path, sha in source_hashes.items()):
        raise ValueError("A source changed during preparation.")
    report = ["# Rescue50 Option-Probe Cohort Alignment", "",
        "Status: CPU preparation only; no new probe result or GPU run.", "",
        "- Original 50 rescue bases, 100 archived Low/Temporal prompts retained.",
        "- 100 pure A/B-permuted control prompts; 200 capture cells, 400 statements in total.",
        "- All 50 joint KO references and 200 singleton KO pairs match these same bases and baselines.",
        "- Reuse historical KO results; no KO rerun is required to align cohorts.",
        "- Keep original videos, timestamps, distractors, settings and grouped folds unchanged.",
        "- Preserve the completed 40-family experiment separately as supplementary evidence.", "",
        f"Archived model margin: Low {levels['low']:.3f}; Temporal {levels['temporal']:.3f}; "
        f"T - L {levels['temporal']-levels['low']:+.3f}.", "",
        "## Capture Readiness", "",
        f"Exact new readouts already present: {len(reusable)}/{len(READOUTS)}.",
        "The old object-difference features cannot reconstruct a last-token hidden state.",
        "Restore and inspect raw per-token checkpoints first; Phase 3B captured all option positions at every block.",
        "Reuse only after checking original prompt/input, token support, site semantics and logit provenance.",
        "Capture genuinely missing positions/views only, following a small exact-input/logit preflight.",
        "This preparation verifies metadata and saved analysis-table checksums, not raw activation/video replay.", "",
        "## Analysis", "",
        "Fit Low-only, Temporal-only and mixed truth probes with base-grouped train-only preprocessing.",
        "Evaluate original archived prompts separately for direct KO comparisons; report A/B controls separately.",
        "Keep every base even if a new runtime replay fails: stop and inspect, never reselect rescue cases.",
        "The 50 original pairs have fixed event order: they provide boundary contrasts, not order reversals.", ""]
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("plan", plan), ("audit", audit)):
        atomic_write(output / f"{name}.json", value)
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source_root", "pairs_path", "mappings_path", "singleton_root", "output_root"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(prepare(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
