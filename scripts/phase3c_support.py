"""Frozen, observed-donor support decomposition after the completed Phase 3C pilot."""

from collections import Counter
from pathlib import Path

try:
    from .analyze_phase3c import verify_inputs
    from .phase3c_core import (CONDITIONS, digest, event_layout, file_hash, frozen_write,
        read_json, support_record)
except ImportError:
    from analyze_phase3c import verify_inputs
    from phase3c_core import (CONDITIONS, digest, event_layout, file_hash, frozen_write,
        read_json, support_record)


SCHEMA = "phase3c_support_followup_v1"
SUPPORTS = ("event2_non_target_complement", "event2_non_target_budget_matched")
LAYERS = (0, 4, 8, 12, 16, 20)


def audit_supports(pair_id, pair, mapping, seed=42):
    supports = mapping["support_audit"]["supports"]
    whole, targets = supports["whole_event2"], supports["both_targets_event2"]
    layouts = [event_layout(pair[condition], mapping["processor_records"][condition]["video_metadata"],
        mapping["processor_records"][condition]["visual_positions"],
        mapping["processor_records"][condition]["prompt_token_count"]) for condition in CONDITIONS]
    pairs = list(zip(whole["low_positions"], whole["temporal_positions"]))
    target_pairs = list(zip(targets["low_positions"], targets["temporal_positions"]))
    if (not pairs or len(set(pairs)) != len(pairs) or not target_pairs or
            len(whole["low_positions"]) != len(whole["temporal_positions"]) or
            len(targets["low_positions"]) != len(targets["temporal_positions"]) or
            not set(target_pairs) <= set(pairs)):
        raise ValueError("Target union must be a nonempty, aligned subset of whole Event 2.")
    for side in ("low_positions", "temporal_positions"):
        if len(set(whole[side])) != len(whole[side]) or len(set(targets[side])) != len(targets[side]):
            raise ValueError("Support correspondence is not one-to-one.")
    if (set(whole["low_positions"]) != set(layouts[0]["lookup"]) or
            set(whole["temporal_positions"]) != set(layouts[1]["lookup"])):
        raise ValueError("Whole Event-2 support is not the complete audited grid.")
    complement = [item for item in pairs if item not in set(target_pairs)]
    bins, target_counts = {}, Counter()
    expected_bins = {(item["low_temporal_index"], item["temporal_temporal_index"])
                     for item in mapping["support_audit"]["event_bin_pairs"]}
    for left, right in pairs:
        low_cell, temporal_cell = layouts[0]["lookup"][left], layouts[1]["lookup"][right]
        if low_cell[1:] != temporal_cell[1:]:
            raise ValueError("Observed donor mapping changed spatial cells.")
        key = (low_cell[0], temporal_cell[0])
        if key not in expected_bins:
            raise ValueError("Observed donor mapping changed temporal-bin correspondence.")
        bins.setdefault(key, [])
        if (left, right) in target_pairs:
            target_counts[key] += 1
        else:
            bins[key].append((left, right))
    # Choose a single cross-condition mapping, never separate outcome-driven samples.
    selected, bin_audit = [], []
    for key, count in sorted(target_counts.items()):
        pool = bins[key]
        if len(pool) < count:
            raise ValueError("Insufficient non-target cells for exact within-bin token-budget matching.")
        ranked = sorted(pool, key=lambda item: digest([seed, pair_id, key,
            layouts[0]["lookup"][item[0]][1:]]))
        selected.extend(ranked[:count])
        bin_audit.append({"low_temporal_index": key[0], "temporal_temporal_index": key[1],
            "target_union_count": count, "control_count": count, "available_non_target_count": len(pool)})
    if not complement or set(complement) & set(target_pairs) or set(complement) | set(target_pairs) != set(pairs):
        raise ValueError("Target union and non-target complement do not partition whole Event 2.")
    result = {}
    for name, chosen in zip(SUPPORTS, (complement, selected)):
        low, temporal = ([item[side] for item in chosen] for side in (0, 1))
        result[name] = support_record(low, temporal, low, temporal, name)
    if len(selected) != len(target_pairs):
        raise ValueError("Matched context/target token counts differ.")
    return {"pair_id": pair_id, "supports": result, "whole_count": len(pairs),
        "target_union_count": len(target_pairs), "complement_count": len(complement),
        "matched_control_count": len(selected), "token_budget_by_bin": bin_audit,
        "partition_exact": True, "alignment": "source_frozen_event_relative_spatial_cell_mapping",
        "non_target_definition": "complement of cross-condition target-ROI union; may contain distractors",
        "content_limit": "Non-target-position residuals can already encode target information."}


def task_grid(config, stage):
    if stage not in ("preflight", "patch"):
        raise ValueError("Unknown support-followup GPU stage.")
    ids = config["preflight_case_ids"] if stage == "preflight" else config["case_ids"]
    tasks = [{"pair_id": pair_id, "condition": condition,
        "kind": "identity" if stage == "preflight" else "support_patch",
        "support": support, "layer": layer, "location": "block_output"}
        for pair_id in ids for condition in CONDITIONS for support in SUPPORTS for layer in LAYERS]
    for task in tasks:
        task["task_id"] = digest(task)
    return tasks


def code_hashes():
    root = Path(__file__).parent
    return {name: file_hash(root / name) for name in ("phase3c_support.py", "run_phase3c_support.py")}


def prepare(plan_dir, source_run, output, source_backup, freeze=False):
    plan, source, root = (Path(path).resolve() for path in (plan_dir, source_run, output))
    if (root == source or source in root.parents or root in source.parents or
            root == plan or root in plan.parents or plan in root.parents):
        raise ValueError("Use a separate follow-up root; the original plan and execution remain read-only.")
    frozen, pairs, execution, baselines, stages = verify_inputs(plan, source)
    summary = read_json(source / "analysis/aggregate_summary.json")
    if not summary.get("complete") or summary["execution_fingerprint"] != execution["execution_fingerprint"]:
        raise ValueError("A verified, completed Phase 3C pilot is required.")
    for name, expected in summary["output_sha256"].items():
        path = source / "analysis" / name
        if not path.resolve().is_relative_to(source / "analysis") or file_hash(path) != expected:
            raise ValueError("Source analysis output changed.")
    backup = Path(source_backup).resolve()
    manifest = read_json(backup / "backup_manifest.json")
    if (manifest.get("schema") != "phase3c_local_backup_v1" or not manifest.get("complete") or
            manifest.get("reports_only") or manifest.get("source_run_root") != str(source) or
            not manifest.get("activation_tensors_included")):
        raise ValueError("Retain the source pilot's complete activation/video backup before continuing.")
    mappings = load_mappings(plan)
    audits = {key: audit_supports(key, pairs[key], mapping, frozen["settings"]["seed"])
              for key, mapping in mappings.items()}
    capture_hashes = {}
    for pair_id, pair in pairs.items():
        for condition in CONDITIONS:
            row = baselines[pair[condition]["eval_id"]]
            index = read_json(row["capture_index_path"])
            side = "low_positions" if condition == CONDITIONS[0] else "temporal_positions"
            if any(not set(item[side]) <= set(index["positions"])
                   for item in audits[pair_id]["supports"].values()):
                raise ValueError("The source capture union is missing new intervention positions.")
            capture_hashes[row["capture_index_path"]] = row["capture_index_sha256"]
            capture_hashes[index["vectors_path"]] = index["vectors_sha256"]
    config = {"schema": SCHEMA, "artifact_type": "real", "exploratory_after_pilot": True,
        "selection_rule": "reuse every frozen pilot case; no new effect-based selection",
        "source_plan_dir": str(plan), "source_run_root": str(source), "output_root": str(root),
        "source_execution_fingerprint": execution["execution_fingerprint"],
        "source_selection_fingerprint": frozen["selection_fingerprint"],
        "source_analysis_sha256": file_hash(source / "analysis/aggregate_summary.json"),
        "source_backup_dir": str(backup), "source_backup_manifest_sha256": file_hash(backup / "backup_manifest.json"),
        "source_capture_sha256": capture_hashes, "source_execution_code_sha256": execution["execution_code_sha256"],
        "followup_code_sha256": code_hashes(), "case_ids": frozen["case_ids"],
        "preflight_case_ids": frozen["technical_preflight_case_ids"], "layers": list(LAYERS),
        "location": "block_output", "seed": frozen["settings"]["seed"], "support_audit": audits,
        "source_runtime": execution["runtime"], "source_path_map": execution["path_map"],
        "new_activation_capture": False, "source_reference_patch_count": len(stages["patch"]),
        "primary_patch_count": 288, "technical_identity_count": 48,
        "interpretation": "Position-support and budget comparisons do not isolate the information content of residuals."}
    config["fingerprint"] = digest(config)
    path = root / "support_config.json"
    if freeze:
        frozen_write(path, config)
    elif not path.is_file() or read_json(path) != config:
        raise ValueError("Support configuration is missing/changed; explicitly prepare and freeze before running.")
    return config, frozen, pairs, mappings, baselines


def load_mappings(plan):
    try:
        from .phase3c_execution import load_selection
    except ImportError:
        from phase3c_execution import load_selection
    return load_selection(plan)[2]


def validate_config(config):
    if (config.get("schema") != SCHEMA or config.get("artifact_type") != "real" or
            config["fingerprint"] != digest({key: value for key, value in config.items() if key != "fingerprint"}) or
            config["followup_code_sha256"] != code_hashes()):
        raise ValueError("Invalid/changed support-followup binding; preserve it and use a new root.")


def result_failures(row, task, config, pairs, baselines):
    try:
        from .phase3c_execution import technical_failures
        from .phase3c_primary import patch_outcomes
    except ImportError:
        from phase3c_execution import technical_failures
        from phase3c_primary import patch_outcomes
    converted = {**task, "kind": "identity" if task["kind"] == "identity" else "transplant_smoke"}
    converted.pop("task_id")
    converted["task_id"] = digest(converted)
    failures = technical_failures({**row, "spec": converted, "task_id": converted["task_id"],
                                   "is_primary_effect_estimate": False})
    if row.get("spec") != task or row.get("support_fingerprint") != config["fingerprint"]:
        failures.append("task_or_binding_changed")
    if (row.get("task_id") != task["task_id"] or
            row.get("is_primary_effect_estimate") is not (task["kind"] == "support_patch")):
        failures.append("invalid_followup_task_kind")
    pair, condition = pairs[task["pair_id"]], task["condition"]
    donor_condition = condition if task["kind"] == "identity" else CONDITIONS[1 if condition == CONDITIONS[0] else 0]
    support = config["support_audit"][task["pair_id"]]["supports"][task["support"]]
    side = "low_positions" if condition == CONDITIONS[0] else "temporal_positions"
    donor_side = "low_positions" if donor_condition == CONDITIONS[0] else "temporal_positions"
    baseline = baselines[pair[condition]["eval_id"]]
    if (row.get("support_audit") != support or row.get("recipient_positions") != support[side] or
            row.get("donor_positions") != support[donor_side] or row.get("donor_condition") != donor_condition or
            row.get("recipient_condition") != condition or row.get("baseline_decision") != baseline["decision"] or
            row.get("input_tensor_sha256") != baseline["input_tensor_sha256"]):
        failures.append("frozen_support_or_baseline_changed")
    if task["kind"] != "identity" and not failures:
        donor = baselines[pair[donor_condition]["eval_id"]]["decision"]
        expected = patch_outcomes(baseline["decision"]["margin"], row["decision"]["margin"], donor["margin"])
        if row.get("donor_baseline_decision") != donor or any(row.get(key) != value for key, value in expected.items()):
            failures.append("patch_decomposition_changed")
    return failures
