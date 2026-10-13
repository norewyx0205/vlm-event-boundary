"""Frozen whole-Event-2 expansion of the existing 50-rescue cohort."""

import copy
from pathlib import Path

try:
    from .phase3c_core import CONDITIONS, digest, file_hash, frozen_write, prepare_support_audit, read_json, read_jsonl
    from .phase3c_execution import execution_hashes
    from .prepare_phase3c import DEFAULT_SETTINGS, load_archive
except ImportError:
    from phase3c_core import CONDITIONS, digest, file_hash, frozen_write, prepare_support_audit, read_json, read_jsonl
    from phase3c_execution import execution_hashes
    from prepare_phase3c import DEFAULT_SETTINGS, load_archive

SCHEMA = "phase3c_whole_event2_scale50_v1"
LAYERS = (12, 16)


def code_hashes():
    return {name: file_hash(Path(__file__).parent / name) for name in
            ("phase3c_scale.py", "run_phase3c_scale.py", "plot_phase3c_comparison.py")}


def select_cohort(candidates):
    rows = sorted([r for r in candidates if r["stratum"] == "rescue"], key=lambda r: r["base_sample_id"])
    if len(rows) != 50 or len({r["base_sample_id"] for r in rows}) != 50:
        raise ValueError("Use exactly the 50 archived rescue bases, once each; no outcome-based replacements.")
    return rows


def tasks(plan, stage):
    if stage not in ("baseline", "identity", "patch"):
        raise ValueError("Unknown scale stage.")
    ids = plan["representatives"] if stage == "identity" else plan["case_ids"]
    result = []
    for key in ids:
        for condition in CONDITIONS:
            for layer in (None,) if stage == "baseline" else LAYERS:
                item = {"pair_id": key, "condition": condition, "kind": stage,
                        "support": "whole_event2", "location": "block_output", "layer": layer}
                item["task_id"] = digest([plan["fingerprint"], item])
                result.append(item)
    return result


def canonical(candidate, fingerprint):
    key = candidate["pair_id"].replace("phase3b_", "phase3c_", 1)
    return {condition: {**candidate["rows"][condition], "phase3c_pair_id": key,
        "phase3c_analysis_stratum": "rescue", "phase3c_plan_fingerprint": fingerprint,
        "phase3c_video_sha256": candidate["video_provenance"][condition]["sha256"]}
        for condition in CONDITIONS}


def source_artifacts(root, group, baseline_stage, table_name):
    root = Path(root).resolve()
    summary = read_json(root / "analysis/aggregate_summary.json")
    if summary.get("complete") is not True or summary.get("artifact_type") != "real":
        raise ValueError("Reuse only complete real source runs.")
    for name, expected in summary["output_sha256"].items():
        path = (root / "analysis" / name).resolve()
        if not path.is_relative_to(root / "analysis") or file_hash(path) != expected:
            raise ValueError("Source analysis changed.")
    evidence = {str(root / "analysis/aggregate_summary.json"): file_hash(root / "analysis/aggregate_summary.json")}
    baselines = {}
    for row in read_jsonl(root / baseline_stage / "rows.jsonl"):
        if row.get("passed") is not True or row["eval_id"] in baselines:
            raise ValueError("Source baseline failed or duplicated.")
        index = read_json(row["capture_index_path"])
        if file_hash(row["capture_index_path"]) != row["capture_index_sha256"]:
            raise ValueError("Source capture index changed.")
        baselines[row["eval_id"]] = {"row": row, "index": index, "cohort_group": group}
        evidence[row["capture_index_path"]] = row["capture_index_sha256"]
    table = read_json(root / "analysis" / table_name)
    effects = {}
    for row in table:
        if (row["analysis_stratum"] != "rescue" or row.get("anchor", row.get("support")) != "whole_event2"
                or row["layer"] not in LAYERS or row["location"] != "block_output"):
            continue
        key = (row["pair_id"], row["direction"], row["layer"])
        if key in effects:
            raise ValueError("Duplicate source whole-event effect.")
        effects[key] = {"effect": row, "cohort_group": group,
                       "source_table_path": str(root / "analysis" / table_name), "source_row_sha256": digest(row)}
    if len(effects) != 32:
        raise ValueError("Each source must contain exactly 8 rescue bases x 2 layers x 2 directions.")
    evidence[str(root / baseline_stage / "rows.jsonl")] = file_hash(root / baseline_stage / "rows.jsonl")
    for name, expected in summary["output_sha256"].items():
        evidence[str(root / "analysis" / name)] = expected
    for stage in ("preflight", "patch"):
        summary_path = root / stage / "summary.json"
        value = read_json(summary_path)
        if (value.get("complete") is not True or value.get("failures") or value.get("missing_task_ids")):
            raise ValueError("Source technical/patch stage is incomplete.")
        for name, field in (("rows.jsonl", "rows_sha256"), ("task_manifest.jsonl", "manifest_sha256")):
            path = root / stage / name
            if file_hash(path) != value[field]:
                raise ValueError("Source task ledger bytes changed.")
            evidence[str(path)] = value[field]
        evidence[str(summary_path)] = file_hash(summary_path)
    return baselines, effects, evidence


def prepare(args):
    root = Path(args.output_root).resolve()
    for name in ("source_run", "pilot_plan", "pilot_run", "replication_run"):
        src = Path(getattr(args, name)).resolve()
        if root == src or root in src.parents or src in root.parents:
            raise ValueError("Use an isolated expansion root; preserve all previous evidence.")
    config, source_hashes, candidates, _ = load_archive(args.source_run)
    cohort = select_cohort(candidates)
    settings = copy.deepcopy(DEFAULT_SETTINGS)
    for name in ("seed", "roi_padding", "video_fps", "video_num_frames", "video_max_pixels"):
        if settings[name] != config[name]:
            raise ValueError("Archived processor settings changed.")
    # Cached processor audits are read-only. Reconstruct correspondence from them.
    records, audit_files = {}, {}
    for path in (Path(args.pilot_plan) / "processor_audit/mapping_audit.jsonl",
                 Path(args.replication_run) / "preparation/processor_audit/rows.jsonl"):
        audit_files[str(path.resolve())] = file_hash(path)
        for item in read_jsonl(path):
            if item.get("eligible") is True:
                records[item["pair_id"]] = item
    mappings = {}
    for candidate in cohort:
        old = records.get(candidate["pair_id"])
        if old is None:
            raise ValueError("An archived rescue has no eligible processor audit; do not drop it.")
        for record in old["processor_records"].values():
            if file_hash(record["video_path"]) != record["video_sha256"]:
                raise ValueError("Audited video bytes changed.")
        rebuilt = prepare_support_audit(candidate, candidate["archived_mapping"], old["processor_records"],
                                        settings["max_event_progress_error"], settings["seed"])
        if rebuilt != old["support_audit"]:
            raise ValueError("Cached mapping differs from reconstructed observed-donor correspondence.")
        whole = rebuilt["supports"]["whole_event2"]
        if (whole["one_to_one"] is not True or any(value["recipient_coverage"] != 1
                for value in whole["directions"].values())):
            raise ValueError("All 50 cases require complete one-to-one whole-event support.")
        mappings[candidate["pair_id"].replace("phase3b_", "phase3c_", 1)] = {
            **old, "support_audit": {**rebuilt, "supports": {"whole_event2": whole}}}
    baselines, effects, dependencies = {}, {}, {}
    for path, group, stage, table in ((args.pilot_run, "original_pilot", "baseline", "case_patch_effects.json"),
                                    (args.replication_run, "independent_replication", "capture", "anchor_effects.json")):
        source_baselines, source_effects, evidence = source_artifacts(path, group, stage, table)
        if set(effects) & set(source_effects) or set(baselines) & set(source_baselines):
            raise ValueError("Previous cohorts overlap.")
        baselines.update(source_baselines)
        effects.update(source_effects)
        dependencies.update(evidence)
    keys = [row["pair_id"].replace("phase3b_", "phase3c_", 1) for row in cohort]
    if not {key[0] for key in effects} <= set(keys):
        raise ValueError("Reused effects contain a foreign rescue.")
    reused = {key[0] for key in effects}
    if len(reused) != 16:
        raise ValueError("Expected exactly 16 previously tested rescue bases.")
    representatives = [next(keys[i] for i, row in enumerate(cohort) if row["first_object_id"] == mover)
                       for mover in (1, 2)]
    reference = read_json(Path(args.replication_run) / "capture_runtime.json")["runtime"]
    runtime_path = Path(args.replication_run) / "capture_runtime.json"
    dependencies[str(runtime_path.resolve())] = file_hash(runtime_path)
    source_refs = {row["eval_id"]: baselines[row["eval_id"]] for candidate in cohort
                   for row in candidate["rows"].values() if row["eval_id"] in baselines}
    if len(source_refs) != 32:
        raise ValueError("The 16 prior rescue bases must supply both condition captures.")
    groups = {key: next((value["cohort_group"] for (pair, _, _), value in effects.items() if pair == key),
                       "newly_tested") for key in keys}
    plan = {"schema": SCHEMA, "artifact_type": "real", "case_ids": keys, "base_ids": [r["base_sample_id"] for r in cohort],
        "cohort_groups": groups, "representatives": representatives, "settings": settings,
        "layers": list(LAYERS), "primary_layer": 16, "support": "whole_event2", "knockout_enabled": False,
        "capture_scope": "Only whole-event positions at L12/L16 for new cases; previous all-layer stores retained.",
        "capture_reuse_count": len(source_refs), "new_capture_count": 100 - len(source_refs),
        "source_runtime": reference, "source_run_root": str(Path(args.source_run).resolve()),
        "source_evidence_sha256": source_hashes, "dependency_sha256": {**audit_files, **dependencies},
        "measurement_code_sha256": execution_hashes(), "scale_code_sha256": code_hashes(),
        "protocol_sha256": file_hash(Path(__file__).resolve().parents[1] / "docs/phase3c_scale50.md"),
        "source_path_map": config.get("path_map", {}), "candidate_sha256": digest(cohort),
        "mappings_sha256": digest(mappings), "source_capture_refs_sha256": digest(source_refs),
        "prior_effects_sha256": digest([{"key": list(key), **value} for key, value in sorted(effects.items())]),
        "primary_outcome": "L16 temporal-to-low actual wrong-to-correct answer with strict margin crossing",
        "scope_limit": "Existing selective rescue cohort; report prior 8 + replication 8 + new 34 separately."}
    plan["fingerprint"] = digest(plan)
    frozen_write(root / "plan.json", plan)
    frozen_write(root / "candidates.json", cohort)
    frozen_write(root / "pairs.json", {key: canonical(candidate, plan["fingerprint"]) for key, candidate in zip(keys, cohort)})
    frozen_write(root / "mappings.json", mappings)
    frozen_write(root / "source_capture_refs.json", source_refs)
    frozen_write(root / "prior_effects.json", [{"key": list(key), **value} for key, value in sorted(effects.items())])
    return load(root)


def load(root):
    root = Path(root)
    plan = read_json(root / "plan.json")
    if (plan.get("schema") != SCHEMA or plan.get("artifact_type") != "real" or
            plan["fingerprint"] != digest({k: v for k, v in plan.items() if k != "fingerprint"}) or
            plan["scale_code_sha256"] != code_hashes() or plan["measurement_code_sha256"] != execution_hashes() or
            plan["protocol_sha256"] != file_hash(Path(__file__).resolve().parents[1] / "docs/phase3c_scale50.md")):
        raise ValueError("Scale configuration/code binding changed; preserve this root.")
    for name, expected in plan["source_evidence_sha256"].items():
        if file_hash(Path(plan["source_run_root"]) / name) != expected:
            raise ValueError("Phase 3B source evidence changed.")
    for path, expected in plan["dependency_sha256"].items():
        if file_hash(path) != expected:
            raise ValueError("Read-only source dependency changed.")
    pairs, mappings, refs = (read_json(root / name) for name in ("pairs.json", "mappings.json", "source_capture_refs.json"))
    candidates = read_json(root / "candidates.json")
    effects = read_json(root / "prior_effects.json")
    if (digest(candidates) != plan["candidate_sha256"] or digest(mappings) != plan["mappings_sha256"] or
            digest(refs) != plan["source_capture_refs_sha256"] or digest(effects) != plan["prior_effects_sha256"] or
            set(pairs) != set(plan["case_ids"]) or len(pairs) != 50):
        raise ValueError("Scale manifest or source references changed.")
    if pairs != {key: canonical(candidate, plan["fingerprint"]) for key, candidate in zip(plan["case_ids"], candidates)}:
        raise ValueError("Pair annotations differ from the frozen cohort.")
    return plan, pairs, mappings, refs, {tuple(value["key"]): value for value in effects}


def capture_estimate(plan, mappings):
    reused = plan["capture_reuse_count"]
    maximum = max(len(mapping["support_audit"]["supports"]["whole_event2"]["low_positions"])
                  for mapping in mappings.values())
    capture = (100 - reused) * (len(LAYERS) * maximum * 4096 * 2 + 2 * 1024**2)
    return {"new_capture_upper_bytes": capture, "safety_reserve_bytes": 12 * 1024**3,
            "required_free_bytes": int(capture * 2.3) + 12 * 1024**3}


def source_replay_matches(result, expected):
    decision = result["decision"]
    return all(decision[key] == expected[f"patched_{key}"] for key in
               ("margin", "correct_logit", "incorrect_logit", "prediction"))
