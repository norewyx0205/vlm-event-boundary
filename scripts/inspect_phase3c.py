"""Read-only Phase 3C readiness and progress; no processor, weights or API calls."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

try:
    from .audit_phase3c_mappings import validate_plan
    from .phase3c_core import digest, read_json
    from .phase3c_execution import execution_hashes, load_selection, require_stage, technical_tasks
    from .phase3c_primary import PRIMARY_STAGES, primary_tasks
    from .run_phase3c_vm import verify_stage
except ImportError:
    from audit_phase3c_mappings import validate_plan
    from phase3c_core import digest, read_json
    from phase3c_execution import execution_hashes, load_selection, require_stage, technical_tasks
    from phase3c_primary import PRIMARY_STAGES, primary_tasks
    from run_phase3c_vm import verify_stage


PREPARATION_FILES = ("plan_config.json", "archive_audit.jsonl", "candidate_manifest.jsonl", "candidate_summary.json")
SELECTION_FILES = ("selection/case_manifest.jsonl", "selection/selected_mappings.jsonl",
                   "selection/frozen_config.json", "selection/case_selection_summary.json")
STAGES = ("baseline", "preflight", *PRIMARY_STAGES, "analyze")
EXPECTED = {"baseline": 24, "preflight": 152, "patch": 696, "routing": 24, "knockout": 2592}


def optional_json(path, errors):
    if not path.exists():
        return None
    try:
        value = read_json(path)
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object.")
        return value
    except (OSError, ValueError) as exc:
        errors.append({"path": str(path), "error_type": type(exc).__name__, "message": str(exc)})
        return None


def recorded_failure(summary):
    if not summary:
        return False
    missing_ids, failures = summary.get("missing_task_ids", []), summary.get("failures", [])
    if not isinstance(missing_ids, list) or not isinstance(failures, list):
        return True
    missing = {f"{task_id}:missing" for task_id in missing_ids}
    return any(not isinstance(reason, str) or reason not in missing for reason in failures)


def gate(root, files, summary=None, claimed=False):
    missing = [str(root / name) for name in files if not (root / name).is_file()]
    if missing:
        state = "missing"
    elif claimed:
        state = "recorded_complete"
    else:
        state = "recorded_failure" if recorded_failure(summary) else "pending"
    return {"status": state, "verified": False, "missing_paths": missing, "recorded_summary": summary}


def inspect(plan_dir, execution_dir=None, verify=False):
    plan = Path(plan_dir).expanduser().resolve()
    root = Path(execution_dir or plan / "execution_v1").expanduser().resolve()
    errors = []
    candidate = optional_json(plan / "candidate_summary.json", errors)
    mapping = optional_json(plan / "processor_audit/summary.json", errors)
    mapping_progress = optional_json(plan / "processor_audit/progress_status.json", errors)
    selection = optional_json(plan / "selection/case_selection_summary.json", errors)
    stages = {
        "audit": gate(plan, PREPARATION_FILES, candidate, candidate is not None),
        "mapping": gate(plan, ("processor_audit/config.json", "processor_audit/mapping_audit.jsonl"), mapping,
                        bool(mapping_progress and mapping_progress.get("cohort_can_freeze") is True)),
        "freeze": gate(plan, SELECTION_FILES, selection, bool(selection and selection.get("case_ids_frozen") is True)),
    }
    stages["mapping"]["reported_progress"] = mapping_progress
    if not stages["freeze"]["missing_paths"] and selection and selection.get("case_ids_frozen") is True:
        stages["mapping"]["status"] = "recorded_complete"
    for stage in STAGES:
        directory = "analysis" if stage == "analyze" else stage
        name = "aggregate_summary.json" if stage == "analyze" else "summary.json"
        summary = optional_json(root / directory / name, errors)
        files = (f"{directory}/{name}", f"{directory}/analysis_config.json", f"{directory}/analysis_status.json") if stage == "analyze" else (
            "execution_config.json", f"{stage}/summary.json", f"{stage}/task_manifest.jsonl", f"{stage}/rows.jsonl")
        claimed = bool(summary and summary.get("complete" if stage == "analyze" else "passed") is True)
        stages[stage] = gate(root, files, summary, claimed)
        if stage != "analyze":
            stages[stage]["expected_tasks"] = EXPECTED[stage]
            stages[stage]["reported_progress"] = optional_json(root / stage / "progress_status.json", errors)
    report = {"schema": "phase3c_readiness_v1", "read_only": True, "verification_requested": verify,
        "plan_dir": str(plan), "execution_dir": str(root), "stages": stages, "diagnostics": errors,
        "primary_grid_and_analysis_verified": False, "full_launch_prerequisites_verified": False,
        "inspector_started_processor_or_gpu": False, "inspector_took_network_or_lifecycle_action": False,
        "vm_attempt": optional_json(root / "vm_last_status.json", errors),
        "lifecycle": optional_json(root.parent / (root.name + "_lifecycle") / "job_status.json", errors),
        "warning": "Recorded summaries/progress are not independently verified gates. No command is started by this inspector."}
    if verify:
        verify_gates(plan, root, stages, errors)
        report["full_launch_prerequisites_verified"] = all(stages[name]["verified"] for name in ("audit", "freeze", "baseline", "preflight"))
        report["primary_grid_and_analysis_verified"] = report["full_launch_prerequisites_verified"] and all(stages[name]["verified"] for name in STAGES)
        report["warning"] = "Checksum verification is a read-only snapshot, not authorization to launch GPU work or Pause. Runner guards still apply."
    report["next_action"] = next_action(stages, verify)
    return report


def verify_gates(plan, root, stages, errors):
    def check(name, action):
        if stages[name]["missing_paths"]:
            return None
        if name in STAGES and (stages[name]["recorded_summary"] or {}).get("complete" if name == "analyze" else "passed") is not True:
            stages[name]["status"] = "recorded_failure" if recorded_failure(stages[name]["recorded_summary"]) else "incomplete"
            return None
        try:
            result = action()
            stages[name].update(status="verified_complete", verified=True)
            return result
        except Exception as exc:
            stages[name].update(status="invalid", verified=False)
            errors.append({"stage": name, "error_type": type(exc).__name__, "message": str(exc)})
            return None
    if check("audit", lambda: validate_plan(plan)) is None:
        return
    # load_selection independently reconstructs existing freeze files. Never call
    # freeze_selection when any selection file is absent: that would write files.
    selected = check("freeze", lambda: load_selection(plan))
    if selected is None:
        return
    frozen, pairs, mappings = selected
    stages["mapping"].update(status="verified_complete", verified=True)
    try:
        config = read_json(root / "execution_config.json")
        if (config.get("artifact_type") != "real" or config["selection_fingerprint"] != frozen["selection_fingerprint"] or
                config["execution_code_sha256"] != execution_hashes() or
                config["execution_fingerprint"] != digest({k: v for k, v in config.items() if k != "execution_fingerprint"})):
            raise ValueError("Execution provenance is invalid or stale.")
    except FileNotFoundError:
        return
    except Exception as exc:
        stages["baseline"].update(status="invalid", verified=False)
        errors.append({"stage": "execution_config", "error_type": type(exc).__name__, "message": str(exc)})
        return
    baselines = check("baseline", lambda: require_stage(root, config, pairs, "baseline",
        {row["eval_id"] for pair in pairs.values() for row in pair.values()}))
    if baselines is None:
        return
    tasks = technical_tasks(frozen, mappings)
    if check("preflight", lambda: require_stage(root, config, pairs, "preflight",
             {task["task_id"] for task in tasks}, expected_tasks=tasks)) is None:
        return
    for stage in PRIMARY_STAGES:
        tasks = primary_tasks(frozen, mappings, stage)
        if check(stage, lambda: require_stage(root, config, pairs, stage,
                 {task["task_id"] for task in tasks}, mappings, baselines, tasks)) is None:
            return
    if not stages["analyze"]["missing_paths"]:
        def check_analysis():
            analysis = read_json(root / "analysis/analysis_config.json")
            args = SimpleNamespace(plan_dir=str(plan), output_root=str(root),
                bootstrap_repeats=analysis["case_bootstrap_repeats"], no_plots=not analysis["plots"])
            verify_stage(args, "analyze")
            return True
        check("analyze", check_analysis)


def next_action(stages, verify):
    for name, item in stages.items():
        if item["status"] in ("invalid", "recorded_failure"):
            return "diagnose_and_preserve_artifacts"
        if item["status"] not in ("recorded_complete", "verified_complete"):
            return name
    return "review_report_and_verify_local_backup" if verify else "verify_readiness"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan_dir", required=True)
    parser.add_argument("--execution_dir")
    parser.add_argument("--verify", action="store_true", help="Independently check frozen inputs, gates and capture checksums. May read large files.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = inspect(args.plan_dir, args.execution_dir, args.verify)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for stage, item in report["stages"].items():
            progress = item.get("reported_progress") or {}
            print(f"{stage}: {item['status']}; passed={progress.get('passed_tasks', '?')}/{item.get('expected_tasks', '?')}; "
                  f"elapsed_sec={progress.get('elapsed_sec', '?')}; checkpoint={progress.get('checkpoint', 'see stage directory')}")
            for path in item["missing_paths"]:
                print(f"  missing: {path}")
        for error in report["diagnostics"]:
            print(f"  diagnostic: {error}")
        print(f"Next action: {report['next_action']}\n{report['warning']}")
    if args.verify and report["diagnostics"]:
        parser.exit(1)


if __name__ == "__main__":
    main()
