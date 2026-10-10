"""Isolated support-decomposition follow-up; never recapture or modify the pilot."""

import argparse
import gc
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    from .analyze_phase3c import aggregate, patch_table, write_csv
    from .backup_phase3b import create_backup, verify_backup
    from .phase3c_core import CONDITIONS, atomic_write, digest, file_hash, frozen_write, read_json
    from .phase3c_primary import patch_outcomes
    from .phase3c_support import SCHEMA, SUPPORTS, prepare, result_failures, task_grid, validate_config
    from .run_phase3b_vm import run_lock, worker_environment
    from .run_phase3c_preflight import Engine
    from .run_phase3c_unattended import (EMAIL_CONFIG, SURF_CONFIG, EmailNotifier, SurfClient,
        assert_quiescent, check_private_location, lifecycle_lock, load_email_config, load_private_config)
    from .run_phase3c_vm import run_child
except ImportError:
    from analyze_phase3c import aggregate, patch_table, write_csv
    from backup_phase3b import create_backup, verify_backup
    from phase3c_core import CONDITIONS, atomic_write, digest, file_hash, frozen_write, read_json
    from phase3c_primary import patch_outcomes
    from phase3c_support import SCHEMA, SUPPORTS, prepare, result_failures, task_grid, validate_config
    from run_phase3b_vm import run_lock, worker_environment
    from run_phase3c_preflight import Engine
    from run_phase3c_unattended import (EMAIL_CONFIG, SURF_CONFIG, EmailNotifier, SurfClient,
        assert_quiescent, check_private_location, lifecycle_lock, load_email_config, load_private_config)
    from run_phase3c_vm import run_child


def context(args):
    return prepare(args.plan_dir, args.source_run, args.output_root, args.source_backup,
                   freeze=args.action == "prepare")


def saved_rows(root, stage, config, pairs, baselines, complete=False):
    tasks = task_grid(config, stage)
    expected = {task["task_id"]: task for task in tasks}
    rows = {}
    for path in sorted((Path(root) / stage / "task_checkpoints").glob("*.json")):
        row = read_json(path)
        key = row.get("task_id")
        if key not in expected or path.stem != key or key in rows:
            raise ValueError("Foreign/duplicate support checkpoint; preserve and diagnose it.")
        if row.get("passed") is True and result_failures(row, expected[key], config, pairs, baselines):
            raise ValueError("A claimed-passed support checkpoint failed independent validation.")
        rows[key] = row
    if complete and (set(rows) != set(expected) or any(not row.get("passed") for row in rows.values())):
        raise ValueError(f"Incomplete/failed support {stage} gate.")
    return tasks, rows


def run_stage(args, prepared_context):
    config, frozen, pairs, mappings, baselines = prepared_context
    stage, root = args.stage, Path(args.output_root)
    if stage == "patch":
        saved_rows(root, "preflight", config, pairs, baselines, complete=True)
    tasks, saved = saved_rows(root, stage, config, pairs, baselines)
    output = root / stage
    frozen_write(output / "task_manifest.jsonl", tasks, jsonl=True)

    def summarize():
        failures = [{"task_id": key, "reasons": result_failures(row, task, config, pairs, baselines)}
                    for task in tasks if (key := task["task_id"]) in saved
                    for row in [saved[key]] if not row.get("passed") or
                    result_failures(row, task, config, pairs, baselines)]
        missing = [task["task_id"] for task in tasks if task["task_id"] not in saved]
        atomic_write(output / "rows.jsonl", [saved[task["task_id"]] for task in tasks if task["task_id"] in saved], jsonl=True)
        summary = {"schema": SCHEMA, "support_fingerprint": config["fingerprint"], "stage": stage,
            "complete": not missing and not failures, "expected": len(tasks), "completed": len(saved),
            "failures": failures, "missing_task_ids": missing,
            "rows_sha256": file_hash(output / "rows.jsonl"),
            "manifest_sha256": file_hash(output / "task_manifest.jsonl")}
        atomic_write(output / "summary.json", summary)
        return summary

    initial = summarize()
    if initial["complete"]:
        print(f"Support {stage}: reused all {len(tasks)} verified tasks; no model load.", flush=True)
        return initial
    if initial["failures"] and not args.retry_failed:
        raise ValueError("Failed support task preserved; diagnose before explicit --retry_failed.")
    env = worker_environment(args.gpus, args.storage_root)
    os.environ.update({key: value for key, value in env.items() if key in
                       ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "HF_HOME", "TORCH_HOME")})
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        os.environ.pop(name, None)
    engine = Engine(frozen["settings"], config["source_runtime"]["execution_mode"],
                    config["source_runtime"]["gpu_weight_budget_gib"])
    if engine.runtime != config["source_runtime"]:
        raise ValueError("Follow-up runtime/hardware/placement changed relative to source captures.")
    engine.project_root = args.project_root
    engine.path_map = config["source_path_map"]
    current, live, captures, computed, started = None, None, {}, 0, time.perf_counter()
    for task in tasks:
        key = task["task_id"]
        if key in saved and saved[key].get("passed"):
            continue
        pair_id, condition = task["pair_id"], task["condition"]
        row = pairs[pair_id][condition]
        result = {"task_id": key, "spec": task, "support_fingerprint": config["fingerprint"], "passed": False}
        begun = time.perf_counter()
        print(f"Support {stage}: {pair_id} {condition} {task['support']} L{task['layer']}", flush=True)
        try:
            if current != (pair_id, condition):
                live = None
                gc.collect()
                engine.torch.cuda.empty_cache()
                live = engine.prepare(row, mappings[pair_id])
                if live["input_tensor_sha256"] != baselines[row["eval_id"]]["input_tensor_sha256"]:
                    raise ValueError("Live input tensor bytes differ from source baseline.")
                captures = {}
                for side in CONDITIONS:
                    index = read_json(baselines[pairs[pair_id][side]["eval_id"]]["capture_index_path"])
                    captures[side] = engine.torch.load(index["vectors_path"], map_location="cpu", weights_only=True)
                current = (pair_id, condition)
            mapping = {**mappings[pair_id], "support_audit": {**mappings[pair_id]["support_audit"],
                "supports": {**mappings[pair_id]["support_audit"]["supports"],
                             **config["support_audit"][pair_id]["supports"]}}}
            converted = {**task, "kind": "identity" if stage == "preflight" else "transplant_smoke"}
            result.update(engine.technical(live, mapping, converted, captures))
            result.update(spec=task, input_tensor_sha256=live["input_tensor_sha256"],
                          is_primary_effect_estimate=stage == "patch")
            if stage == "patch":
                donor = baselines[pairs[pair_id][result["donor_condition"]]["eval_id"]]["decision"]
                result["donor_baseline_decision"] = donor
                result.update(patch_outcomes(result["baseline_decision"]["margin"], result["decision"]["margin"], donor["margin"]))
            failures = result_failures(result, task, config, pairs, baselines)
            if failures:
                result.update(passed=False, validation_failures=failures)
        except Exception as exc:
            result.update(passed=False, failure_type=type(exc).__name__, failure_message=str(exc),
                          traceback=traceback.format_exc())
            print(result["traceback"], flush=True)
        result["elapsed_sec"] = time.perf_counter() - begun
        if not result["passed"]:
            errors = read_json(output / "errors.json") if (output / "errors.json").exists() else []
            atomic_write(output / "errors.json", [*errors, result])
        saved[key] = result
        atomic_write(output / "task_checkpoints" / f"{key}.json", result)
        computed += 1
        passed = sum(item.get("passed") is True for item in saved.values())
        elapsed = time.perf_counter() - started
        remaining = len(tasks) - passed
        atomic_write(output / "progress_status.json", {"stage": stage, "expected": len(tasks), "passed": passed,
            "failed": len(saved) - passed, "elapsed_sec": elapsed, "newly_computed": computed,
            "eta_sec": elapsed / computed * remaining,
            "checkpoint": str(output / "task_checkpoints")})
        print(f"Support {stage}: {passed}/{len(tasks)} passed; elapsed={elapsed/60:.1f} min; "
              f"ETA={elapsed/computed*remaining/60:.1f} min; checkpoint={output/'task_checkpoints'}", flush=True)
        if not result["passed"]:
            break
    return summarize()


def analyze(args, prepared_context):
    config, _, pairs, _, baselines = prepared_context
    root = Path(args.output_root)
    saved_rows(root, "preflight", config, pairs, baselines, complete=True)
    _, rows = saved_rows(root, "patch", config, pairs, baselines, complete=True)
    new = patch_table(rows, pairs)
    references = [row for row in read_json(Path(args.source_run) / "analysis/case_patch_effects.json")
        if row["support"] in ("whole_event2", "both_targets_event2") and row["layer"] in config["layers"]
        and row["location"] == "block_output"]
    if len(new) != 288 or len(references) != 288:
        raise ValueError("Incomplete new/reference support grid; analysis blocked.")
    table = [*references, *new]
    lookup = {(row["pair_id"], row["direction"], row["layer"], row["support"]): row for row in table}
    contrasts = []
    for row in new:
        if row["support"] != SUPPORTS[0]:
            continue
        prefix = (row["pair_id"], row["direction"], row["layer"])
        full = lookup[(*prefix, "whole_event2")]
        target = lookup[(*prefix, "both_targets_event2")]
        matched = lookup[(*prefix, SUPPORTS[1])]
        if target["replaced_token_count"] != matched["replaced_token_count"]:
            raise ValueError("Actual target/context token budgets differ.")
        if full["replaced_token_count"] != target["replaced_token_count"] + row["replaced_token_count"]:
            raise ValueError("Actual target/complement supports do not partition full support.")
        contrasts.append({**{key: row[key] for key in ("pair_id", "base_sample_id", "analysis_stratum", "first_object_id",
            "direction", "layer")}, "target_minus_budget_context": target["margin_delta"] - matched["margin_delta"],
            "whole_minus_target": full["margin_delta"] - target["margin_delta"],
            "complement_minus_target": row["margin_delta"] - target["margin_delta"],
            "non_additivity": full["margin_delta"] - target["margin_delta"] - row["margin_delta"]})
    summaries = aggregate(table, ("analysis_stratum", "direction", "support", "layer", "location"),
        ("margin_delta", "source_aligned_margin_delta", "recovery", "strict_incorrect_to_correct",
         "strict_sign_crossing", "replaced_token_count"))
    contrasts_summary = aggregate(contrasts, ("analysis_stratum", "direction", "layer"),
        ("target_minus_budget_context", "whole_minus_target", "complement_minus_target", "non_additivity"))
    output = root / "analysis"
    for name, items in (("case_support_effects", table), ("support_summary", summaries),
                        ("case_support_contrasts", contrasts), ("contrast_summary", contrasts_summary)):
        atomic_write(output / f"{name}.json", items)
        write_csv(output / f"{name}.csv", items)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"whole_event2": "Whole Event 2 (reused)", "both_targets_event2": "Target union (reused)",
              SUPPORTS[0]: "Non-target complement", SUPPORTS[1]: "Non-target, target-matched budget"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for i, stratum in enumerate(("rescue", "stable")):
        for j, direction in enumerate(("temporal_to_low", "low_to_temporal")):
            ax = axes[i, j]
            for support, label in labels.items():
                items = sorted([row for row in summaries if row["analysis_stratum"] == stratum and
                    row["direction"] == direction and row["support"] == support], key=lambda row: row["layer"])
                x = [row["layer"] for row in items]
                line, = ax.plot(x, [row["margin_delta_mean"] for row in items], marker="o", label=label)
                ax.fill_between(x, [row["margin_delta_ci_low"] for row in items],
                    [row["margin_delta_ci_high"] for row in items], color=line.get_color(), alpha=0.1)
            ax.axhline(0, color="black", linewidth=0.7)
            ax.set(title=f"{stratum} | {direction.replace('_', ' ')}", xlabel="Decoder layer (0-based)",
                   ylabel="Correct-option margin change", xticks=config["layers"])
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Exploratory support decomposition | Same 12 bases | Case-bootstrap pilot intervals")
    fig.savefig(output / "support_decomposition.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    report = """# Exploratory Phase 3C Support Follow-Up

The same 8 rescue and 4 stable bases are reused, not new independent replication.
Whole-Event-2 and both-target effects are read-only references from the completed pilot.
Two new bidirectional observed-donor interventions use the original six-layer grid.
The non-target complement and target union exactly partition the matched Event-2 grid.
The smaller context control matches target-union token counts within each paired temporal bin.
Non-target positions may contain distractors and may already encode target information.
Equal token counts do not guarantee equal attention, content, spatial layout or causal potency.
The full-minus-target-minus-complement contrast describes non-additivity, not mediation.
L0 block-output interventions retain the recipient DeepStack addition; L4+ do not.
Intervals are exploratory case-bootstrap diagnostics, not population/multiplicity-adjusted inference.
Retain the separate, locally verified source Phase 3C activation/video backup.
"""
    (output / "report.md").write_text(report, encoding="ascii")
    summary = {"schema": SCHEMA, "artifact_type": "real", "support_fingerprint": config["fingerprint"],
        "complete": True, "exploratory_after_pilot": True, "case_count": 12, "new_patch_rows": len(new),
        "reused_reference_rows": len(references), "new_activation_capture": False,
        "output_sha256": {path.name: file_hash(path) for path in output.iterdir() if path.is_file()}}
    summary["output_sha256"].pop("aggregate_summary.json", None)
    atomic_write(output / "aggregate_summary.json", summary)
    return summary


def verify_analysis(args, config):
    output = Path(args.output_root) / "analysis"
    summary = read_json(output / "aggregate_summary.json")
    if not summary.get("complete") or summary.get("support_fingerprint") != config["fingerprint"]:
        raise ValueError("Support analysis is incomplete or incompatible.")
    for name, expected in summary["output_sha256"].items():
        path = output / name
        if not path.resolve().is_relative_to(output) or file_hash(path) != expected:
            raise ValueError("Support analysis bytes changed.")
    return summary


def command(args, action, stage="full"):
    result = [sys.executable, "-u", str(Path(__file__).resolve()), "--action", action, "--stage", stage,
        "--plan_dir", args.plan_dir, "--source_run", args.source_run, "--source_backup", args.source_backup,
        "--output_root", args.output_root, "--project_root", args.project_root, "--storage_root", args.storage_root,
        "--gpus", args.gpus]
    if args.retry_failed:
        result.append("--retry_failed")
    if getattr(args, "attempt_id", None):
        result += ["--attempt_id", args.attempt_id]
    if action == "job":
        result += ["--backup_dir", args.backup_dir]
        if args.email_notify:
            result += ["--email_notify", "--email_config", args.email_config]
        if args.pause_after:
            result += ["--pause_after", "--surf_config", args.surf_config, "--confirm_exclusive_workspace"]
    return result


def private_services(args):
    forbidden = (args.project_root, args.plan_dir, args.source_run, args.output_root, args.backup_dir,
                 str(args.output_root) + "_lifecycle")
    notifier = client = None
    if args.email_notify:
        check_private_location(args.email_config, forbidden)
        notifier = EmailNotifier(load_email_config(args.email_config, forbidden))
        notifier.check()
    if args.pause_after:
        if not args.confirm_exclusive_workspace:
            raise ValueError("Whole-VM Pause requires --confirm_exclusive_workspace.")
        check_private_location(args.surf_config, forbidden)
        client = SurfClient(load_private_config(args.surf_config, forbidden))
        client.check(require_pause=True)
    return notifier, client


def launch(args):
    config, _, pairs, _, baselines = context(args)
    saved_rows(args.output_root, "preflight", config, pairs, baselines, complete=True)
    if not shutil.which("tmux"):
        raise ValueError("Install tmux first.")
    if subprocess.run(["tmux", "has-session", "-t", "=phase3c_support"], capture_output=True).returncode == 0:
        raise ValueError("The support session already exists; do not launch a duplicate.")
    directory = Path(str(args.output_root) + "_lifecycle")
    with lifecycle_lock(directory), run_lock(args.output_root):
        private_services(args)
        shell = f"exec {shlex.join(command(args, 'job'))} >> {shlex.quote(str(directory/'job.log'))} 2>&1"
        subprocess.run(["tmux", "new-session", "-d", "-s", "phase3c_support", "-c", args.project_root, shell], check=True)
    print(f"Detached support job started; status={directory/'job_status.json'}", flush=True)


def job(args):
    directory = Path(str(args.output_root) + "_lifecycle")
    started = time.perf_counter()
    status = {"schema": SCHEMA, "started_at": datetime.now(timezone.utc).isoformat(),
        "state": "prechecking", "output_root": args.output_root, "backup_verified": False}
    notifier = client = None

    def save(state, **fields):
        status.update(state=state, elapsed_sec=time.perf_counter() - started, **fields)
        atomic_write(directory / "job_status.json", status)
        print(f"Support lifecycle: {state}; elapsed={status['elapsed_sec']/60:.1f} min", flush=True)

    def notify():
        if notifier is None:
            return
        label = "SUCCESS" if status.get("experiment_success") else "FAILED" if "experiment_success" in status else "NOT STARTED"
        body = (f"Experiment: {label}\nExploratory Phase 3C support decomposition\n"
            f"State: {status['state']}\nRun: {args.output_root}\nElapsed: {status['elapsed_sec']/3600:.2f} hours\n"
            f"Verified follow-up backup: {status['backup_verified']}\nBackup: {status.get('backup_dir','not available')}\n"
            f"Required source activation/video backup: {args.source_backup}\n"
            "The follow-up reuses source captures; its package does not duplicate them. Retain both backups.\n"
            "Local backup and Pause/billing stop are not confirmed by this email. Check the portal.\n"
            "The Pause request follows this email only after backup verification and idle checks.\n"
            "Technical status only; no credentials or tensors attached.")
        try:
            status["email"] = notifier.send(f"[Phase 3C Support] {label}", body)
        except Exception as exc:
            status["email"] = {"error_type": type(exc).__name__}
        atomic_write(directory / "job_status.json", status)

    with lifecycle_lock(directory):
        save("prechecking")
        try:
            config, _, pairs, _, baselines = context(args)
            saved_rows(args.output_root, "preflight", config, pairs, baselines, complete=True)
            notifier, client = private_services(args)
            if args.pause_after:
                assert_quiescent()
            save("experiment_running")
            args.attempt_id = uuid.uuid4().hex
            code = run_child(args, command(args, "run"), "support_full")
            success = False
            if code == 0:
                try:
                    with run_lock(args.output_root):
                        attempt = read_json(Path(args.output_root) / "last_attempt.json")
                        if (attempt.get("attempt_id") != args.attempt_id or not attempt.get("complete") or
                                attempt.get("support_fingerprint") != config["fingerprint"]):
                            raise ValueError("No new successful child attempt; old analysis cannot establish success.")
                        saved_rows(args.output_root, "patch", config, pairs, baselines, complete=True)
                        verify_analysis(args, config)
                        success = True
                except Exception as exc:
                    status["verification_error_type"] = type(exc).__name__
            save("experiment_finished", experiment_return_code=code, experiment_success=success)
            with run_lock(args.output_root):
                atomic_write(Path(args.output_root) / "unattended_status.json", status)
                shutil.copyfile(directory / "job.log", Path(args.output_root) / "unattended_job.log")
            save("backup_running")
            bundle = create_backup(args.output_root, args.backup_dir, bundle_prefix="phase3c_backup",
                metadata={"schema": "phase3c_support_backup_v1", "support_fingerprint": config["fingerprint"],
                    "new_activation_file_count": 0, "source_activation_backup_included": False,
                    "source_backup_dir": args.source_backup,
                    "source_backup_manifest_sha256": config["source_backup_manifest_sha256"]})
            manifest = verify_backup(bundle)
            if (not manifest["complete"] or manifest["reports_only"] or
                    manifest["support_fingerprint"] != config["fingerprint"]):
                raise ValueError("Follow-up backup verification failed.")
            save("backup_verified", backup_verified=True, backup_dir=str(bundle))
            if args.pause_after:
                with run_lock(args.output_root):
                    assert_quiescent()
                    save("pause_request_pending")
                    notify()
                    save("pause_requested", pause_result=client.request_pause())
            else:
                save("done_without_pause")
                notify()
            return 0 if success else code if code > 0 else 1
        except Exception as exc:
            print(traceback.format_exc(), flush=True)
            save("needs_attention", error_type=type(exc).__name__, pause_not_confirmed=True)
            notify()
            return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "run", "analyze", "verify", "launch", "job"), required=True)
    parser.add_argument("--stage", choices=("preflight", "patch", "full"), default="preflight")
    parser.add_argument("--plan_dir", required=True)
    parser.add_argument("--source_run", required=True)
    parser.add_argument("--source_backup", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--storage_root", default="/data/yuxuanstorage")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--backup_dir", default="/data/yuxuanstorage/backups")
    parser.add_argument("--retry_failed", action="store_true")
    parser.add_argument("--email_notify", action="store_true")
    parser.add_argument("--email_config", default=EMAIL_CONFIG)
    parser.add_argument("--pause_after", action="store_true")
    parser.add_argument("--surf_config", default=SURF_CONFIG)
    parser.add_argument("--confirm_exclusive_workspace", action="store_true")
    parser.add_argument("--attempt_id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("plan_dir", "source_run", "source_backup", "output_root", "project_root", "storage_root", "backup_dir"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    storage = Path(args.storage_root)
    if not storage.is_dir() or storage not in Path(args.output_root).parents:
        parser.error("Store follow-up artifacts on existing persistent storage.")
    if not Path(args.backup_dir).is_relative_to(storage) or Path(args.backup_dir).is_relative_to(args.output_root):
        parser.error("Backups must use a separate persistent directory.")
    if shutil.disk_usage(storage).free < 12 * 1024**3:
        parser.error("Keep at least 12 GiB free for outputs/backup and safety reserve; no files are auto-deleted.")
    if args.action == "launch":
        launch(args)
        return
    if args.action == "job":
        sys.exit(job(args))
    with run_lock(args.output_root):
        prepared_context = context(args)
        validate_config(prepared_context[0])
        if args.action == "prepare":
            print(json.dumps({"fingerprint": prepared_context[0]["fingerprint"], "new_patches": 288,
                "identity_preflight": 48, "new_activation_capture": False,
                "support_counts": {key: {name: item[name] for name in ("whole_count", "target_union_count",
                    "complement_count", "matched_control_count")} for key, item in prepared_context[0]["support_audit"].items()}}, indent=2))
        elif args.action == "run":
            full = args.stage == "full"
            for stage in ("patch",) if args.stage == "full" else (args.stage,):
                args.stage = stage
                if not run_stage(args, prepared_context)["complete"]:
                    sys.exit(1)
            if stage == "patch":
                analyze(args, prepared_context)
            if full:
                atomic_write(Path(args.output_root) / "last_attempt.json", {"attempt_id": args.attempt_id or uuid.uuid4().hex,
                    "complete": True, "support_fingerprint": prepared_context[0]["fingerprint"],
                    "analysis_sha256": file_hash(Path(args.output_root) / "analysis/aggregate_summary.json")})
        elif args.action == "analyze":
            analyze(args, prepared_context)
        else:
            config, _, pairs, _, baselines = prepared_context
            saved_rows(args.output_root, "preflight", config, pairs, baselines, complete=True)
            saved_rows(args.output_root, "patch", config, pairs, baselines, complete=True)
            print(json.dumps(verify_analysis(args, config), indent=2))


if __name__ == "__main__":
    main()
