"""Checkpointed, artifact-isolated Phase 3C fixed-budget follow-up."""

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
    from . import phase3c_budget as protocol
    from . import run_phase3c_support as shared
except ImportError:
    import phase3c_budget as protocol
    import run_phase3c_support as shared

CONDITIONS = shared.CONDITIONS
atomic_write, read_json, file_hash, digest = shared.atomic_write, shared.read_json, shared.file_hash, shared.digest
run_lock, lifecycle_lock, frozen_write = shared.run_lock, shared.lifecycle_lock, shared.frozen_write


def context(args):
    return protocol.prepare(args, freeze=args.action == "prepare")


def result_failures(row, task, config, pairs, baselines):
    if task["kind"] != "capture_replay":
        return shared.result_failures(row, task, config, pairs, baselines)
    expected = config["expected_replay_signatures"][pairs[task["pair_id"]][task["condition"]]["eval_id"]]
    baseline = baselines[pairs[task["pair_id"]][task["condition"]]["eval_id"]]
    failures = []
    if (row.get("spec") != task or row.get("task_id") != task["task_id"] or
            row.get("support_fingerprint") != config["fingerprint"]):
        failures.append("replay_binding_changed")
    if row.get("replay_signatures") != expected:
        failures.append("source_logits_or_donor_state_bytes_changed")
    if (row.get("input_tensor_sha256") != baseline["input_tensor_sha256"] or
            row.get("decision") != baseline["decision"] or row.get("is_primary_effect_estimate") is not False):
        failures.append("replay_input_or_decision_changed")
    hook = row.get("hook_audit", {})
    if hook.get("block_output_sites") != 36 or hook.get("post_deepstack_layers") != [0, 1, 2]:
        failures.append("replay_hook_incomplete")
    return failures


def saved_rows(root, stage, config, pairs, baselines, complete=False):
    tasks = protocol.task_grid(config, stage)
    expected, rows = {task["task_id"]: task for task in tasks}, {}
    for path in sorted((Path(root) / stage / "task_checkpoints").glob("*.json")):
        row = read_json(path)
        key = row.get("task_id")
        if key not in expected or path.stem != key or key in rows:
            raise ValueError("Foreign/duplicate budget checkpoint; preserve and diagnose it.")
        if row.get("passed") is True and result_failures(row, expected[key], config, pairs, baselines):
            raise ValueError("A claimed-passed budget checkpoint failed independent validation.")
        rows[key] = row
    if complete and (set(rows) != set(expected) or any(not row.get("passed") for row in rows.values())):
        raise ValueError(f"Incomplete/failed budget {stage} gate.")
    return tasks, rows


def replay(engine, live, mapping, task, config):
    try:
        from .phase3c_interventions import ResidualSites, selected_state
    except ImportError:
        from phase3c_interventions import ResidualSites, selected_state
    record = mapping["processor_records"][task["condition"]]
    expected = config["expected_replay_signatures"][live["row"]["eval_id"]]
    controller = ResidualSites(engine.text, len(record["input_ids"]), record["visual_positions"], expected["positions"])
    with controller.installed():
        logits = engine.forward(live)
    signatures = {"logits": protocol.tensor_signature(logits), "positions": expected["positions"],
        "sites": {str(layer): protocol.tensor_signature(selected_state(controller.states, layer,
                    "block_output", expected["positions"])["vectors"]) for layer in protocol.LAYERS}}
    return {"replay_signatures": signatures, "hook_audit": controller.validate(),
        "input_tensor_sha256": live["input_tensor_sha256"], "decision": engine.decide(logits, live["row"]),
        "is_primary_effect_estimate": False, "passed": signatures == expected}


def run_stage(args, prepared, stage):
    config, frozen, pairs, mappings, baselines = prepared
    root, output = Path(args.output_root), Path(args.output_root) / stage
    for gate in (() if stage == "replay" else ("replay",) if stage == "preflight" else ("replay", "preflight")):
        saved_rows(root, gate, config, pairs, baselines, complete=True)
    tasks, saved = saved_rows(root, stage, config, pairs, baselines)
    frozen_write(output / "task_manifest.jsonl", tasks, jsonl=True)

    def summarize():
        missing = [task["task_id"] for task in tasks if task["task_id"] not in saved]
        failures = [{"task_id": task["task_id"], "reasons": result_failures(saved[task["task_id"]], task,
            config, pairs, baselines)} for task in tasks if task["task_id"] in saved and
            (not saved[task["task_id"]].get("passed") or result_failures(saved[task["task_id"]], task, config, pairs, baselines))]
        atomic_write(output / "rows.jsonl", [saved[task["task_id"]] for task in tasks if task["task_id"] in saved], jsonl=True)
        summary = {"schema": protocol.SCHEMA, "support_fingerprint": config["fingerprint"], "stage": stage,
            "complete": not missing and not failures, "expected": len(tasks), "completed": len(saved),
            "missing_task_ids": missing, "failures": failures,
            "rows_sha256": file_hash(output / "rows.jsonl"), "manifest_sha256": file_hash(output / "task_manifest.jsonl")}
        atomic_write(output / "summary.json", summary)
        return summary

    initial = summarize()
    if initial["complete"]:
        print(f"Budget {stage}: reused all {len(tasks)} verified tasks; no model load.", flush=True)
        return initial
    if initial["failures"] and not args.retry_failed:
        raise ValueError("Failed budget task preserved; diagnose before explicit --retry_failed.")
    env = shared.worker_environment(args.gpus, args.storage_root)
    os.environ.update({key: value for key, value in env.items() if key in
                       ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "HF_HOME", "TORCH_HOME")})
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        os.environ.pop(name, None)
    engine = shared.Engine(frozen["settings"], config["source_runtime"]["execution_mode"],
                           config["source_runtime"]["gpu_weight_budget_gib"])
    compatibility = protocol.runtime_compatibility(config["source_runtime"], engine.runtime)
    frozen_write(root / "observed_runtime.json", {"runtime": engine.runtime, "compatibility": compatibility,
        "support_fingerprint": config["fingerprint"]})
    engine.project_root, engine.path_map = args.project_root, config["source_path_map"]
    current, live, captures, computed, started = None, None, {}, 0, time.perf_counter()
    for task in tasks:
        key = task["task_id"]
        if key in saved and saved[key].get("passed"):
            continue
        pair_id, condition = task["pair_id"], task["condition"]
        result = {"task_id": key, "spec": task, "support_fingerprint": config["fingerprint"], "passed": False}
        begun = time.perf_counter()
        print(f"Budget {stage}: {pair_id} {condition} {task.get('support', 'exact donor replay')} L{task.get('layer', '12/16')}", flush=True)
        try:
            if current != (pair_id, condition):
                live = None
                gc.collect()
                engine.torch.cuda.empty_cache()
                live = engine.prepare(pairs[pair_id][condition], mappings[pair_id])
                baseline = baselines[live["row"]["eval_id"]]
                if live["input_tensor_sha256"] != baseline["input_tensor_sha256"]:
                    raise ValueError("Live input tensor bytes differ from frozen source baseline.")
                captures = {}
                if stage != "replay":
                    for side in CONDITIONS:
                        index = read_json(baselines[pairs[pair_id][side]["eval_id"]]["capture_index_path"])
                        captures[side] = engine.torch.load(index["vectors_path"], map_location="cpu", weights_only=True)
                current = (pair_id, condition)
            if stage == "replay":
                result.update(replay(engine, live, mappings[pair_id], task, config))
            else:
                mapping = {**mappings[pair_id], "support_audit": {**mappings[pair_id]["support_audit"],
                    "supports": {**mappings[pair_id]["support_audit"]["supports"],
                                 **config["support_audit"][pair_id]["supports"]}}}
                converted = {**task, "kind": "identity" if stage == "preflight" else "transplant_smoke"}
                result.update(engine.technical(live, mapping, converted, captures))
                result.update(spec=task, input_tensor_sha256=live["input_tensor_sha256"], is_primary_effect_estimate=stage == "patch")
                if stage == "patch":
                    donor = baselines[pairs[pair_id][result["donor_condition"]]["eval_id"]]["decision"]
                    result["donor_baseline_decision"] = donor
                    result.update(shared.patch_outcomes(result["baseline_decision"]["margin"], result["decision"]["margin"], donor["margin"]))
            failures = result_failures(result, task, config, pairs, baselines)
            if failures:
                result.update(passed=False, validation_failures=failures)
        except Exception as exc:
            result.update(passed=False, failure_type=type(exc).__name__, failure_message=str(exc), traceback=traceback.format_exc())
            print(result["traceback"], flush=True)
        result["elapsed_sec"] = time.perf_counter() - begun
        if not result["passed"]:
            errors = read_json(output / "errors.json") if (output / "errors.json").exists() else []
            atomic_write(output / "errors.json", [*errors, result])
        saved[key] = result
        atomic_write(output / "task_checkpoints" / f"{key}.json", result)
        computed += 1
        passed, elapsed = sum(item.get("passed") is True for item in saved.values()), time.perf_counter() - started
        eta = elapsed / computed * (len(tasks) - passed)
        atomic_write(output / "progress_status.json", {"stage": stage, "expected": len(tasks), "passed": passed,
            "failed": len(saved) - passed, "elapsed_sec": elapsed, "newly_computed": computed, "eta_sec": eta,
            "checkpoint": str(output / "task_checkpoints")})
        print(f"Budget {stage}: {passed}/{len(tasks)} passed; elapsed={elapsed/60:.1f} min; ETA={eta/60:.1f} min; checkpoint={output/'task_checkpoints'}", flush=True)
        if not result["passed"]:
            break
    return summarize()


def analyze(args, prepared):
    config, _, pairs, _, baselines = prepared
    validate_observed_runtime(args, config)
    for stage in ("replay", "preflight", "patch"):
        _, rows = saved_rows(args.output_root, stage, config, pairs, baselines, complete=True)
    table = shared.patch_table(rows, pairs)
    for row in table:
        row.update(config["support_audit"][row["pair_id"]]["support_specs"][row["support"]])
        if row["replaced_token_count"] != row["budget"]:
            raise ValueError("Actual replacement count differs from fixed budget.")
    cases = protocol.collapse_seeds(table)
    lookup = {(row["pair_id"], row["direction"], row["layer"], row["budget"], row["family"]): row for row in cases}
    contrasts = []
    for row in cases:
        if row["family"] != "targets_plus_context":
            continue
        control = lookup[(row["pair_id"], row["direction"], row["layer"], row["budget"], "context_only")]
        contrasts.append({**{key: row[key] for key in ("pair_id", "base_sample_id", "analysis_stratum", "first_object_id",
            "direction", "layer", "budget")}, "mixed_minus_context": row["margin_delta"] - control["margin_delta"],
            "mixed_minus_context_source_aligned": row["source_aligned_margin_delta"] - control["source_aligned_margin_delta"]})
    if len(table) != config["primary_patch_count"] or len(cases) != 288 or len(contrasts) != 144:
        raise ValueError("Incomplete seed/composition/budget analysis matrix.")
    summaries = shared.aggregate(cases, ("analysis_stratum", "direction", "family", "layer", "budget"),
        ("margin_delta", "source_aligned_margin_delta", "recovery", "strict_incorrect_to_correct",
         "strict_sign_crossing", "seed_margin_sd", "replaced_token_count"))
    contrasts_summary = shared.aggregate(contrasts, ("analysis_stratum", "direction", "layer", "budget"),
        ("mixed_minus_context", "mixed_minus_context_source_aligned"))
    refs = [row for row in read_json(Path(args.source_followup) / "analysis/case_support_effects.json")
            if row["support"] in ("whole_event2", "both_targets_event2") and row["layer"] in protocol.LAYERS]
    if len(refs) != 96:
        raise ValueError("Missing prior reference effects.")
    output = Path(args.output_root) / "analysis"
    for name, items in (("seed_effects", table), ("case_seed_means", cases), ("budget_summary", summaries),
                        ("case_budget_contrasts", contrasts), ("contrast_summary", contrasts_summary), ("source_references", refs)):
        atomic_write(output / f"{name}.json", items)
        shared.write_csv(output / f"{name}.csv", items)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for i, stratum in enumerate(("rescue", "stable")):
        for j, direction in enumerate(("temporal_to_low", "low_to_temporal")):
            ax = axes[i, j]
            for family, color in zip(protocol.FAMILIES, ("#007F83", "#BC365D")):
                for layer, style in zip(protocol.LAYERS, ("--", "-")):
                    selected = sorted([row for row in summaries if row["analysis_stratum"] == stratum and
                        row["direction"] == direction and row["family"] == family and row["layer"] == layer],
                        key=lambda row: row["budget"])
                    x = [row["budget"] for row in selected]
                    ax.plot(x, [row["margin_delta_mean"] for row in selected], color=color, linestyle=style,
                        marker="o", label=f"{family.replace('_', ' ')} | L{layer}")
                    ax.fill_between(x, [row["margin_delta_ci_low"] for row in selected],
                        [row["margin_delta_ci_high"] for row in selected], color=color, alpha=0.07)
            ax.axhline(0, color="black", linewidth=0.7)
            ax.set(title=f"{stratum} | {direction.replace('_', ' ')}", xlabel="Exact replaced token count",
                   ylabel="Correct-option margin change", xticks=protocol.BUDGETS)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Exploratory fixed-budget dose | 3 seeds averaged within case | Case-bootstrap pilot intervals")
    fig.savefig(output / "budget_dose.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    (output / "report.md").write_text("""# Exploratory Phase 3C Fixed-Budget Dose Follow-Up

The same 8 rescue and 4 stable bases are reused. This is post-pilot exploration, not an independent replication.
Budgets 64/128/256 and sites L12/L16 were fixed before these intervention outcomes.
Both compositions match exact counts in every paired temporal bin. Three fixed seeds select nested supports.
Seeds are averaged within case before case-level bootstrap; strict rescue metrics are fractions across seed runs.
No seed, case or layer was selected for a favorable effect. No dummy/interpolated/duplicated donor vectors are used.
Non-target-position residuals may contain target information; equal counts do not equalize information or attention.
Capacity-only runtime metadata changes require exact replay of every condition's full logits and donor states.
Observed runtime and replay signatures remain separate from the unchanged source activation artifacts.
The prior whole-event/target effects are references only, not new independent observations or equal-budget contrasts.
Intervals are exploratory diagnostics, not population/multiplicity-adjusted inference or additive causal mediation.
Retain the separate complete Phase 3C activation/video backup and the prior support follow-up backup.
""", encoding="ascii")
    summary = {"schema": protocol.SCHEMA, "artifact_type": "real", "support_fingerprint": config["fingerprint"],
        "complete": True, "case_count": 12, "rescue_cases": 8, "stable_cases": 4, "new_patch_rows": len(table),
        "within_case_seed_mean_rows": len(cases), "sampling_seeds": list(protocol.SEEDS),
        "capture_replay_count": 24, "identity_preflight_count": 144, "new_activation_capture": False,
        "reused_reference_rows": len(refs), "exploratory_after_pilot": True,
        "output_sha256": {path.name: file_hash(path) for path in output.iterdir() if path.is_file()
                          and path.name != "aggregate_summary.json"}}
    atomic_write(output / "aggregate_summary.json", summary)
    return summary


def verify_analysis(args, config):
    validate_observed_runtime(args, config)
    output = Path(args.output_root) / "analysis"
    summary = read_json(output / "aggregate_summary.json")
    if not summary.get("complete") or summary.get("support_fingerprint") != config["fingerprint"]:
        raise ValueError("Budget analysis incomplete or incompatible.")
    for name, expected in summary["output_sha256"].items():
        path = output / name
        if not path.resolve().is_relative_to(output) or file_hash(path) != expected:
            raise ValueError("Budget analysis bytes changed.")
    return summary


def validate_observed_runtime(args, config):
    record = read_json(Path(args.output_root) / "observed_runtime.json")
    if (record.get("support_fingerprint") != config["fingerprint"] or
            record.get("compatibility") != protocol.runtime_compatibility(config["source_runtime"], record["runtime"])):
        raise ValueError("Observed runtime binding changed.")


def private_services(args):
    for name in ("email_config", "surf_config"):
        if (name == "email_config" and args.email_notify) or (name == "surf_config" and args.pause_after):
            shared.check_private_location(getattr(args, name), (args.source_followup,))
    return shared.private_services(args)


def command(args, action):
    result = [sys.executable, "-u", str(Path(__file__).resolve()), "--action", action, "--stage", args.stage]
    for name in ("plan_dir", "source_run", "source_followup", "source_backup", "output_root", "project_root", "storage_root", "gpus"):
        result.extend([f"--{name}", getattr(args, name)])
    if args.retry_failed:
        result.append("--retry_failed")
    if getattr(args, "attempt_id", None):
        result.extend(["--attempt_id", args.attempt_id])
    if action == "job":
        result.extend(["--backup_dir", args.backup_dir])
        if args.email_notify:
            result.extend(["--email_notify", "--email_config", args.email_config])
        if args.pause_after:
            result.extend(["--pause_after", "--surf_config", args.surf_config, "--confirm_exclusive_workspace"])
    return result


def launch(args):
    context(args)
    if args.stage != "full" or not shutil.which("tmux"):
        raise ValueError("Unattended launch requires full gated sequence and tmux.")
    if subprocess.run(["tmux", "has-session", "-t", "=phase3c_budget"], capture_output=True).returncode == 0:
        raise ValueError("Budget session already exists; do not launch a duplicate.")
    directory = Path(str(args.output_root) + "_lifecycle")
    with lifecycle_lock(directory), run_lock(args.output_root):
        private_services(args)
        shell = f"exec {shlex.join(command(args, 'job'))} >> {shlex.quote(str(directory/'job.log'))} 2>&1"
        subprocess.run(["tmux", "new-session", "-d", "-s", "phase3c_budget", "-c", args.project_root, shell], check=True)
    print(f"Detached budget job started; replay/preflight gates precede patch; status={directory/'job_status.json'}", flush=True)


def job(args):
    directory = Path(str(args.output_root) + "_lifecycle")
    started = time.perf_counter()
    status = {"schema": protocol.SCHEMA, "started_at": datetime.now(timezone.utc).isoformat(),
              "state": "prechecking", "output_root": args.output_root, "backup_verified": False}
    notifier = client = None

    def save(state, **fields):
        status.update(state=state, elapsed_sec=time.perf_counter() - started, **fields)
        atomic_write(directory / "job_status.json", status)
        print(f"Budget lifecycle: {state}; elapsed={status['elapsed_sec']/60:.1f} min", flush=True)

    def notify():
        if notifier is None:
            return
        label = "SUCCESS" if status.get("experiment_success") else "FAILED" if "experiment_success" in status else "NOT STARTED"
        body = (f"Experiment: {label}\nExploratory Phase 3C fixed-budget dose\nState: {status['state']}\n"
            f"Run: {args.output_root}\nElapsed: {status['elapsed_sec']/3600:.2f} hours\n"
            "Cohort: 8 rescue + 4 stable bases; 3 seeds are within-case repeats, not independent cases.\n"
            "Required gates: 24 exact donor/logit replays + 144 identity patches; primary grid: 864 patches.\n"
            f"Verified new backup: {status['backup_verified']}\nBackup: {status.get('backup_dir','not available')}\n"
            f"Required source activation/video backup: {args.source_backup}\n"
            "The new package does not duplicate source tensors/videos. Retain the prior complete backups.\n"
            "Local backup and Pause/billing stop are not confirmed by this email. Check the portal.\n"
            "The Pause request follows this email only after verified backup and idle checks.\n"
            "Technical status only; no credentials or tensors attached.")
        try:
            status["email"] = notifier.send(f"[Phase 3C Budget] {label}", body)
        except Exception as exc:
            status["email"] = {"error_type": type(exc).__name__}
        atomic_write(directory / "job_status.json", status)

    with lifecycle_lock(directory):
        save("prechecking")
        try:
            config, _, pairs, _, baselines = context(args)
            notifier, client = private_services(args)
            if args.pause_after:
                shared.assert_quiescent()
            save("experiment_running")
            args.attempt_id = uuid.uuid4().hex
            code = shared.run_child(args, command(args, "run"), "budget_full")
            success = False
            if code == 0:
                try:
                    with run_lock(args.output_root):
                        attempt = read_json(Path(args.output_root) / "last_attempt.json")
                        if (attempt.get("attempt_id") != args.attempt_id or not attempt.get("complete") or
                                attempt.get("support_fingerprint") != config["fingerprint"]):
                            raise ValueError("Old analysis cannot establish a new successful budget attempt.")
                        for stage in ("replay", "preflight", "patch"):
                            saved_rows(args.output_root, stage, config, pairs, baselines, complete=True)
                        verify_analysis(args, config)
                        success = True
                except Exception as exc:
                    status["verification_error_type"] = type(exc).__name__
            save("experiment_finished", experiment_return_code=code, experiment_success=success)
            with run_lock(args.output_root):
                atomic_write(Path(args.output_root) / "unattended_status.json", status)
                shutil.copyfile(directory / "job.log", Path(args.output_root) / "unattended_job.log")
            save("backup_running")
            bundle = shared.create_backup(args.output_root, args.backup_dir, bundle_prefix="phase3c_backup",
                metadata={"schema": "phase3c_budget_backup_v1", "support_fingerprint": config["fingerprint"],
                    "new_activation_file_count": 0, "source_activation_backup_included": False,
                    "source_backup_dir": args.source_backup,
                    "source_backup_manifest_sha256": config["source_backup_manifest_sha256"],
                    "source_followup_root": args.source_followup})
            manifest = shared.verify_backup(bundle)
            if not manifest["complete"] or manifest["reports_only"] or manifest["support_fingerprint"] != config["fingerprint"]:
                raise ValueError("Budget backup verification failed.")
            save("backup_verified", backup_verified=True, backup_dir=str(bundle))
            if args.pause_after:
                with run_lock(args.output_root):
                    shared.assert_quiescent()
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
    parser.add_argument("--stage", choices=("replay", "preflight", "patch", "full"), default="full")
    for name in ("plan_dir", "source_run", "source_followup", "source_backup", "output_root"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--storage_root", default="/data/yuxuanstorage")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--backup_dir", default="/data/yuxuanstorage/backups")
    parser.add_argument("--retry_failed", action="store_true")
    parser.add_argument("--email_notify", action="store_true")
    parser.add_argument("--email_config", default=shared.EMAIL_CONFIG)
    parser.add_argument("--pause_after", action="store_true")
    parser.add_argument("--surf_config", default=shared.SURF_CONFIG)
    parser.add_argument("--confirm_exclusive_workspace", action="store_true")
    parser.add_argument("--attempt_id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("plan_dir", "source_run", "source_followup", "source_backup", "output_root", "project_root", "storage_root", "backup_dir"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    storage = Path(args.storage_root)
    if not storage.is_dir() or storage not in Path(args.output_root).parents:
        parser.error("Use existing persistent storage for new artifacts.")
    if not Path(args.backup_dir).is_relative_to(storage) or Path(args.backup_dir).is_relative_to(args.output_root):
        parser.error("Backups must use a separate persistent directory.")
    if shutil.disk_usage(storage).free < 12 * 1024**3:
        parser.error("Keep at least 12 GiB free; no files are auto-deleted.")
    if args.action == "launch":
        launch(args)
        return
    if args.action == "job":
        sys.exit(job(args))
    with run_lock(args.output_root):
        prepared = context(args)
        protocol.validate_config(prepared[0])
        if args.action == "prepare":
            print(json.dumps({"fingerprint": prepared[0]["fingerprint"], "new_patches": 864,
                "identity_preflight": 144, "exact_capture_replays": 24, "new_activation_capture": False,
                "budgets": list(protocol.BUDGETS), "layers": list(protocol.LAYERS), "seeds": list(protocol.SEEDS)}, indent=2))
        elif args.action == "run":
            for stage in ("replay", "preflight", "patch") if args.stage == "full" else (args.stage,):
                if not run_stage(args, prepared, stage)["complete"]:
                    sys.exit(1)
            if stage == "patch":
                analyze(args, prepared)
            if args.stage == "full":
                atomic_write(Path(args.output_root) / "last_attempt.json", {"attempt_id": args.attempt_id or uuid.uuid4().hex,
                    "complete": True, "support_fingerprint": prepared[0]["fingerprint"],
                    "analysis_sha256": file_hash(Path(args.output_root) / "analysis/aggregate_summary.json")})
        elif args.action == "analyze":
            analyze(args, prepared)
        else:
            for stage in ("replay", "preflight", "patch"):
                saved_rows(args.output_root, stage, prepared[0], prepared[2], prepared[4], complete=True)
            print(json.dumps(verify_analysis(args, prepared[0]), indent=2))


if __name__ == "__main__":
    main()
