"""Independent-case replication with live eligibility, new captures and isolated backups."""

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
    from . import phase3c_replication as protocol
    from . import run_phase3c_budget as runtime
    from . import run_phase3c_support as shared
    from .phase3c_core import atomic_write, baseline_failures, digest, file_hash, frozen_write, read_json, read_jsonl, select_balanced
    from .run_phase3c_preflight import logits_parity
except ImportError:
    import phase3c_replication as protocol
    import run_phase3c_budget as runtime
    import run_phase3c_support as shared
    from phase3c_core import atomic_write, baseline_failures, digest, file_hash, frozen_write, read_json, read_jsonl, select_balanced
    from run_phase3c_preflight import logits_parity

CONDITIONS = protocol.CONDITIONS


def make_engine(args, plan):
    import torch
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    env = shared.worker_environment(args.gpus, args.storage_root)
    for name in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "HF_HOME", "TORCH_HOME"):
        os.environ[name] = env[name]
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        os.environ.pop(name, None)
    expected = plan["reference_runtime"]
    engine = shared.Engine(plan["settings"], expected["execution_mode"], expected["gpu_weight_budget_gib"])
    compatibility = protocol.budget.runtime_compatibility(expected, engine.runtime)
    frozen_write(Path(args.output_root) / "capture_runtime.json", {"runtime": engine.runtime,
        "reference_compatibility": compatibility, "plan_fingerprint": plan["plan_fingerprint"]})
    engine.project_root = args.project_root
    engine.path_map = read_json(Path(args.output_root) / "preparation/processor_audit/config.json")["path_map"]
    return engine


def progress(root, stage, completed, total, started):
    elapsed = time.perf_counter() - started
    atomic_write(Path(root) / stage / "progress_status.json", {"stage": stage, "completed": completed,
        "expected": total, "elapsed_sec": elapsed, "eta_sec": elapsed / completed * (total - completed) if completed else None})
    print(f"Replication {stage}: {completed}/{total}; elapsed={elapsed/60:.1f} min; checkpoint={Path(root)/stage/'rows.jsonl'}", flush=True)


def live_condition(engine, candidate, mapping, condition, plan):
    try:
        from .phase3c_interventions import ResidualSites
    except ImportError:
        from phase3c_interventions import ResidualSites
    pair = protocol.canonical_pair(candidate, plan)
    live = engine.prepare(pair[condition], mapping)
    plain = engine.forward(live)
    record = mapping["processor_records"][condition]
    controller = ResidualSites(engine.text, len(record["input_ids"]), record["visual_positions"])
    with controller.installed():
        captured = engine.forward(live)
    try:
        from .probe_attention_roi import standard_first_token
    except ImportError:
        from probe_attention_roi import standard_first_token
    token, standard = standard_first_token(engine.model, live["inputs"])
    parity = logits_parity(captured, standard)
    decision = engine.decide(captured, live["row"])
    parity["first_token_match"] = token == decision["predicted_first_token_id"]
    noop = logits_parity(plain, captured, exact=True)
    if not noop["passed"] or not parity["passed"] or not parity["first_token_match"]:
        raise RuntimeError("Technical live-baseline parity failed; do not skip this candidate as behavioral ineligibility.")
    result = {"eval_id": live["row"]["eval_id"], "decision": decision, "standard_parity": parity,
        "capture_noop_parity": noop, "hook_audit": controller.validate(),
        "archived_input_parity": live["archived_input_parity"], "input_tensor_sha256": live["input_tensor_sha256"],
        "logits_signature": protocol.budget.tensor_signature(captured)}
    result["behavioral_failures"] = protocol.live_failures(live["row"], result)
    return result


def screen(args):
    plan, candidates, mappings = protocol.audited_candidates(args)
    root, path = Path(args.output_root), Path(args.output_root) / "live_screening/rows.jsonl"
    records = {}
    for row in read_jsonl(path) if path.exists() else []:
        if row["pair_id"] in records:
            raise ValueError("Duplicate live screening checkpoint.")
        records[row["pair_id"]] = row
    lookup = {row["pair_id"]: row for row in candidates}
    if not set(records) <= set(lookup):
        raise ValueError("Foreign live screening candidate.")
    for key, row in records.items():
        if row["candidate_sha256"] != digest(lookup[key]) or row["plan_fingerprint"] != plan["plan_fingerprint"]:
            raise ValueError("Live screening binding changed.")
        if row.get("technical_failure"):
            raise ValueError("A failed live audit is preserved; diagnose before resuming.")
    selected, missing = select_balanced(candidates, {key for key, row in records.items() if row.get("eligible")})
    if not missing:
        return protocol.freeze_cohort(args, records)
    engine, started = make_engine(args, plan), time.perf_counter()
    for number, candidate in enumerate(candidates, 1):
        key = candidate["pair_id"]
        if key in records or not mappings[key]["eligible"]:
            continue
        print(f"Replication live baseline {number}/{len(candidates)}: {key}; no patch outcomes consulted", flush=True)
        row = {"pair_id": key, "candidate_sha256": digest(candidate), "plan_fingerprint": plan["plan_fingerprint"], "eligible": False}
        try:
            row["conditions"] = {condition: live_condition(engine, candidate, mappings[key], condition, plan) for condition in CONDITIONS}
            row["eligible"] = not any(value["behavioral_failures"] for value in row["conditions"].values())
        except Exception as exc:
            row.update(technical_failure=True, failure_type=type(exc).__name__, failure_message=str(exc), traceback=traceback.format_exc())
        records[key] = row
        atomic_write(path, [records[value["pair_id"]] for value in candidates if value["pair_id"] in records], jsonl=True)
        progress(root, "live_screening", len(records), len(candidates), started)
        if row.get("technical_failure"):
            raise RuntimeError("Live technical gate failed; evidence preserved and intervention not started.")
        selected, missing = select_balanced(candidates, {key for key, value in records.items() if value.get("eligible")})
        atomic_write(root / "live_screening/selection_progress.json", {"audited_candidates": len(records),
            "eligible_candidates": sum(value.get("eligible") is True for value in records.values()),
            "missing_quotas": missing, "provisional_base_ids": [value["base_sample_id"] for value in selected],
            "cohort_frozen": False, "no_intervention_results_used": True})
        print(f"Replication live quotas remaining: {missing}", flush=True)
        if not missing:
            break
    del engine
    gc.collect()
    return protocol.freeze_cohort(args, records)


def capture(args):
    plan, _, _ = protocol.audited_candidates(args)
    selection, pairs, mappings = protocol.selected(args)
    root, path = Path(args.output_root), Path(args.output_root) / "capture/rows.jsonl"
    saved = {}
    for row in read_jsonl(path) if path.exists() else []:
        if row["eval_id"] in saved:
            raise ValueError("Duplicate replication capture.")
        saved[row["eval_id"]] = row
    expected = {row["eval_id"] for pair in pairs.values() for row in pair.values()}
    if not set(saved) <= expected:
        raise ValueError("Foreign replication capture.")
    if any(row.get("passed") is not True for row in saved.values()):
        raise ValueError("A failed capture is preserved; diagnose before continuing.")
    if set(saved) == expected:
        return protocol.build_budget_context(args, freeze=True)
    engine, started = make_engine(args, plan), time.perf_counter()
    fingerprint = protocol.capture_binding(args, selection, engine.runtime)
    for pair_id, pair in pairs.items():
        for condition, annotation in pair.items():
            if annotation["eval_id"] in saved:
                continue
            print(f"Replication all-layer capture: {pair_id} {condition}; 36 blocks + 3 post-DeepStack sites", flush=True)
            live = engine.prepare(annotation, mappings[pair_id])
            result = engine.baseline(live, mappings[pair_id], root, fingerprint)
            result["capture_fingerprint"] = fingerprint
            saved[annotation["eval_id"]] = result
            atomic_write(path, list(saved.values()), jsonl=True)
            progress(root, "capture", len(saved), len(expected), started)
            if not result["passed"]:
                raise RuntimeError("Frozen-case capture gate failed; do not replace or drop this case.")
    del engine
    gc.collect()
    return protocol.build_budget_context(args, freeze=True)


def analyze(args, prepared):
    config, _, pairs, _, baselines = prepared
    runtime.validate_observed_runtime(args, config)
    for stage in ("replay", "preflight", "patch"):
        _, saved = runtime.saved_rows(args.output_root, stage, config, pairs, baselines, complete=True)
    table = shared.patch_table(saved, pairs)
    for row in table:
        row.update(config["support_audit"][row["pair_id"]]["support_specs"][row["support"]])
    anchors = [row for row in table if row["family"] == "anchor"]
    dose = [row for row in table if row["family"] != "anchor"]
    cases = protocol.budget.collapse_seeds(dose)
    if (len(table), len(anchors), len(dose), len(cases)) != (816, 240, 576, 192):
        raise ValueError("Incomplete independent replication grid.")
    summary = shared.aggregate(cases, ("analysis_stratum", "direction", "family", "layer", "budget"),
        ("margin_delta", "source_aligned_margin_delta", "strict_incorrect_to_correct", "seed_margin_sd"))
    anchor_summary = shared.aggregate(anchors, ("analysis_stratum", "direction", "anchor", "layer"),
        ("margin_delta", "source_aligned_margin_delta", "strict_incorrect_to_correct", "replaced_token_count", "recipient_coverage"))
    role_summary = shared.aggregate([row for row in anchors if row["mover_role"] is not None],
        ("analysis_stratum", "direction", "mover_role", "layer"),
        ("margin_delta", "source_aligned_margin_delta", "replaced_token_count", "recipient_coverage"))
    lookup = {(row["pair_id"], row["direction"], row["layer"], row["budget"], row["family"]): row for row in cases}
    anchor_lookup = {(row["pair_id"], row["direction"], row["layer"], row["anchor"]): row for row in anchors}
    contrasts = []
    for key, row in lookup.items():
        if row["family"] != "targets_plus_context":
            continue
        control = lookup[(*key[:4], "context_only")]
        contrasts.append({**{name: row[name] for name in ("pair_id", "base_sample_id", "analysis_stratum", "first_object_id", "direction", "layer", "budget")},
            "mixed_minus_context": row["margin_delta"] - control["margin_delta"]})
    contrast_summary = shared.aggregate(contrasts, ("analysis_stratum", "direction", "layer", "budget"), ("mixed_minus_context",))
    coverage = []
    for pair_id, pair in pairs.items():
        meta = shared.patch_table({key: value for key, value in saved.items() if value["spec"]["pair_id"] == pair_id}, pairs)[0]
        for direction in ("temporal_to_low", "low_to_temporal"):
            for layer in protocol.budget.LAYERS:
                coverage.append({**{name: meta[name] for name in ("base_sample_id", "analysis_stratum", "first_object_id")}, "pair_id": pair_id,
                    "direction": direction, "layer": layer,
                    "context_dose_256_minus_64": lookup[(pair_id, direction, layer, 256, "context_only")]["margin_delta"] - lookup[(pair_id, direction, layer, 64, "context_only")]["margin_delta"],
                    "whole_minus_target_union": anchor_lookup[(pair_id, direction, layer, "whole_event2")]["margin_delta"] - anchor_lookup[(pair_id, direction, layer, "both_targets_event2")]["margin_delta"]})
    coverage_summary = shared.aggregate(coverage, ("analysis_stratum", "direction", "layer"), ("context_dose_256_minus_64", "whole_minus_target_union"))
    output = Path(args.output_root) / "analysis"
    for name, values in (("seed_effects", dose), ("anchor_effects", anchors), ("case_seed_means", cases),
                         ("dose_summary", summary), ("anchor_summary", anchor_summary), ("mover_role_summary", role_summary), ("case_composition_contrasts", contrasts),
                         ("composition_summary", contrast_summary), ("case_coverage_contrasts", coverage), ("coverage_summary", coverage_summary)):
        atomic_write(output / f"{name}.json", values)
        shared.write_csv(output / f"{name}.csv", values)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, stratum in zip(axes, ("rescue", "stable")):
        for family, color in zip(protocol.budget.FAMILIES, ("#007F83", "#BC365D")):
            selected = sorted([row for row in summary if row["analysis_stratum"] == stratum and row["direction"] == "temporal_to_low" and row["family"] == family and row["layer"] == 16], key=lambda row: row["budget"])
            x = [row["budget"] for row in selected]
            ax.plot(x, [row["margin_delta_mean"] for row in selected], marker="o", color=color, label=family.replace("_", " "))
            ax.fill_between(x, [row["margin_delta_ci_low"] for row in selected], [row["margin_delta_ci_high"] for row in selected], color=color, alpha=0.1)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set(title=f"{stratum} | new independent bases | L16", xlabel="Exact token budget", ylabel="Correct-option margin change", xticks=protocol.BUDGETS)
    axes[0].legend(fontsize=8)
    fig.suptitle("Independent-case replication | 3 seeds averaged within case | Pilot case-bootstrap intervals")
    fig.savefig(output / "replication_dose.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    result = {"schema": protocol.SCHEMA, "artifact_type": "real", "complete": True, "support_fingerprint": config["fingerprint"],
        "case_count": 12, "new_patch_rows": 816, "dose_rows": 576, "anchor_rows": 240, "knockout_rows": 0,
        "primary_contrast": config["primary_contrast"], "secondary_contrasts": config["secondary_contrasts"],
        "source_independent_case_overlap": 0, "scope": "New stimuli from the same rescue pool; not external population inference.",
        "output_sha256": {path.name: file_hash(path) for path in output.iterdir() if path.is_file() and path.name != "aggregate_summary.json"}}
    atomic_write(output / "aggregate_summary.json", result)
    return result


def command(args, action, stage=None):
    result = [sys.executable, "-u", str(Path(__file__).resolve()), "--action", action, "--stage", stage or args.stage]
    for name in ("source_run", "exclude_plan", "reference_run", "output_root", "project_root", "storage_root", "gpus"):
        result.extend([f"--{name}", getattr(args, name)])
    if getattr(args, "attempt_id", None):
        result.extend(["--attempt_id", args.attempt_id])
    if action == "job":
        result.extend(["--backup_dir", args.backup_dir])
        if args.email_notify:
            result.extend(["--email_notify", "--email_config", args.email_config])
        if args.pause_after:
            result.extend(["--pause_after", "--surf_config", args.surf_config, "--confirm_exclusive_workspace"])
    return result


def services(args):
    forbidden = (args.project_root, args.source_run, args.exclude_plan, args.reference_run, args.output_root, args.backup_dir,
                 str(args.output_root) + "_lifecycle")
    notifier = client = None
    if args.email_notify:
        shared.check_private_location(args.email_config, forbidden)
        notifier = shared.EmailNotifier(shared.load_email_config(args.email_config, forbidden))
        notifier.check()
    if args.pause_after:
        if not args.confirm_exclusive_workspace:
            raise ValueError("Whole-VM Pause requires exclusive-workspace authorization.")
        shared.check_private_location(args.surf_config, forbidden)
        client = shared.SurfClient(shared.load_private_config(args.surf_config, forbidden))
        client.check(require_pause=True)
    return notifier, client


def storage_gate(args):
    _, _, records = protocol.audited_candidates(args)
    estimate = protocol.storage_estimate(records)
    free = shutil.disk_usage(args.storage_root).free
    atomic_write(Path(args.output_root) / "storage_preflight.json", {**estimate, "observed_free_bytes": free})
    if free < estimate["required_free_bytes"]:
        raise ValueError("Insufficient storage for new all-layer captures, full backup and 12 GiB reserve; no files deleted.")
    return estimate


def verify_analysis(args, prepared):
    config, _, pairs, _, baselines = prepared
    for stage in ("replay", "preflight", "patch"):
        runtime.saved_rows(args.output_root, stage, config, pairs, baselines, complete=True)
    runtime.validate_observed_runtime(args, config)
    output = Path(args.output_root) / "analysis"
    result = read_json(output / "aggregate_summary.json")
    if not result.get("complete") or result["support_fingerprint"] != config["fingerprint"] or result["source_independent_case_overlap"] != 0:
        raise ValueError("Replication analysis incomplete/changed.")
    for name, expected in result["output_sha256"].items():
        path = output / name
        if not path.resolve().is_relative_to(output) or file_hash(path) != expected:
            raise ValueError("Replication analysis bytes changed.")
    return result


def backup_files(root, reports_only):
    try:
        from .backup_phase3b import backup_files as files_under_root
    except ImportError:
        from backup_phase3b import backup_files as files_under_root
    root = Path(root)
    files = files_under_root(root, reports_only)
    if (root / "selection/mappings.json").exists() and not reports_only:
        mapping = read_json(root / "selection/mappings.json")
        videos = {record["video_path"]: record["video_sha256"] for row in mapping.values() for record in row["processor_records"].values()}
        for index, (name, expected) in enumerate(sorted(videos.items())):
            path = Path(name)
            if path.is_symlink() or file_hash(path) != expected:
                raise ValueError("A selected replication video changed; full backup and Pause are blocked.")
            files[f"source_videos/{index:03d}_{path.name}"] = path
    return files


def job(args):
    root, directory = Path(args.output_root), Path(str(args.output_root) + "_lifecycle")
    started = time.perf_counter()
    status = {"schema": protocol.SCHEMA, "started_at": datetime.now(timezone.utc).isoformat(), "backup_verified": False, "output_root": str(root)}
    notifier = client = None

    def save(state, **values):
        status.update(state=state, elapsed_sec=time.perf_counter() - started, **values)
        atomic_write(directory / "job_status.json", status)
        print(f"Replication lifecycle: {state}; elapsed={status['elapsed_sec']/60:.1f} min", flush=True)

    def notify():
        if notifier is None:
            return
        label = "SUCCESS" if status.get("experiment_success") else "FAILED" if "experiment_success" in status else "NOT STARTED"
        body = (f"Experiment: {label}\nPhase 3C independent-case replication\nRun: {root}\n"
            f"State: {status['state']}\nElapsed: {status['elapsed_sec']/3600:.2f} hours\n"
            "Planned cohort: 8 new rescue + 4 new stable bases; excluded prior 12 bases.\n"
            "Primary patches: 816; knockout not run. Live eligibility precedes freeze and new all-layer captures.\n"
            f"Verified complete new-evidence/video backup: {status['backup_verified']}\nBackup: {status.get('backup_dir','not available')}\n"
            "Download every part, manifest and SHA256SUMS, then verify locally. Retain prior Phase 3B/3C backups.\n"
            "Local backup and Pause/billing stop are not confirmed. A Pause request follows this email only after verified backup/idle checks.\n"
            "Technical status only, not proof of a mechanistic explanation; no credentials or tensors attached.")
        try:
            status["email"] = notifier.send(f"[Phase 3C Replication] {label}", body)
        except Exception as exc:
            status["email"] = {"error_type": type(exc).__name__}
        atomic_write(directory / "job_status.json", status)

    with shared.lifecycle_lock(directory):
        save("prechecking")
        try:
            storage_gate(args)
            notifier, client = services(args)
            if args.pause_after:
                shared.assert_quiescent()
            args.attempt_id = uuid.uuid4().hex
            code = 0
            for stage in ("screen", "capture", "interventions"):
                save("experiment_running", stage=stage)
                code = shared.run_child(args, command(args, "run", stage), f"replication_{stage}")
                if code != 0:
                    break
            success = False
            if code == 0:
                try:
                    prepared = protocol.build_budget_context(args)
                    verify_analysis(args, prepared)
                    attempt = read_json(root / "last_attempt.json")
                    success = (attempt.get("attempt_id") == args.attempt_id and attempt.get("complete") is True and
                               attempt.get("support_fingerprint") == prepared[0]["fingerprint"])
                except Exception as exc:
                    status["verification_error_type"] = type(exc).__name__
            save("experiment_finished", experiment_return_code=code, experiment_success=success)
            with shared.run_lock(root):
                atomic_write(root / "unattended_status.json", status)
                shutil.copyfile(directory / "job.log", root / "unattended_job.log")
            save("backup_running")
            bundle = shared.create_backup(root, args.backup_dir, bundle_prefix="phase3c_backup", file_provider=backup_files,
                metadata={"schema": "phase3c_replication_backup_v1", "available_activation_file_count": len(list((root / "captures").rglob("*.pt"))),
                    "selected_videos_included": (root / "selection/mappings.json").is_file(), "previous_phase3b_phase3c_backups_included": False,
                    "scope": "All available new replication artifacts/captures and selected videos; completion is verified separately."})
            manifest = shared.verify_backup(bundle)
            if not manifest["complete"] or manifest["reports_only"] or manifest["schema"] != "phase3c_replication_backup_v1":
                raise ValueError("Full replication backup verification failed.")
            if success and (manifest.get("available_activation_file_count") != 24 or manifest.get("selected_videos_included") is not True):
                raise ValueError("Successful replication must back up all 24 captures and selected videos before Pause.")
            save("backup_verified", backup_verified=True, backup_dir=str(bundle))
            if args.pause_after:
                with shared.run_lock(root):
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
    parser.add_argument("--action", choices=("prepare", "run", "verify", "launch", "job"), required=True)
    parser.add_argument("--stage", choices=("screen", "capture", "interventions"), default="interventions")
    for name in ("source_run", "exclude_plan", "reference_run", "output_root"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--storage_root", default="/data/yuxuanstorage")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--backup_dir", default="/data/yuxuanstorage/backups")
    parser.add_argument("--email_notify", action="store_true")
    parser.add_argument("--email_config", default=shared.EMAIL_CONFIG)
    parser.add_argument("--pause_after", action="store_true")
    parser.add_argument("--surf_config", default=shared.SURF_CONFIG)
    parser.add_argument("--confirm_exclusive_workspace", action="store_true")
    parser.add_argument("--attempt_id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.retry_failed = False
    for name in ("source_run", "exclude_plan", "reference_run", "output_root", "project_root", "storage_root", "backup_dir"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    if Path(args.storage_root) not in Path(args.output_root).parents or not Path(args.storage_root).is_dir():
        parser.error("Use existing persistent storage.")
    if Path(args.backup_dir).is_relative_to(args.output_root) or not Path(args.backup_dir).is_relative_to(args.storage_root):
        parser.error("Use a separate backup directory on persistent storage.")
    if shutil.disk_usage(args.storage_root).free < 24 * 1024**3 and args.action in ("prepare", "launch", "job"):
        parser.error("Keep 24 GiB free before replication to reserve capture/backup space; no files are auto-deleted.")
    root = Path(args.output_root)
    if args.action == "job":
        sys.exit(job(args))
    if args.action == "launch":
        storage_gate(args)
        services(args)
        if not shutil.which("tmux") or subprocess.run(["tmux", "has-session", "-t", "=phase3c_replication"], capture_output=True).returncode == 0:
            parser.error("tmux unavailable or replication session already exists.")
        directory = Path(str(root) + "_lifecycle")
        with shared.lifecycle_lock(directory), shared.run_lock(root):
            shell = f"exec {shlex.join(command(args, 'job'))} >> {shlex.quote(str(directory/'job.log'))} 2>&1"
            subprocess.run(["tmux", "new-session", "-d", "-s", "phase3c_replication", "-c", args.project_root, shell], check=True)
        print(f"Detached replication started; status={directory/'job_status.json'}", flush=True)
        return
    with shared.run_lock(root):
        if args.action == "prepare":
            protocol.create_plan(args)
            print(json.dumps(protocol.processor_audit(args), indent=2))
        elif args.action == "verify":
            print(json.dumps(verify_analysis(args, protocol.build_budget_context(args)), indent=2))
        elif args.stage == "screen":
            screen(args)
        elif args.stage == "capture":
            capture(args)
        else:
            prepared = protocol.build_budget_context(args)
            for stage in ("replay", "preflight", "patch"):
                if not runtime.run_stage(args, prepared, stage)["complete"]:
                    sys.exit(1)
            analyze(args, prepared)
            atomic_write(root / "last_attempt.json", {"attempt_id": args.attempt_id or uuid.uuid4().hex,
                "complete": True, "support_fingerprint": prepared[0]["fingerprint"]})


if __name__ == "__main__":
    main()
