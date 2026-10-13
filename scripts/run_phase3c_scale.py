"""Detached, gated whole-Event-2 expansion: 50 frozen rescues, two layers, no KO."""

import argparse
import gc
import json
import os
import re
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
    from . import phase3c_scale as protocol
    from . import run_phase3c_support as shared
    from .phase3c_budget import runtime_compatibility, tensor_signature
    from .phase3c_core import atomic_write, baseline_failures, digest, file_hash, frozen_write, read_json
    from .phase3c_interventions import ResidualSites, selected_state
    from .phase3c_primary import finite_decision, patch_outcomes
    from .plot_phase3c_comparison import generate
    from .run_phase3c_preflight import logits_parity, save_tensors
except ImportError:
    import phase3c_scale as protocol
    import run_phase3c_support as shared
    from phase3c_budget import runtime_compatibility, tensor_signature
    from phase3c_core import atomic_write, baseline_failures, digest, file_hash, frozen_write, read_json
    from phase3c_interventions import ResidualSites, selected_state
    from phase3c_primary import finite_decision, patch_outcomes
    from plot_phase3c_comparison import generate
    from run_phase3c_preflight import logits_parity, save_tensors

CONDITIONS = protocol.CONDITIONS


class CompactSites(ResidualSites):
    """Retain the full hook audit but save only the two approved patch sites."""

    def visit(self, hidden, layer, location):
        positions = self.positions
        if location != "block_output" or layer not in protocol.LAYERS:
            self.positions = None
        try:
            return super().visit(hidden, layer, location)
        finally:
            self.positions = positions

    def validate(self):
        positions, self.positions = self.positions, None
        try:
            result = super().validate()
        finally:
            self.positions = positions
        if set(self.states) != {(layer, "block_output") for layer in protocol.LAYERS}:
            raise RuntimeError("Compact capture missed a planned intervention site.")
        return {**result, "saved_sites": [[layer, "block_output"] for layer in protocol.LAYERS]}


def seal(row):
    return {**row, "record_sha256": digest({k: v for k, v in row.items() if k != "record_sha256"})}


def checked_payload(row, torch):
    path = row["capture_index_path"]
    if file_hash(path) != row["capture_index_sha256"]:
        raise ValueError("Capture index checksum changed.")
    index = read_json(path)
    if (index["artifact_type"] != "real" or index["eval_id"] != row["eval_id"] or
            index["input_tensor_sha256"] != row["input_tensor_sha256"] or
            file_hash(index["vectors_path"]) != index["vectors_sha256"]):
        raise ValueError("Capture tensor checksum or input binding changed.")
    return torch.load(index["vectors_path"], map_location="cpu", weights_only=True)


def validate_row(row, spec, plan, pairs, mappings, baselines, prior):
    if (row.get("record_sha256") != seal(row)["record_sha256"] or row.get("spec") != spec or
            row.get("plan_fingerprint") != plan["fingerprint"] or row.get("passed") is not True):
        raise ValueError("Foreign, altered or failed checkpoint; preserve and diagnose it.")
    annotation = pairs[spec["pair_id"]][spec["condition"]]
    if spec["kind"] == "baseline":
        if (baseline_failures(annotation, row) or not row["capture_noop_parity"]["exact_match"] or
                not row["hook_audit"]["block_output_sites"] == 36 or
                row["hook_audit"]["post_deepstack_layers"] != [0, 1, 2] or
                (annotation["condition"] == CONDITIONS[0]) != (row["decision"]["margin"] < 0)):
            raise ValueError("Saved baseline violates strict rescue/parity/hook gates.")
        index = read_json(row["capture_index_path"])
        if (file_hash(row["capture_index_path"]) != row["capture_index_sha256"] or
                index["input_tensor_sha256"] != row["input_tensor_sha256"] or
                file_hash(index["vectors_path"]) != index["vectors_sha256"] or
                index["eval_id"] != annotation["eval_id"]):
            raise ValueError("Saved baseline capture changed.")
        if row["capture_reused"] and not all(row["source_replay"].values()):
            raise ValueError("Prior capture was not exactly replayed.")
    else:
        baseline = baselines[annotation["eval_id"]]
        support = mappings[spec["pair_id"]]["support_audit"]["supports"]["whole_event2"]
        side = "low_positions" if spec["condition"] == CONDITIONS[0] else "temporal_positions"
        if (row["baseline_record_sha256"] != baseline["record_sha256"] or
                row["input_tensor_sha256"] != baseline["input_tensor_sha256"] or
                row["baseline_decision"] != baseline["decision"] or row["support_audit"] != support or
                row["recipient_positions"] != support[side] or
                row["hook_audit"]["patch_applied_count"] != 1 or
                row["hook_audit"]["block_output_sites"] != 36 or
                row["hook_audit"]["post_deepstack_layers"] != [0, 1, 2] or
                not finite_decision(row["decision"])):
            raise ValueError("Saved intervention input, hook, support or baseline binding changed.")
        if spec["kind"] == "identity":
            if not row["noop_parity"]["exact_match"] or row["noop_parity"]["max_abs_diff"] != 0:
                raise ValueError("Identity patch must be an exact no-op.")
        else:
            opposite = CONDITIONS[1] if spec["condition"] == CONDITIONS[0] else CONDITIONS[0]
            donor = baselines[pairs[spec["pair_id"]][opposite]["eval_id"]]
            expected = patch_outcomes(baseline["decision"]["margin"], row["decision"]["margin"], donor["decision"]["margin"])
            if (row["donor_baseline_decision"] != donor["decision"] or
                    row["donor_record_sha256"] != donor["record_sha256"] or
                    any(row[key] != value for key, value in expected.items())):
                raise ValueError("Donor binding or patch decomposition changed.")
            direction = "temporal_to_low" if spec["condition"] == CONDITIONS[0] else "low_to_temporal"
            previous = prior.get((spec["pair_id"], direction, spec["layer"]))
            if bool(previous) != row["prior_effect_replayed"] or previous and not protocol.source_replay_matches(row, previous["effect"]):
                raise ValueError("Prior patch effect did not exactly replay.")


def saved_rows(root, stage, context, baselines=None, complete=False):
    plan, pairs, mappings, _, prior = context
    tasks = protocol.tasks(plan, stage)
    expected = {item["task_id"]: item for item in tasks}
    rows = {}
    for path in sorted((Path(root) / stage / "task_checkpoints").glob("*.json")):
        row = read_json(path)
        key = row.get("task_id")
        if key not in expected or key in rows or path.stem != key:
            raise ValueError("Foreign/duplicate scale checkpoint.")
        validate_row(row, expected[key], plan, pairs, mappings, baselines, prior)
        rows[key] = row
    if complete and set(rows) != set(expected):
        raise ValueError(f"Incomplete {stage} gate: {len(rows)}/{len(expected)}.")
    return tasks, rows


def baseline_lookup(args, context):
    _, rows = saved_rows(args.output_root, "baseline", context, complete=True)
    return {value["eval_id"]: value for value in rows.values()}


def make_engine(args, plan, stage):
    env = shared.worker_environment(args.gpus, args.storage_root)
    os.environ.update({key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "HF_HOME", "TORCH_HOME")})
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        os.environ.pop(key, None)
    expected = plan["source_runtime"]
    engine = shared.Engine(plan["settings"], expected["execution_mode"], expected["gpu_weight_budget_gib"])
    compatibility = runtime_compatibility(expected, engine.runtime)
    record = {"runtime": engine.runtime, "reference_compatibility": compatibility,
              "plan_fingerprint": plan["fingerprint"], "stage": stage}
    atomic_write(Path(args.output_root) / "runtime_observations" / f"{stage}_{time.time_ns()}.json", record)
    engine.project_root, engine.path_map = args.project_root, plan["source_path_map"]
    return engine


def capture(engine, live, mapping, reference, root, plan):
    try:
        from .probe_attention_roi import standard_first_token
    except ImportError:
        from probe_attention_roi import standard_first_token
    row = live["row"]
    record = mapping["processor_records"][row["condition"]]
    side = "low_positions" if row["condition"] == CONDITIONS[0] else "temporal_positions"
    positions = sorted(mapping["support_audit"]["supports"]["whole_event2"][side])
    plain = engine.forward(live)
    controller = CompactSites(engine.text, len(record["input_ids"]), record["visual_positions"], positions)
    with controller.installed():
        logits = engine.forward(live)
    hook_audit = controller.validate()
    token, standard = standard_first_token(engine.model, live["inputs"])
    parity = logits_parity(logits, standard)
    decision = engine.decide(logits, row)
    parity["first_token_match"] = token == decision["predicted_first_token_id"]
    result = {"eval_id": row["eval_id"], "decision": decision, "standard_parity": parity,
        "capture_noop_parity": logits_parity(plain, logits, exact=True), "hook_audit": hook_audit,
        "input_tensor_sha256": live["input_tensor_sha256"], "archived_input_parity": live["archived_input_parity"],
        "input_metadata": live["input_metadata"], "capture_reused": reference is not None}
    if baseline_failures(row, result) or not result["capture_noop_parity"]["exact_match"]:
        raise RuntimeError("Live baseline parity/category failed; do not skip or replace this rescue.")
    if (row["condition"] == CONDITIONS[0]) != (decision["margin"] < 0):
        raise RuntimeError("Frozen pair no longer satisfies rescue criterion.")
    if reference:
        previous = reference["row"]
        payload = checked_payload(previous, engine.torch)
        result["source_replay"] = {"logits_exact": logits_parity(payload["logits"], logits, exact=True)["passed"],
            "input_bytes_exact": previous["input_tensor_sha256"] == live["input_tensor_sha256"],
            "decision_exact": previous["decision"] == decision,
            "states_exact": all(engine.torch.equal(selected_state(payload["states"], layer, "block_output", positions)["vectors"],
                controller.states[(layer, "block_output")]["vectors"]) for layer in protocol.LAYERS)}
        if not all(result["source_replay"].values()):
            raise RuntimeError("Old capture exact replay failed; preserve it and stop before patching.")
        result.update({key: previous[key] for key in ("capture_index_path", "capture_index_sha256")})
    else:
        path = root / "captures" / row["phase3c_pair_id"] / row["condition"] / "states.pt"
        index_path = path.with_name("index.json")
        if path.exists() or index_path.exists():
            raise ValueError("Uncheckpointed compact capture exists; preserve and diagnose before retry.")
        save_tensors(path, {"states": controller.states, "logits": logits})
        frozen_write(index_path, {"schema": protocol.SCHEMA, "artifact_type": "real", "eval_id": row["eval_id"],
            "plan_fingerprint": plan["fingerprint"], "positions": positions, "site_count": 2,
            "hook_audit": hook_audit, "input_tensor_sha256": live["input_tensor_sha256"],
            "logits_signature": tensor_signature(logits), "vectors_path": str(path.resolve()), "vectors_sha256": file_hash(path)})
        result.update(capture_index_path=str(index_path.resolve()), capture_index_sha256=file_hash(index_path))
    result["passed"] = True
    return result


def summarize(root, stage, plan, tasks, rows):
    output = root / stage
    atomic_write(output / "rows.jsonl", [rows[task["task_id"]] for task in tasks if task["task_id"] in rows], jsonl=True)
    summary = {"schema": protocol.SCHEMA, "plan_fingerprint": plan["fingerprint"], "expected": len(tasks),
        "complete": len(rows) == len(tasks) and all(row["passed"] for row in rows.values()),
        "completed": len(rows), "failed": sum(not row["passed"] for row in rows.values()),
        "missing_task_ids": [task["task_id"] for task in tasks if task["task_id"] not in rows],
        "rows_sha256": file_hash(output / "rows.jsonl"), "manifest_sha256": file_hash(output / "task_manifest.jsonl")}
    atomic_write(output / "summary.json", summary)
    return summary


def run_stage(args, context, stage, engine=None):
    plan, pairs, mappings, refs, prior = context
    root = Path(args.output_root)
    baselines = baseline_lookup(args, context) if stage != "baseline" else None
    if stage == "patch":
        saved_rows(root, "identity", context, baselines, complete=True)
    tasks, rows = saved_rows(root, stage, context, baselines)
    frozen_write(root / stage / "task_manifest.jsonl", tasks, jsonl=True)
    initial = summarize(root, stage, plan, tasks, rows)
    if initial["complete"]:
        print(f"Scale {stage}: all {len(tasks)} verified checkpoints reused, no forwards.", flush=True)
        return engine
    engine = engine or make_engine(args, plan, stage)
    current, live, captures = None, None, {}
    started, computed = time.perf_counter(), 0
    for spec in tasks:
        key, pair_id, condition = spec["task_id"], spec["pair_id"], spec["condition"]
        if key in rows:
            continue
        annotation = pairs[pair_id][condition]
        result = {"task_id": key, "spec": spec, "plan_fingerprint": plan["fingerprint"], "passed": False}
        begun = time.perf_counter()
        print(f"Scale {stage} {len(rows)+1}/{len(tasks)}: {pair_id} {condition} L{spec['layer']}; checkpoint={root/stage/'task_checkpoints'}", flush=True)
        try:
            if current != (pair_id, condition):
                live, captures = None, {}
                gc.collect()
                engine.torch.cuda.empty_cache()
                live = engine.prepare(annotation, mappings[pair_id])
                current = (pair_id, condition)
            if stage == "baseline":
                result.update(capture(engine, live, mappings[pair_id], refs.get(annotation["eval_id"]), root, plan))
            else:
                baseline = baselines[annotation["eval_id"]]
                if live["input_tensor_sha256"] != baseline["input_tensor_sha256"]:
                    raise ValueError("Live intervention input bytes changed.")
                if not captures:
                    captures = {side: checked_payload(baselines[pairs[pair_id][side]["eval_id"]], engine.torch) for side in CONDITIONS}
                result.update(engine.technical(live, mappings[pair_id], {**spec,
                    "kind": "identity" if stage == "identity" else "transplant_smoke"}, captures))
                result.update(spec=spec, baseline_record_sha256=baseline["record_sha256"],
                    input_tensor_sha256=live["input_tensor_sha256"], is_primary_effect_estimate=stage == "patch")
                if stage == "patch":
                    opposite = CONDITIONS[1] if condition == CONDITIONS[0] else CONDITIONS[0]
                    donor = baselines[pairs[pair_id][opposite]["eval_id"]]
                    result.update(donor_baseline_decision=donor["decision"], donor_record_sha256=donor["record_sha256"])
                    result.update(patch_outcomes(result["baseline_decision"]["margin"], result["decision"]["margin"], donor["decision"]["margin"]))
                    direction = "temporal_to_low" if condition == CONDITIONS[0] else "low_to_temporal"
                    previous = prior.get((pair_id, direction, spec["layer"]))
                    result["prior_effect_replayed"] = previous is not None
                result["passed"] = bool(result["passed"])
            result = seal(result)
            validate_row(result, spec, plan, pairs, mappings, baselines, prior)
        except Exception as exc:
            result.update(passed=False, error_type=type(exc).__name__, error_message=str(exc), traceback=traceback.format_exc())
            print(result["traceback"], flush=True)
        result["elapsed_sec"] = time.perf_counter() - begun
        result = seal(result)
        frozen_write(root / stage / "task_checkpoints" / f"{key}.json", result)
        rows[key] = result
        computed += 1
        elapsed = time.perf_counter() - started
        atomic_write(root / stage / "progress_status.json", {"stage": stage, "expected": len(tasks), "completed": len(rows),
            "failed": sum(not row["passed"] for row in rows.values()), "elapsed_sec": elapsed,
            "eta_sec": elapsed/computed * (len(tasks)-len(rows)), "checkpoint": str(root/stage/'task_checkpoints')})
        print(f"Scale {stage}: {len(rows)}/{len(tasks)}; elapsed={elapsed/60:.1f} min; ETA={elapsed/computed*(len(tasks)-len(rows))/60:.1f} min", flush=True)
        summarize(root, stage, plan, tasks, rows)
        if not result["passed"]:
            raise RuntimeError("Scale task failed; checkpoint preserved, no silent retry or sample replacement.")
    return engine


def analyze(args, context):
    plan, pairs, _, _, _ = context
    baselines = baseline_lookup(args, context)
    saved_rows(args.output_root, "identity", context, baselines, complete=True)
    _, patches = saved_rows(args.output_root, "patch", context, baselines, complete=True)
    table = shared.patch_table(patches, pairs)
    for row in table:
        row["cohort_group"] = plan["cohort_groups"][row["pair_id"]]
        row["strict_incorrect_to_correct"] = (row["baseline_margin"] < 0 < row["patched_margin"] and
            row["baseline_prediction"] != row["correct_option"] and row["patched_prediction"] == row["correct_option"])
        row["strict_correct_to_incorrect"] = (row["baseline_margin"] > 0 > row["patched_margin"] and
            row["baseline_prediction"] == row["correct_option"] and row["patched_prediction"] in ("A", "B") and
            row["patched_prediction"] != row["correct_option"])
    metrics = ("margin_delta", "source_aligned_margin_delta", "strict_incorrect_to_correct", "strict_correct_to_incorrect",
               "non_ab_first_token", "zero_margin_tie", "recipient_coverage", "replaced_token_count")
    all_summary = shared.aggregate(table, ("direction", "layer"), metrics)
    groups = shared.aggregate(table, ("cohort_group", "direction", "layer"), metrics)
    role_summary = shared.aggregate(table, ("first_object_id", "direction", "layer"), metrics)
    prompt_summary = shared.aggregate(table, ("prompt_variant", "direction", "layer"), metrics)
    output = Path(args.output_root) / "analysis"
    for name, values in (("case_patch_effects", table), ("all_rescue_summary", all_summary),
                         ("cohort_summary", groups), ("mover_summary", role_summary), ("prompt_summary", prompt_summary)):
        atomic_write(output / f"{name}.json", values)
        shared.write_csv(output / f"{name}.csv", values)
    result = {"schema": protocol.SCHEMA, "artifact_type": "real", "complete": True, "plan_fingerprint": plan["fingerprint"],
        "case_count": 50, "baseline_conditions": 100, "identity_tasks": 8, "patch_rows": 200,
        "prior_effect_replays": sum(row["prior_effect_replayed"] for row in patches.values()),
        "new_effect_rows": sum(not row["prior_effect_replayed"] for row in patches.values()),
        "new_capture_count": sum(not row["capture_reused"] for row in baselines.values()), "knockout_rows": 0,
        "interpretation": "Selective existing rescue cohort; 8 pilot + 8 replication + 34 newly tested. Not 50 new confirmation cases.",
        "output_sha256": {p.name: file_hash(p) for p in output.iterdir() if p.is_file() and p.name != "aggregate_summary.json"}}
    atomic_write(output / "aggregate_summary.json", result)
    generate(Path(args.pilot_run)/"analysis", Path(args.replication_run)/"analysis", output/"plots", output)
    result["output_sha256"].update({str(p.relative_to(output)): file_hash(p) for p in (output/"plots").iterdir() if p.is_file()})
    atomic_write(output / "aggregate_summary.json", result)
    return result


def verify_analysis(args, context):
    baseline = baseline_lookup(args, context)
    for stage in ("identity", "patch"):
        saved_rows(args.output_root, stage, context, baseline, complete=True)
    output = Path(args.output_root)/"analysis"
    summary = read_json(output/"aggregate_summary.json")
    if not summary.get("complete") or summary["plan_fingerprint"] != context[0]["fingerprint"]:
        raise ValueError("Scale analysis not complete or bound to this plan.")
    for name, expected in summary["output_sha256"].items():
        path = (output/name).resolve()
        if not path.is_relative_to(output) or file_hash(path) != expected:
            raise ValueError("Scale reporting bytes changed.")
    if (summary["case_count"], summary["patch_rows"], summary["prior_effect_replays"], summary["new_effect_rows"], summary["new_capture_count"]) != (50, 200, 64, 136, 68):
        raise ValueError("Incorrect expansion/reuse counts.")
    return summary


def storage_gate(args, context):
    estimate = protocol.capture_estimate(context[0], context[2])
    free = shutil.disk_usage(args.storage_root).free
    atomic_write(Path(args.output_root)/"storage_preflight.json", {**estimate, "observed_free_bytes": free})
    if free < estimate["required_free_bytes"]:
        raise ValueError("Storage insufficient for compact captures, full backup and 12 GiB reserve; nothing deleted.")


def services(args):
    forbidden = [getattr(args, name) for name in ("project_root", "source_run", "pilot_plan", "pilot_run", "replication_run", "output_root", "backup_dir")]
    forbidden.append(args.output_root + "_lifecycle")
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


def assert_idle():
    shared.assert_quiescent()
    result = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True, timeout=30)
    active = re.compile(r"(?:^|[ /])run_phase3c_scale\.py(?:\s|$)")
    tmux = shutil.which("tmux")
    for line in result.stdout.splitlines():
        if not active.search(line):
            continue
        pid = line.split(maxsplit=1)[0]
        if pid == str(os.getpid()):
            continue
        try:
            executable = os.readlink(f"/proc/{pid}/exe")
        except OSError:
            executable = None
        if tmux and executable == str(Path(tmux).resolve()):
            continue
        raise RuntimeError("Another scale worker is active; launch/Pause blocked.")


def backup_files(root, reports_only):
    try:
        from .backup_phase3b import backup_files as files_under_root
    except ImportError:
        from backup_phase3b import backup_files as files_under_root
    files = files_under_root(root, reports_only)
    if not reports_only:
        mappings = read_json(Path(root)/"mappings.json")
        videos = {record["video_path"]: record["video_sha256"] for mapping in mappings.values()
                  for record in mapping["processor_records"].values()}
        for number, (name, expected) in enumerate(sorted(videos.items())):
            path = Path(name)
            if path.is_symlink() or file_hash(path) != expected:
                raise ValueError("Selected video changed; full backup/Pause blocked.")
            files[f"source_videos/{number:03d}_{path.name}"] = path
    return files


def command(args, action, stage=None):
    result = [sys.executable, "-u", str(Path(__file__).resolve()), "--action", action, "--stage", stage or args.stage]
    for name in ("source_run", "pilot_plan", "pilot_run", "replication_run", "output_root", "project_root", "storage_root", "gpus", "backup_dir"):
        result.extend([f"--{name}", getattr(args, name)])
    if getattr(args, "attempt_id", None):
        result.extend(["--attempt_id", args.attempt_id])
    if action == "job":
        if args.email_notify:
            result.extend(["--email_notify", "--email_config", args.email_config])
        if args.pause_after:
            result.extend(["--pause_after", "--surf_config", args.surf_config, "--confirm_exclusive_workspace"])
    return result


def job(args):
    root, directory = Path(args.output_root), Path(args.output_root + "_lifecycle")
    started = time.perf_counter()
    notifier = client = None
    status = {"schema": protocol.SCHEMA, "started_at": datetime.now(timezone.utc).isoformat(), "backup_verified": False}

    def save(state, **values):
        status.update(state=state, elapsed_sec=time.perf_counter()-started, **values)
        atomic_write(directory/"job_status.json", status)
        print(f"Scale lifecycle: {state}; elapsed={status['elapsed_sec']/60:.1f} min", flush=True)

    def notify():
        if notifier is None:
            return
        label = "SUCCESS" if status.get("experiment_success") else "FAILED" if "experiment_success" in status else "NOT STARTED"
        body = (f"Experiment: {label}\nPhase 3C whole-Event-2 expansion\nRun: {root}\nState: {status['state']}\n"
            f"Elapsed: {status['elapsed_sec']/3600:.2f} hours\n50 existing rescue bases; L16 primary/L12 comparison, both directions.\n"
            "200 patches (64 exact old-effect replays + 136 new); 68 compact new captures. No KO rerun.\n"
            f"Verified complete new-evidence/video backup: {status['backup_verified']}\nBackup: {status.get('backup_dir','not available')}\n"
            "Retain old Phase 3B/3C activation backups. Download all new parts, manifest and SHA256SUMS and verify locally.\n"
            "Local backup and billing stop are not confirmed. Pause is requested after this email only after verified backup/idle checks.\n"
            "Technical completion only; not a mechanistic conclusion. No credentials or tensors attached.")
        try:
            status["email"] = notifier.send(f"[Phase 3C Scale50] {label}", body)
        except Exception as exc:
            status["email"] = {"error_type": type(exc).__name__}
        atomic_write(directory/"job_status.json", status)

    with shared.lifecycle_lock(directory):
        save("prechecking")
        try:
            notifier, client = services(args)
            context = protocol.load(root)
            storage_gate(args, context)
            assert_idle()
            args.attempt_id = uuid.uuid4().hex
            code = 0
            for stage in ("baseline", "patch"):
                save("experiment_running", stage=stage)
                code = shared.run_child(args, command(args, "run", stage), f"scale_{stage}")
                if code != 0:
                    break
            success = False
            if code == 0:
                try:
                    verify_analysis(args, context)
                    attempt = read_json(root/"last_attempt.json")
                    success = attempt.get("attempt_id") == args.attempt_id and attempt.get("complete") is True
                except Exception as exc:
                    status["verification_error_type"] = type(exc).__name__
            save("experiment_finished", experiment_return_code=code, experiment_success=success)
            with shared.run_lock(root):
                atomic_write(root/"unattended_status.json", status)
                shutil.copyfile(directory/"job.log", root/"unattended_job.log")
                save("backup_running")
                bundle = shared.create_backup(root, args.backup_dir, bundle_prefix="phase3c_backup", file_provider=backup_files,
                    metadata={"schema": "phase3c_scale50_backup_v1", "available_activation_file_count": len(list((root/"captures").rglob("*.pt"))),
                        "selected_videos_included": True, "old_activation_stores_included": False})
                manifest = shared.verify_backup(bundle)
            if not manifest["complete"] or manifest["reports_only"] or manifest["schema"] != "phase3c_scale50_backup_v1":
                raise ValueError("Full backup verification failed.")
            if success and manifest["available_activation_file_count"] != 68:
                raise ValueError("Successful expansion must back up all 68 new compact captures.")
            save("backup_verified", backup_verified=True, backup_dir=str(bundle))
            if args.pause_after:
                with shared.run_lock(root):
                    assert_idle()
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
    parser.add_argument("--action", required=True, choices=("prepare", "run", "verify", "launch", "job"))
    parser.add_argument("--stage", choices=("baseline", "patch"), default="patch")
    for name in ("source_run", "pilot_plan", "pilot_run", "replication_run", "output_root"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--storage_root", default="/data/yuxuanstorage")
    parser.add_argument("--backup_dir", default="/data/yuxuanstorage/backups")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--email_notify", action="store_true")
    parser.add_argument("--email_config", default=shared.EMAIL_CONFIG)
    parser.add_argument("--pause_after", action="store_true")
    parser.add_argument("--surf_config", default=shared.SURF_CONFIG)
    parser.add_argument("--confirm_exclusive_workspace", action="store_true")
    parser.add_argument("--attempt_id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("source_run", "pilot_plan", "pilot_run", "replication_run", "output_root", "project_root", "storage_root", "backup_dir"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    root = Path(args.output_root)
    if not Path(args.storage_root).is_dir() or not root.is_relative_to(args.storage_root) or root == Path(args.storage_root):
        parser.error("Use an isolated directory on existing persistent storage.")
    if not Path(args.backup_dir).is_relative_to(args.storage_root) or Path(args.backup_dir).is_relative_to(root):
        parser.error("Backup must be separate from the run, on persistent storage.")
    if args.action == "job":
        sys.exit(job(args))
    if args.action == "launch":
        context = protocol.load(root)
        storage_gate(args, context)
        services(args)
        assert_idle()
        if not shutil.which("tmux") or subprocess.run(["tmux", "has-session", "-t", "=phase3c_scale50"], capture_output=True).returncode == 0:
            parser.error("tmux unavailable or scale50 session already exists.")
        directory = Path(str(root)+"_lifecycle")
        with shared.lifecycle_lock(directory), shared.run_lock(root):
            shell = f"exec {shlex.join(command(args, 'job'))} >> {shlex.quote(str(directory/'job.log'))} 2>&1"
            subprocess.run(["tmux", "new-session", "-d", "-s", "phase3c_scale50", "-c", args.project_root, shell], check=True)
        print(f"Detached scale50 started; status={directory/'job_status.json'}", flush=True)
        return
    with shared.run_lock(root):
        if args.action == "prepare":
            context = protocol.prepare(args)
            storage_gate(args, context)
            print(json.dumps({k: context[0][k] for k in ("schema", "case_ids", "cohort_groups", "capture_reuse_count", "new_capture_count", "fingerprint")}, indent=2))
        elif args.action == "verify":
            print(json.dumps(verify_analysis(args, protocol.load(root)), indent=2))
        else:
            context = protocol.load(root)
            if args.stage == "baseline":
                run_stage(args, context, "baseline")
            else:
                engine = run_stage(args, context, "identity")
                engine = run_stage(args, context, "patch", engine)
                del engine
                gc.collect()
                analyze(args, context)
                atomic_write(root/"last_attempt.json", {"attempt_id": args.attempt_id or uuid.uuid4().hex, "complete": True,
                                                       "plan_fingerprint": context[0]["fingerprint"]})


if __name__ == "__main__":
    main()
