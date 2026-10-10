"""Run an explicit Phase 3C stage, verify backup, notify, then optionally Pause."""

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from .backup_phase3c import create_backup, verify_backup
    from .phase3b_email import DEFAULT_CONFIG as EMAIL_CONFIG, EmailNotifier, load_private_config as load_email_config
    from .phase3c_core import atomic_write, file_hash, read_json
    from .run_phase3b_unattended import assert_quiescent as legacy_idle, lifecycle_directory, lifecycle_lock
    from .run_phase3b_vm import run_lock
    from .run_phase3c_vm import STAGES, add_arguments, build_plan, normalized_args, verify_stage, vm_lock
    from .surf_workspace import DEFAULT_CONFIG as SURF_CONFIG, SurfClient, check_private_location, load_private_config
except ImportError:
    from backup_phase3c import create_backup, verify_backup
    from phase3b_email import DEFAULT_CONFIG as EMAIL_CONFIG, EmailNotifier, load_private_config as load_email_config
    from phase3c_core import atomic_write, file_hash, read_json
    from run_phase3b_unattended import assert_quiescent as legacy_idle, lifecycle_directory, lifecycle_lock
    from run_phase3b_vm import run_lock
    from run_phase3c_vm import STAGES, add_arguments, build_plan, normalized_args, verify_stage, vm_lock
    from surf_workspace import DEFAULT_CONFIG as SURF_CONFIG, SurfClient, check_private_location, load_private_config


def add_job_arguments(parser):
    add_arguments(parser)
    parser.set_defaults(stage="full")
    parser.add_argument("--backup_dir", default="/data/yuxuanstorage/backups")
    parser.add_argument("--pause_policy", choices=("off", "success", "finished"), default="off")
    parser.add_argument("--surf_config", default=SURF_CONFIG)
    parser.add_argument("--email_notify", action="store_true")
    parser.add_argument("--email_config", default=EMAIL_CONFIG)
    parser.add_argument("--confirm_exclusive_workspace", action="store_true",
                        help="Authorize Pause of this entire VM; no colleagues or other jobs need it running.")


def forbidden_locations(args):
    return (args.project_root, args.plan_dir, args.output_root, args.backup_dir, lifecycle_directory(args))


def validate_job_args(args):
    normalized_args(args)
    if args.stage not in STAGES:
        raise ValueError("Use the VM plan CLI for planning, or the launcher --dry_run. A lifecycle requires an explicit stage.")
    args.backup_dir = str(Path(args.backup_dir).expanduser().resolve())
    backup = Path(args.backup_dir)
    if not backup.is_relative_to(args.storage_root):
        raise ValueError("Backups must be on persistent storage.")
    for path in (args.output_root, args.plan_dir, args.project_root, lifecycle_directory(args)):
        if backup.is_relative_to(path) or Path(path).is_relative_to(backup):
            raise ValueError("Use a separate sibling backup directory, not a preparation/repository/run/lifecycle directory.")
    for name in ("email_config", "surf_config"):
        setattr(args, name, str(Path(getattr(args, name)).expanduser().absolute()))
    if args.email_notify:
        check_private_location(args.email_config, forbidden_locations(args))
    if args.pause_policy != "off":
        if not args.confirm_exclusive_workspace:
            raise ValueError("Pause stops the whole VM; --confirm_exclusive_workspace is required.")
        check_private_location(args.surf_config, forbidden_locations(args))
    return args


def vm_options(args):
    options = ["--stage", args.stage, "--plan_dir", args.plan_dir, "--output_root", args.output_root,
        "--storage_root", args.storage_root, "--project_root", args.project_root, "--gpus", args.gpus,
        "--execution_mode", args.execution_mode, "--bootstrap_repeats", str(args.bootstrap_repeats)]
    parser = argparse.ArgumentParser(add_help=False)
    add_arguments(parser)
    default_budget = parser.get_default("gpu_weight_budget_gib")
    # JSON fingerprints distinguish an omitted default (10) from parsed "10" (10.0).
    if (type(args.gpu_weight_budget_gib) is not type(default_budget) or
            args.gpu_weight_budget_gib != default_budget):
        options.extend(["--gpu_weight_budget_gib", str(args.gpu_weight_budget_gib)])
    if args.path_map:
        options.extend(["--path_map", args.path_map])
    if args.retry_failed:
        options.append("--retry_failed")
    if args.no_plots:
        options.append("--no_plots")
    return options


def vm_command(args):
    return [sys.executable, "-u", str(Path(args.project_root) / "scripts/run_phase3c_vm.py"), *vm_options(args)]


def assert_quiescent():
    legacy_idle()
    processes = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, timeout=30, check=True)
    active = re.compile(r"(?:^|[ /])(?:run_phase3c_vm|run_phase3c_preflight|run_phase3c|audit_phase3c_mappings|"
                        r"prepare_phase3c|analyze_phase3c|backup_phase3c|run_phase3c_unattended|"
                        r"run_phase3b_unattended|launch_phase3c_vm|launch_phase3b_vm)\.py(?:\s|$)")
    tmux = shutil.which("tmux")
    tmux_executable = str(Path(tmux).resolve()) if tmux else None
    for line in processes.stdout.splitlines():
        if not active.search(line):
            continue
        pid = line.split(maxsplit=1)[0]
        if pid == str(os.getpid()):
            continue
        # A tmux server's argv contains its launch command, not a live Python worker.
        try:
            executable = os.readlink(f"/proc/{pid}/exe")
        except OSError:
            executable = None
        if tmux_executable is not None and executable == tmux_executable:
            continue
        # Unknown/unreadable executables remain fail-closed.
        raise RuntimeError("Another Phase 3C job is active; automatic Pause is blocked.")


def run_child(args):
    env = os.environ.copy()
    for name in ("RESEARCH_CLOUD_TOKEN", "SURF_API_TOKEN"):
        env.pop(name, None)
    process = subprocess.Popen(vm_command(args), cwd=args.project_root, env=env, start_new_session=True)
    previous, cancelled = {}, False
    def stop(signum, _frame):
        nonlocal cancelled
        cancelled = True
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, stop)
    try:
        code = process.wait()
        return 130 if cancelled else code
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def completion_message(status, args):
    success = status.get("experiment_success", False)
    experiment = "SUCCESS" if success else "FAILED" if "experiment_success" in status else "NOT STARTED"
    outcome = "NEEDS ATTENTION" if status["state"] == "needs_attention" else experiment
    directory = lifecycle_directory(args)
    subject = f"[Phase 3C] {args.stage} {outcome} - {Path(args.output_root).name}"
    body = [f"Experiment: {experiment}", f"Stage: {args.stage}", f"Lifecycle state: {status['state']}",
        f"Exit code: {status.get('experiment_return_code', 'not started')}",
        f"Started (UTC): {status['started_at']}", f"Elapsed: {status.get('elapsed_sec', 0) / 3600:.2f} hours",
        "Frozen cohort: 8 rescue + 4 stable controls (12 independent bases; 24 condition rows)",
        f"Run: {args.output_root}", f"Log: {directory / 'job.log'}", f"Status: {directory / 'job_status.json'}",
        f"Verified VM backup: {status.get('backup_verified', False)}", f"Backup: {status.get('backup_dir', 'not available')}",
        "Local backup is NOT confirmed: download every part, manifest and SHA256SUMS; verify on your computer.",
        "Retain the separate frozen Phase 3B backup. The Phase 3C package does not duplicate all Phase 3B activations."]
    if args.stage in ("full", "analyze") and success:
        body.extend(["Complete fixed grid verified: 696 patches, 24 intact routing diagnostics, 2592 knockout forwards.",
                     f"Analysis: {Path(args.output_root) / 'analysis/aggregate_summary.json'}"])
    else:
        body.append("This message does NOT establish a completed Phase 3C pilot. A stage gate or partial run is not the full result.")
    if args.pause_policy == "off":
        body.append("Automatic Pause is disabled. The VM may still be running and charging.")
    else:
        body.append("Pause/billing stop is NOT confirmed by this email. Check the SURF portal. "
                    "A pending request is made only after verified full backup and a quiescence check.")
    if status.get("state") == "needs_attention":
        body.append(f"Lifecycle error type: {status.get('error_type')}; inspect public logs. No automatic Pause was confirmed.")
    body.append("Technical completion summary only; not a scientific interpretation. No credentials, videos or activations are attached.")
    return subject, "\n".join(body)


def verify_attempt(args, previous_attempt):
    record = read_json(Path(args.output_root) / "vm_last_status.json")
    config = read_json(Path(args.output_root) / "phase3c_vm_config.json")
    _, steps = build_plan(args)
    if (not record.get("attempt_id") or record["attempt_id"] == previous_attempt or
            not record.get("complete") or record.get("stage") != args.stage or
            record.get("vm_fingerprint") != config["vm_fingerprint"] or record.get("completed_steps") != steps):
        raise ValueError("No new, independently verified successful attempt; old reports cannot establish success.")
    for stage in steps:
        verify_stage(args, stage)
    return record


def execute_job(args, client=None, child=run_child, backup=create_backup, verify=verify_backup,
                idle_check=assert_quiescent, notifier=None):
    directory, root = lifecycle_directory(args), Path(args.output_root)
    with lifecycle_lock(directory):
        with vm_lock(root), run_lock(root):
            build_plan(args)
        prior = read_json(root / "vm_last_status.json").get("attempt_id") if (root / "vm_last_status.json").exists() else None
        started = time.perf_counter()
        status = {"schema": "phase3c_unattended_v1", "artifact_type": "real", "stage": args.stage,
            "output_root": args.output_root, "started_at": datetime.now(timezone.utc).isoformat(),
            "wrapper_code_sha256": file_hash(__file__), "wrapper_pid": os.getpid(), "pause_policy": args.pause_policy,
            "state": "prechecking", "backup_verified": False, "local_backup_confirmed": False, "email_notifications": {}}
        path = directory / "job_status.json"
        def save(state, **fields):
            status.update(state=state, elapsed_sec=time.perf_counter() - started, **fields)
            atomic_write(path, status)
            print(f"[Phase 3C lifecycle] {state}; elapsed={status['elapsed_sec']/60:.1f} min; status={path}", flush=True)
        def notify(event):
            if not args.email_notify or event in status["email_notifications"]:
                return
            record = {"state": "attempting", "attempted_at": datetime.now(timezone.utc).isoformat()}
            status["email_notifications"][event] = record
            atomic_write(path, status)
            try:
                subject, body = completion_message(status, args)
                record.update(notifier.send(subject, body), state="smtp_accepted")
            except Exception as exc:
                record.update(state="failed_or_unconfirmed", error_type=type(exc).__name__)
            atomic_write(path, status)
        save("prechecking")
        try:
            if args.email_notify:
                notifier = notifier or EmailNotifier(load_email_config(args.email_config, forbidden_locations(args)))
                notifier.check()
            if args.pause_policy != "off":
                client = client or SurfClient(load_private_config(args.surf_config, forbidden_locations(args)))
                status["workspace"] = client.check(require_pause=True)
                idle_check()
            save("experiment_running")
            code = child(args)
            attempt = read_json(root / "vm_last_status.json")
            if not attempt.get("attempt_id") or attempt["attempt_id"] == prior or attempt.get("stage") != args.stage:
                raise ValueError("The child did not record a new attempt; stale run status cannot authorize Pause.")
            success = False
            if code == 0:
                try:
                    verified = verify_attempt(args, prior)
                    success = True
                    status["attempt_id"] = verified["attempt_id"]
                except Exception as exc:
                    status["verification_error_type"] = type(exc).__name__
            save("experiment_finished", experiment_return_code=code, experiment_success=success)
            with vm_lock(root), run_lock(root):
                atomic_write(root / "unattended_run_status.json", status)
                if (directory / "job.log").is_file():
                    shutil.copyfile(directory / "job.log", root / "unattended_job.log")
            save("backup_running")
            bundle = backup(root, args.backup_dir, reports_only=False)
            manifest = verify(bundle)
            if (manifest.get("schema") != "phase3c_local_backup_v1" or not manifest.get("complete") or
                    manifest.get("reports_only") or not manifest.get("activation_tensors_included") or
                    manifest.get("vm_fingerprint") != read_json(root / "phase3c_vm_config.json")["vm_fingerprint"] or
                    manifest.get("run_status") != read_json(root / "vm_last_status.json")):
                raise ValueError("Only a verified full backup of this stopped attempt can authorize Pause.")
            save("backup_verified", backup_verified=True, backup_dir=str(bundle))
            eligible = args.pause_policy == "finished" or (args.pause_policy == "success" and success)
            if eligible:
                with vm_lock(root), run_lock(root):
                    idle_check()
                    save("pause_request_pending")
                    notify("completion")
                    # The VM may disconnect immediately. Never retry an ambiguous Pause response.
                    result = client.request_pause()
                    save("pause_requested", pause_result=result, billing_stop_confirmed=result.get("billing_stop_confirmed", False))
            else:
                save("done_without_pause")
                notify("completion")
            return 0 if success else code if code > 0 else 1
        except Exception as exc:
            save("needs_attention", error_type=type(exc).__name__, pause_not_confirmed=True)
            notify("attention")
            return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_job_arguments(parser)
    args = validate_job_args(parser.parse_args())
    sys.exit(execute_job(args))


if __name__ == "__main__":
    main()
