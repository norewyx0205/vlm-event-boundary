"""Wrap a VM stage with verified backup and optional, explicit SURF Pause."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

try:
    from .backup_phase3b import create_backup, verify_backup
    from .common import PROJECT_ROOT
    from .run_phase3b_vm import run_lock, write_json
    from .surf_workspace import DEFAULT_CONFIG, SurfClient, check_private_location, load_private_config
except ImportError:
    from backup_phase3b import create_backup, verify_backup
    from common import PROJECT_ROOT
    from run_phase3b_vm import run_lock, write_json
    from surf_workspace import DEFAULT_CONFIG, SurfClient, check_private_location, load_private_config


def add_job_arguments(parser):
    parser.add_argument("--stage", choices=("preflight", "full", "analyze"), default="full")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--storage_root", default="/data/yuxuanstorage")
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--backup_dir", default="/data/yuxuanstorage/backups")
    parser.add_argument("--pause_policy", choices=("off", "success", "finished"), default="off",
                        help="success: pause only after success; finished: also after failure. Both require a verified full backup.")
    parser.add_argument("--surf_config", default=DEFAULT_CONFIG)
    parser.add_argument("--confirm_exclusive_workspace", action="store_true",
                        help="Confirm this is the intended VM and no colleagues/other jobs need it to stay running.")
    parser.add_argument("runner_args", nargs=argparse.REMAINDER, help="Pass VM runner options after --.")


def lifecycle_directory(args):
    root = Path(args.output_root)
    return root.parent / (root.name + "_lifecycle")


def validate_job_args(args):
    for name in ("output_root", "storage_root", "project_root", "backup_dir"):
        setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
    args.surf_config = str(Path(args.surf_config).expanduser().absolute())
    for name in ("output_root", "backup_dir"):
        if not Path(getattr(args, name)).is_relative_to(args.storage_root):
            raise ValueError("Run artifacts and backups must be on persistent --storage_root.")
    if Path(args.backup_dir).is_relative_to(args.output_root):
        raise ValueError("Backups must be outside the run directory.")
    if not Path(args.project_root, "scripts/run_phase3b_vm.py").is_file():
        raise FileNotFoundError("Set --project_root to the VM repository.")
    if args.pause_policy != "off":
        if not args.confirm_exclusive_workspace:
            raise ValueError("Automatic Pause requires --confirm_exclusive_workspace; it stops the whole VM, not just your GPU job.")
        check_private_location(args.surf_config, (args.project_root, args.output_root, args.backup_dir, lifecycle_directory(args)))
    forwarded = args.runner_args[1:] if args.runner_args[:1] == ["--"] else args.runner_args
    reserved = {"--stage", "--output_root", "--storage_root", "--project_root"}
    if any(item.split("=", 1)[0] in reserved for item in forwarded):
        raise ValueError("Set stage/output/storage/project only on this wrapper, not again after --.")
    args.runner_args = forwarded
    return args


def vm_command(args):
    return [sys.executable, "-u", "scripts/run_phase3b_vm.py", "--stage", args.stage,
            "--output_root", args.output_root, "--storage_root", args.storage_root,
            "--project_root", args.project_root, *args.runner_args]


@contextmanager
def lifecycle_lock(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)
    with Path(directory, ".unattended.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("An unattended job already owns this run. Do not launch a duplicate.") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def assert_quiescent():
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30, check=True)
    if gpu.stdout.strip():
        raise RuntimeError("GPU compute processes still exist; automatic Pause is blocked.")
    processes = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, timeout=30, check=True)
    active = re.compile(r"(?:^|[ /])(?:run_phase3b_vm|run_phase3b_patching|run_phase3b_relocation_control|run_phase3b_screening|run_eval|probe_attention_roi)\.py(?:\s|$)")
    if any(active.search(line) for line in processes.stdout.splitlines()):
        raise RuntimeError("Another research runner is active; automatic Pause is blocked.")


def run_child(args):
    env = os.environ.copy()
    for name in ("RESEARCH_CLOUD_TOKEN", "SURF_API_TOKEN"):
        env.pop(name, None)
    process = subprocess.Popen(vm_command(args), cwd=args.project_root, env=env, start_new_session=True)
    previous = {}
    def terminate(signum, _frame):
        # Propagate SIGINT so the VM orchestrator can cancel its separate GPU groups.
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, terminate)
    try:
        return process.wait()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def execute_job(args, client=None, child=run_child, backup=create_backup, verify=verify_backup, idle_check=assert_quiescent):
    directory = lifecycle_directory(args)
    with lifecycle_lock(directory):
        # A running direct CLI/notebook invocation must not be adopted or duplicated.
        with run_lock(args.output_root):
            pass
        status = {
            "schema": "phase3b_unattended_v1", "stage": args.stage, "output_root": args.output_root,
            "started_at": datetime.now(timezone.utc).isoformat(), "pause_policy": args.pause_policy,
            "wrapper_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "state": "prechecking", "backup_verified": False, "local_backup_confirmed": False,
            "wrapper_pid": os.getpid(), "workspace": None,
        }
        started = time.perf_counter()
        path = directory / "job_status.json"
        write_json(path, status)
        def save(state, **fields):
            status.update(state=state, elapsed_sec=time.perf_counter() - started, **fields)
            write_json(path, status)
            print(f"[lifecycle] {state}; elapsed={status['elapsed_sec']/60:.1f} min; status={path}", flush=True)
        try:
            if args.pause_policy != "off":
                client = client or SurfClient(load_private_config(args.surf_config, (
                    args.project_root, args.output_root, args.backup_dir, directory,
                )))
                workspace = client.check(require_pause=True)
                idle_check()
            else:
                workspace = None
            save("experiment_running", workspace=workspace)
            return_code = child(args)
            saved = Path(args.output_root, "vm_last_status.json")
            run_status = json.loads(saved.read_text()) if saved.is_file() else {}
            success = return_code == 0 and run_status.get("complete") is True and run_status.get("stage") == args.stage
            save("experiment_finished", experiment_return_code=return_code, experiment_success=success)
            if not Path(args.output_root, "vm_run_config.json").is_file():
                raise RuntimeError("No VM run configuration exists; no verified research backup can be made.")
            with run_lock(args.output_root):
                write_json(Path(args.output_root, "unattended_run_status.json"), status)
                log = directory / "job.log"
                if log.is_file():
                    # Snapshot only after workers stop; later Pause messages stay in the sibling journal.
                    shutil.copyfile(log, Path(args.output_root, "unattended_job.log"))
            save("backup_running")
            bundle = backup(args.output_root, args.backup_dir, reports_only=False)
            manifest = verify(bundle)
            if manifest.get("reports_only") or not manifest.get("activation_tensors_included") or not manifest.get("complete"):
                raise RuntimeError("A verified full backup is required; reports-only/incomplete exports cannot authorize Pause.")
            save("backup_verified", backup_dir=str(bundle), backup_verified=True)
            eligible = args.pause_policy == "finished" or (args.pause_policy == "success" and success)
            if not eligible:
                save("done_without_pause", pause_skipped_reason="disabled_or_experiment_failed")
            else:
                idle_check()
                with run_lock(args.output_root):
                    save("pause_request_pending")
                    result = client.request_pause()
                    save("pause_requested", pause_result=result,
                         note="Verify paused in the SURF portal; request acceptance alone does not confirm billing has stopped.")
            return 0 if success else (return_code if return_code > 0 else 1)
        except Exception as exc:
            # Exceptions from API code are sanitized before reaching this public record.
            save("needs_attention", error=str(exc), pause_not_confirmed=True)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_job_arguments(parser)
    args = validate_job_args(parser.parse_args())
    sys.exit(execute_job(args))


if __name__ == "__main__":
    main()
