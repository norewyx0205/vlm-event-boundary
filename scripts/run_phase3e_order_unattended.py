"""tmux order-reversal capture, checkpoint backup, completion email and idle Pause."""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_phase3e_order_capture as capture
from scripts.backup_phase3b import create_backup, verify_backup
from scripts.phase3c_core import atomic_write, digest, read_json
from scripts.run_phase3c_vm import run_child
from scripts.run_phase3d_vm import lock

SCHEMA = "phase3e_rescue50_direct_order_lifecycle_v1"


def directory(args):
    root = Path(args.output_root)
    return root.parent/(root.name+"_lifecycle")


def idle():
    from scripts.run_phase3e_rescue50_unattended import assert_idle
    assert_idle()
    output = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True, timeout=30)
    pattern = re.compile(r"(?:^|[ /])(?:scripts\.)?(?:run_phase3e_order_(?:capture|probes|unattended)|phase3e_order_materials)(?:\.py)?(?:\s|$)")
    for line in output.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or int(fields[0]) == os.getpid() or not pattern.search(fields[1]) or "--action launch" in fields[1]:
            continue
        try:
            executable = Path(os.readlink(f"/proc/{fields[0]}/exe")).name
        except OSError:
            raise RuntimeError("Cannot prove order-study workspace quiescence.") from None
        if executable != "tmux":
            raise RuntimeError("Another order-study worker is active; Pause is blocked.")


def validate_args(args):
    from scripts.run_phase3e_paired_capture import persistent_volume
    for name in ("output_root", "storage_root", "project_root", "backup_dir"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    root, storage, backup = (Path(getattr(args, n)) for n in ("output_root", "storage_root", "backup_dir"))
    persistent_volume(root, storage)
    if (not backup.is_relative_to(storage) or backup == storage or root == backup or
            root in backup.parents or backup in root.parents or
            not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.session_name) or
            not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", args.gpus) or len(set(args.gpus.split(","))) != len(args.gpus.split(","))):
        raise ValueError("Invalid persistent paths, session name or GPU selection.")
    if args.pause_after and not args.confirm_exclusive_workspace:
        raise ValueError("Whole-VM Pause requires exclusive-workspace confirmation.")
    return args


def services(args):
    from scripts.phase3b_email import EmailNotifier, load_private_config as email_config
    from scripts.surf_workspace import SurfClient, load_private_config as surf_config
    forbidden = [args.project_root, args.output_root, args.backup_dir, str(directory(args))]
    notifier = EmailNotifier(email_config(args.email_config, forbidden)) if args.email_notify else None
    client = SurfClient(surf_config(args.surf_config, forbidden)) if args.pause_after else None
    if notifier:
        notifier.check()
    if client:
        client.check(require_pause=True)
    return notifier, client


def commands(args):
    base = [sys.executable, "-u", str(Path(args.project_root)/"scripts/run_phase3e_order_capture.py"),
            "--output_root", args.output_root]
    result = [base+["--action", "run", "--stage", "preflight"]]
    if args.stage == "full":
        result += [base+["--action", "run", "--stage", "full"], base+["--action", "export"],
                   base+["--action", "verify", "--stage", "full"]]
    return result


def snapshot(args):
    target = Path(args.output_root)/"code_snapshot"
    files = [(p, Path("scripts")/p.name) for p in Path(args.project_root, "scripts").glob("*.py")]
    files.append((Path(args.project_root)/"docs/phase3e_paired_difference_training.md",
                  Path("docs/phase3e_paired_difference_training.md")))
    from scripts.phase3c_core import file_hash
    for source, relative in files:
        path = target/relative
        if path.exists():
            if file_hash(path) != file_hash(source):
                raise ValueError("Order-study code snapshot changed.")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, path)


def message(status, args):
    label = "SUCCESS" if status.get("experiment_success") else "FAILED" if status.get("experiment_started") else "NOT STARTED"
    return f"[Phase 3E direct D] {args.stage} capture {label}", (
        f"Experiment: {label}\nRun: {args.output_root}\nStage: {args.stage}\n"
        f"Started UTC: {status['started_utc']}\nElapsed: {status['elapsed_sec']/3600:.2f} hours\n"
        "Cohort: same original 50 rescue cases; no new cases or performance screening.\n"
        "New material: motion-order-reversed partners with unchanged object paths and option text.\n"
        "New scope: 200 prompt endpoints, eight-prompt real-model preflight first. No new KO or patch.\n"
        "Primary next CPU fit uses D = h(original order) - h(reversed order), with opposite-label -D views.\n"
        "This email reports capture/export status, NOT completed direct-D probe fitting.\n"
        f"Verified full VM backup: {status.get('backup_verified', False)}\n"
        f"Backup: {status.get('backup_dir', 'not available')}\n"
        f"Status: {directory(args)/'job_status.json'}\nLog: {directory(args)/'job.log'}\n"
        "Off-VM copy is not confirmed. Keep prior full backups.\n"
        "Pause is requested only after worker completion and an idle check; check the portal for final status.\n"
        "No credentials, videos or activation tensors are attached.\n")


def job(args, child=run_child, get_services=services, check_idle=idle, pack=create_backup, check=verify_backup):
    started, root = time.perf_counter(), Path(args.output_root)
    status = {"schema": SCHEMA, "started_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage, "backup_verified": False, "experiment_started": False, "experiment_success": False}
    notifier = client = None

    def save(state, **values):
        status.update(state=state, elapsed_sec=time.perf_counter()-started, **values)
        atomic_write(directory(args)/"job_status.json", status)
        print("[direct-D lifecycle]", state, flush=True)

    with lock(args.storage_root, ".phase3e_workspace.lock"), lock(directory(args), ".lifecycle.lock"):
        save("prechecking")
        try:
            notifier, client = get_services(args)
            config, _, _, _ = capture.context(root)
            capture.space(root)
            check_idle()
            snapshot(args)
            code = 0
            for index, command in enumerate(commands(args)):
                save("experiment_running", experiment_started=True, child_index=index)
                code = child(args, command, "order_"+str(index))
                if code:
                    break
            if code == 0:
                capture.verify(root, args.stage == "preflight")
                if args.stage == "full":
                    capture.export(root)
            save("experiment_finished", experiment_success=code == 0, experiment_return_code=code)
        except Exception as exc:
            save("needs_attention", error_type=type(exc).__name__, experiment_success=False, experiment_return_code=1)
        try:
            check_idle()
            atomic_write(root/"job_status.json", status)
            logfile = directory(args)/"job.log"
            if logfile.exists():
                shutil.copyfile(logfile, root/"unattended_job.log")
            files = capture.public_files(root)
            metadata = {"schema": SCHEMA, "requested_stage": args.stage, "stage_verified": status["experiment_success"]}
            bundle = pack(root, args.backup_dir, file_provider=capture.public_files, bundle_prefix="phase3c_backup", metadata=metadata)
            manifest = check(bundle)
            if (manifest.get("complete") is not True or manifest.get("reports_only") is not False or
                    manifest.get("activation_tensors_included") is not True or
                    manifest.get("source_run_root") != str(root.resolve()) or
                    any(manifest.get(k) != v for k, v in metadata.items()) or
                    {f["path"] for f in manifest["files"]} != set(files)):
                raise ValueError("Order backup coverage or metadata changed.")
            save("backup_verified", backup_verified=True, backup_dir=str(bundle))
        except Exception as exc:
            save("backup_needs_attention", backup_error_type=type(exc).__name__)
        if notifier:
            try:
                status["email"] = notifier.send(*message(status, args))
            except Exception as exc:
                status["email"] = {"error_type": type(exc).__name__}
        if args.pause_after and client:
            try:
                check_idle()
                save("pause_request_pending")
                save("pause_requested", pause_result=client.request_pause())
            except Exception as exc:
                save("pause_needs_attention", pause_error_type=type(exc).__name__)
        else:
            save("done_without_pause")
    pause_ok = not args.pause_after or "pause_result" in status
    return 0 if status["experiment_success"] and status["backup_verified"] and pause_ok else 1


def launch(args):
    if not shutil.which("tmux") or subprocess.run(["tmux", "has-session", "-t", "="+args.session_name], capture_output=True).returncode == 0:
        raise RuntimeError("tmux missing or requested session already active.")
    with lock(args.storage_root, ".phase3e_workspace.lock"), lock(directory(args), ".lifecycle.lock"):
        services(args)
        idle()
        capture.context(args.output_root)
        capture.space(args.output_root)
        snapshot(args)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--action", "job", "--stage", args.stage]
    for key in ("output_root", "storage_root", "project_root", "backup_dir", "gpus", "session_name"):
        command += ["--"+key, str(getattr(args, key))]
    if args.email_notify:
        command += ["--email_notify", "--email_config", args.email_config]
    if args.pause_after:
        command += ["--pause_after", "--surf_config", args.surf_config, "--confirm_exclusive_workspace"]
    shell = f"exec {shlex.join(command)} >> {shlex.quote(str(directory(args)/'job.log'))} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", args.session_name, "-c", args.project_root, shell], check=True)
    return {"status": "detached_job_started", "session": args.session_name, "log": str(directory(args)/"job.log")}


if __name__ == "__main__":
    from scripts.phase3b_email import DEFAULT_CONFIG as EMAIL
    from scripts.surf_workspace import DEFAULT_CONFIG as SURF
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("launch", "job"), required=True)
    parser.add_argument("--stage", choices=("preflight", "full"), default="full")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--storage_root", default="/data/yuxuanstorage")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--backup_dir", default="/data/yuxuanstorage/backups")
    parser.add_argument("--session_name", default="phase3e-direct-D")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--email_notify", action="store_true")
    parser.add_argument("--pause_after", action="store_true")
    parser.add_argument("--confirm_exclusive_workspace", action="store_true")
    parser.add_argument("--email_config", default=EMAIL)
    parser.add_argument("--surf_config", default=SURF)
    args = validate_args(parser.parse_args())
    if args.action == "launch":
        print(json.dumps(launch(args), indent=2))
    else:
        sys.exit(job(args))
