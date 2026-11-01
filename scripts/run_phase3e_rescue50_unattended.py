"""Detached rescue50 supplementation: preflight, capture, backup, email, Pause."""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_phase3e_rescue50_capture as capture
from scripts.backup_phase3b import create_backup, verify_backup
from scripts.phase3c_core import atomic_write, digest, file_hash, read_json
from scripts.run_phase3c_vm import run_child
from scripts.run_phase3d_vm import lock

SCHEMA = "phase3e_rescue50_supplement_lifecycle_v1"


def directory(args):
    root = Path(args.output_root)
    return root.parent/(root.name+"_lifecycle")


def assert_idle():
    from scripts.run_phase3e_paired_unattended import assert_idle as previous
    previous()
    result = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True, timeout=30)
    pattern = re.compile(r"(?:^|[ /])(?:scripts\.)?run_phase3e_rescue50_(?:capture|unattended)(?:\.py)?(?:\s|$)")
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or int(parts[0]) == os.getpid() or not pattern.search(parts[1]) or "--action launch" in parts[1]:
            continue
        try:
            executable = Path(os.readlink(f"/proc/{parts[0]}/exe")).name
        except OSError:
            raise RuntimeError("Cannot prove rescue50 workspace quiescence.") from None
        if executable != "tmux":
            raise RuntimeError("Another rescue50 capture or CPU worker is active.")


def validate_args(args):
    from scripts.run_phase3e_paired_capture import persistent_volume
    for name in ("output_root", "storage_root", "backup_dir", "project_root"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    root, storage, backup = (Path(getattr(args, n)) for n in ("output_root", "storage_root", "backup_dir"))
    persistent_volume(root, storage)
    if (not backup.is_relative_to(storage) or backup == storage or root == backup or
            root in backup.parents or backup in root.parents or
            not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.session_name) or
            not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", args.gpus) or
            len(set(args.gpus.split(","))) != len(args.gpus.split(","))):
        raise ValueError("Invalid run/backup location, session name or GPU selection.")
    if args.pause_after and not args.confirm_exclusive_workspace:
        raise ValueError("Whole-VM Pause requires exclusive-workspace confirmation.")
    return args


def services(args):
    from scripts.phase3b_email import EmailNotifier, load_private_config as email_config
    from scripts.surf_workspace import SurfClient, load_private_config as surf_config
    forbidden = [args.output_root, args.project_root, args.backup_dir, str(directory(args))]
    notifier = EmailNotifier(email_config(args.email_config, forbidden)) if args.email_notify else None
    client = SurfClient(surf_config(args.surf_config, forbidden)) if args.pause_after else None
    if notifier:
        notifier.check()
    if client:
        client.check(require_pause=True)
    return notifier, client


def precheck(args):
    config, plan, context, _ = capture.input_context(args.output_root)
    capture.space(args.output_root, 1024**3)
    specs = capture.tasks(context, plan)
    for spec in specs:
        capture.source_audit(config["source_root"], context, spec)
    if read_json(Path(args.output_root)/"task_manifest.json") != specs:
        raise ValueError("Supplementation task manifest changed.")
    return config, plan


def commands(args):
    base = [sys.executable, "-u", str(Path(args.project_root)/"scripts/run_phase3e_rescue50_capture.py"),
            "--output_root", args.output_root]
    result = [base+["--action", "run", "--stage", "preflight"]]
    if args.stage == "full":
        result += [base+["--action", "run", "--stage", "full"], base+["--action", "export"],
                   base+["--action", "verify", "--stage", "full"]]
    return result


def message(status, args):
    label = "SUCCESS" if status.get("experiment_success") else "FAILED" if "experiment_success" in status else "NOT STARTED"
    return f"[Phase 3E rescue50] {args.stage} {label}", (
        f"Experiment: {label}\nStage: {args.stage}\nRun: {args.output_root}\n"
        f"Lifecycle state: {status['state']}\nElapsed: {status['elapsed_sec']/3600:.2f} hours\n"
        "Cohort: original 50 rescue pairs, unchanged videos and prompts.\n"
        "Scope: 100 pure A/B-permuted prompt views; four-prompt preflight first.\n"
        "Original 100 prompt states are verified CPU reuse. No new KO or patch.\n"
        f"Capture/export verified: {status.get('experiment_success', False)}\n"
        f"Verified VM backup: {status.get('backup_verified', False)}\nBackup: {status.get('backup_dir', 'not available')}\n"
        f"Log: {directory(args)/'job.log'}\n"
        "Probe fitting has not been performed. A failed run backup can contain partial data.\n"
        "Off-VM backup and Pause/billing stop are not confirmed by this email.\n")


def snapshot(args):
    root = Path(args.output_root)/"code_snapshot"
    files = [(p, Path("scripts")/p.name) for p in sorted((Path(args.project_root)/"scripts").glob("*.py"))]
    files += [(Path(args.project_root)/"docs"/n, Path("docs")/n) for n in
              ("phase3e_rescue50_option_protocol.md", "phase3e_rescue50_recovery.md", "phase3e_rescue50_capture.md")]
    for source, relative in files:
        target = root/relative
        if target.exists():
            if file_hash(target) != file_hash(source):
                raise ValueError("Code snapshot changed.")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)


def job(args, child=run_child, pack=create_backup, check=verify_backup, idle=assert_idle, get_services=services):
    root, started = Path(args.output_root), time.perf_counter()
    status = {"schema": SCHEMA, "stage": args.stage, "backup_verified": False, "output_root": str(root)}
    notifier = client = None

    def save(state, **extra):
        status.update(state=state, elapsed_sec=time.perf_counter()-started, **extra)
        atomic_write(directory(args)/"job_status.json", status)
        print(f"[rescue50 lifecycle] {state}", flush=True)

    def notify():
        if notifier:
            try:
                status["email"] = notifier.send(*message(status, args))
            except Exception as exc:
                status["email"] = {"error_type": type(exc).__name__}
            atomic_write(directory(args)/"job_status.json", status)

    with lock(args.storage_root, ".phase3e_workspace.lock"), lock(directory(args), ".lifecycle.lock"):
        save("prechecking")
        try:
            notifier, client = get_services(args)
            config, _ = precheck(args)
            idle()
            snapshot(args)
            success, code = True, 0
            for index, command in enumerate(commands(args)):
                save("experiment_running", child_index=index)
                try:
                    code = child(args, command, f"rescue50_{index}")
                except Exception as exc:
                    status["child_error_type"] = type(exc).__name__
                    code = 1
                if code:
                    success = False
                    break
            if success:
                try:
                    proof = capture.verify(root, args.stage == "preflight")
                    if args.stage == "full":
                        store = capture.export(root)
                        if not store["complete"] or store["statement_rows"] != 400:
                            raise ValueError("Combined store export is incomplete.")
                    atomic_write(root/"verification.json", proof)
                except Exception as exc:
                    success, code = False, 1
                    status["verification_error_type"] = type(exc).__name__
            save("experiment_finished", experiment_success=success, experiment_return_code=code)
            idle()
            atomic_write(root/"job_status.json", status)
            if (directory(args)/"job.log").exists():
                shutil.copyfile(directory(args)/"job.log", root/"unattended_job.log")
            files = capture.public_files(root)
            capture.space(root, int(1.1*sum(p.stat().st_size for p in files.values())))
            save("backup_running")
            metadata = {"schema": SCHEMA, "capture_config_sha256": digest(config),
                        "stage_verified": success, "requested_stage": args.stage}
            bundle = pack(root, args.backup_dir, file_provider=capture.public_files,
                          bundle_prefix="phase3c_backup", metadata=metadata)
            manifest = check(bundle)
            if (manifest.get("schema") != SCHEMA or manifest.get("source_run_root") != str(root.resolve()) or
                    manifest.get("complete") is not True or manifest.get("reports_only") is not False or
                    manifest.get("activation_tensors_included") is not True or
                    any(manifest.get(k) != v for k, v in metadata.items()) or
                    {f["path"] for f in manifest["files"]} != set(files)):
                raise ValueError("Backup does not cover this supplementation attempt.")
            save("backup_verified", backup_verified=True, backup_dir=str(bundle))
            if args.pause_after:
                idle()
                save("pause_request_pending")
                notify()
                idle()
                save("pause_requested", pause_result=client.request_pause())
            else:
                save("done_without_pause")
                notify()
            return 0 if success else max(1, code)
        except Exception as exc:
            save("needs_attention", error_type=type(exc).__name__)
            notify()
            return 1


def launch(args):
    if not shutil.which("tmux") or subprocess.run(
            ["tmux", "has-session", "-t", "="+args.session_name], capture_output=True).returncode == 0:
        raise RuntimeError("tmux unavailable or session already exists.")
    with lock(args.storage_root, ".phase3e_workspace.lock"), lock(directory(args), ".lifecycle.lock"):
        precheck(args)
        services(args)
        assert_idle()
        snapshot(args)
    invocation = [sys.executable, "-u", str(Path(__file__).resolve()), "--action", "job", "--stage", args.stage,
        "--output_root", args.output_root, "--storage_root", args.storage_root, "--project_root", args.project_root,
        "--backup_dir", args.backup_dir, "--gpus", args.gpus, "--session_name", args.session_name]
    if args.email_notify:
        invocation += ["--email_notify", "--email_config", args.email_config]
    if args.pause_after:
        invocation += ["--pause_after", "--surf_config", args.surf_config, "--confirm_exclusive_workspace"]
    shell = f"exec {shlex.join(invocation)} >> {shlex.quote(str(directory(args)/'job.log'))} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", args.session_name, "-c", args.project_root, shell], check=True)
    return {"status": "detached_job_started", "stage": args.stage, "session": args.session_name,
            "log": str(directory(args)/"job.log")}


def main():
    from scripts.phase3b_email import DEFAULT_CONFIG as EMAIL
    from scripts.surf_workspace import DEFAULT_CONFIG as SURF
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("launch", "job"), required=True)
    parser.add_argument("--stage", choices=("preflight", "full"), default="full")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--storage_root", default="/data/yuxuanstorage")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--backup_dir", default="/data/yuxuanstorage/backups")
    parser.add_argument("--session_name", default="phase3e-rescue50-supplement")
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


if __name__ == "__main__":
    main()
