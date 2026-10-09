"""Launch one isolated Phase 3C lifecycle in tmux, with explicit opt-in side effects."""

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from .phase3c_core import atomic_write
    from .run_phase3b_vm import run_lock
    from .run_phase3c_unattended import (EmailNotifier, SurfClient, add_job_arguments, forbidden_locations,
        lifecycle_directory, lifecycle_lock, load_email_config, load_private_config, validate_job_args, vm_options)
    from .run_phase3c_vm import build_plan, vm_lock
except ImportError:
    from phase3c_core import atomic_write
    from run_phase3b_vm import run_lock
    from run_phase3c_unattended import (EmailNotifier, SurfClient, add_job_arguments, forbidden_locations,
        lifecycle_directory, lifecycle_lock, load_email_config, load_private_config, validate_job_args, vm_options)
    from run_phase3c_vm import build_plan, vm_lock


def launch_command(args):
    command = [sys.executable, "-u", str(Path(args.project_root) / "scripts/run_phase3c_unattended.py"),
               *vm_options(args), "--backup_dir", args.backup_dir, "--pause_policy", args.pause_policy]
    if args.email_notify:
        command.extend(["--email_notify", "--email_config", args.email_config])
    if args.pause_policy != "off":
        command.extend(["--surf_config", args.surf_config, "--confirm_exclusive_workspace"])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_job_arguments(parser)
    parser.add_argument("--session_name", default="phase3c_full")
    parser.add_argument("--dry_run", action="store_true", help="Validate local artifacts and print commands; no API, GPU, tmux or writes.")
    args = validate_job_args(parser.parse_args())
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.session_name):
        parser.error("Use 1-64 letters, digits, underscores or hyphens for the tmux session.")
    directory, command = lifecycle_directory(args), launch_command(args)
    if args.dry_run:
        config, steps = build_plan(args)
        print(json.dumps({"session": args.session_name, "command": command, "steps": steps,
                          "vm_fingerprint": config["vm_fingerprint"], "log": str(directory / "job.log")}, indent=2))
        return
    if not shutil.which("tmux"):
        parser.error("Install tmux on the VM first.")
    if subprocess.run(["tmux", "has-session", "-t", "=" + args.session_name], capture_output=True, check=False).returncode == 0:
        parser.error("Session already exists; attach to it instead of starting a duplicate.")
    with lifecycle_lock(directory), vm_lock(args.output_root), run_lock(args.output_root):
        config, steps = build_plan(args)
        if args.email_notify:
            EmailNotifier(load_email_config(args.email_config, forbidden_locations(args))).check()
        if args.pause_policy != "off":
            SurfClient(load_private_config(args.surf_config, forbidden_locations(args))).check(require_pause=True)
        atomic_write(directory / "launch_plan.json", {"session": args.session_name, "command": command,
                    "steps": steps, "vm_fingerprint": config["vm_fingerprint"]})
    shell = f"exec {shlex.join(command)} >> {shlex.quote(str(directory / 'job.log'))} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", args.session_name, "-c", args.project_root, shell], check=True)
    print(f"Detached job started. Attach: tmux attach -t {args.session_name}")
    print(f"Log: {directory / 'job.log'}\nStatus: {directory / 'job_status.json'}")
    print("Closing the browser/computer is safe; do NOT manually Pause the VM while work is active. "
          "VM packaging is not a locally verified backup.")


if __name__ == "__main__":
    main()
