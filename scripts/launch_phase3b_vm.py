"""Start one detached tmux job; closing the browser does not stop the VM run."""

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from .run_phase3b_unattended import add_job_arguments, lifecycle_directory, validate_job_args
    from .run_phase3b_vm import write_json
    from .surf_workspace import SurfClient, load_private_config
except ImportError:
    from run_phase3b_unattended import add_job_arguments, lifecycle_directory, validate_job_args
    from run_phase3b_vm import write_json
    from surf_workspace import SurfClient, load_private_config


def launch_command(args):
    command = [sys.executable, "-u", str(Path(args.project_root, "scripts/run_phase3b_unattended.py")),
               "--stage", args.stage, "--output_root", args.output_root, "--storage_root", args.storage_root,
               "--project_root", args.project_root, "--backup_dir", args.backup_dir,
               "--pause_policy", args.pause_policy]
    if args.pause_policy != "off":
        command.extend(["--surf_config", args.surf_config, "--confirm_exclusive_workspace"])
    return command + ["--", *args.runner_args]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session_name", default="phase3b_full")
    parser.add_argument("--dry_run", action="store_true", help="Print the launch plan without API calls, tmux or file writes.")
    add_job_arguments(parser)
    args = validate_job_args(parser.parse_args())
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.session_name):
        parser.error("Use a short session name containing letters, digits, underscores or hyphens.")
    directory = lifecycle_directory(args)
    command = launch_command(args)
    if args.dry_run:
        print(json.dumps({"session": args.session_name, "command": command, "log": str(directory / "job.log")}, indent=2))
        return
    if not shutil.which("tmux"):
        parser.error("Install tmux on the VM first (sudo apt install tmux), then rerun this launcher.")
    if subprocess.run(["tmux", "has-session", "-t", "=" + args.session_name], capture_output=True).returncode == 0:
        parser.error("This tmux session already exists. Attach to it; do not start a duplicate experiment.")
    if args.pause_policy != "off":
        SurfClient(load_private_config(args.surf_config, (
            args.project_root, args.output_root, args.backup_dir, directory,
        ))).check(require_pause=True)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "launch_plan.json", {"session": args.session_name, "command": command})
    shell_command = f"exec {shlex.join(command)} >> {shlex.quote(str(directory / 'job.log'))} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", args.session_name,
                    "-c", args.project_root, shell_command], check=True)
    print(f"Detached job started. Attach: tmux attach -t {args.session_name}")
    print(f"Log: {directory / 'job.log'}")
    print(f"Status: {directory / 'job_status.json'}")
    print("The session may disappear when the command finishes; persistent logs/status remain. Automatic packaging is NOT a local backup.")


if __name__ == "__main__":
    main()
