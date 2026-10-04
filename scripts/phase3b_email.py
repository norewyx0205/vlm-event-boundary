"""Private TLS SMTP configuration and small, attachment-free VM notifications."""

import argparse
import getpass
import json
import os
import smtplib
import ssl
import stat
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from email.headerregistry import Address
from email.message import EmailMessage
from email.utils import format_datetime, make_msgid
from pathlib import Path

try:
    from .surf_workspace import check_private_location
except ImportError:
    from surf_workspace import check_private_location

DEFAULT_CONFIG = "/data/yuxuanstorage/.phase3b_private/email.json"


def validate_config(config):
    if config.get("schema") != "phase3b_email_private_v1":
        raise ValueError("Unsupported private email configuration schema.")
    config = dict(config)
    host = config.get("smtp_host")
    if not isinstance(host, str) or not host or any(c.isspace() or c in "/\\:\0" for c in host):
        raise ValueError("Set an SMTP hostname, without a URL or embedded port.")
    if config.get("tls") not in {"ssl", "starttls"}:
        raise ValueError("Email requires TLS: choose ssl or starttls; plaintext is not supported.")
    port = config.get("port")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("SMTP port must be an integer between 1 and 65535.")
    for key in ("from_addr", "to_addr"):
        value = config.get(key)
        try:
            if not isinstance(value, str) or not value.isascii() or any(c in value for c in "\r\n\0"):
                raise ValueError
            address = Address(addr_spec=value)
            if not address.username or not address.domain or address.addr_spec != value:
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError("Set one plain email address for sender and recipient; no display names or lists.") from None
    for key in ("username", "password"):
        value = config.get(key)
        if not isinstance(value, str) or not value or len(value) > 4096 or any(c in value for c in "\r\n\0"):
            raise ValueError("Set the SMTP username and a provider-issued app password in the private config.")
    return config


def load_private_config(path, forbidden_roots=()):
    path = check_private_location(path, forbidden_roots)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "r", encoding="utf-8") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise PermissionError("Email config must be a file owned by you with permissions 600.")
        return validate_config(json.load(handle))


def save_private_config(path, config, forbidden_roots=()):
    path = check_private_location(path, forbidden_roots)
    config = validate_config(config)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent = path.parent.stat()
    if parent.st_uid != os.getuid() or parent.st_mode & 0o077:
        raise PermissionError("Use a private directory owned by you with permissions 700.")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


class EmailNotifier:
    def __init__(self, config):
        self.config = validate_config(config)

    @contextmanager
    def connection(self):
        config = self.config
        context = ssl.create_default_context()
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        smtp = None
        try:
            if config["tls"] == "ssl":
                smtp = smtplib.SMTP_SSL(config["smtp_host"], config["port"], timeout=20, context=context)
            else:
                smtp = smtplib.SMTP(config["smtp_host"], config["port"], timeout=20)
                smtp.ehlo()
                smtp.starttls(context=context)
                smtp.ehlo()
            smtp.login(config["username"], config["password"])
            yield smtp
        except (OSError, smtplib.SMTPException, ValueError) as exc:
            # Server responses can echo authentication data; never publish them.
            raise RuntimeError(f"Email SMTP operation failed ({type(exc).__name__}); check app-password, TLS and provider permissions.") from None
        finally:
            if smtp is not None:
                # Closing cannot turn an accepted DATA response into a failed send.
                try:
                    smtp.close()
                except OSError:
                    pass

    def check(self):
        with self.connection():
            pass
        return {"authenticated": True, "smtp_host": self.config["smtp_host"], "tls": self.config["tls"],
                "from_addr": self.config["from_addr"], "to_addr": self.config["to_addr"]}

    def send(self, subject, body):
        message = EmailMessage()
        message["From"] = self.config["from_addr"]
        message["To"] = self.config["to_addr"]
        message["Subject"] = subject
        message["Date"] = format_datetime(datetime.now(timezone.utc))
        message["Message-ID"] = make_msgid()
        message.set_content(body)
        with self.connection() as smtp:
            refused = smtp.send_message(message, from_addr=self.config["from_addr"], to_addrs=[self.config["to_addr"]])
            if refused:
                raise RuntimeError("Email recipient was refused; check recipient/provider settings.")
        return {"smtp_accepted": True, "inbox_delivery_confirmed": False, "message_id": message["Message-ID"]}


def read_public_json(path):
    try:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        return result if isinstance(result, dict) else {}
    except (OSError, ValueError):
        return {}


def completion_message(status, run_root, lifecycle_dir):
    root = Path(run_root)
    success = status.get("experiment_success")
    result = "SUCCESS" if success is True else "FAILED" if success is False else "NOT COMPLETED"
    outcome = "NEEDS ATTENTION" if status["state"] == "needs_attention" else result
    subject = f"[Phase 3B] {status['stage']} {outcome} - {root.name}"
    config = read_public_json(root / "vm_run_config.json")
    lines = [
        f"Experiment: {result}", f"Stage: {status['stage']}", f"Lifecycle state: {status['state']}",
        f"Exit code: {status.get('experiment_return_code', 'not available')}",
        f"Started (UTC): {status['started_at']}", f"Elapsed: {status.get('elapsed_sec', 0) / 3600:.2f} hours",
        f"Frozen full cohort: {config.get('primary_count', 'unknown')} primary / {config.get('pair_count', 'unknown')} total pairs",
        f"Run directory: {root}", f"Log: {Path(lifecycle_dir) / 'job.log'}",
        f"Status: {Path(lifecycle_dir) / 'job_status.json'}",
        f"Verified VM backup: {status.get('backup_verified', False)}",
        f"Backup directory: {status.get('backup_dir', 'not available')}",
        "Local backup is NOT confirmed: download all parts and verify on your computer.",
    ]
    if status.get("error"):
        lines.append(f"Error: {str(status['error'])[:1000]}")
    if success is True:
        summaries = [root / "primary/analysis/aggregate_summary.json"] if status["stage"] != "preflight" else sorted(root.glob("preflight/*/analysis/aggregate_summary.json"))
        for path in summaries:
            summary = read_public_json(path)
            if summary:
                lines.append(f"Analysis: {path}")
                for key in ("capture_conditions", "divergence_rows", "patch_rows", "missing_patch_count",
                            "missing_capture_count", "missing_divergence_count", "missing_technical_control_count"):
                    if key in summary:
                        lines.append(f"  {key}: {summary[key]}")
    else:
        lines.append("Any prior analysis in this directory is not a completed result for this attempt. See checkpoints/logs.")
    if status["state"] == "pause_request_pending":
        lines.append("Automatic Pause: next action is to REQUEST SURF Pause after this email attempt. Pause and billing stop are NOT yet confirmed; check the portal.")
    elif status["state"] == "needs_attention":
        lines.append("Automatic Pause: NOT confirmed. The VM may still be running and charging; check the portal.")
    else:
        lines.append(f"Automatic Pause: not requested (policy={status['pause_policy']}). The VM remains running.")
    lines.append("This email is a technical completion summary, not a scientific interpretation of patch effects. No activations, videos or credentials are attached.")
    return subject, "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("setup", "check", "test"))
    parser.add_argument("--config_path", default=DEFAULT_CONFIG)
    parser.add_argument("--smtp_host", default="smtp.gmail.com")
    parser.add_argument("--tls", choices=("ssl", "starttls"), default="ssl")
    parser.add_argument("--port", type=int)
    parser.add_argument("--username")
    parser.add_argument("--from_addr")
    parser.add_argument("--to_addr")
    args = parser.parse_args()
    project = Path(__file__).resolve().parent.parent
    forbidden = (project, "/data/yuxuanstorage/vlm_phase3b", "/data/yuxuanstorage/backups")
    if args.action == "setup":
        if not args.username or not args.to_addr:
            parser.error("setup requires --username and --to_addr; no password command-line argument is supported.")
        if not sys.stdin.isatty():
            parser.error("Run setup in an interactive VM terminal; never pipe a password or put it in a notebook.")
        password = getpass.getpass("SMTP app password (hidden, not your normal account password): ").strip()
        if args.smtp_host == "smtp.gmail.com":
            password = password.replace(" ", "")
        config = validate_config({"schema": "phase3b_email_private_v1", "smtp_host": args.smtp_host,
                                  "tls": args.tls, "port": args.port if args.port is not None else (465 if args.tls == "ssl" else 587),
                                  "username": args.username, "password": password,
                                  "from_addr": args.from_addr or args.username, "to_addr": args.to_addr})
        public = EmailNotifier(config).check()
        save_private_config(args.config_path, config, forbidden)
        print(f"Private config saved to {args.config_path}. No email was sent; use test to check inbox delivery.")
    else:
        notifier = EmailNotifier(load_private_config(args.config_path, forbidden))
        if args.action == "check":
            public = notifier.check()
        else:
            public = notifier.send("[Phase 3B] Notification test", "The VM notification sender accepted this test. Check your inbox/spam folder. No experiment started and no Pause was requested.\n")
    print(json.dumps(public, indent=2))


if __name__ == "__main__":
    main()
