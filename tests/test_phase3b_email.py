import contextlib
import io
import json
import smtplib
import ssl
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import phase3b_email as email


def config(tls="ssl"):
    return {"schema": "phase3b_email_private_v1", "smtp_host": "smtp.example.com", "tls": tls,
            "port": 465 if tls == "ssl" else 587, "username": "sender@example.com",
            "password": "unit-test-app-password-not-a-real-credential", "from_addr": "sender@example.com",
            "to_addr": "recipient@example.org"}


class EmailTest(unittest.TestCase):
    def test_private_configuration_permissions_no_overwrite_and_forbidden_locations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "private/email.json"
            email.save_private_config(path, config())
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(path.parent.stat().st_mode & 0o777, 0o700)
            self.assertEqual(email.load_private_config(path), config())
            with self.assertRaises(FileExistsError):
                email.save_private_config(path, config())
            with self.assertRaises(ValueError):
                email.load_private_config(path, (root,))
            link = root / "link.json"
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                email.load_private_config(link)
            path.chmod(0o644)
            with self.assertRaises(PermissionError):
                email.load_private_config(path)

    def test_invalid_address_injection_plaintext_and_configuration_are_rejected(self):
        for change in ({"tls": "none"}, {"to_addr": "a@b.com\nBcc: other@b.com"},
                       {"to_addr": "a@b.com,other@b.com"}, {"from_addr": "not-an-email"},
                       {"password": ""}, {"port": 0}, {"smtp_host": "https://smtp.example.com"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                email.validate_config({**config(), **change})

    def test_starttls_precedes_authentication_and_only_one_recipient_gets_no_attachments(self):
        smtp = Mock()
        smtp.send_message.return_value = {}
        events = []
        smtp.ehlo.side_effect = lambda: events.append("ehlo")
        smtp.starttls.side_effect = lambda **kwargs: events.append("tls")
        smtp.login.side_effect = lambda *_: events.append("login")
        smtp.send_message.side_effect = lambda *args, **kwargs: events.append("send") or {}
        with patch.object(email.smtplib, "SMTP", return_value=smtp) as connect:
            result = email.EmailNotifier(config("starttls")).send("Experiment completed", "A small report.\n")
        self.assertEqual(events, ["ehlo", "tls", "ehlo", "login", "send"])
        self.assertEqual(connect.call_args.kwargs["timeout"], 20)
        context = smtp.starttls.call_args.kwargs["context"]
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        message = smtp.send_message.call_args.args[0]
        self.assertEqual(message["To"], "recipient@example.org")
        self.assertEqual(list(message.iter_attachments()), [])
        self.assertNotIn(config()["password"], message.as_string())
        self.assertEqual(smtp.send_message.call_args.kwargs["to_addrs"], ["recipient@example.org"])
        self.assertTrue(result["smtp_accepted"])
        self.assertFalse(result["inbox_delivery_confirmed"])
        smtp.set_debuglevel.assert_not_called()

    def test_ssl_check_authenticates_without_sending_and_server_errors_are_sanitized(self):
        smtp = Mock()
        with patch.object(email.smtplib, "SMTP_SSL", return_value=smtp) as connect:
            checked = email.EmailNotifier(config()).check()
        self.assertTrue(checked["authenticated"])
        self.assertNotIn(config()["password"], json.dumps(checked))
        self.assertTrue(connect.call_args.kwargs["context"].check_hostname)
        smtp.send_message.assert_not_called()
        smtp.login.side_effect = smtplib.SMTPAuthenticationError(535, config()["password"].encode())
        with patch.object(email.smtplib, "SMTP_SSL", return_value=smtp), self.assertRaises(RuntimeError) as error:
            email.EmailNotifier(config()).check()
        self.assertNotIn(config()["password"], str(error.exception))
        self.assertIn("SMTPAuthenticationError", str(error.exception))

    def test_no_retry_after_ambiguous_send_or_fallback_if_tls_fails(self):
        smtp = Mock()
        smtp.send_message.side_effect = TimeoutError("secret server response")
        with patch.object(email.smtplib, "SMTP_SSL", return_value=smtp), self.assertRaises(RuntimeError):
            email.EmailNotifier(config()).send("Completed", "Report")
        self.assertEqual(smtp.send_message.call_count, 1)
        smtp.starttls.side_effect = smtplib.SMTPNotSupportedError("TLS unavailable")
        smtp.login.reset_mock()
        with patch.object(email.smtplib, "SMTP", return_value=smtp), self.assertRaises(RuntimeError):
            email.EmailNotifier(config("starttls")).check()
        smtp.login.assert_not_called()

    def test_summary_reports_analysis_and_unconfirmed_pause_not_stale_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "primary/analysis").mkdir(parents=True)
            (root / "vm_run_config.json").write_text(json.dumps({"primary_count": 50, "pair_count": 65}))
            (root / "primary/analysis/aggregate_summary.json").write_text(json.dumps({"patch_rows": 14170, "missing_patch_count": 0}))
            status = {"stage": "full", "state": "pause_request_pending", "experiment_success": True,
                      "experiment_return_code": 0, "started_at": "2026-10-04T10:00:00Z", "elapsed_sec": 72000,
                      "backup_verified": True, "backup_dir": "/data/backups/verified", "pause_policy": "finished"}
            subject, body = email.completion_message(status, root, root / "journal")
            self.assertIn("SUCCESS", subject)
            self.assertIn("patch_rows: 14170", body)
            self.assertIn("20.00 hours", body)
            self.assertIn("NOT yet confirmed", body)
            self.assertIn("Local backup is NOT confirmed", body)
            status.update(state="needs_attention", experiment_success=False, experiment_return_code=1, error="disk full")
            subject, body = email.completion_message(status, root, root / "journal")
            self.assertIn("NEEDS ATTENTION", subject)
            self.assertIn("Experiment: FAILED", body)
            self.assertNotIn("patch_rows: 14170", body)
            self.assertIn("disk full", body)

    def test_setup_requires_interactive_terminal_before_prompting_for_password(self):
        argv = ["email", "setup", "--username", "sender@example.com", "--to_addr", "recipient@example.org"]
        with patch.object(sys, "argv", argv), patch.object(sys.stdin, "isatty", return_value=False), \
                patch.object(email.getpass, "getpass") as prompt, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                email.main()
            prompt.assert_not_called()

    def test_check_and_test_are_explicit_and_setup_prints_no_password(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "private/email.json")
            notifier = Mock()
            notifier.check.return_value = {"authenticated": True}
            notifier.send.return_value = {"smtp_accepted": True, "inbox_delivery_confirmed": False}
            output = io.StringIO()
            argv = ["email", "setup", "--config_path", str(path), "--username", "sender@example.com",
                    "--to_addr", "recipient@example.org"]
            with patch.object(sys, "argv", argv), patch.object(sys.stdin, "isatty", return_value=True), \
                    patch.object(email.getpass, "getpass", return_value=config()["password"]), \
                    patch.object(email, "EmailNotifier", return_value=notifier), contextlib.redirect_stdout(output):
                email.main()
            notifier.send.assert_not_called()
            self.assertNotIn(config()["password"], output.getvalue())
            for action in ("check", "test"):
                with patch.object(sys, "argv", ["email", action, "--config_path", str(path)]), \
                        patch.object(email, "EmailNotifier", return_value=notifier), contextlib.redirect_stdout(io.StringIO()):
                    email.main()
            notifier.send.assert_called_once()


if __name__ == "__main__":
    unittest.main()
