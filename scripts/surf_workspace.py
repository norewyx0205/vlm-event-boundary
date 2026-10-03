"""Minimal SURF client: read workspace status and request Pause, never Delete."""

import argparse
import getpass
import json
import os
import stat
import sys
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

API_BASE = "https://gw.live.surfresearchcloud.nl/v1/workspace/"
DEFAULT_CONFIG = "/data/yuxuanstorage/.phase3b_private/surf.json"


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_config(config):
    if config.get("schema") != "phase3b_surf_private_v1":
        raise ValueError("Unsupported private SURF configuration schema.")
    config = dict(config)
    config["workspace_id"] = str(uuid.UUID(config["workspace_id"]))
    if not isinstance(config.get("workspace_name"), str) or not config["workspace_name"].strip():
        raise ValueError("Set the expected workspace name, exactly as shown in the portal.")
    token = config.get("token")
    if not isinstance(token, str) or not token or len(token) > 4096 or any(c in token for c in "\r\n\0"):
        raise ValueError("Invalid API token; enter it only through the hidden setup prompt.")
    return config


def check_private_location(path, forbidden_roots=()):
    path = Path(path).expanduser().absolute()
    if path.is_symlink() or any(path.resolve().is_relative_to(Path(root).resolve()) for root in forbidden_roots):
        raise ValueError("Private SURF config must be outside the repository, run and backup directories, without symlinks.")
    return path


def load_private_config(path, forbidden_roots=()):
    path = check_private_location(path, forbidden_roots)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "r", encoding="utf-8") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise PermissionError("SURF config must be a file owned by you with permissions 600.")
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


class SurfClient:
    def __init__(self, config, opener=None):
        self.config = validate_config(config)
        self.opener = opener or build_opener(NoRedirects())

    def _request(self, action=False):
        path = f"workspaces/{self.config['workspace_id']}/"
        body = None
        if action:
            path += "actions/"
            body = json.dumps([{"action": "pause", "parameters": {}}]).encode("utf-8")
        request = Request(API_BASE + path, data=body, method="POST" if action else "GET", headers={
            "authorization": self.config["token"], "Accept": "application/json", "Content-Type": "application/json",
        })
        try:
            with self.opener.open(request, timeout=30) as response:
                payload = json.loads(response.read())
        except HTTPError as exc:
            # Never expose a server response that could echo authorization headers.
            raise RuntimeError(f"SURF HTTP {exc.code}; check token expiry, workspace permissions and API availability.") from None
        except (URLError, TimeoutError, OSError):
            raise RuntimeError("SURF connection failed. A timed-out Pause may already be queued; check the portal before retrying.") from None
        except (ValueError, TypeError):
            raise RuntimeError("SURF returned an invalid JSON response; check the portal.") from None
        if not isinstance(payload, dict) or payload.get("id") != self.config["workspace_id"] or payload.get("name") != self.config["workspace_name"]:
            raise RuntimeError("SURF workspace ID/name does not match the explicitly configured target.")
        return payload

    def check(self, require_pause=False):
        workspace = self._request()
        result = {key: workspace.get(key) for key in ("id", "name", "status")}
        result["pause_allowed"] = "pause" in workspace.get("allowed_actions", [])
        if require_pause and (result["status"] != "running" or not result["pause_allowed"]):
            raise RuntimeError("Target must be running and your token must have Pause permission before unattended launch.")
        return result

    def request_pause(self):
        before = self.check()
        if before["status"] in {"paused", "pausing"}:
            return {"request_accepted": False, "already_paused_or_pausing": True, "observed_status": before["status"],
                    "billing_stop_confirmed": before["status"] == "paused"}
        if before["status"] != "running" or not before["pause_allowed"]:
            raise RuntimeError("SURF workspace is not running or Pause permission is missing.")
        result = self._request(action=True)
        return {"request_accepted": True, "observed_status": result.get("status"),
                "billing_stop_confirmed": result.get("status") == "paused"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("setup", "check"))
    parser.add_argument("--config_path", default=DEFAULT_CONFIG)
    parser.add_argument("--workspace_id")
    parser.add_argument("--workspace_name")
    args = parser.parse_args()
    project = Path(__file__).resolve().parent.parent
    if args.action == "setup":
        if not args.workspace_id or not args.workspace_name:
            parser.error("setup requires --workspace_id and --workspace_name; no token command-line argument is supported.")
        if not sys.stdin.isatty():
            parser.error("Run setup in an interactive VM terminal; never pass the token through a notebook or pipe.")
        config = validate_config({"schema": "phase3b_surf_private_v1", "workspace_id": args.workspace_id,
                                  "workspace_name": args.workspace_name, "token": getpass.getpass("SURF API token (hidden): ").strip()})
        public = SurfClient(config).check(require_pause=True)
        save_private_config(args.config_path, config, (project,))
        print(f"Private config saved to {args.config_path}; token was not printed. No Pause was requested.")
    else:
        public = SurfClient(load_private_config(args.config_path, (project,))).check()
    print(json.dumps(public, indent=2))


if __name__ == "__main__":
    main()
