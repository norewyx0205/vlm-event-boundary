import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


VALID_MODES = ("skip", "reuse", "analyze", "run")
ARTIFACT_SCHEMA_VERSION = 2
STRICT_MANIFEST_FIELDS = {
    "artifact_schema_version",
    "artifact_type",
    "created_at",
    "source_commit",
    "model_name",
    "model_revision",
    "config_fingerprint",
}

EXPERIMENT_ALLOWED_MODES = {
    "baseline": {"skip", "reuse", "run"},
    "synthetic": {"skip", "reuse", "run"},
    "ladder": set(VALID_MODES),
    "ladder_smoke": {"skip", "run"},
    "feature_ablation": set(VALID_MODES),
    "size_stress": set(VALID_MODES),
    "size_clear_contrast": set(VALID_MODES),
    "diagnostics": set(VALID_MODES),
    "roi_perturbation": set(VALID_MODES),
    "attention_phase0": set(VALID_MODES),
    "attention_phase1": set(VALID_MODES),
    "activation_patching_phase3": set(VALID_MODES),
}


def _profile(default="skip", **overrides):
    profile = {name: default for name in EXPERIMENT_ALLOWED_MODES}
    profile.update(overrides)
    return profile


PROFILE_MODES = {
    "part1_reuse": _profile(
        default="reuse",
        ladder_smoke="skip",
        attention_phase0="skip",
        activation_patching_phase3="skip",
    ),
    "analysis_only": _profile(
        default="analyze",
        baseline="reuse",
        synthetic="reuse",
        ladder_smoke="skip",
        activation_patching_phase3="skip",
    ),
    "full_reproduction": _profile(
        default="run",
        ladder_smoke="skip",
        activation_patching_phase3="skip",
    ),
    "smoke": _profile(
        ladder_smoke="run",
    ),
}


class ArtifactError(RuntimeError):
    pass


def resolve_experiment_modes(profile_name, overrides=None):
    if profile_name not in PROFILE_MODES:
        choices = ", ".join(sorted(PROFILE_MODES))
        raise ValueError(f"Unknown pipeline profile {profile_name!r}; choose from {choices}.")
    overrides = overrides or {}
    unknown = sorted(set(overrides) - set(EXPERIMENT_ALLOWED_MODES))
    if unknown:
        raise ValueError(f"Unknown experiment override(s): {', '.join(unknown)}")

    modes = dict(PROFILE_MODES[profile_name])
    modes.update(overrides)
    for experiment, mode in modes.items():
        if mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode {mode!r} for {experiment}; choose from {VALID_MODES}."
            )
        allowed = EXPERIMENT_ALLOWED_MODES[experiment]
        if mode not in allowed:
            raise ValueError(
                f"Mode {mode!r} is not meaningful for {experiment}; "
                f"choose from {sorted(allowed)}."
            )
    return modes


def mode_runs(mode):
    return mode == "run"


def mode_analyzes(mode):
    return mode in {"run", "analyze"}


def mode_uses_artifacts(mode):
    return mode in {"reuse", "analyze"}


def announce_experiment(name, mode):
    descriptions = {
        "skip": "skipped",
        "reuse": "using archived artifacts after provenance audit",
        "analyze": "reusing raw artifacts and rebuilding CPU analysis",
        "run": "executing the real experiment",
    }
    print(f"[{name}] mode={mode}: {descriptions[mode]}")


def require_paths(experiment, paths, hint=None):
    resolved = [Path(path) for path in paths]
    missing = [str(path) for path in resolved if not path.exists()]
    if missing:
        message = (
            f"[{experiment}] required research artifact(s) are missing:\n- "
            + "\n- ".join(missing)
        )
        if hint:
            message += f"\n{hint}"
        raise ArtifactError(message)
    return resolved


def matching_paths(root, pattern):
    return sorted(Path(root).glob(pattern))


def require_matches(experiment, root, pattern, minimum=1, hint=None):
    matches = matching_paths(root, pattern)
    if len(matches) < minimum:
        message = (
            f"[{experiment}] expected at least {minimum} research artifact(s) matching "
            f"{Path(root) / pattern}, found {len(matches)}."
        )
        if hint:
            message += f"\n{hint}"
        raise ArtifactError(message)
    return matches


def latest_match(experiment, root, pattern, hint=None):
    return require_matches(experiment, root, pattern, hint=hint)[-1]


def latest_results_by_dataset(
    experiment,
    model_result_root,
    dataset_pattern,
    minimum=1,
    hint=None,
):
    model_result_root = Path(model_result_root)
    latest = []
    for dataset_dir in sorted(model_result_root.glob(dataset_pattern)):
        if not dataset_dir.is_dir():
            continue
        runs = sorted(dataset_dir.glob("*/raw_results.jsonl"))
        if runs:
            latest.append(runs[-1])
    if len(latest) < minimum:
        message = (
            f"[{experiment}] expected at least {minimum} distinct completed datasets "
            f"matching {model_result_root / dataset_pattern}, found {len(latest)}."
        )
        if hint:
            message += f"\n{hint}"
        raise ArtifactError(message)
    return latest


def stable_fingerprint(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def discover_latest_archive(search_roots, patterns=None):
    patterns = patterns or ("*vlm*results*.zip", "*output*.zip")
    candidates = []
    for root in search_roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for pattern in patterns:
            candidates.extend(path for path in root.glob(pattern) if path.is_file())
    unique = {path.resolve(): path for path in candidates}
    if not unique:
        return None
    return max(unique.values(), key=lambda path: (path.stat().st_mtime, path.name))


def _archive_manifest(archive):
    try:
        payload = json.loads(archive.read("archive_manifest.json"))
    except KeyError as exc:
        raise ArtifactError("Artifact archive has no archive_manifest.json.") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactError("Artifact archive manifest is not valid JSON.") from exc
    artifact_type = payload.get("artifact_type")
    if artifact_type not in {None, "real"}:
        raise ArtifactError(
            f"Refusing artifact_type={artifact_type!r}; research runs require real artifacts."
        )
    schema_version = payload.get("artifact_schema_version")
    if schema_version is None:
        payload["_provenance_status"] = "legacy_unverified"
        payload["_validation_warnings"] = [
            "Legacy archive has no artifact_schema_version; provenance fields "
            "are audited when present but cannot be treated as strictly validated."
        ]
        return payload
    if schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactError(
            f"Unsupported artifact_schema_version={schema_version!r}; "
            f"expected {ARTIFACT_SCHEMA_VERSION}."
        )
    missing = sorted(
        field
        for field in STRICT_MANIFEST_FIELDS
        if payload.get(field) in {None, ""}
    )
    if missing:
        raise ArtifactError(
            "Strict artifact manifest is missing required provenance field(s): "
            + ", ".join(missing)
        )
    payload["_provenance_status"] = "strict_validated"
    payload["_validation_warnings"] = []
    return payload


def validate_archive_manifest(manifest, expected=None):
    mismatches = {}
    missing_expected = []
    for key, expected_value in (expected or {}).items():
        archived_value = manifest.get(key)
        if expected_value is None:
            continue
        if archived_value is None:
            missing_expected.append(key)
            continue
        if archived_value != expected_value:
            mismatches[key] = {
                "expected": expected_value,
                "archived": archived_value,
            }
    if mismatches:
        details = "; ".join(
            f"{key}: expected {values['expected']!r}, archived {values['archived']!r}"
            for key, values in mismatches.items()
        )
        raise ArtifactError(f"Artifact archive provenance mismatch: {details}")
    if missing_expected:
        if manifest.get("_provenance_status") == "strict_validated":
            raise ArtifactError(
                "Strict artifact archive cannot verify expected provenance field(s): "
                + ", ".join(sorted(missing_expected))
            )
        manifest.setdefault("_validation_warnings", []).append(
            "Legacy archive is missing expected provenance field(s): "
            + ", ".join(sorted(missing_expected))
        )
    return manifest


def _safe_members(archive):
    members = []
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ArtifactError(f"Unsafe path in artifact archive: {member.filename}")
        members.append(member)
    return members


def restore_artifact_archive(archive_path, project_root, expected=None):
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise ArtifactError(f"Artifact archive does not exist: {archive_path}")
    project_root = Path(project_root)
    project_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        manifest = validate_archive_manifest(_archive_manifest(archive), expected)
        members = [
            member
            for member in _safe_members(archive)
            if member.filename != "archive_manifest.json"
        ]
        archive.extractall(project_root, members=members)
    return {
        "archive_path": str(archive_path),
        "artifact_type": (
            "real"
            if manifest.get("_provenance_status") == "strict_validated"
            else "legacy_unverified"
        ),
        "provenance_status": manifest.get("_provenance_status"),
        "validation_warnings": list(manifest.get("_validation_warnings", [])),
        "source_commit": manifest.get("source_commit"),
        "model_name": manifest.get("model_name"),
        "model_revision": manifest.get("model_revision"),
        "created_at": manifest.get("created_at"),
        "manifest": manifest,
    }
