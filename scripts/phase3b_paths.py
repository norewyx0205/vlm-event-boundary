"""Resolve relocated Phase 3B videos without rewriting frozen manifests."""

import json
from pathlib import Path


def load_path_map(path=None):
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Video path map must be an object of absolute prefix mappings.")
    result = {}
    for source, target in payload.items():
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError("Video path prefixes must be strings.")
        if not Path(source).is_absolute() or not Path(target).is_absolute():
            raise ValueError("Video path prefixes must be absolute.")
        result[str(Path(source))] = str(Path(target).resolve())
    return result


def resolve_video_path(video_path, project_root, path_map=None):
    path = Path(video_path)
    if path.is_absolute():
        for source, target in sorted((path_map or {}).items(), key=lambda item: len(Path(item[0]).parts), reverse=True):
            try:
                suffix = path.relative_to(source)
            except ValueError:
                continue
            mapped = Path(target) / suffix
            if not mapped.is_file():
                raise FileNotFoundError(f"Mapped video is missing: {path} -> {mapped}. Copy the frozen source video; do not regenerate it.")
            return mapped.resolve()
        candidates = [path]
    else:
        candidates = [Path(project_root) / path, path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Video is missing: {video_path}; project root={project_root}.")
