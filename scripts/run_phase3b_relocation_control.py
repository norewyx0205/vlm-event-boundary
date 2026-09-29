"""Preflight cross-position control with codec-matched global event relocation."""

import argparse
import copy
import hashlib
import json
import math
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import torch
import transformers

try:
    from .activation_patching_core import (
        atomic_write_json, atomic_write_jsonl, decision_from_logits,
        validate_archived_input_metadata, validate_archived_prediction,
    )
    from .common import PROJECT_ROOT, read_jsonl
    from .phase3b_core import VIDEO_GROUPS, build_pair_mappings, prepare_example
    from .run_eval import configure_reproducibility, load_model
    from .run_phase3b_patching import (
        _capture_all_layers, activation_root, load_capture, no_patch_forward, patch_forward,
        repo_commit, validate_patch_controls,
    )
except ImportError:
    from activation_patching_core import (
        atomic_write_json, atomic_write_jsonl, decision_from_logits,
        validate_archived_input_metadata, validate_archived_prediction,
    )
    from common import PROJECT_ROOT, read_jsonl
    from phase3b_core import VIDEO_GROUPS, build_pair_mappings, prepare_example
    from run_eval import configure_reproducibility, load_model
    from run_phase3b_patching import (
        _capture_all_layers, activation_root, load_capture, no_patch_forward, patch_forward,
        repo_commit, validate_patch_controls,
    )


def decode_frames(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot decode {path}.")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}.")
    return frames, fps


def encode_frames(path, frames, fps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot encode {path}.")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def shifted_row(row, path, shift_frames):
    changed = copy.deepcopy(row)
    changed["video_path"] = str(path)
    changed["video_id"] = Path(path).name
    changed["eval_id"] = row["eval_id"] + f"_shift_{shift_frames}"
    for obj in changed["target_objects"] + changed.get("distractors", []):
        for key in ("start_frame", "end_frame"):
            if isinstance(obj.get(key), int) and obj[key] >= 0:
                obj[key] += shift_frames
    for section in ("event_timing", "boundary_timing"):
        for key, value in changed.get(section, {}).items():
            if key.endswith("_frame") and isinstance(value, int):
                changed[section][key] += shift_frames
    changed.pop("archived_prediction", None)
    changed.pop("archived_input_metadata", None)
    return changed


def relocation_pairs(manifest_rows, max_cases, requested_shift=None):
    by_pair = defaultdict(dict)
    for row in manifest_rows:
        pair = by_pair[row["phase3b_pair_id"]]
        if row["condition"] in pair:
            raise ValueError(f"Duplicate {row['condition']} row in {row['phase3b_pair_id']}.")
        pair[row["condition"]] = row
    selected = []
    for pair_id, pair in by_pair.items():
        if set(pair) != {"low_boundary", "temporal_boundary"}:
            raise ValueError(f"Relocation requires both boundary rows for {pair_id}.")
        low, temporal = pair["low_boundary"], pair["temporal_boundary"]
        for key in ("first_event_start_frame", "first_event_end_frame"):
            if low["event_timing"].get(key) != temporal["event_timing"].get(key):
                raise ValueError(f"Event 1 is not time-aligned in matched pair {pair_id}.")
        shift = (int(temporal["event_timing"]["second_event_start_frame"])
                 - int(low["event_timing"]["second_event_start_frame"]))
        if shift <= 0:
            raise ValueError(f"Event 2 is not delayed in temporal boundary for {pair_id}.")
        if (int(temporal["event_timing"]["second_event_end_frame"])
                - int(low["event_timing"]["second_event_end_frame"])) != shift:
            raise ValueError(f"Event 2 duration differs in matched pair {pair_id}.")
        if requested_shift is not None and requested_shift != shift:
            raise ValueError(f"Requested shift {requested_shift} differs from matched Event 2 shift {shift} for {pair_id}.")
        selected.append((low, shift))
    return selected[:max_cases]


def decoded_video_psnr(original_frames, reencoded_frames):
    if len(original_frames) != len(reencoded_frames):
        raise RuntimeError("Re-encode control changed the decoded frame count.")
    squared_error = 0.0
    pixel_count = 0
    for original, reencoded in zip(original_frames, reencoded_frames):
        if original.shape != reencoded.shape:
            raise RuntimeError("Re-encode control changed the decoded frame dimensions.")
        difference = original.astype(np.float32) - reencoded.astype(np.float32)
        squared_error += float(np.square(difference).sum())
        pixel_count += difference.size
    mse = squared_error / pixel_count
    return None if mse == 0 else 10 * math.log10((255 ** 2) / mse)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--model_revision", default="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
    parser.add_argument("--max_cases", type=int, default=2)
    parser.add_argument("--shift_frames", type=int, default=None,
                        help="Optional assertion; must equal the pair-derived Event 2 displacement.")
    parser.add_argument("--video_fps", type=float, default=None)
    parser.add_argument("--video_num_frames", type=int, default=None)
    parser.add_argument("--video_max_pixels", type=int, default=None)
    parser.add_argument("--roi_padding", type=int, default=8)
    parser.add_argument("--min_mapping_coverage", type=float, default=0.5)
    parser.add_argument("--layers", default="0,16,32")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (args.shift_frames is not None and args.shift_frames < 1) or args.max_cases < 1:
        parser.error("shift_frames, when supplied, and max_cases must be positive.")
    pairs = relocation_pairs(read_jsonl(args.manifest_path), args.max_cases, args.shift_frames)
    if not pairs:
        parser.error("Manifest contains no low-boundary cases.")
    layers = [int(part) for part in args.layers.split(",")]
    if any(layer not in range(0, 36, 4) for layer in layers):
        parser.error("Relocation layers must belong to the fixed 0,4,...,32 grid.")
    configure_reproducibility(args.seed, deterministic=True)
    output_dir = Path(args.output_dir)
    source_paths = {}
    source_hashes = {}
    for row, _ in pairs:
        path = Path(args.project_root) / row["video_path"]
        if not path.is_file():
            path = Path(row["video_path"])
        if not path.is_file():
            raise FileNotFoundError(row["video_path"])
        pair_id = row["phase3b_pair_id"]
        source_paths[pair_id] = path
        source_hashes[pair_id] = hashlib.sha256(path.read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps({
        "schema": "phase3b_temporal_relocation_control_v2",
        "source_commit": repo_commit(),
        "code_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("run_phase3b_relocation_control.py", "phase3b_core.py", "run_phase3b_patching.py")
        },
        "model_name": args.model_name, "revision": args.model_revision,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "qwen_vl_utils_version": version("qwen-vl-utils"),
        "video_fps": args.video_fps, "video_num_frames": args.video_num_frames,
        "video_max_pixels": args.video_max_pixels, "roi_padding": args.roi_padding,
        "min_mapping_coverage": args.min_mapping_coverage,
        "shift_frames_by_pair": {row["phase3b_pair_id"]: shift for row, shift in pairs},
        "seed": args.seed, "layers": layers,
        "source_ids": [row["eval_id"] for row, _ in pairs],
        "source_video_sha256_by_pair": source_hashes,
    }, sort_keys=True).encode()).hexdigest()
    config_path = output_dir / "relocation_config.json"
    if config_path.exists():
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        if prior.get("fingerprint") != fingerprint:
            raise RuntimeError("Existing relocation control has a different configuration fingerprint.")
        if (output_dir / "relocation_audit.jsonl").is_file() and (
            output_dir / "temporal_relocation_control.jsonl"
        ).is_file() and (output_dir / "relocation_summary.json").is_file():
            print("Relocation control is already complete; reusing the saved output.")
            return
    model, processor = load_model(
        args.model_name, model_revision=args.model_revision, attn_implementation="eager"
    )
    results = []
    audits = []
    for row, shift_frames in pairs:
        pair_id = row["phase3b_pair_id"]
        original = source_paths[pair_id]
        frames, fps = decode_frames(original)
        if shift_frames >= len(frames) - max(
            obj["end_frame"] for obj in row["target_objects"]
        ):
            raise RuntimeError(f"No safe post-event tail for {pair_id} relocation.")
        tail_difference = max(
            float(cv2.absdiff(frame, frames[-1]).mean())
            for frame in frames[-shift_frames:]
        )
        if tail_difference > 3:
            raise RuntimeError(f"Tail is not stationary for {pair_id}; mean difference={tail_difference}.")
        case_dir = output_dir / "videos" / pair_id / source_hashes[pair_id][:12] / f"event2_shift_{shift_frames}"
        base_video = case_dir / "reencoded_base.mp4"
        shifted_video = case_dir / "reencoded_shifted.mp4"
        if not base_video.exists():
            encode_frames(base_video, frames, fps)
        if not shifted_video.exists():
            encode_frames(shifted_video, [frames[0]] * shift_frames + frames[:-shift_frames], fps)
        reencoded_frames, reencoded_fps = decode_frames(base_video)
        if abs(fps - reencoded_fps) > 0.01:
            raise RuntimeError(f"Re-encode control changed FPS for {pair_id}.")
        codec_psnr = decoded_video_psnr(frames, reencoded_frames)
        base_row = copy.deepcopy(row)
        base_row.update({"video_path": str(base_video), "video_id": base_video.name,
                         "eval_id": row["eval_id"] + "_reencoded"})
        base_row.pop("archived_prediction", None)
        base_row.pop("archived_input_metadata", None)
        shift_row = shifted_row(row, shifted_video, shift_frames)
        original_prepared = prepare_example(
            row, processor, args.project_root, args.video_fps,
            args.video_num_frames, args.video_max_pixels, args.roi_padding, model.device,
        )
        prepared = {
            "base": prepare_example(base_row, processor, args.project_root, args.video_fps,
                                    args.video_num_frames, args.video_max_pixels, args.roi_padding, model.device),
            "shifted": prepare_example(shift_row, processor, args.project_root, args.video_fps,
                                       args.video_num_frames, args.video_max_pixels, args.roi_padding, model.device),
        }
        original_decision = decision_from_logits(
            no_patch_forward(model, original_prepared), processor, row["correct_option"]
        )
        validate_archived_prediction(original_prepared, original_decision)
        archived_input_parity = validate_archived_input_metadata(original_prepared)
        mappings = build_pair_mappings(prepared["base"], prepared["shifted"], args.min_mapping_coverage)
        pair_roots = {
            condition: activation_root(output_dir, pair_id, condition)
            for condition in ("base", "shifted")
        }
        for condition in ("base", "shifted"):
            _capture_all_layers(model, processor, prepared[condition], pair_roots[condition], fingerprint, True)
        controls = {condition: validate_patch_controls(
            model, processor, prepared[condition], pair_roots[condition]
        ) for condition in ("base", "shifted")}
        base_decision = json.loads((pair_roots["base"] / "index.json").read_text(encoding="utf-8"))["decision"]
        codec_parity = {
            "original_prediction": original_decision["prediction"],
            "reencoded_prediction": base_decision["prediction"],
            "prediction_match": original_decision["prediction"] == base_decision["prediction"],
            "original_margin": original_decision["margin"],
            "reencoded_margin": base_decision["margin"],
            "margin_delta": base_decision["margin"] - original_decision["margin"],
            "decoded_psnr_db": codec_psnr,
            "archived_input_parity": archived_input_parity,
        }
        audits.append({
            "pair_id": pair_id, "event_2_shift_frames": shift_frames,
            "tail_mean_absolute_difference": tail_difference,
            "mapping": mappings, "identity_controls": controls,
            "reencode_parity": codec_parity,
        })
        atomic_write_jsonl(output_dir / "relocation_audit.jsonl", audits)
        if not codec_parity["prediction_match"]:
            raise RuntimeError(f"Re-encoding changed the prediction for {pair_id}; relocation is uninterpretable.")
        for group in (name for name in VIDEO_GROUPS if name.endswith("_e2")):
            mapping = mappings[group]
            if not mapping["eligible"]:
                continue
            for layer in layers:
                for direction, source_name, target_name, source_positions, target_positions in (
                    ("base_to_shifted", "base", "shifted", mapping["source_positions"], mapping["target_positions"]),
                    ("shifted_to_base", "shifted", "base", mapping["target_positions"], mapping["source_positions"]),
                ):
                    source_values = load_capture(pair_roots[source_name], layer, source_positions)
                    _, patched = patch_forward(
                        model, processor, prepared[target_name], layer, target_positions, source_values
                    )
                    target_index = json.loads((pair_roots[target_name] / "index.json").read_text(encoding="utf-8"))
                    results.append({
                        "pair_id": pair_id, "group": group, "layer": layer, "direction": direction,
                        "absolute_temporal_shift_frames": shift_frames,
                        "source_positions": source_positions, "target_positions": target_positions,
                        "mapping_coverage_source": mapping["source_coverage"],
                        "mapping_coverage_target": mapping["target_coverage"],
                        "baseline_margin": target_index["decision"]["margin"],
                        "baseline_prediction": target_index["decision"]["prediction"],
                        "patched_margin": patched["margin"],
                        "patched_prediction": patched["prediction"],
                        "categorical_flip": patched["prediction"] != target_index["decision"]["prediction"],
                        "margin_change": patched["margin"] - target_index["decision"]["margin"],
                    })
                    atomic_write_jsonl(output_dir / "temporal_relocation_control.jsonl", results)
    atomic_write_json(output_dir / "relocation_config.json", {
        "model_name": args.model_name, "model_revision": args.model_revision,
        "fingerprint": fingerprint,
        "event_2_shift_frames_by_pair": {row["phase3b_pair_id"]: shift for row, shift in pairs},
        "source_video_sha256_by_pair": source_hashes,
        "layers": layers, "case_count": len(pairs),
    })
    grouped = defaultdict(list)
    for item in results:
        grouped[(item["group"], item["layer"], item["direction"])].append(item)
    summary = [
        {
            "group": group, "layer": layer, "direction": direction,
            "n": len(items),
            "mean_margin_change": sum(item["margin_change"] for item in items) / len(items),
            "mean_absolute_margin_change": sum(abs(item["margin_change"]) for item in items) / len(items),
            "max_absolute_margin_change": max(abs(item["margin_change"]) for item in items),
            "categorical_flips": sum(bool(item["categorical_flip"]) for item in items),
        }
        for (group, layer, direction), items in sorted(grouped.items())
    ]
    atomic_write_json(output_dir / "relocation_summary.json", {
        "schema": "phase3b_temporal_relocation_control_v2",
        "fingerprint": fingerprint, "case_count": len(pairs),
        "event_2_shift_frames_by_pair": {row["phase3b_pair_id"]: shift for row, shift in pairs},
        "reencode_prediction_matches": sum(item["reencode_parity"]["prediction_match"] for item in audits),
        "reencode_parity_by_pair": {item["pair_id"]: item["reencode_parity"] for item in audits},
        "patch_rows": len(results), "by_group_layer_direction": summary,
    })
    print(f"Wrote {len(results)} temporal relocation patch rows to {output_dir}.")


if __name__ == "__main__":
    main()
