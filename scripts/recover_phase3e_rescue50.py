"""Recover unchanged rescue50 option tokens from verified local backup members.

Only raw Phase 3B per-token states are eligible. Phase 3D compact means do not
stand in for missing contextual vectors. Partial exports cannot train probes.
"""

import argparse
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path, PurePosixPath

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts.phase3c_core import atomic_write, digest, file_hash, read_json
from scripts.phase3d_execution import checked_seal
from scripts.phase3d_single_layer_ko import check_logits
from scripts.phase3e_option_features import archived_readouts
from scripts.prepare_phase3e_rescue50 import READOUTS, SCHEMA
from scripts.run_phase3d_probes import save_npz

OUTPUT_SCHEMA = "phase3e_rescue50_raw_recovery_v1"
LAYERS = (16, 17, 18, 20)


class CheckedBackup:
    """Read checksum-verified members without extracting arbitrary ZIP paths."""

    def __init__(self, manifest_path, part_paths=None):
        self.path = Path(manifest_path).resolve()
        self.manifest_sha256 = file_hash(self.path)
        self.manifest = read_json(self.path)
        m = self.manifest
        if (m.get("artifact_type") != "real" or m.get("complete") is not True or
                m.get("reports_only") is not False or m.get("activation_tensors_included") is not True or
                len(m["files"]) != m["expected_file_count"] or
                sum(f["size_bytes"] for f in m["files"]) != m["expected_total_bytes"]):
            raise ValueError("Require a complete full-activation backup manifest.")
        self.files = {f["path"]: f for f in m["files"]}
        self.parts = {f["name"]: f for f in m["archives"]}
        if len(self.files) != len(m["files"]) or len(self.parts) != len(m["archives"]):
            raise ValueError("Duplicate backup paths or parts.")
        for name in self.files:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe backup member path.")
        if any(PurePosixPath(n).name != n for n in self.parts) or any(
                f["part"] not in self.parts for f in self.files.values()):
            raise ValueError("Invalid backup part inventory.")
        paths = list(part_paths) if part_paths is not None else [self.path.parent/n for n in self.parts]
        self.paths = {Path(p).name: Path(p).resolve() for p in paths}
        if len(self.paths) != len(paths) or not set(self.paths) <= set(self.parts):
            raise ValueError("Duplicate or foreign backup part supplied.")
        self.archives, self.used = {}, {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for archive in self.archives.values():
            archive.close()

    def read(self, name):
        if name not in self.files:
            raise ValueError("Requested backup member is missing.")
        spec = self.files[name]
        part = spec["part"]
        if part not in self.archives:
            path = self.paths.get(part)
            if path is None or not path.is_file():
                raise ValueError(f"Required local backup part is unavailable: {part}")
            expected = self.parts[part]
            print(f"VERIFY_ARCHIVE {path}", flush=True)
            if path.stat().st_size != expected["size_bytes"] or file_hash(path) != expected["sha256"]:
                raise ValueError("Backup part checksum/size changed.")
            archive = zipfile.ZipFile(path)
            members = {n for n, f in self.files.items() if f["part"] == part}
            if len(archive.namelist()) != len(members) or set(archive.namelist()) != members:
                archive.close()
                raise ValueError("ZIP member inventory differs from its manifest.")
            self.archives[part] = archive
        value = self.archives[part].read(name)
        if len(value) != spec["size_bytes"] or hashlib.sha256(value).hexdigest() != spec["sha256"]:
            raise ValueError("Backup member checksum/size changed.")
        self.used[name] = {"part": part, "sha256": spec["sha256"], "size_bytes": len(value)}
        return value

    def json(self, name):
        return json.loads(self.read(name))

    def tensor(self, name):
        return torch.load(io.BytesIO(self.read(name)), map_location="cpu", weights_only=True)

    def receipt(self):
        if file_hash(self.path) != self.manifest_sha256:
            raise ValueError("Backup manifest changed during recovery.")
        return {"manifest_sha256": self.manifest_sha256, "manifest_path": str(self.path),
            "verified_parts": {n: {"path": str(self.paths[n]), **self.parts[n]} for n in self.archives},
            "unopened_parts": sorted(set(self.parts)-set(self.archives)), "verified_members": self.used,
            "verification_scope": "whole SHA256 of opened parts and independent hashes of consumed members"}


def load_plan(path):
    path = Path(path).resolve()
    plan = read_json(path)
    if (plan.get("schema") != SCHEMA or plan.get("cohort") != "original_frozen_rescue50" or
            plan.get("fingerprint") != digest({k: v for k, v in plan.items() if k != "fingerprint"}) or
            plan["readouts"] != READOUTS or len(plan["case_ids"]) != 50 or len(plan["tasks"]) != 200 or
            len(plan["statement_examples"]) != 400):
        raise ValueError("Require the frozen original rescue50 preparation.")
    for name, sha in plan["source_files_sha256"].items():
        if file_hash(Path(name)) != sha:
            raise ValueError("Preparation source bytes changed.")
    return plan


def catalog(backup, stage, plan):
    prefix = f"run/{stage}/task_checkpoints/"
    result = {}
    for name in backup.files:
        if not name.startswith(prefix) or not name.endswith(".json"):
            continue
        value = checked_seal(backup.json(name))
        spec = value["spec"]
        if (value.get("passed") is not True or spec["stage"] != stage or
                spec["intervention"] != "baseline" or
                value["plan_fingerprint"] != plan["source_plan_fingerprint"] or
                spec["task_id"] != digest([value["plan_fingerprint"], {k: v for k, v in spec.items() if k != "task_id"}])):
            raise ValueError("Foreign, failed or altered historical baseline checkpoint.")
        key = spec["pair_id"], spec["condition"], spec["view"]
        if key in result:
            raise ValueError("Duplicated historical task cell.")
        result[key] = value
    expected = {(t["pair_id"], t["condition"], t["view"]) for t in plan["tasks"]}
    if set(result) != expected:
        raise ValueError("Historical audits/captures do not cover the unchanged 200 prompt views.")
    return result


def checked_mapping(task, row, audit, capture, frozen_mapping):
    if capture["audit_record_sha256"] != audit["record_sha256"] or capture["spec"]["view"] != task["view"]:
        raise ValueError("Capture does not reference its processor audit/view.")
    record = audit["processor_record"]
    if (record["video_sha256"] != task["video_sha256"] or
            audit["input_fingerprint"] != digest(record["input_tensor_sha256"]) or
            capture["input_fingerprint"] != audit["input_fingerprint"]):
        raise ValueError("Video/input provenance does not match this task.")
    original = frozen_mapping["processor_records"][task["condition"]]
    common_fields = ("input_ids", "prompt_input_ids_sha256", "attention_mask", "visual_positions",
                     "group_positions", "video_metadata", "video_sha256")
    if task["view"] == "archived" and any(record[k] != original[k] for k in common_fields):
        raise ValueError("Archived processor mapping differs from the frozen KO50 mapping.")
    groups, tokens = archived_readouts(row, {"processor_record": record, "capture_groups": audit["groups"]})
    return {"groups": groups, "option_readout_audit": tokens, "input_fingerprint": audit["input_fingerprint"],
            "input_tensor_sha256": record["input_tensor_sha256"],
            "source_capture_record_sha256": capture["record_sha256"],
            "source_audit_record_sha256": audit["record_sha256"]}


def mean_and_bound(raw, positions):
    x = raw.float()
    count = len(positions)
    selected = x[positions]
    mean = selected.mean(0)
    # Two FP32 reductions may differ in ordering/scale placement. Derive a
    # componentwise roundoff envelope from n and the observed absolute inputs.
    eps = torch.finfo(torch.float32).eps
    gamma = (count+3)*eps/(1-(count+3)*eps)
    bound = gamma*selected.double().abs().mean(0)
    return mean, bound


def replay_layer(payload, groups, expected, names, layer):
    positions, raw = payload["positions"], payload["vectors"]
    if (len(positions) != len(set(positions)) or raw.ndim != 2 or raw.shape[0] != len(positions) or
            raw.shape[1] != 4096 or raw.dtype != torch.float16 or not bool(torch.isfinite(raw).all())):
        raise ValueError("Invalid raw per-token hidden states.")
    lookup = {p: i for i, p in enumerate(positions)}
    means, bounds = {}, {}
    for name in ["decision_position", "query_all"] + [f"option_{o}_{k}" for o in ("A", "B")
                                                       for k in ("t1", "t2", "relation")]:
        if not set(groups[name]) <= set(lookup):
            raise ValueError("Raw activation is missing a historical text group.")
        means[name], bounds[name] = mean_and_bound(raw, [lookup[p] for p in groups[name]])
    observed = {"decision": means["decision_position"], "query": means["query_all"]}
    envelopes = {"decision": bounds["decision_position"]*2, "query": bounds["query_all"]*2}
    for option in ("A", "B"):
        a, b = (f"option_{option}_t{target}" for target in (1, 2))
        observed[f"option_{option}_pair"] = means[a]-means[b]
        envelopes[f"option_{option}_pair"] = 2*(bounds[a]+bounds[b])+2*torch.finfo(torch.float32).eps*(
            means[a].double().abs()+means[b].double().abs())
        relation = f"option_{option}_relation"
        observed[relation] = means[relation]
        envelopes[relation] = 2*bounds[relation]
    errors = []
    exact = 0
    for name, values in observed.items():
        saved = expected[names.index(f"{name}:L{layer}_output")]
        if name in ("decision", "option_A_relation", "option_B_relation") and len(
                groups["decision_position" if name == "decision" else name]) == 1:
            if not torch.equal(values, saved):
                raise ValueError("Single-token historical anchor did not replay exactly.")
            exact += 1
        else:
            if not bool(((values.double()-saved.double()).abs() <= envelopes[name]+1e-40).all()):
                raise ValueError("Historical pooled feature differs beyond the FP32 reduction error bound.")
        errors.append(float((values-saved).abs().max()))
    return {"layer": layer, "exact_single_token_anchors": exact,
            "features_replayed": len(observed), "maximum_pooling_absolute_difference": max(errors),
            "pooling_tolerance": "componentwise FP32 reduction/subtraction roundoff envelope"}


def recover(plan_path, phase3b_manifest, phase3d_manifest, phase3d_parts, output_root):
    plan_path, output = Path(plan_path).resolve(), Path(output_root).resolve()
    preparation_sha256 = file_hash(plan_path)
    plan = load_plan(plan_path)
    sources = [plan_path.parent, Path(phase3b_manifest).resolve().parent, Path(phase3d_manifest).resolve().parent]
    sources += [Path(p).resolve() for p in phase3d_parts]
    if any(output == p or output in p.parents or p in output.parents for p in sources):
        raise ValueError("Recovery outputs must be separate from every immutable input.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty recovery directory.")
    with ExitStack() as stack:
        old = stack.enter_context(CheckedBackup(phase3b_manifest))
        newer = stack.enter_context(CheckedBackup(phase3d_manifest, phase3d_parts))
        if newer.manifest.get("phase3d_plan_fingerprint") != plan["source_plan_fingerprint"]:
            raise ValueError("Phase 3D backup belongs to a different cohort/plan.")
        original_config = old.json("run/vm_run_config.json")
        settings = plan["settings"]
        for key in ("model_name", "model_revision", "dtype", "attn_implementation", "video_fps",
                    "video_num_frames", "video_max_pixels", "roi_padding"):
            if original_config[key] != settings[key]:
                raise ValueError("Raw activation model/input settings differ from the frozen experiment.")
        pairs, mappings = newer.json("run/pairs.json"), newer.json("run/mappings.json")
        for name in ("pairs.json", "mappings.json"):
            expected = [sha for path, sha in plan["source_files_sha256"].items() if Path(path).name == name]
            if len(expected) != 1 or newer.files["run/"+name]["sha256"] != expected[0]:
                raise ValueError("Raw backup source metadata differs from the frozen preparation.")
        names = newer.json("run/feature_store/manifest.json")["feature_names"]
        audits, captures = (catalog(newer, stage, plan) for stage in ("audit", "capture"))
        wanted = {t["eval_id"] for t in plan["tasks"] if t["view"] == "archived"}
        indices = {}
        for path in old.files:
            if path.startswith("run/primary/") and "/activations/" in path and path.endswith("/index.json"):
                index = old.json(path)
                if index["eval_id"] in wanted:
                    if index["eval_id"] in indices:
                        raise ValueError("Ambiguous duplicate historical per-token capture.")
                    indices[index["eval_id"]] = (path, index)
        values, examples, decisions, recovered, missing, readout_maps = [], [], {}, [], [], {}
        layers_replayed, verified_videos = [], set()
        for task in plan["tasks"]:
            key = task["pair_id"], task["condition"], task["view"]
            row = dict(pairs[key[0]][key[1]])
            row.update(option_A=task["option_A"], option_B=task["option_B"], correct_option=task["correct_option"])
            audit, capture = audits[key], captures[key]
            mapping = checked_mapping(task, row, audit, capture, mappings[key[0]])
            readout_maps[task["eval_id"]] = mapping
            if task["view"] != "archived" or task["eval_id"] not in indices:
                missing.append({**task, "required_readouts": READOUTS, "mapping": mapping,
                    "baseline_decision": capture["decision"], "baseline_payload": capture["payload"],
                    "reason": "no verified individual contextual reference-token vectors; pooled differences are not eligible"})
                continue
            index_path, index = indices[task["eval_id"]]
            parent = str(PurePosixPath(index_path).parent)
            record = audit["processor_record"]
            if (index["group_positions"] != record["group_positions"] or index["video_metadata"] != record["video_metadata"] or
                    index["decision"] != capture["decision"] or not index["standard_parity"]["first_token_match"] or
                    not index["standard_parity"]["logits_allclose"] or index["hidden_dtype"] != "torch.float16"):
                raise ValueError("Raw activation index differs from the original contextual baseline.")
            source_video = original_config["video_paths_by_eval_id"][task["eval_id"]]
            if original_config["video_sha256"][source_video] != task["video_sha256"]:
                raise ValueError("Original video hash differs from the KO50 stimulus.")
            video_files = [f for f in old.files.values() if f["source_path"] == source_video and
                           f["path"].startswith("source_videos/")]
            if len(video_files) != 1 or video_files[0]["sha256"] != task["video_sha256"]:
                raise ValueError("Original frozen video is missing from the full backup.")
            if task["video_sha256"] not in verified_videos:
                old.read(video_files[0]["path"])
                verified_videos.add(task["video_sha256"])
            payload_name = "run/"+capture["payload"]["path"]
            if newer.files[payload_name]["sha256"] != capture["payload"]["sha256"]:
                raise ValueError("Historical baseline tensor hash binding changed.")
            measured = newer.tensor(payload_name)
            baseline = old.tensor(parent+"/baseline.pt")["logits"]
            check_logits(baseline, capture["decision"])
            if (not all(torch.equal(baseline, measured[k]) for k in
                        ("logits", "plain_logits", "identity_logits", "disabled_ko_logits")) or
                    not capture["gates"]["capture_noop"]["passed"]):
                raise ValueError("Original first-token logits differ from the KO50 baseline chain.")
            vectors = {o: [] for o in ("A", "B")}
            for layer in LAYERS:
                payload = old.tensor(parent+f"/layer_{layer:02d}.pt")
                if payload["positions"] != index["positions"]:
                    raise ValueError("Layer tensor and index position support differ.")
                replay = replay_layer(payload, audit["groups"], measured["features"], names, layer)
                layers_replayed.append({"eval_id": task["eval_id"], **replay})
                for option in ("A", "B"):
                    position = mapping["option_readout_audit"][option]["reference_last_position"]
                    vectors[option].append(payload["vectors"][payload["positions"].index(position)].float().numpy())
            for option in ("A", "B"):
                example = next(e for e in plan["statement_examples"] if e["eval_id"] == task["eval_id"] and e["option"] == option)
                examples.append(example)
                values.append(np.stack(vectors[option]))
            decisions[task["eval_id"]] = capture["decision"]
            recovered.append({"eval_id": task["eval_id"], "index_member": index_path,
                "index_sha256": old.files[index_path]["sha256"], "mapping": mapping})
            print(f"RECOVERED {len(recovered)}/100 archived prompts {task['eval_id']}", flush=True)
        for contrast in plan["boundary_pairs"]:
            if contrast["view"] != "archived":
                continue
            low, temporal = (next(e for e in examples if e["example_id"] == contrast[k])
                             for k in ("low_example_id", "temporal_example_id")) if len(recovered) == 100 else (None, None)
            if low is not None and (readout_maps[low["eval_id"]]["option_readout_audit"][low["option"]] !=
                                    readout_maps[temporal["eval_id"]]["option_readout_audit"][temporal["option"]]):
                raise ValueError("Low/Temporal reference token alignment changed.")
        if file_hash(plan_path) != preparation_sha256:
            raise ValueError("Preparation changed during recovery.")
        source_receipts = {"phase3b": old.receipt(), "phase3d": newer.receipt()}
    array = np.stack(values) if values else np.empty((0, len(READOUTS), 4096), dtype=np.float32)
    if len(examples) != len(array) or not np.isfinite(array).all():
        raise ValueError("Recovered option array is incomplete/nonfinite.")
    output.mkdir(parents=True, exist_ok=True)
    save_npz(output/"recovered_features.npz", features=array)
    for name, data in (("examples", examples), ("decisions", decisions), ("recovered_prompts", recovered),
                       ("missing_capture_tasks", missing), ("readout_maps", readout_maps),
                       ("layer_replay", layers_replayed), ("backup_receipts", source_receipts)):
        atomic_write(output/f"{name}.json", data)
    text = ["# Rescue50 Raw Option-State Recovery", "",
        f"Recovered {len(recovered)}/200 prompt views and {len(examples)}/400 statement rows.",
        f"Remaining: {len(missing)} contextual prompt captures; existing KO results remain unchanged.",
        "", "Original video bytes, reference token positions and full first-answer logits were checked.",
        "Single-token historical anchors replay exactly; pooled anchors use a derived FP32 roundoff envelope.",
        "", "No GPU forward or new probe fitting was performed. This partial store cannot train formal probes.",
        "Missing pure A/B views cannot be reconstructed by permuting old contextual vectors.", ""]
    (output/"report.md").write_text("\n".join(text), encoding="utf-8")
    summary = {"schema": OUTPUT_SCHEMA, "status": "partial_raw_recovery_complete", "complete": False,
        "ready_for_formal_probe_training": False, "preparation_sha256": file_hash(plan_path),
        "preparation_fingerprint": plan["fingerprint"], "independent_bases": 50,
        "recovered_prompt_cells": len(recovered), "recovered_statement_rows": len(examples),
        "missing_prompt_cells": len(missing), "missing_by_view": dict(Counter(t["view"] for t in missing)),
        "per_token_layers_replayed": len(layers_replayed), "frozen_video_files_verified": len(verified_videos),
        "exact_anchor_comparisons": sum(r["exact_single_token_anchors"] for r in layers_replayed),
        "historical_feature_vectors_replayed": sum(r["features_replayed"] for r in layers_replayed),
        "maximum_pooling_roundoff_difference": max((r["maximum_pooling_absolute_difference"] for r in layers_replayed), default=0),
        "shape": list(array.shape), "feature_names": READOUTS, "new_GPU_forwards": 0,
        "probe_fitting_performed": False, "code_sha256": file_hash(Path(__file__)),
        "files": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}}
    atomic_write(output/"recovery_manifest.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan_path", "phase3b_manifest", "phase3d_manifest", "output_root"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--phase3d_parts", nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(recover(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
