"""Two-scene factorial fixtures and read-only attention-site capture for Phase 3E."""

import copy
import inspect
import random
from collections import Counter
from contextlib import ExitStack, contextmanager
from itertools import product
from pathlib import Path

import numpy as np
import torch

from . import generate_ladder_dataset as generator
from .phase3c_core import atomic_write, checked_positions, digest, file_hash, read_json
from .phase3c_interventions import assert_hidden, hidden_output
from .phase3d_core import audit_row

SEEDS = (2026102701, 2026102702)
CONDITIONS = ("low_boundary", "temporal_boundary")
LAYERS = (0, 16, 17, 18, 20)
SCHEMA = "phase3e_two_scene_technical_preflight_v1"


def scene(seed):
    """Use existing geometry helpers without changing the caller's RNG state."""
    state = random.getstate()
    try:
        random.seed(seed)
        level = generator.LEVELS[4]
        identity = generator.make_target_identity_specs(1)[0]
        one, two, _, occupied = generator.make_target_objects(level, 1, identity)
        distractors = generator.make_distractors(level, occupied, [one, two], 2, 2, False,
            static_count_override=2, moving_count_override=2)
        low = timing(CONDITIONS[0])
        return {"seed": seed, "targets": [one, two],
                "distractors": generator.timed_distractors(distractors, low)}
    finally:
        random.setstate(state)


def timing(condition):
    return generator.get_timing(condition, 15, 18, 30, 45, 0, False)


def factorial_rows(root, seeds=SEEDS):
    root = Path(root).resolve()
    rows, scenes = [], [scene(seed) for seed in seeds]
    for item, first, condition in product(scenes, (1, 2), CONDITIONS):
        schedule = timing(condition)
        objects, annotated = [], []
        for obj in item["targets"]:
            event = "first" if obj["id"] == first else "second"
            start, end = (schedule[f"{event}_event_{key}_frame"] for key in ("start", "end"))
            objects.append({**copy.deepcopy(obj), "motion": generator.make_motion(obj["path"], start, end)})
            annotated.append(generator.serialize_obj(obj, start, end))
        video_id = f"seed_{item['seed']}_first_{first}_{condition}"
        base = {"scene_seed": item["seed"], "base_sample_id": item["seed"],
            "dataset_version": SCHEMA, "difficulty_level": 5, "feature_variant": "full",
            "condition": condition, "fps": 15, "duration_sec": 18, "total_frames": 270,
            "first_object_id": first, "second_object_id": 3-first,
            "target_objects": annotated, "distractors": [], "event_timing": schedule,
            "boundary_timing": {k: schedule[k] for k in ("boundary_start_frame", "boundary_end_frame")},
            "video_id": video_id, "video_path": str(root / "videos" / (video_id + ".mp4")),
            "geometry_sha256": digest(item), "render_objects": objects,
            "render_distractors": item["distractors"]}
        for dist in item["distractors"]:
            base["distractors"].append({"id": dist["id"], "shape": dist["shape"],
                "color": dist["color"], "label": f"the {dist['color']} {dist['shape']}",
                "radius": dist.get("radius", 23), "motion_kind": dist["motion_kind"],
                **dist["motion"]})
        labels = {obj["id"]: obj["label"] for obj in annotated}
        for subject, layout in product((1, 2), ("before_first", "after_first")):
            relations = ("before", "after") if layout == "before_first" else ("after", "before")
            truth = "before" if subject == first else "after"
            options = {key: f"{labels[subject].capitalize()} moves {rel} {labels[3-subject]}."
                       for key, rel in zip(("A", "B"), relations)}
            row = {**copy.deepcopy(base), "subject_id": subject, "layout": layout,
                "prompt_variant": f"subject_{subject}_{layout}", "correct_relation": truth,
                "option_A": options["A"], "option_B": options["B"],
                "correct_option": "A" if relations[0] == truth else "B",
                "eval_id": f"{video_id}_subject_{subject}_{layout}"}
            audit_row(row)
            rows.append(row)
    audit_factorial(rows, seeds)
    return scenes, rows


def audit_factorial(rows, seeds=SEEDS):
    expected = set(product(seeds, (1, 2), CONDITIONS, (1, 2), ("before_first", "after_first")))
    actual = [(r["scene_seed"], r["first_object_id"], r["condition"], r["subject_id"], r["layout"])
              for r in rows]
    if len(actual) != len(expected) or set(actual) != expected or len({r["eval_id"] for r in rows}) != len(rows):
        raise ValueError("Incomplete/duplicated factorial cells; do not outcome-select scenes.")
    for seed in seeds:
        subset = [r for r in rows if r["scene_seed"] == seed]
        if len({r["geometry_sha256"] for r in subset}) != 1:
            raise ValueError("Scene geometry changed between factorial variants.")
        geometry = lambda r: [{k: v for k, v in obj.items() if k not in ("start_frame", "end_frame")}
                              for obj in r["target_objects"]]
        if any(geometry(r) != geometry(subset[0]) or r["distractors"] != subset[0]["distractors"] for r in subset):
            raise ValueError("Static identities/paths or fixed absolute distractor schedule changed.")
        for r in subset:
            audit_row(r)
            other = next(q for q in subset if q["first_object_id"] == r["first_object_id"] and
                         q["condition"] == r["condition"] and q["subject_id"] == r["subject_id"] and
                         q["layout"] != r["layout"])
            if (r["option_A"], r["option_B"], r["correct_option"]) != (
                    other["option_B"], other["option_A"], "B" if other["correct_option"] == "A" else "A"):
                raise ValueError("A/B views must be literal permutations, not role changes.")
    return {"independent_scene_seeds": list(seeds), "prompt_cells": len(rows),
            "unique_videos": len({r["video_id"] for r in rows}),
            "correct_option_counts": dict(Counter(r["correct_option"] for r in rows)),
            "canonical_order_counts": dict(Counter(audit_row(r)["canonical_relation"] for r in rows)),
            "distractor_schedule": "absolute frames fixed within each scene across all variants",
            "event2_mover_shortcut_eliminated": False}


def video_audit(path, expected_frames=270):
    import cv2
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("Rendered video cannot be decoded.")
    fps = capture.get(cv2.CAP_PROP_FPS)
    count, extrema, first, last, motion = 0, [], None, None, 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != (512, 512, 3):
                raise ValueError("Unexpected rendered video dimensions.")
            if first is None:
                first = frame.copy()
            if last is not None and not np.array_equal(last, frame):
                motion += 1
            last = frame.copy()
            extrema.append(float(frame.std()))
            count += 1
    finally:
        capture.release()
    if count != expected_frames or abs(fps-15) > .01 or min(extrema, default=0) <= 1 or motion == 0:
        raise ValueError("Video has wrong duration/FPS or blank/static content.")
    return {"decoded_frames": count, "fps": fps, "shape": [512, 512, 3],
            "changing_frame_pairs": motion, "min_frame_std": min(extrema), "sha256": file_hash(path)}


def prepare(root):
    root = Path(root).resolve()
    if (root / "plan.json").exists():
        return load_plan(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a new isolated output root, not old artifacts.")
    (root / "videos").mkdir(parents=True)
    scenes, rows = factorial_rows(root)
    videos = {}
    for row in rows:
        if row["video_id"] in videos:
            continue
        path = Path(row["video_path"])
        generator.render_video(path, row["condition"], 15, 18, row["render_objects"],
                               row["render_distractors"], row["event_timing"])
        videos[row["video_id"]] = video_audit(path)
    for row in rows:
        row.pop("render_objects")
        row.pop("render_distractors")
    plan = {"schema": SCHEMA, "scope": "technical preflight only; no probes/patch/KO",
            "scenes": scenes, "rows": rows, "videos": videos,
            "factorial_audit": audit_factorial(rows), "layers": list(LAYERS),
            "attention_layers": [17, 18], "standard_generation_gate": "all 32 prompts",
            "full_40_scene_run_authorized": False}
    plan["fingerprint"] = digest(plan)
    atomic_write(root / "plan.json", plan)
    return load_plan(root)


def load_plan(root):
    plan = read_json(Path(root) / "plan.json")
    if plan.get("schema") != SCHEMA or plan.get("fingerprint") != digest(
            {k: v for k, v in plan.items() if k != "fingerprint"}):
        raise ValueError("Changed preflight plan.")
    audit_factorial(plan["rows"])
    for row in plan["rows"]:
        if file_hash(row["video_path"]) != plan["videos"][row["video_id"]]["sha256"]:
            raise ValueError("Frozen video bytes changed.")
    return plan


class AttentionSites:
    """Read actual residual sites, never return a replacement from any hook."""

    def __init__(self, text_model, length, groups):
        self.layers, self.length = text_model.layers, length
        self.groups = {k: checked_positions(list(v), k, length) for k, v in groups.items()}
        self.union = sorted({p for values in self.groups.values() for p in values})
        self.raw, self.means, self.counts = {}, {}, Counter()

    def take(self, name, hidden):
        assert_hidden(hidden, self.length)
        self.counts[name] += 1
        if self.counts[name] != 1:
            raise RuntimeError("Repeated capture site visit; prefill only is required.")
        selected = hidden[0, self.union].detach().cpu().clone()
        if not bool(torch.isfinite(selected).all()):
            raise RuntimeError("Nonfinite representation.")
        if name.startswith(("L17_", "L18_")):
            self.raw[name] = selected
        self.means[name] = {k: hidden[0, values].detach().float().mean(0).cpu().clone()
                            for k, values in self.groups.items()}

    @contextmanager
    def installed(self):
        def before(layer):
            def hook(module, args, kwargs):
                bound = inspect.signature(module.forward).bind(*args, **kwargs).arguments
                self.take(f"L{layer}_input", bound["hidden_states"])
            return hook

        def output(name):
            def hook(module, args, result):
                self.take(name, hidden_output(result))
            return hook

        def post_residual(layer):
            def hook(module, args):
                self.take(f"L{layer}_post_attention_residual", args[0])
            return hook

        with ExitStack() as stack:
            for layer in LAYERS:
                block = self.layers[layer]
                stack.callback(block.register_forward_hook(output(f"L{layer}_block_output")).remove)
                if layer in (16, 17, 18):
                    stack.callback(block.register_forward_pre_hook(before(layer), with_kwargs=True).remove)
                if layer in (17, 18):
                    stack.callback(block.self_attn.register_forward_hook(
                        output(f"L{layer}_attention_contribution")).remove)
                    stack.callback(block.post_attention_layernorm.register_forward_pre_hook(
                        post_residual(layer)).remove)
            yield self

    def validate(self):
        expected = {f"L{l}_block_output" for l in LAYERS}
        expected |= {f"L{l}_input" for l in (16, 17, 18)}
        expected |= {f"L{l}_{s}" for l in (17, 18) for s in (
            "attention_contribution", "post_attention_residual")}
        if set(self.means) != expected or any(self.counts[s] != 1 for s in expected):
            raise RuntimeError("Missing capture site.")
        for layer in (17, 18):
            before = self.raw[f"L{layer}_input"]
            contribution = self.raw[f"L{layer}_attention_contribution"]
            after = self.raw[f"L{layer}_post_attention_residual"]
            if not torch.equal(before+contribution, after):
                raise RuntimeError("Actual post-attention residual != input + attention contribution.")
        if not torch.equal(self.raw["L17_block_output"], self.raw["L18_input"]):
            raise RuntimeError("Adjacent block output/input mismatch.")
        for group in self.groups:
            if not torch.equal(self.means["L16_block_output"][group], self.means["L17_input"][group]):
                raise RuntimeError("L16 output/L17 input mismatch.")
        return {"sites": sorted(expected), "one_visit_each": True,
                "residual_addition_exact_at_union_positions": True,
                "adjacent_sites_exact": True, "union_position_count": len(self.union)}

    def payload(self):
        return {"groups": self.groups, "union_positions": self.union,
                "means": self.means, "timing_raw": self.raw, "audit": self.validate()}
