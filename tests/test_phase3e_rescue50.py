"""Synthetic metadata fixtures only; no GPU or scientific-result generation."""

import copy
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts import prepare_phase3e_rescue50 as core
from scripts.phase3c_core import atomic_write, digest, file_hash, read_json
from scripts.phase3d_core import audit_pairs, base_folds
from tests.test_phase3d import row


def fixture(root):
    source, singleton = root/"source", root/"single"
    annotations, mappings = {}, {}
    for base in range(1, 51):
        pair_id = f"pair_{base}"
        annotations[pair_id] = {}
        for condition in core.CONDITIONS:
            value = row(base, first=1 if base % 2 else 2,
                        correct="A" if base <= 46 else "B", condition=condition)
            value.update(video_path=f"original_videos/{base}_{condition}.mp4",
                         phase3c_video_sha256=digest([base, condition]))
            annotations[pair_id][condition] = value
        mappings[pair_id] = {"processor_records": {c: {
            "video_sha256": annotations[pair_id][c]["phase3c_video_sha256"]} for c in core.CONDITIONS},
            "support_audit": {"supports": {"whole_event2": {"one_to_one": True,
                "low_positions": list(range(512)), "temporal_positions": list(range(512))}}}}
    pairs_path, maps_path = root/"pairs.json", root/"mappings.json"
    atomic_write(pairs_path, annotations)
    atomic_write(maps_path, mappings)
    cases = audit_pairs(annotations)
    folds = base_folds(cases)
    plan = {"artifact_type": "real", "case_ids": [c["pair_id"] for c in cases],
        "settings": {"fixture_only": True}, "source_runtime": {"fixture_only": True},
        "source_files": {"pairs.json": file_hash(pairs_path), "mappings.json": file_hash(maps_path)},
        "case_metadata_sha256": digest(cases), "folds_sha256": digest(folds)}
    plan["fingerprint"] = digest(plan)
    for name, value in (("plan", plan), ("case_metadata", cases), ("outer_folds", folds)):
        atomic_write(source/f"{name}.json", value)
    atomic_write(source/"feature_store/manifest.json", {"complete": True, "artifact_type": "real",
        "plan_fingerprint": plan["fingerprint"], "feature_names": ["option_pair:L18_output"]})
    single_plan = {"artifact_type": "real", "source_plan_fingerprint": plan["fingerprint"],
                   "case_ids": plan["case_ids"], "settings": plan["settings"]}
    single_plan["fingerprint"] = digest(single_plan)
    atomic_write(singleton/"plan.json", single_plan)
    joint, singles = [], []
    for case in cases:
        result = {**{k: case[k] for k in
            ("pair_id", "base_sample_id", "first_object_id", "prompt_variant", "correct_option")},
            **core.ROUTE, "window_start": 16, "window_end": 19,
            "advantage_base": 3., "advantage_KO": 1., "compression": 2.,
            "delta_M_low": .5, "delta_M_temporal": -1.5}
        for side in ("low", "temporal"):
            good, bad = ((1., 2.) if side == "low" else (4., 2.))
            for stage in ("base", "ko"):
                delta = result[f"delta_M_{side}"] if stage == "ko" else 0
                result.update({f"{side}_{stage}_correct_logit": good+delta,
                    f"{side}_{stage}_incorrect_logit": bad, f"{side}_{stage}_margin": good+delta-bad,
                    f"{side}_{stage}_prediction": case["correct_option"] if good+delta > bad else
                        ("B" if case["correct_option"] == "A" else "A")})
            result[f"{side}_key_count"] = 512
        joint.append(result)
        singles.extend([{**copy.deepcopy(result), "window_start": layer, "window_end": layer}
                        for layer in (16, 17, 18, 19)]+[copy.deepcopy(result)])
    for directory, rows, fingerprint in ((source/"knockout/analysis", joint, plan["fingerprint"]),
                                         (singleton/"analysis", singles, single_plan["fingerprint"])):
        path = directory/"case_decomposition.json"
        atomic_write(path, rows)
        atomic_write(directory/"aggregate_summary.json", {"complete": True, "artifact_type": "real",
            "missing_count": 0, "plan_fingerprint": fingerprint,
            "output_sha256": {path.name: file_hash(path)}})
    return {"source_root": source, "pairs_path": pairs_path, "mappings_path": maps_path,
            "singleton_root": singleton, "output_root": root/"output"}


class Rescue50PreparationTest(unittest.TestCase):
    def test_exact_cohort_original_prompts_AB_control_and_frozen_folds(self):
        with tempfile.TemporaryDirectory() as temp:
            args = fixture(Path(temp))
            before = {p: file_hash(p) for p in Path(temp).rglob("*") if p.is_file()}
            audit = core.prepare(**args)
            plan = read_json(args["output_root"]/"plan.json")
            self.assertEqual(len(plan["tasks"]), 200)
            self.assertEqual(len(plan["statement_examples"]), 400)
            self.assertEqual(audit["original_low_temporal_pairs"], 50)
            self.assertEqual(audit["joint_KO_pairs"], 50)
            self.assertEqual(audit["singleton_KO_pairs"], 200)
            self.assertEqual(audit["archived_correct_answer_positions"], {"A": 46, "B": 4})
            self.assertEqual(audit["correct_answer_positions_with_permutation_per_condition"], {"A": 50, "B": 50})
            self.assertEqual(audit["existing_exact_readouts"], [])
            self.assertEqual(plan["outer_folds"], read_json(args["source_root"]/"outer_folds.json"))
            self.assertEqual(plan["fingerprint"], digest({k: v for k, v in plan.items() if k != "fingerprint"}))
            self.assertFalse(plan["new_order_reversed_videos"])
            original = read_json(args["pairs_path"])
            examples = {e["example_id"]: e for e in plan["statement_examples"]}
            self.assertEqual(Counter(e["y_statement_true"] for e in examples.values()), {0: 200, 1: 200})
            for task in plan["tasks"]:
                row_ = original[task["pair_id"]][task["condition"]]
                self.assertEqual(task["video_sha256"], row_["phase3c_video_sha256"])
                self.assertEqual(task["video_path"], row_["video_path"])
                if task["view"] == "archived":
                    self.assertTrue(task["require_rescue_behavior"])
                    self.assertEqual(task["eval_id"], row_["eval_id"])
                    self.assertEqual(task["option_A"], row_["option_A"])
                else:
                    self.assertFalse(task["require_rescue_behavior"])
                    self.assertEqual(task["option_A"], row_["option_B"])
            for contrast in plan["boundary_pairs"]:
                left, right = (examples[contrast[key]] for key in ("low_example_id", "temporal_example_id"))
                self.assertEqual(left["option_text"], right["option_text"])
                self.assertEqual(left["y_statement_true"], right["y_statement_true"])
            self.assertTrue(all(file_hash(p) == sha for p, sha in before.items()))
            with self.assertRaisesRegex(ValueError, "empty"):
                core.prepare(**args)

    def test_tampered_originals_or_folds_fail_before_output(self):
        for kind in ("pairs", "mapping", "folds", "plan", "singleton_plan"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                args = fixture(Path(temp))
                if kind == "pairs":
                    path = args["pairs_path"]
                    value = read_json(path)
                    value.pop("pair_1")
                elif kind == "mapping":
                    path = args["mappings_path"]
                    value = read_json(path)
                    value["pair_1"]["processor_records"]["low_boundary"]["video_sha256"] = "changed"
                elif kind == "folds":
                    path = args["source_root"]/"outer_folds.json"
                    value = read_json(path)
                    value[0]["test_base_ids"][0] = 999
                else:
                    path = args["source_root" if kind == "plan" else "singleton_root"]/"plan.json"
                    value = read_json(path)
                    value["settings"]["changed"] = True
                atomic_write(path, value)
                with self.assertRaises(ValueError):
                    core.prepare(**args)
                self.assertFalse(args["output_root"].exists())

    def test_KO_tampering_detected_even_if_table_hash_is_refreshed(self):
        for kind in ("checksum", "duplicate", "baseline", "route", "margin", "compression", "window"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                args = fixture(Path(temp))
                path = args["singleton_root"]/"analysis/case_decomposition.json"
                rows = read_json(path)
                if kind == "duplicate":
                    rows[1] = copy.deepcopy(rows[0])
                elif kind == "baseline":
                    rows[0]["low_base_correct_logit"] += .25
                    rows[0]["low_base_incorrect_logit"] += .25
                elif kind == "route":
                    rows[0]["query_group"] = "query_all"
                elif kind == "margin":
                    rows[0]["low_ko_margin"] += .5
                elif kind == "compression":
                    rows[0]["compression"] += .5
                elif kind == "window":
                    rows[0]["window_end"] = 20
                else:
                    rows[0]["pair_id"] = "foreign"
                atomic_write(path, rows)
                if kind != "checksum":
                    summary_path = path.parent/"aggregate_summary.json"
                    summary = read_json(summary_path)
                    summary["output_sha256"][path.name] = file_hash(path)
                    atomic_write(summary_path, summary)
                with self.assertRaises(ValueError):
                    core.prepare(**args)
                self.assertFalse(args["output_root"].exists())

    def test_source_output_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args = fixture(Path(temp))
            for target in (args["source_root"], args["source_root"]/"new", Path(temp), args["pairs_path"]):
                with self.subTest(target=target), self.assertRaises(ValueError):
                    core.prepare(**{**args, "output_root": target})


if __name__ == "__main__":
    unittest.main()
