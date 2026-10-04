import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import phase3b_baseline as baseline
from scripts import phase3b_checkpoint_reuse as reuse
from scripts import repair_phase3b_cohort as repair
from scripts import run_phase3b_baseline as gate
from scripts import run_phase3b_vm as vm
from scripts import analyze_phase3b
from scripts.phase3b_core import GROUPS, PATCH_LAYERS
from scripts.common import read_jsonl


def pair(base=1, prompt="original", stratum="primary_rescue", behavior="temporal_rescue"):
    result = {}
    for condition, correct in zip(baseline.CONDITIONS, baseline.BEHAVIORS[behavior]):
        correct_option = "A" if prompt == "original" else "B"
        result[condition] = {
            "eval_id": f"{base}_{prompt}_{condition}", "base_sample_id": base, "prompt_variant": prompt,
            "condition": condition, "phase3b_pair_id": f"pair_{base:03d}_{prompt}",
            "correct_option": correct_option, "first_object_id": 1 if base % 2 else 2,
            "archived_prediction": correct_option if correct else ("B" if correct_option == "A" else "A"),
            "phase3b_analysis_stratum": stratum, "phase3b_prompt_pair_behavior": behavior,
        }
    return result


def baseline_rows(rows):
    return {
        row["eval_id"]: {
            "eval_id": row["eval_id"],
            "capture_decision": {
                "prediction": row["archived_prediction"], "correct_option": row["correct_option"],
                "margin": 1.0 if row["archived_prediction"] == row["correct_option"] else -1.0,
            },
            "standard_parity": {"first_token_match": True, "logits_allclose": True},
            "archived_input_parity": {"available": True, "matches": True},
        } for row in rows.values()
    }


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class BaselineTest(unittest.TestCase):
    def test_strict_gate_for_each_prompt_pair_not_base_category(self):
        for prompt in ("original", "swapped"):
            rows = pair(prompt=prompt)
            self.assertEqual(baseline.pair_failures(rows, baseline_rows(rows)), [])
        control = pair(prompt="swapped", stratum="mirrored_prompt_control", behavior="stable_both_correct")
        self.assertEqual(baseline.pair_failures(control, baseline_rows(control)), [])
        control["low_boundary"]["phase3b_prompt_pair_behavior"] = "temporal_rescue"
        self.assertTrue(baseline.pair_failures(control, baseline_rows(pair(prompt="swapped", behavior="stable_both_correct"))))

    def test_tie_nonfinite_mismatch_and_missing_evidence_block(self):
        rows = pair()
        for value in (0.0, float("nan"), float("inf"), -0.125):
            saved = baseline_rows(rows)
            saved[rows["temporal_boundary"]["eval_id"]]["capture_decision"]["margin"] = value
            self.assertTrue(baseline.pair_failures(rows, saved))
        for field, key in (("standard_parity", "logits_allclose"), ("standard_parity", "first_token_match"),
                           ("archived_input_parity", "matches")):
            saved = baseline_rows(rows)
            saved[rows["low_boundary"]["eval_id"]][field][key] = False
            self.assertTrue(baseline.pair_failures(rows, saved))
        saved = baseline_rows(rows)
        saved[rows["temporal_boundary"]["eval_id"]]["capture_decision"]["prediction"] = "B"
        self.assertTrue(baseline.pair_failures(rows, saved))
        self.assertTrue(baseline.pair_failures(rows, {}))

    def test_replacements_ascending_independent_original_preferred(self):
        candidates = [
            {"base_sample_id": base, "prompt_variant": prompt, "eligible": True}
            for base, prompt in ((449, "swapped"), (447, "swapped"), (447, "original"), (446, "original"), (451, "original"))
        ]
        selected = repair.pick_replacements(candidates)
        self.assertEqual([(row["base_sample_id"], row["prompt_variant"]) for row in selected],
                         [(446, "original"), (447, "original"), (449, "swapped")])
        candidates[3]["eligible"] = False
        self.assertEqual([row["base_sample_id"] for row in repair.pick_replacements(candidates)], [447, 449, 451])

    def test_gate_coverage_staleness_and_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = pair()
            write_rows(root / "selection/analysis_case_manifest.jsonl", rows.values())
            plan = {"pipeline_fingerprint": "new", "pair_count": 1}
            with self.assertRaisesRegex(RuntimeError, "Missing"):
                gate.require_gate(root, plan)
            output = root / "baseline_audit"
            write_rows(output / "rows.jsonl", baseline_rows(rows).values())
            vm.write_json(output / "config.json", {"pipeline_fingerprint": "new", "measurement_code_sha256": baseline.measurement_hashes()})
            summary = {"passed": True, "pipeline_fingerprint": "new", "pair_count": 1,
                       "rows_sha256": baseline.sha256(output / "rows.jsonl"), "config_sha256": baseline.sha256(output / "config.json")}
            vm.write_json(output / "summary.json", summary)
            self.assertTrue(gate.require_gate(root, plan)["passed"])
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                gate.require_gate(root, {**plan, "pipeline_fingerprint": "old"})
            saved = list(baseline_rows(rows).values())
            saved[-1]["capture_decision"]["margin"] = 0
            write_rows(output / "rows.jsonl", saved)
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                gate.require_gate(root, plan)
            summary["rows_sha256"] = baseline.sha256(output / "rows.jsonl")
            vm.write_json(output / "summary.json", summary)
            with self.assertRaisesRegex(RuntimeError, "strict"):
                gate.require_gate(root, plan)

    def test_amendment_preserves_controls_preflight_and_old_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / "old", Path(temporary) / "new"
            pairs = {rows["low_boundary"]["phase3b_pair_id"]: rows for rows in (pair(base) for base in range(1, 51))}
            control = pair(100, stratum="stable_both_correct_control", behavior="stable_both_correct")
            pairs[control["low_boundary"]["phase3b_pair_id"]] = control
            selection = source / "selection"
            write_rows(selection / "analysis_case_manifest.jsonl", [row for rows in pairs.values() for row in rows.values()])
            write_rows(selection / "preflight_case_manifest.jsonl", [row for key in list(pairs)[:2] for row in pairs[key].values()])
            write_rows(selection / "selected_video_mappings.jsonl", [{"pair_id": key, "eligible": True} for key in pairs])
            vm.write_json(selection / "case_selection_summary.json", {
                "primary_bases": list(range(1, 51)), "representative_pair_ids": {"target_1_first": list(pairs)[0], "target_2_first": list(pairs)[1]},
            })
            source_hash = baseline.sha256(selection / "analysis_case_manifest.jsonl")
            replacements = []
            for base in (446, 447, 450):
                rows = pair(base)
                key = rows["low_boundary"]["phase3b_pair_id"]
                replacements.append({"base_sample_id": base, "pair_id": key, "eligible": True,
                                     "manifest_rows": list(rows.values()), "baseline_rows": list(baseline_rows(rows).values()),
                                     "mapping": {"pair_id": key, "eligible": True}})
            excluded = [{"pair_id": key} for key in list(pairs)[-4:-1]]
            repair.freeze_amendment(source, output, pairs, excluded, replacements, {"order": "ascending"})
            manifest = vm.manifest_pairs(output / "analysis_case_manifest.jsonl")
            self.assertEqual(len(manifest), 51)
            self.assertEqual(manifest[control["low_boundary"]["phase3b_pair_id"]], control)
            self.assertEqual(baseline.sha256(selection / "analysis_case_manifest.jsonl"), source_hash)
            self.assertEqual(read_jsonl(selection / "preflight_case_manifest.jsonl"), read_jsonl(output / "preflight_case_manifest.jsonl"))
            # Identical resume is allowed; a changed frozen selection is not.
            repair.freeze_amendment(source, output, pairs, excluded, replacements, {"order": "ascending"})
            bad = copy.deepcopy(replacements)
            bad[0]["base_sample_id"] = 1
            with self.assertRaisesRegex(RuntimeError, "duplicates"):
                repair.freeze_amendment(source, output, pairs, excluded, bad, {})

    def test_byte_identical_reuse_checks_every_file_no_fingerprint_rewrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / "old", root / "new"
            source.mkdir()
            (source / "run_config.json").write_text('{"run_fingerprint":"old"}')
            (source / "layer_00.pt").write_bytes(b"tensor")
            records = {}
            reuse.copy_verified(source, target / "primary/checkpoints/shard_00", records, target)
            vm.write_json(target / "checkpoint_reuse.json", {
                "schema": "phase3b_checkpoint_reuse_v1", "old_fingerprints_rewritten": False,
                "measurement_code_sha256": baseline.measurement_hashes(), "files_sha256": records,
                "target_pipeline_fingerprint": "new", "manifest_sha256": "manifest", "shards": {},
            })
            self.assertIsNotNone(reuse.validate_reuse(target, {"pipeline_fingerprint": "new"}))
            self.assertEqual(json.loads((target / "primary/checkpoints/shard_00/run_config.json").read_text())["run_fingerprint"], "old")
            with self.assertRaisesRegex(RuntimeError, "another"):
                reuse.validate_reuse(target, {"pipeline_fingerprint": "changed"})
            (target / "primary/checkpoints/shard_00/layer_00.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                reuse.validate_reuse(target)
            with self.assertRaisesRegex(RuntimeError, "overwrite"):
                reuse.copy_verified(source, target / "primary/checkpoints/shard_00", {}, target)

    def test_full_skips_only_certified_completed_shard(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(stage="full", output_root=temporary, selection_dir=temporary,
                                   gpus=["0", "1"], execution_mode="model_parallel")
            runner = SimpleNamespace(started=0, calls=[], run=lambda *call: runner.calls.append(call), cancel=lambda: None)
            certificate = {"shards": {"shard_00": {"pair_ids": ["old"]}}}
            with patch.object(reuse, "validate_reuse", return_value=certificate), patch.object(vm, "model_options", return_value=[]):
                vm.run_shards(args, runner, Path(temporary) / "manifest", lambda _: Path(temporary) / "checkpoints", {"model_parallel": [0, 1]})
            self.assertEqual(len(runner.calls), 2)
            self.assertTrue(all("--shard_index" in command and command[command.index("--shard_index") + 1] == "1" for command, *_ in runner.calls))

    def test_merge_keeps_shard_fingerprints_and_only_allows_explicit_cohort_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in (0, 1):
                shard = root / f"shard_{index:02d}"
                vm.write_json(shard / "run_config.json", {
                    "run_fingerprint": f"fingerprint_{index}", "shard_index": index,
                    "pair_ids": [f"pair_{index}"], "manifest_sha256": f"cohort_{index}",
                    "mapping_sha256": f"mappings_{index}", "repo_commit": f"commit_{index}",
                    "patching_code_sha256": baseline.measurement_hashes(), "model_revision": "pinned",
                })
                write_rows(shard / "patches" / f"pair_{index}" / "temporal_to_low.jsonl", [
                    {"run_fingerprint": f"fingerprint_{index}", "phase3b_pair_id": f"pair_{index}"},
                ])
            with self.assertRaisesRegex(ValueError, "incompatible"):
                analyze_phase3b.read_shards(root)
            _, patches, _, configs, _ = analyze_phase3b.read_shards(root, {"shards": {}})
            self.assertEqual([item["run_fingerprint"] for item in patches], ["fingerprint_0", "fingerprint_1"])
            config = configs[-1]
            config["model_revision"] = "other"
            vm.write_json(root / "shard_01/run_config.json", config)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                analyze_phase3b.read_shards(root, {"shards": {}})


if __name__ == "__main__":
    unittest.main()
