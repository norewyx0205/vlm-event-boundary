"""Backup/member and numeric-replay tests; no live inference or network."""

import copy
import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import torch
import numpy as np

from scripts import recover_phase3e_rescue50 as core
from scripts.phase3c_core import atomic_write, digest, file_hash
from scripts.phase3d_execution import seal
from scripts import verify_phase3e_rescue50_recovery as independent
from scripts.run_phase3d_probes import save_npz
from tests.test_phase3d import row


def backup_fixture(root, members=None):
    members = members or {"run/example.json": b'{"test": true}', "run/raw.pt": b"not_loaded"}
    root.mkdir(parents=True, exist_ok=True)
    path = root/"part_001.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    manifest = {"artifact_type": "real", "complete": True, "reports_only": False,
        "activation_tensors_included": True, "expected_file_count": len(members),
        "expected_total_bytes": sum(len(v) for v in members.values()),
        "archives": [{"name": path.name, "size_bytes": path.stat().st_size, "sha256": file_hash(path)}],
        "files": [{"path": n, "part": path.name, "size_bytes": len(v),
                   "sha256": hashlib.sha256(v).hexdigest()} for n, v in members.items()]}
    atomic_write(root/"backup_manifest.json", manifest)
    return root/"backup_manifest.json", manifest


class CheckedBackupTest(unittest.TestCase):
    def test_checks_whole_part_and_used_member_without_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            path, _ = backup_fixture(Path(temp))
            with core.CheckedBackup(path) as backup:
                self.assertEqual(backup.json("run/example.json"), {"test": True})
                self.assertEqual(list(backup.receipt()["verified_members"]), ["run/example.json"])
                self.assertEqual(backup.receipt()["unopened_parts"], [])
            self.assertFalse((Path(temp)/"run").exists())

    def test_whole_part_member_inventory_and_member_hash_tampering(self):
        for kind in ("part", "member", "inventory"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                path, manifest = backup_fixture(root)
                if kind == "part":
                    with (root/"part_001.zip").open("ab") as handle:
                        handle.write(b"tamper")
                elif kind == "member":
                    manifest["files"][0]["sha256"] = "0"*64
                else:
                    manifest["files"][0]["path"] = "run/foreign.json"
                atomic_write(path, manifest)
                with core.CheckedBackup(path) as backup, self.assertRaises(ValueError):
                    backup.read(manifest["files"][0]["path"])

    def test_invalid_paths_partial_manifests_and_missing_parts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, manifest = backup_fixture(root)
            for change in ("unsafe", "partial", "duplicate"):
                bad = copy.deepcopy(manifest)
                if change == "unsafe":
                    bad["files"][0]["path"] = "../outside"
                elif change == "partial":
                    bad["complete"] = False
                else:
                    bad["files"][1]["path"] = bad["files"][0]["path"]
                atomic_write(path, bad)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    core.CheckedBackup(path)
            atomic_write(path, manifest)
            with core.CheckedBackup(path, []) as backup, self.assertRaisesRegex(ValueError, "unavailable"):
                backup.read("run/example.json")


class ReplayTest(unittest.TestCase):
    def test_old_and_new_processor_schemas_match_by_exact_behavioral_fields(self):
        record = {k: [1, 2] for k in ("input_ids", "attention_mask", "visual_positions")}
        record.update(prompt_input_ids_sha256="ids", group_positions={"options_all": [1, 2]},
                      video_metadata="video", video_sha256="sha", input_tensor_sha256={"input_ids": "sha"})
        audit = {"record_sha256": "audit", "processor_record": record, "groups": {},
                 "input_fingerprint": digest(record["input_tensor_sha256"])}
        capture = {"spec": {"view": "archived"}, "audit_record_sha256": "audit",
                   "record_sha256": "capture", "input_fingerprint": audit["input_fingerprint"]}
        task = {"view": "archived", "condition": "low_boundary", "video_sha256": "sha"}
        old = {k: v for k, v in record.items() if k != "input_tensor_sha256"}
        old.update(video_path="historical/absolute/path", prompt_token_count=2)
        mapping = {"processor_records": {"low_boundary": old}}
        with patch.object(core, "archived_readouts", return_value=({}, {})):
            core.checked_mapping(task, {}, audit, capture, mapping)
            for name in ("input_ids", "attention_mask", "group_positions", "video_sha256"):
                altered = copy.deepcopy(mapping)
                altered["processor_records"]["low_boundary"][name] = "changed"
                with self.subTest(name=name), self.assertRaises(ValueError):
                    core.checked_mapping(task, {}, audit, capture, altered)

    def layer_fixture(self):
        torch.manual_seed(2)
        raw = torch.randn(18, 4096).half()
        groups = {"decision_position": [0], "query_all": [1, 2, 3]}
        for i, option in enumerate(("A", "B")):
            offset = 4+i*7
            groups.update({f"option_{option}_t1": list(range(offset, offset+3)),
                f"option_{option}_t2": list(range(offset+3, offset+6)), f"option_{option}_relation": [offset+6]})
        means = {k: raw[v].float().mean(0) for k, v in groups.items()}
        features = {"decision": means["decision_position"], "query": means["query_all"]}
        for option in ("A", "B"):
            features[f"option_{option}_pair"] = means[f"option_{option}_t1"]-means[f"option_{option}_t2"]
            features[f"option_{option}_relation"] = means[f"option_{option}_relation"]
        names = [name+":L18_output" for name in features]
        return {"positions": list(range(18)), "vectors": raw}, groups, torch.stack(list(features.values())), names

    def test_exact_anchors_and_derived_roundoff_gate(self):
        payload, groups, values, names = self.layer_fixture()
        result = core.replay_layer(payload, groups, values, names, 18)
        self.assertEqual(result["exact_single_token_anchors"], 3)
        self.assertEqual(result["features_replayed"], 6)
        tiny = values.clone()
        tiny[1, 0] = torch.nextafter(tiny[1, 0], torch.tensor(float("inf")))
        core.replay_layer(payload, groups, tiny, names, 18)
        for name in ("decision", "query", "option_A_pair"):
            bad = values.clone()
            bad[names.index(name+":L18_output"), 0] += .1
            with self.subTest(name=name), self.assertRaises(ValueError):
                core.replay_layer(payload, groups, bad, names, 18)
        missing = copy.deepcopy(groups)
        missing["option_A_t1"] = [100]
        with self.assertRaisesRegex(ValueError, "missing"):
            core.replay_layer(payload, missing, values, names, 18)

    def test_catalog_checks_seals_exact_original_views_and_no_selection(self):
        plan = {"source_plan_fingerprint": "old", "tasks": [{"pair_id": "one", "condition": "low_boundary", "view": "archived"}]}
        spec = {**plan["tasks"][0], "stage": "audit", "intervention": "baseline"}
        spec["task_id"] = digest(["old", spec])
        record = seal({"plan_fingerprint": "old", "passed": True, "spec": spec})
        encoded = __import__("json").dumps(record).encode()
        with tempfile.TemporaryDirectory() as temp:
            path, _ = backup_fixture(Path(temp), {"run/audit/task_checkpoints/one.json": encoded})
            with core.CheckedBackup(path) as backup:
                result = core.catalog(backup, "audit", plan)
                self.assertEqual(list(result), [("one", "low_boundary", "archived")])
                altered = copy.deepcopy(plan)
                altered["tasks"].append({"pair_id": "two", "condition": "low_boundary", "view": "archived"})
                with self.assertRaisesRegex(ValueError, "200"):
                    core.catalog(backup, "audit", altered)


class IndependentReplayTest(unittest.TestCase):
    def test_reference_lookup_uses_identity_and_literal_option_section(self):
        index = {"group_positions": {"options_all": list(range(9))+list(range(10, 19)),
            "option_target_1_mentions": [1, 2, 11, 12], "option_target_2_mentions": [5, 6, 15, 16]}}
        self.assertEqual(independent.reference_position(index, row(subject=1), "A"), 6)
        self.assertEqual(independent.reference_position(index, row(subject=1), "B"), 16)
        self.assertEqual(independent.reference_position(index, row(subject=2), "A"), 2)
        index["group_positions"]["options_all"] = list(range(19))
        with self.assertRaisesRegex(ValueError, "spans"):
            independent.reference_position(index, row(), "A")

    def test_partial_store_hashes_and_readiness_cannot_be_promoted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save_npz(root/"recovered_features.npz", features=np.zeros((200, 4, 4096), dtype=np.float32))
            names = {"examples.json", "decisions.json", "recovered_prompts.json", "readout_maps.json",
                     "layer_replay.json", "backup_receipts.json", "missing_capture_tasks.json"}
            for name in names:
                atomic_write(root/name, [])
            (root/"report.md").write_text("Synthetic fixture", encoding="ascii")
            m = {"schema": core.OUTPUT_SCHEMA, "complete": False, "ready_for_formal_probe_training": False,
                 "recovered_prompt_cells": 100, "missing_prompt_cells": 100, "recovered_statement_rows": 200,
                 "shape": [200, 4, 4096], "missing_by_view": {"option_order_permuted": 100},
                 "files": {p.name: file_hash(p) for p in root.iterdir()}}
            path = root/"recovery_manifest.json"
            atomic_write(path, m)
            self.assertEqual(independent.load_partial(path)[1].shape, (200, 4, 4096))
            for change in ("ready", "unbound", "altered"):
                bad = copy.deepcopy(m)
                if change == "ready":
                    bad["complete"] = True
                elif change == "unbound":
                    bad["files"].pop("examples.json")
                else:
                    bad["files"]["examples.json"] = "changed"
                atomic_write(path, bad)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    independent.load_partial(path)


if __name__ == "__main__":
    unittest.main()
