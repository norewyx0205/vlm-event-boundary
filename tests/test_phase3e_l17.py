import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import extend_phase3e_l17_probes as runner
from scripts import reanalyze_phase3d_balanced_probes as base
from scripts.phase3c_core import atomic_write, file_hash, read_json
from tests.test_phase3e_balanced_probes import sample, SETTINGS


class L17ExtensionTest(unittest.TestCase):
    def reference(self, root):
        store = root / "store"
        atomic_write(store / "manifest.json", {"fixture": True})
        audit = {"complete": True, "resources": [
            {"path": str(store / "manifest.json"), "sha256": file_hash(store / "manifest.json")} ]}
        reference = root / "reference"
        config = {"source_manifest_sha256": file_hash(store / "manifest.json"),
                  "settings": {**SETTINGS, "label_permutations": 0},
                  "code_sha256": {"reanalyze_phase3d_balanced_probes.py": file_hash(Path(base.__file__))},
                  "training": "All balanced baseline views", "runtime": runner.runtime_info()}
        atomic_write(reference / "run_config.json", config)
        atomic_write(reference / "input_audit.json", audit)
        atomic_write(reference / "summary.json", {"schema": "phase3e_balanced_probe_reanalysis_v1",
                     "status": "complete", "base_count": 50, "baseline_rows": 200, "source_bytes_unchanged": True})
        atomic_write(reference / "output_manifest.json", {"files": [
            {"path": path.name, "sha256": file_hash(path)} for path in sorted(reference.iterdir())]})
        return store / "manifest.json", reference, audit

    def test_reference_tamper_and_different_sources_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, reference, audit = self.reference(Path(directory))
            with patch.object(base, "SETTINGS", SETTINGS), patch.object(base, "input_audit", return_value=audit):
                self.assertEqual(runner.verify_reference(reference, source)[0]["settings"]["C_grid"], [.001])
                other = source.with_name("other.json")
                atomic_write(other, {"different": True})
                with self.assertRaisesRegex(ValueError, "exact same"):
                    runner.verify_reference(reference, other)
                summary = read_json(reference / "summary.json")
                summary["changed"] = True
                atomic_write(reference / "summary.json", summary)
                with self.assertRaisesRegex(ValueError, "Changed"):
                    runner.verify_reference(reference, source)

    def test_full_l17_extension_preserves_reference_and_exports_models(self):
        rows, values, cases, folds = sample(50)
        names = runner.PRIMARY + runner.SECONDARY
        features = np.stack([values] * len(names), axis=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, reference, audit = self.reference(root)
            before = {path.name: file_hash(path) for path in reference.iterdir()}
            with patch.object(base, "SETTINGS", SETTINGS), patch.object(base, "input_audit", return_value=audit), \
                    patch.object(base, "load_dataset", return_value=({}, rows, cases, folds, names, features)), \
                    redirect_stdout(io.StringIO()):
                result = runner.run(source, reference, root / "extension")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(result["results"]), 7)
            self.assertEqual(before, {path.name: file_hash(path) for path in reference.iterdir()})
            self.assertEqual(len(read_json(root / "extension/option_pair__L17_output/held_out_predictions.json")), 300)
            self.assertEqual(len(read_json(root / "extension/delta_source_pair__L17/held_out_predictions.json")), 100)
            for item in read_json(root / "extension/output_manifest.json")["files"]:
                self.assertEqual(file_hash(root / "extension" / item["path"]), item["sha256"])
            with self.assertRaises(ValueError):
                runner.run(source, reference, root / "extension")

    def test_reference_parent_child_outputs_rejected_before_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, reference, audit = self.reference(root)
            for output in (reference, reference / "new"):
                with self.assertRaises(ValueError):
                    runner.run(source, reference, output)


if __name__ == "__main__":
    unittest.main()
