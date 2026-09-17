import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts import experiment_artifacts as artifacts


class ExperimentArtifactsTest(unittest.TestCase):
    def test_part1_profile_reuses_completed_experiments(self):
        modes = artifacts.resolve_experiment_modes("part1_reuse")

        self.assertEqual(modes["ladder"], "reuse")
        self.assertEqual(modes["attention_phase1"], "reuse")
        self.assertEqual(modes["attention_phase0"], "skip")
        self.assertEqual(modes["activation_patching_phase3"], "skip")
        self.assertEqual(modes["ladder_smoke"], "skip")

    def test_overrides_are_validated(self):
        modes = artifacts.resolve_experiment_modes(
            "part1_reuse",
            {"attention_phase1": "analyze", "ladder": "skip"},
        )
        self.assertEqual(modes["attention_phase1"], "analyze")
        self.assertEqual(modes["ladder"], "skip")
        phase3_modes = artifacts.resolve_experiment_modes(
            "part1_reuse",
            {"activation_patching_phase3": "run"},
        )
        self.assertEqual(phase3_modes["activation_patching_phase3"], "run")
        with self.assertRaisesRegex(ValueError, "not meaningful"):
            artifacts.resolve_experiment_modes(
                "part1_reuse",
                {"baseline": "analyze"},
            )

    def test_require_paths_fails_before_expensive_work(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "raw_results.jsonl"
            with self.assertRaisesRegex(artifacts.ArtifactError, "required research artifact"):
                artifacts.require_paths("ladder", [missing])

    def test_latest_results_are_deduplicated_by_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dataset, timestamps in {
                "ladder_level_1": ["20260101", "20260201"],
                "ladder_level_2": ["20260101"],
            }.items():
                for timestamp in timestamps:
                    path = root / dataset / timestamp / "raw_results.jsonl"
                    path.parent.mkdir(parents=True)
                    path.write_text("{}\n", encoding="utf-8")

            latest = artifacts.latest_results_by_dataset(
                "ladder", root, "ladder_level_*", minimum=2
            )

            self.assertEqual(len(latest), 2)
            self.assertIn("20260201", str(latest[0]))

    def test_restore_real_archive_and_validate_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "results.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "archive_manifest.json",
                    json.dumps({
                        "artifact_schema_version": artifacts.ARTIFACT_SCHEMA_VERSION,
                        "artifact_type": "real",
                        "created_at": "20260910_120000",
                        "model_name": "Qwen/test",
                        "model_revision": "revision-1",
                        "source_commit": "abc123",
                        "config_fingerprint": "fingerprint-1",
                    }),
                )
                archive.writestr("analysis/summary.json", "{}")

            restored = artifacts.restore_artifact_archive(
                archive_path,
                root / "project",
                expected={
                    "model_name": "Qwen/test",
                    "model_revision": "revision-1",
                },
            )

            self.assertEqual(restored["artifact_type"], "real")
            self.assertEqual(restored["provenance_status"], "strict_validated")
            self.assertEqual(restored["validation_warnings"], [])
            self.assertTrue((root / "project/analysis/summary.json").is_file())
            self.assertFalse((root / "project/archive_manifest.json").exists())

    def test_legacy_archive_is_allowed_but_not_labelled_validated_real(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "legacy.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "archive_manifest.json",
                    json.dumps({
                        "model_name": "Qwen/test",
                        "source_commit": "abc123",
                    }),
                )
                archive.writestr("analysis/summary.json", "{}")

            restored = artifacts.restore_artifact_archive(
                archive_path,
                root / "project",
                expected={
                    "model_name": "Qwen/test",
                    "model_revision": "revision-1",
                },
            )

            self.assertEqual(restored["artifact_type"], "legacy_unverified")
            self.assertEqual(restored["provenance_status"], "legacy_unverified")
            self.assertTrue(restored["validation_warnings"])

    def test_strict_archive_rejects_missing_required_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "incomplete.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "archive_manifest.json",
                    json.dumps({
                        "artifact_schema_version": artifacts.ARTIFACT_SCHEMA_VERSION,
                        "artifact_type": "real",
                    }),
                )
            with self.assertRaisesRegex(artifacts.ArtifactError, "missing required"):
                artifacts.restore_artifact_archive(archive_path, root / "project")

    def test_mock_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "mock.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "archive_manifest.json",
                    json.dumps({"artifact_type": "mock"}),
                )
            with self.assertRaisesRegex(artifacts.ArtifactError, "research runs require real"):
                artifacts.restore_artifact_archive(archive_path, root / "project")

    def test_unsafe_archive_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "unsafe.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "archive_manifest.json",
                    json.dumps({"artifact_type": "real"}),
                )
                archive.writestr("../outside.txt", "unsafe")
            with self.assertRaisesRegex(artifacts.ArtifactError, "Unsafe path"):
                artifacts.restore_artifact_archive(archive_path, root / "project")


if __name__ == "__main__":
    unittest.main()
