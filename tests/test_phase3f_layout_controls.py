"""CPU tests for the Phase 3F grid, route controls and layout-symmetric summaries; no GPU or SURF."""

import collections
import json
import unittest
from pathlib import Path

from scripts import phase3f_layout_controls as core
from scripts import run_phase3f_layout_controls as runner

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "analysis/phase3c/depth50_ko25_results_20261014/run"


class GridTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"fingerprint": "f", "case_ids": [f"base_{i:03d}" for i in range(50)]}

    def test_counts_match_protocol(self):
        gates, tasks = core.specs(self.plan, "gate"), core.specs(self.plan, "intervention")
        self.assertEqual(len(gates), 200)
        self.assertEqual(len(tasks), core.EXPECTED_FORWARDS)
        kinds = collections.Counter((t["kind"], t["route"]) for t in tasks)
        self.assertEqual(kinds, {("patch", None): 100, ("ko", "event2"): 700, ("ko", "event1"): 400,
                                 ("ko", "random"): 400, ("patch_ko", "event2"): 400})
        self.assertEqual(len({t["task_id"] for t in tasks + gates}), 2200)

    def test_patch_x_ko_has_matching_ko_alone_cells(self):
        tasks = core.specs(self.plan, "intervention", ["base_000"])
        patched = {(t["view"], tuple(t["window"])) for t in tasks if t["kind"] == "patch_ko"}
        alone = {(t["view"], tuple(t["window"])) for t in tasks
                 if t["kind"] == "ko" and t["route"] == "event2" and t["condition"] == core.LOW}
        archived_refs = {(core.ARCHIVED, (17,)), (core.ARCHIVED, (18,))}
        self.assertTrue(patched <= alone | archived_refs)
        self.assertTrue(all(t["condition"] == core.LOW and t["patch"] == "temporal_to_low_L16"
                            for t in tasks if t["kind"] == "patch_ko"))

    def test_controls_cover_both_views_and_event2_comparators_exist(self):
        tasks = core.specs(self.plan, "intervention", ["base_000"])
        for route in ("event1", "random"):
            cells = {(t["view"], t["condition"], tuple(t["window"])) for t in tasks if t["route"] == route}
            self.assertEqual(len(cells), 8)
        permuted_e2 = {(t["condition"], tuple(t["window"])) for t in tasks
                       if t["route"] == "event2" and t["kind"] == "ko" and t["view"] == core.PERMUTED}
        self.assertTrue({(c, (18,)) for c in core.CONDITIONS} | {(c, tuple(core.JOINT)) for c in core.CONDITIONS}
                        <= permuted_e2)

    def test_rejects_unknown_subset(self):
        with self.assertRaises(ValueError):
            core.specs(self.plan, "intervention", ["nope"])


@unittest.skipUnless((FROZEN / "mappings.json").exists(), "frozen local mappings not available")
class RouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mappings = json.loads((FROZEN / "mappings.json").read_text())
        cls.pairs = json.loads((FROZEN / "pairs.json").read_text())
        audits = {p: {v: {c: {"processor_record": m["processor_records"][c]} for c in core.CONDITIONS}
                      for v in (core.ARCHIVED, core.PERMUTED)} for p, m in cls.mappings.items()}
        cls.routes = core.routes(cls.pairs, cls.mappings, audits)

    def test_all_bases_have_equal_disjoint_routes(self):
        self.assertEqual(len(self.routes), 50)
        for pair_id, views in self.routes.items():
            for condition, item in views[core.ARCHIVED].items():
                e1, e2, rnd = (set(item[k]) for k in ("event1", "event2", "random"))
                self.assertEqual(len(e1), 512)
                self.assertEqual(len(e2), 512)
                self.assertEqual(len(rnd), 512)
                self.assertFalse(e1 & e2 or rnd & (e1 | e2))
                visual = set(self.mappings[pair_id]["processor_records"][condition]["visual_positions"])
                self.assertTrue(e1 | e2 | rnd <= visual)
                self.assertLess(max(e1 | e2 | rnd), min(item["queries"]))
                budgets = {b["all_head_visible_causal_edges_per_layer"] for b in item["budgets"].values()}
                self.assertEqual(len(budgets), 1)

    def test_random_control_is_deterministic(self):
        pair_id = next(iter(self.mappings))
        mapping = self.mappings[pair_id]
        record = mapping["processor_records"][core.LOW]
        args = (pair_id, self.pairs[pair_id][core.LOW], record, mapping["support_audit"]["supports"], core.LOW)
        self.assertEqual(core.route_keys(*args), core.route_keys(*args))

    def test_event1_spans_whole_frame_groups(self):
        pair_id = next(iter(self.mappings))
        item = self.routes[pair_id][core.ARCHIVED][core.LOW]
        self.assertEqual(len(item["event1_frame_groups"]), 2)
        self.assertEqual(item["event1"], self.routes[pair_id][core.ARCHIVED][core.TEMPORAL]["event1"])


class SummaryTests(unittest.TestCase):
    def test_constant_letter_preference_cancels(self):
        ids = list(range(50))
        bias = 3.0
        # Archived correct = A (margin lowered by B preference), permuted correct = B (raised).
        effect = lambda p, view: 1.0 + (-bias if view == core.ARCHIVED else bias)
        result = core.layout_summary(ids, effect)
        self.assertAlmostEqual(result["symmetric"]["mean"], 1.0)
        self.assertAlmostEqual(result["archived"]["mean"], -2.0)

    def test_bootstrap_interval_contains_mean(self):
        cell = core.bootstrap([0.0, 1.0, 2.0, 3.0])
        self.assertLessEqual(cell["ci_low"], cell["mean"])
        self.assertGreaterEqual(cell["ci_high"], cell["mean"])


class AnalysisTests(unittest.TestCase):
    def test_full_grid_analysis_on_synthetic_margins(self):
        import tempfile
        from unittest import mock
        ids = [f"base_{i:03d}" for i in range(50)]
        plan = {"fingerprint": "f", "case_ids": ids}
        letter = lambda view: "A" if view == core.ARCHIVED else "B"
        bias = lambda view: -2.0 if view == core.ARCHIVED else 2.0
        base = {core.LOW: 0.5, core.TEMPORAL: 2.0}
        records = {p: {v: {c: {"baseline": {"decision": {"margin": base[c]+bias(v), "correct_option": letter(v)}}}
                           for c in core.CONDITIONS} for v in (core.ARCHIVED, core.PERMUTED)} for p in ids}
        references = {"patch": {p: {core.LOW: {"margin": base[core.LOW]+bias(core.ARCHIVED)+1.0},
                                    core.TEMPORAL: {"margin": base[core.TEMPORAL]+bias(core.ARCHIVED)-1.0}} for p in ids},
                      "event2_ko": {p: {c: {w: {"margin": base[c]+bias(core.ARCHIVED)-0.5}
                                            for w in ("16", "17", "18", "19", "16-19")} for c in core.CONDITIONS} for p in ids}}
        rows = []
        for spec in core.specs(plan, "gate") + core.specs(plan, "intervention"):
            value = base[spec["condition"]] + bias(spec["view"])
            value += {"gate": 0.0, "patch": 1.0 if spec["condition"] == core.LOW else -1.0,
                      "ko": -0.5 if spec["route"] == "event2" else -0.1, "patch_ko": 0.5}[spec["kind"]]
            rows.append({"spec": spec, "decision": {"margin": value}})
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "source_records.json").write_text(json.dumps(records))
            Path(tmp, "references.json").write_text(json.dumps(references))
            with mock.patch.object(core, "verify_rows", return_value=rows):
                core.analyze(tmp, (plan, {}, {}))
            result = json.loads(Path(tmp, "analysis/results.json").read_text())
            self.assertTrue(Path(tmp, "analysis/report.md").read_text().startswith("# Phase 3F"))
        self.assertAlmostEqual(result["patch"][core.LOW]["symmetric"]["mean"], 1.0)
        self.assertAlmostEqual(result["event2_ko"]["16-19"]["delta_low"]["symmetric"]["mean"], -0.5)
        self.assertAlmostEqual(result["controls"]["event1:16-19"]["event2_minus_control_temporal"]["symmetric"]["mean"], -0.4)
        self.assertAlmostEqual(result["patch_x_ko"]["17-23"]["rescue_under_ko"]["symmetric"]["mean"], 1.0)
        self.assertAlmostEqual(result["patch_x_ko"]["17-23"]["retained_fraction"]["permuted"], 1.0)
        self.assertEqual(result["correct_A_counts"], {core.ARCHIVED: 50, core.PERMUTED: 0})


class RunnerTests(unittest.TestCase):
    def test_backup_refuses_unexpected_directory(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "stray").mkdir()
            (Path(tmp) / "stray" / "x.json").write_text("{}")
            with self.assertRaises(ValueError):
                runner.backup_files(tmp, reports_only=True)


if __name__ == "__main__":
    unittest.main()
