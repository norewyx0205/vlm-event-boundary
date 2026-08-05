import unittest

import numpy as np

from scripts import make_roi_perturbation_dataset as perturb


def temporal_row():
    return {
        "condition": "temporal_boundary",
        "total_frames": 270,
        "fps": 15,
        "target_objects": [
            {
                "id": 1,
                "shape": "circle",
                "radius": 14,
                "from": [100, 100],
                "to": [100, 150],
                "start_frame": 30,
                "end_frame": 60,
            },
            {
                "id": 2,
                "shape": "circle",
                "radius": 48,
                "from": [300, 300],
                "to": [250, 300],
                "start_frame": 105,
                "end_frame": 135,
            },
        ],
        "distractors": [
            {
                "id": 1,
                "shape": "circle",
                "motion_kind": "unrelated_motion",
                "from": [50, 200],
                "to": [100, 200],
                "start_frame": 30,
                "end_frame": 60,
            },
            {
                "id": 2,
                "shape": "circle",
                "motion_kind": "unrelated_motion",
                "from": [200, 200],
                "to": [250, 200],
                "start_frame": 105,
                "end_frame": 135,
            },
        ],
        "event_timing": {
            "first_event_start_frame": 30,
            "first_event_end_frame": 60,
            "second_event_start_frame": 105,
            "second_event_end_frame": 135,
            "unrelated_event_start_frame": None,
            "unrelated_event_end_frame": None,
        },
        "boundary_timing": {
            "boundary_start_frame": 60,
            "boundary_end_frame": 105,
            "gap_frames": 45,
            "visual_marker": "none",
            "audio_marker": "none",
        },
    }


class TemporalInterventionTest(unittest.TestCase):
    def test_gap_removed_preserves_duration_and_advances_second_event(self):
        plan, updates = perturb.temporal_intervention(
            temporal_row(), "gap_removed", 270, 15, 1.0
        )

        self.assertEqual(len(plan), 270)
        self.assertEqual(updates["boundary_timing"]["gap_frames"], 0)
        self.assertEqual(updates["event_timing"]["second_event_start_frame"], 60)
        self.assertEqual(updates["target_objects"][1]["start_frame"], 60)
        self.assertEqual(updates["distractors"][0]["start_frame"], 30)
        self.assertEqual(updates["distractors"][1]["start_frame"], 60)
        self.assertEqual(updates["distractors"][1]["end_frame"], 90)

    def test_gap_shortened_keeps_one_second_between_events(self):
        plan, updates = perturb.temporal_intervention(
            temporal_row(), "gap_shortened", 270, 15, 1.0
        )

        self.assertEqual(len(plan), 270)
        self.assertEqual(updates["boundary_timing"]["gap_frames"], 15)
        self.assertEqual(updates["event_timing"]["second_event_start_frame"], 75)
        self.assertEqual(updates["distractors"][0]["start_frame"], 30)
        self.assertEqual(updates["distractors"][1]["start_frame"], 75)
        self.assertEqual(updates["distractors"][1]["end_frame"], 105)

    def test_gap_shifted_places_hold_before_first_event(self):
        plan, updates = perturb.temporal_intervention(
            temporal_row(), "gap_shifted", 270, 15, 1.0
        )

        self.assertEqual(len(plan), 270)
        self.assertEqual(updates["boundary_timing"]["temporal_gap_location"], "before_first_event")
        self.assertEqual(updates["event_timing"]["first_event_start_frame"], 75)
        self.assertEqual(updates["event_timing"]["second_event_start_frame"], 105)
        self.assertEqual(updates["distractors"][0]["start_frame"], 75)
        self.assertEqual(updates["distractors"][0]["end_frame"], 105)
        self.assertEqual(updates["distractors"][1]["start_frame"], 105)

    def test_background_sham_matches_reference_area_without_object_overlap(self):
        reference = np.zeros((64, 64), dtype=np.uint8)
        reference[20:30, 20:30] = 255
        occupied = np.zeros((64, 64), dtype=np.uint8)
        occupied[24:40, 24:40] = 255

        sham = perturb.area_matched_background_mask(reference, occupied, (20, 0))

        self.assertEqual(np.count_nonzero(sham), np.count_nonzero(reference))
        self.assertEqual(np.count_nonzero((sham > 0) & (occupied > 0)), 0)


if __name__ == "__main__":
    unittest.main()
