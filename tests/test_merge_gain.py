from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.inference.merge import new_detection_gain_after_merge, new_detection_utility_after_merge


class MergeGainTest(unittest.TestCase):
    def test_replacement_of_existing_same_class_box_is_not_new(self) -> None:
        gain = new_detection_gain_after_merge(
            image_shape=(100, 100),
            merge_iou=0.5,
            previous_boxes_parts=[np.array([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32)],
            previous_scores_parts=[np.array([0.4], dtype=np.float32)],
            previous_classes_parts=[np.array([0.0], dtype=np.float32)],
            candidate_boxes=np.array([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32),
            candidate_scores=np.array([0.9], dtype=np.float32),
            candidate_classes=np.array([0.0], dtype=np.float32),
        )

        self.assertEqual(gain, 0)

    def test_shifted_replacement_of_existing_same_class_box_is_not_new(self) -> None:
        gain = new_detection_gain_after_merge(
            image_shape=(100, 100),
            merge_iou=0.5,
            previous_boxes_parts=[np.array([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32)],
            previous_scores_parts=[np.array([0.4], dtype=np.float32)],
            previous_classes_parts=[np.array([0.0], dtype=np.float32)],
            candidate_boxes=np.array([[12.0, 12.0, 32.0, 32.0]], dtype=np.float32),
            candidate_scores=np.array([0.9], dtype=np.float32),
            candidate_classes=np.array([0.0], dtype=np.float32),
            duplicate_iou=0.5,
        )

        self.assertEqual(gain, 0)

    def test_spatially_new_same_class_box_counts_as_new(self) -> None:
        gain = new_detection_gain_after_merge(
            image_shape=(100, 100),
            merge_iou=0.5,
            previous_boxes_parts=[np.array([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32)],
            previous_scores_parts=[np.array([0.9], dtype=np.float32)],
            previous_classes_parts=[np.array([0.0], dtype=np.float32)],
            candidate_boxes=np.array([[60.0, 60.0, 80.0, 80.0]], dtype=np.float32),
            candidate_scores=np.array([0.8], dtype=np.float32),
            candidate_classes=np.array([0.0], dtype=np.float32),
        )

        self.assertEqual(gain, 1)

    def test_spatially_new_utility_uses_candidate_confidence(self) -> None:
        utility = new_detection_utility_after_merge(
            image_shape=(100, 100),
            merge_iou=0.5,
            previous_boxes_parts=[np.array([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32)],
            previous_scores_parts=[np.array([0.9], dtype=np.float32)],
            previous_classes_parts=[np.array([0.0], dtype=np.float32)],
            candidate_boxes=np.array([[60.0, 60.0, 80.0, 80.0]], dtype=np.float32),
            candidate_scores=np.array([0.8], dtype=np.float32),
            candidate_classes=np.array([0.0], dtype=np.float32),
        )

        self.assertAlmostEqual(utility, 0.8, places=6)

    def test_same_location_different_class_counts_as_new(self) -> None:
        gain = new_detection_gain_after_merge(
            image_shape=(100, 100),
            merge_iou=0.5,
            previous_boxes_parts=[np.array([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32)],
            previous_scores_parts=[np.array([0.9], dtype=np.float32)],
            previous_classes_parts=[np.array([0.0], dtype=np.float32)],
            candidate_boxes=np.array([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32),
            candidate_scores=np.array([0.8], dtype=np.float32),
            candidate_classes=np.array([2.0], dtype=np.float32),
        )

        self.assertEqual(gain, 1)


if __name__ == "__main__":
    unittest.main()
