from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.cache import DetectionCache, detection_cache_is_current, load_detection_cache, save_detection_cache


def _cache(metadata: dict) -> DetectionCache:
    return DetectionCache(
        image_path="synthetic.jpg",
        image_shape=(32, 32),
        boxes=np.zeros((0, 4), dtype=np.float32),
        scores=np.zeros((0,), dtype=np.float32),
        classes=np.zeros((0,), dtype=np.float32),
        feature=np.zeros((4,), dtype=np.float32),
        feature_layers=(10,),
        objectness_map=np.zeros((1, 16, 16), dtype=np.float32),
        spatial_feature_map=np.zeros((4, 16, 16), dtype=np.float32),
        metadata=metadata,
    )


class DetectionCacheMetadataTest(unittest.TestCase):
    def test_expected_metadata_mismatch_invalidates_cache(self) -> None:
        metadata = {"imgsz": 640, "feature_layers": (10,), "weights": {"path": "model.pt"}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "det.npz"
            save_detection_cache(path, _cache(metadata))

            self.assertTrue(detection_cache_is_current(path, metadata))
            self.assertFalse(detection_cache_is_current(path, {**metadata, "imgsz": 320}))

    def test_load_round_trips_metadata(self) -> None:
        metadata = {"imgsz": 640, "conf": 0.01}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "det.npz"
            save_detection_cache(path, _cache(metadata))

            loaded = load_detection_cache(path)

        self.assertEqual(loaded.metadata, metadata)


if __name__ == "__main__":
    unittest.main()
