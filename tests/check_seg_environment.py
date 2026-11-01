"""Run with env-seg Python; no inference, download or GPU allocation."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from totalsegmentator.config import send_usage_stats
from totalsegmentator.map_to_binary import class_map
from totalsegmentator.postprocessing import postprocess_vertebrae_pp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.totalsegmentator_patch import check_patch  # noqa: E402


class SegmentationEnvironment(unittest.TestCase):
    def test_single_connected_component_postprocessing(self):
        self.assertEqual(len(check_patch()), 64)
        array = np.zeros((20, 20, 20), dtype=np.uint8)
        array[3:10, 3:10, 3:7] = 21
        array[3:10, 3:10, 7:11] = 22
        result = postprocess_vertebrae_pp(array, class_map["vertebrae_pp"], dilation_mm=0)
        self.assertEqual(result.shape, array.shape)
        self.assertEqual(set(np.unique(result)), {0, 22})

    def test_telemetry_disabled_setting_sends_nothing(self):
        with patch("requests.post", side_effect=AssertionError("Unexpected network request")):
            send_usage_stats({"send_usage_stats": False}, {})


if __name__ == "__main__":
    unittest.main()
