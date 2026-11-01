"""Run with env-radiomics Python; synthetic masks only."""
import runpy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
extract = runpy.run_path(str(ROOT / "03_feature_extraction/extract_shape_features.py"))
PARAMS = str(ROOT / "03_feature_extraction/pyradiomics_shape.yaml")


class RadiomicsEnvironment(unittest.TestCase):
    def test_small_structure_has_finite_features(self):
        array = np.zeros((9, 9, 9), dtype=np.uint8)
        array[2:6, 2:7, 2:7] = 1
        row = extract["structure_row"]("1", "T0", "sternum", 1, sitk.GetImageFromArray(array), PARAMS)
        self.assertEqual(row["status"], "computed")
        self.assertEqual(row["error"], "")
        self.assertTrue(np.isfinite([row[k] for k in extract["SHAPE_FEATURES"]]).all())

    def test_permutation_and_flip_preserve_all_features(self):
        array = np.zeros((19, 23, 27), dtype=np.uint8)
        array[3:15, 4:19, 5:20] = 1
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((0.7, 0.8, 2.5))
        results = []
        with tempfile.TemporaryDirectory() as directory:
            for i, mask in enumerate((image, sitk.PermuteAxes(image, [2, 0, 1]),
                                      sitk.Flip(image, [True, False, True]))):
                path = Path(directory) / f"mask{i}.nii.gz"
                sitk.WriteImage(mask, str(path))
                exam = {"pid": "1", "timepoint": "T0", "sternum_path": path.name, "vertebrae_path": "",
                        "structures": {"sternum": 1}}
                row = extract["extract_exam"](directory, exam, PARAMS)[0]
                self.assertEqual(row["status"], "computed")
                results.append([row[k] for k in extract["SHAPE_FEATURES"]])
        np.testing.assert_allclose(results, np.tile(results[0], (3, 1)), atol=1e-10, rtol=0)

    def test_empty_label_is_a_structure_error(self):
        array = np.zeros((9, 9, 9), dtype=np.uint8)
        row = extract["structure_row"]("1", "T0", "sternum", 1, sitk.GetImageFromArray(array), PARAMS)
        self.assertEqual(row["status"], "error")
        self.assertTrue(row["error"])
        self.assertTrue(all(row[k] == "" for k in extract["SHAPE_FEATURES"]))


if __name__ == "__main__":
    unittest.main()
