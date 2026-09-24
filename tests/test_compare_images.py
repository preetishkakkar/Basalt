import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "tools" / "compare_images.py"
SPEC = importlib.util.spec_from_file_location("compare_images", MODULE_PATH)
compare_images = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_images)


class CompareImagesTests(unittest.TestCase):
    def test_equal_luminance_different_colors_fail_rgb_comparison(self):
        red = np.zeros((8, 8, 3), dtype=np.float32)
        green = np.zeros_like(red)
        red[:, :, 0] = 1.0
        green[:, :, 1] = 0.2126 / 0.7152
        self.assertGreater(compare_images.compare(red, green, 8)["aggregate"], 0.5)

    def test_partial_edge_blocks_are_included(self):
        a = np.ones((9, 9, 3), dtype=np.float32)
        b = a.copy()
        b[8, :, :] = 100.0
        b[:, 8, :] = 100.0
        result = compare_images.compare(a, b, 8)
        self.assertGreater(result["aggregate"], 0.1)
        self.assertEqual(result["worst_index"], (0, 1))

    def test_nonfinite_pixels_are_rejected(self):
        a = np.ones((2, 2, 3), dtype=np.float32)
        a[0, 0, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            compare_images.compare(a, np.ones_like(a), 1)

    def test_mean_rgb_energy_error_is_reported_independently(self):
        a = np.ones((4, 4, 3), dtype=np.float32)
        b = np.full_like(a, 1.02)
        result = compare_images.compare(a, b, 2)
        self.assertAlmostEqual(result["energy_error"], 0.02 / 1.01, places=6)

    def test_zero_block_size_is_rejected(self):
        a = np.ones((2, 2, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "block size"):
            compare_images.compare(a, a, 0)

    def test_truncated_pfm_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.pfm"
            path.write_bytes(b"PF\n2 2\n-1.0\n" + np.zeros(3, dtype="<f4").tobytes())
            with self.assertRaisesRegex(ValueError, "expected 12"):
                compare_images.read_pfm(str(path))

    def test_pfm_orientation_and_endianness(self):
        top_to_bottom = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.pfm"
            with path.open("wb") as output:
                output.write(b"PF\n3 2\n-1.0\n")
                output.write(top_to_bottom[::-1].astype("<f4").tobytes())
            np.testing.assert_array_equal(compare_images.read_pfm(str(path)), top_to_bottom)


if __name__ == "__main__":
    unittest.main()
