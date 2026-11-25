"""Unit and hardware tests for the native-format NPU mirror effect."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from linux_studio_effects.npu import probe_npu
from linux_studio_effects.video import FrameContext, MirrorEffect as CpuMirrorEffect
from linux_studio_effects.video.effects.mirror.npu import (
    MirrorEffect as NpuMirrorEffect,
    _RGB_MIRROR_C_SRC,
    _UYVY_MIRROR_C_SRC,
    _YUYV_MIRROR_C_SRC,
    select_mirror_workers,
)


class NpuMirrorTests(unittest.TestCase):
    def test_worker_count_divides_frame_rows(self) -> None:
        for height, workers in ((1080, 4), (720, 4), (10, 2), (7, 1), (1, 1)):
            count = select_mirror_workers(height)
            self.assertEqual(count, workers)
            self.assertEqual(height % count, 0)
        with self.assertRaises(ValueError):
            select_mirror_workers(0)

    def test_kernel_sources_reverse_pixels_without_converting_format(self) -> None:
        self.assertIn("(width - 1 - x) * 3", _RGB_MIRROR_C_SRC)
        self.assertIn("(pairs - 1 - x) * 4", _YUYV_MIRROR_C_SRC)
        self.assertIn("dst[0] = src[2]", _YUYV_MIRROR_C_SRC)
        self.assertIn("dst[1] = src[3]", _UYVY_MIRROR_C_SRC)

    def test_npu_mirror_matches_cpu_native_formats(self) -> None:
        if not probe_npu().available:
            self.skipTest("No AMD NPU available on this host")

        rng = np.random.default_rng(12)
        width, height = 32, 8
        for pixel_format, shape in (
            ("RGB3", (height, width, 3)),
            ("BGR3", (height, width, 3)),
            ("YUYV", (height, width // 2, 4)),
            ("UYVY", (height, width // 2, 4)),
        ):
            frame = rng.integers(0, 256, size=shape, dtype=np.uint8)
            ctx = FrameContext(1, 0.0, width, height, 30.0, pixel_format=pixel_format)
            expected = CpuMirrorEffect().process(frame, ctx)
            effect = NpuMirrorEffect()
            try:
                actual = effect.process(frame, ctx)
            finally:
                effect.teardown()
            np.testing.assert_array_equal(actual, expected, err_msg=pixel_format)


if __name__ == "__main__":
    unittest.main()
