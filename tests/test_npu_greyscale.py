"""Tests for NPU-accelerated RGB -> Greyscale video effect."""

from __future__ import annotations

import pathlib
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from linux_studio_effects.npu import probe_npu
from linux_studio_effects.video import (
    CameraConfig,
    DeviceUnavailableError,
    FrameContext,
    GreyscaleEffect,
    PipelineConfig,
    V4L2Camera,
    VideoPipeline,
    check_camera_deps,
    pick_default_loopback,
)
from linux_studio_effects.video.effects.greyscale.npu import select_npu_tiling


def _ref_greyscale(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.int32)
    y = ((77 * arr[:, :, 0] + 150 * arr[:, :, 1] + 29 * arr[:, :, 2] + 128) >> 8).astype(np.uint8)
    return np.repeat(y[:, :, np.newaxis], 3, axis=2)


class NpuGreyscaleTests(unittest.TestCase):
    """Verify NPU tiling, bit-exact greyscale kernel, and live video pipeline."""

    def test_tiling_selector(self) -> None:
        self.assertEqual(select_npu_tiling(1280 * 720), (4, 2560))
        self.assertEqual(select_npu_tiling(1920 * 1080), (4, 1280))
        self.assertEqual(select_npu_tiling(640 * 480), (4, 2560))
        self.assertEqual(select_npu_tiling(13 * 7), (1, 1))

    def test_cpu_device_rejected_for_npu_only_effect(self) -> None:
        with self.assertRaises(DeviceUnavailableError):
            GreyscaleEffect(device="cpu")

    def test_npu_greyscale_matches_reference_exact(self) -> None:
        if not probe_npu().available:
            self.skipTest("No AMD NPU available on this host")

        rng = np.random.default_rng(7)
        for h, w in ((120, 160), (720, 1280)):
            rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
            eff = GreyscaleEffect()
            eff.setup(w, h, 30.0)
            try:
                out = eff.process(rgb, FrameContext(1, 0.0, w, h, 30.0))
            finally:
                eff.teardown()
            self.assertEqual(out.shape, (h, w, 3))
            self.assertEqual(out.dtype, np.uint8)
            np.testing.assert_array_equal(out, _ref_greyscale(rgb))

    def test_live_camera_npu_greyscale_to_loopback(self) -> None:
        if not probe_npu().available:
            self.skipTest("No AMD NPU available on this host")
        deps = check_camera_deps()
        loopback = pick_default_loopback(list(deps.devices))
        if not deps.ok or deps.default_device is None or loopback is None:
            self.skipTest("Requires both an accessible camera and a loaded v4l2loopback device")

        eff = GreyscaleEffect()
        cfg = PipelineConfig(
            input_device=deps.default_device.path,
            input_width=1280,
            input_height=720,
            input_format="MJPG",
            input_fps=30.0,
            output_device=loopback.path,
            output_width=640,
            output_height=480,
            output_format="YUYV",
            output_fps=30.0,
        )
        pipe = VideoPipeline(cfg, effect=eff)
        holder: dict[str, object] = {}

        t = threading.Thread(target=lambda: holder.setdefault("stats", pipe.run(duration_s=3.5)), daemon=True)
        t.start()

        deadline = time.monotonic() + 10.0
        while pipe.negotiated_output is None and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.5)

        received = []
        with V4L2Camera(
            CameraConfig(device=loopback.path, width=640, height=480, pixel_format="YUYV", fps=30.0)
        ) as consumer:
            for _ in range(3):
                frame = consumer.read_frame(decode=True, timeout_s=4.0)
                assert frame.data is not None
                received.append(frame.data)

        t.join(timeout=10.0)
        stats = holder.get("stats")
        self.assertIsNotNone(stats)
        self.assertEqual(len(received), 3)
        # After YUYV roundtrip on greyscale RGB, R, G, and B channels stay within +-2.
        sample = received[-1].astype(np.int32)
        self.assertLessEqual(int(np.max(np.abs(sample[:, :, 0] - sample[:, :, 1]))), 3)
        self.assertLessEqual(int(np.max(np.abs(sample[:, :, 1] - sample[:, :, 2]))), 3)


if __name__ == "__main__":
    unittest.main()
