"""Unit and NPU tests for AutomaticFramingEffect."""

from __future__ import annotations

import pathlib
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from linux_studio_effects.npu import probe_npu
from linux_studio_effects.video import (
    DeviceUnavailableError,
    FrameContext,
    FrameGraph,
    create_effect,
)
from linux_studio_effects.video.effects.automatic_framing.cpu import (
    AutomaticFramingEffect as CpuAutomaticFramingEffect,
    estimate_framing_target,
)
from linux_studio_effects.video.effects.automatic_framing.npu import (
    AutomaticFramingEffect as NpuAutomaticFramingEffect,
)
from linux_studio_effects.video.effects.mirror.cpu import MirrorEffect
from linux_studio_effects.video.models.modnet import create_synthetic_weights, get_modnet_config


class AutomaticFramingUnitTests(unittest.TestCase):
    """Unit tests for subject localization, smooth pan/zoom, and native formats."""

    def test_estimate_framing_target_tracks_subject_and_empty_frame(self) -> None:
        alpha = np.zeros((128, 256), dtype=np.float32)
        # Empty frame returns None
        self.assertIsNone(estimate_framing_target(alpha))

        # Place off-center subject on right side (x in [160, 224], y in [32, 120])
        alpha[32:120, 160:224] = 1.0
        est = estimate_framing_target(alpha, min_zoom=1.0, max_zoom=2.0, headroom=0.08, side_margin=0.10)
        self.assertIsNotNone(est)
        assert est is not None
        (tx, ty, tz), bbox = est
        self.assertGreater(tx, 0.55)
        self.assertGreater(tz, 1.05)
        self.assertLessEqual(tz, 2.0)
        self.assertLessEqual(bbox[1], 33 / 128.0)

    def test_smooth_pan_zoom_deadzone_and_lost_hold(self) -> None:
        cfg = get_modnet_config("webcam", max_size=128)
        weights = create_synthetic_weights(cfg, seed=3)
        eff = CpuAutomaticFramingEffect(
            max_size=128,
            weights=weights,
            async_tracking=False,
            pan_smoothing=0.25,
            zoom_smoothing=0.25,
            deadzone=0.02,
            lost_hold_frames=2,
            min_zoom=1.0,
            max_zoom=2.0,
        )

        a_left = np.zeros((64, 128), dtype=np.float32)
        a_left[16:56, 16:48] = 1.0
        cx0, cy0, z0 = eff.update_framing_from_alpha(a_left)
        self.assertLess(cx0, 0.4)
        self.assertGreater(z0, 1.0)

        # Tiny jitter within deadzone should keep pose unchanged
        a_jitter = np.zeros((64, 128), dtype=np.float32)
        a_jitter[16:56, 17:49] = 1.0
        cx_j, cy_j, z_j = eff.update_framing_from_alpha(a_jitter)
        self.assertAlmostEqual(cx_j, cx0, places=5)

        # Shift right -> smooth interpolation moves gradually toward right
        a_right = np.zeros((64, 128), dtype=np.float32)
        a_right[16:56, 80:112] = 1.0
        cx1, _, _ = eff.update_framing_from_alpha(a_right)
        cx2, _, _ = eff.update_framing_from_alpha(a_right)
        self.assertGreater(cx1, cx0)
        self.assertGreater(cx2, cx1)
        self.assertLess(cx2, 0.75)

        # Subject disappears -> holds for lost_hold_frames=2, then drifts back toward wide 1.0
        empty = np.zeros((64, 128), dtype=np.float32)
        for _ in range(2):
            eff.update_framing_from_alpha(empty)
        _, _, z_hold = eff.update_framing_from_alpha(empty)
        for _ in range(15):
            _, _, z_after = eff.update_framing_from_alpha(empty)
        self.assertLess(z_after, z_hold)

    def test_automatic_framing_native_formats_async_and_frame_graph(self) -> None:
        cfg = get_modnet_config("webcam", max_size=128)
        weights = create_synthetic_weights(cfg, seed=5)
        rng = np.random.default_rng(5)

        for fmt, shape in (
            ("RGB3", (120, 160, 3)),
            ("BGR3", (120, 160, 3)),
            ("YUYV", (120, 80, 4)),
            ("UYVY", (120, 80, 4)),
        ):
            frame = rng.integers(0, 256, size=shape, dtype=np.uint8)
            eff = CpuAutomaticFramingEffect(max_size=128, weights=weights, async_tracking=False, zoom_bias=1.2)
            eff.setup(160, 120, 30.0)
            try:
                out = eff.process(frame, FrameContext(1, 0.0, 160, 120, 30.0, pixel_format=fmt))
                self.assertEqual(out.shape, shape, msg=fmt)
                self.assertEqual(out.dtype, np.uint8, msg=fmt)
                self.assertIsNotNone(eff.last_crop, msg=fmt)
            finally:
                eff.teardown()

        yuyv = rng.integers(0, 256, size=(72, 64, 4), dtype=np.uint8)
        graph = FrameGraph()
        graph.add("mirror", MirrorEffect(device="cpu"))
        graph.add(
            "framing",
            create_effect(
                "automatic_framing",
                device="cpu",
                max_size=128,
                weights=weights,
                async_tracking=True,
                track_interval_s=0.0,
            ),
            source="mirror",
        )
        graph.set_output("framing")
        graph.setup(128, 72, 30.0)
        try:
            for idx in range(4):
                out = graph.process(yuyv, FrameContext(idx + 1, idx / 30.0, 128, 72, 30.0, pixel_format="YUYV"))
                self.assertEqual(out.shape, yuyv.shape)
                self.assertEqual(out.dtype, np.uint8)
                time.sleep(0.02)
            node = graph.nodes[1][2]
            self.assertGreaterEqual(node.tracker_updates, 1)  # type: ignore[attr-defined]
        finally:
            graph.teardown()

    def test_gpu_backend_rejected_for_automatic_framing(self) -> None:
        with self.assertRaises(DeviceUnavailableError):
            create_effect("automatic_framing", device="gpu")


class AutomaticFramingNpuTests(unittest.TestCase):
    """Hardware tests verifying NPU1 AutomaticFramingEffect parity."""

    def test_npu_vs_cpu_automatic_framing_parity(self) -> None:
        if not probe_npu().available:
            self.skipTest("No AMD NPU available on this host")

        cfg = get_modnet_config("webcam", max_size=384)
        weights = create_synthetic_weights(cfg, seed=23)
        rgb = np.random.default_rng(23).integers(0, 256, size=(240, 320, 3), dtype=np.uint8)

        cpu_eff = CpuAutomaticFramingEffect(max_size=384, weights=weights, async_tracking=False)
        npu_eff = NpuAutomaticFramingEffect(max_size=384, weights=weights, async_tracking=False)
        try:
            cpu_out = cpu_eff.process(rgb, FrameContext(1, 0.0, 320, 240, 30.0))
            npu_out = npu_eff.process(rgb, FrameContext(1, 0.0, 320, 240, 30.0))
        finally:
            cpu_eff.teardown()
            npu_eff.teardown()

        assert cpu_eff.last_result is not None and npu_eff.last_result is not None
        corr = float(np.corrcoef(cpu_eff.last_result.alpha.ravel(), npu_eff.last_result.alpha.ravel())[0, 1])
        self.assertGreater(corr, 0.99)
        self.assertEqual(cpu_out.shape, npu_out.shape)
        self.assertEqual(cpu_eff.last_crop, npu_eff.last_crop)


if __name__ == "__main__":
    unittest.main()
