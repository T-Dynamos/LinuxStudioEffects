"""Unit and NPU parity tests for PortraitLightEffect."""

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
from linux_studio_effects.video.effects.mirror.cpu import MirrorEffect
from linux_studio_effects.video.effects.portrait_light import cpu as light_cpu
from linux_studio_effects.video.effects.portrait_light.cpu import (
    PortraitLightEffect as CpuPortraitLightEffect,
)
from linux_studio_effects.video.effects.portrait_light.npu import (
    PortraitLightEffect as NpuPortraitLightEffect,
)
from linux_studio_effects.video.models.video_depth_anything import (
    create_synthetic_weights,
    get_video_depth_config,
)


class PortraitLightUnitTests(unittest.TestCase):
    """CPU unit tests for Portrait Light relighting kernels, 3D geometry, and native formats."""

    def test_relight_kernels_match_numpy_reference(self) -> None:
        rng = np.random.default_rng(23)
        rgb = rng.integers(0, 256, size=(47, 83, 3), dtype=np.uint8)
        gain_3 = rng.uniform(0.3, 2.5, size=(47, 83, 3)).astype(np.float32)

        got_rgb = light_cpu._relight_rgb_u8(rgb, gain_3)
        y = rng.integers(0, 256, size=(47, 83), dtype=np.uint8)
        gain_1 = gain_3[:, :, 0]
        got_y = light_cpu._relight_luma_u8(y, gain_1)

        orig_lib = light_cpu._LIGHT_LIB
        light_cpu._LIGHT_LIB = None
        try:
            ref_rgb = light_cpu._relight_rgb_u8(rgb, gain_3)
            ref_y = light_cpu._relight_luma_u8(y, gain_1)
        finally:
            light_cpu._LIGHT_LIB = orig_lib

        self.assertEqual(got_rgb.dtype, np.uint8)
        self.assertEqual(got_y.dtype, np.uint8)
        self.assertLessEqual(int(np.max(np.abs(got_rgb.astype(np.int32) - ref_rgb.astype(np.int32)))), 1)
        self.assertLessEqual(int(np.max(np.abs(got_y.astype(np.int32) - ref_y.astype(np.int32)))), 1)

    def test_set_params_and_quality_presets(self) -> None:
        fast = CpuPortraitLightEffect(quality="fast", use_real_weights=False)
        balanced = CpuPortraitLightEffect(quality="balanced", use_real_weights=False)
        high = CpuPortraitLightEffect(quality="high", use_real_weights=False)
        self.assertEqual((fast.depth_config.variant, fast.depth_config.input_size), ("small", 196))
        self.assertEqual((balanced.depth_config.variant, balanced.depth_config.input_size), ("base", 252))
        self.assertEqual((high.depth_config.variant, high.depth_config.input_size), ("base", 336))

        fast.set_params(
            light_pos=(0.2, 0.3, 1.0),
            intensity=2.1,
            ambient=0.85,
            bg_dim=0.35,
            light_radius=0.9,
            light_color=(1.0, 0.9, 0.8),
            normal_strength=4.0,
            normal_smooth_radius=3,
            depth_focus=0.5,
            depth_softness=0.25,
            depth_percentiles=(2.0, 98.0),
            temporal_alpha=0.5,
            visualize_light=True,
            async_depth=False,
            depth_interval_s=0.1,
        )
        self.assertEqual(fast.light_pos, (0.2, 0.3, 1.0))
        self.assertAlmostEqual(fast.intensity, 2.1)
        self.assertTrue(fast.visualize_light)
        self.assertFalse(fast.async_depth)

        for bad in (
            {"light_pos": (0.5, 0.5, 0.01)},
            {"intensity": -0.1},
            {"ambient": -0.2},
            {"bg_dim": 1.2},
            {"light_radius": 0.01},
            {"light_color": (1.0, -0.1, 1.0)},
            {"depth_focus": 1.5},
            {"depth_softness": 0.0},
            {"depth_percentiles": (90.0, 10.0)},
            {"temporal_alpha": 1.2},
            {"unknown_param": 42},
        ):
            with self.assertRaises(ValueError):
                fast.set_params(**bad)

    def test_compute_light_map_directional_response(self) -> None:
        cfg = get_video_depth_config("small", input_size=140)
        weights = create_synthetic_weights(cfg, seed=3)
        eff = CpuPortraitLightEffect(
            variant="small",
            input_size=140,
            weights=weights,
            temporal_alpha=1.0,
            ambient=0.85,
            intensity=2.0,
            bg_dim=0.30,
            async_depth=False,
        )
        eff.setup(160, 120, 30.0)
        try:
            yy, xx = np.mgrid[0:140, 0:140].astype(np.float32) / 140.0
            # Centered smooth portrait disparity mound on top of flat background
            r2 = (xx - 0.5) ** 2 + (yy - 0.5) ** 2
            depth_raw = np.where(r2 < 0.09, 1.0 - 2.5 * r2, 0.05).astype(np.float32)

            eff.set_params(light_pos=(0.10, 0.50, 0.90))
            left_map = eff.compute_light_map(depth_raw, 120, 160)

            eff.set_params(light_pos=(0.90, 0.50, 0.90))
            right_map = eff.compute_light_map(depth_raw, 120, 160)
        finally:
            eff.teardown()

        self.assertEqual(left_map.shape, (120, 160))
        self.assertGreater(float(left_map[40:80, 40:80].mean()), float(right_map[40:80, 40:80].mean()))
        self.assertGreater(float(right_map[40:80, 80:120].mean()), float(left_map[40:80, 80:120].mean()))
        # Foreground portrait mound stays brighter than dimmed corners
        self.assertGreater(float(left_map[50:70, 70:90].mean()), float(left_map[0:15, 0:15].mean()))

    def test_native_formats_and_visualize_light(self) -> None:
        cfg = get_video_depth_config("small", input_size=140)
        weights = create_synthetic_weights(cfg, seed=7)
        rng = np.random.default_rng(7)

        for fmt, shape in (
            ("RGB3", (120, 160, 3)),
            ("BGR3", (120, 160, 3)),
            ("YUYV", (120, 80, 4)),
            ("UYVY", (120, 80, 4)),
        ):
            frame = rng.integers(0, 256, size=shape, dtype=np.uint8)
            eff = CpuPortraitLightEffect(
                variant="small",
                input_size=140,
                weights=weights,
                async_depth=False,
            )
            eff.setup(160, 120, 30.0)
            try:
                out = eff.process(frame, FrameContext(1, 0.0, 160, 120, 30.0, pixel_format=fmt))
                self.assertEqual(out.shape, shape, msg=fmt)
                self.assertEqual(out.dtype, np.uint8, msg=fmt)
                self.assertIsNotNone(eff.last_light_map, msg=fmt)

                eff.set_params(visualize_light=True)
                vis = eff.process(frame, FrameContext(2, 0.033, 160, 120, 30.0, pixel_format=fmt))
                self.assertEqual(vis.shape, shape, msg=fmt)
                if fmt in ("RGB3", "BGR3"):
                    np.testing.assert_array_equal(vis[:, :, 0], vis[:, :, 1])
                    np.testing.assert_array_equal(vis[:, :, 1], vis[:, :, 2])
            finally:
                eff.teardown()

    def test_async_depth_worker_and_frame_graph(self) -> None:
        cfg = get_video_depth_config("small", input_size=140)
        weights = create_synthetic_weights(cfg, seed=11)
        yuyv = np.random.default_rng(11).integers(0, 256, size=(72, 64, 4), dtype=np.uint8)

        graph = FrameGraph()
        graph.add("mirror", MirrorEffect(device="cpu"))
        graph.add(
            "portrait_light",
            create_effect(
                "portrait_light",
                device="cpu",
                variant="small",
                input_size=140,
                weights=weights,
                async_depth=True,
                depth_interval_s=0.0,
            ),
            source="mirror",
        )
        graph.set_output("portrait_light")
        graph.setup(128, 72, 30.0)
        try:
            for idx in range(4):
                out = graph.process(yuyv, FrameContext(idx + 1, idx / 30.0, 128, 72, 30.0, pixel_format="YUYV"))
                self.assertEqual(out.shape, yuyv.shape)
                self.assertEqual(out.dtype, np.uint8)
                time.sleep(0.02)
            node = graph.nodes[1][2]
            self.assertGreaterEqual(node.depth_updates, 1)  # type: ignore[attr-defined]
        finally:
            graph.teardown()

    def test_gpu_backend_rejected_for_portrait_light(self) -> None:
        with self.assertRaises(DeviceUnavailableError):
            create_effect("portrait_light", device="gpu")


class PortraitLightNpuTests(unittest.TestCase):
    """Hardware tests verifying NPU1 Portrait Light parity with CPU."""

    def test_npu_vs_cpu_portrait_light_parity(self) -> None:
        if not probe_npu().available:
            self.skipTest("No AMD NPU available on this host")

        cfg = get_video_depth_config("small", input_size=140)
        weights = create_synthetic_weights(cfg, seed=19)
        rgb = np.random.default_rng(19).integers(0, 256, size=(120, 160, 3), dtype=np.uint8)

        auto_eff = create_effect("portrait_light", variant="small", input_size=140, weights=weights, async_depth=False)
        self.assertIsInstance(auto_eff, NpuPortraitLightEffect)
        auto_eff.teardown()

        cpu_eff = CpuPortraitLightEffect(variant="small", input_size=140, weights=weights, async_depth=False)
        npu_eff = NpuPortraitLightEffect(variant="small", input_size=140, weights=weights, async_depth=False)
        try:
            cpu_out = cpu_eff.process(rgb, FrameContext(1, 0.0, 160, 120, 30.0))
            npu_out = npu_eff.process(rgb, FrameContext(1, 0.0, 160, 120, 30.0))
            assert cpu_eff.last_result is not None and npu_eff.last_result is not None
            corr = float(np.corrcoef(cpu_eff.last_result.depth.ravel(), npu_eff.last_result.depth.ravel())[0, 1])
            self.assertGreater(corr, 0.99)
            assert cpu_eff.last_light_map is not None and npu_eff.last_light_map is not None
            self.assertLess(float(np.max(np.abs(cpu_eff.last_light_map - npu_eff.last_light_map))), 0.05)
            self.assertEqual(cpu_out.shape, npu_out.shape)
        finally:
            cpu_eff.teardown()
            npu_eff.teardown()


if __name__ == "__main__":
    unittest.main()
