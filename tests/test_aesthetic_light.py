"""Unit and NPU parity tests for AestheticLightEffect."""

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
from linux_studio_effects.video.effects.aesthetic_light.cpu import (
    LIGHT_COLOR_PRESETS,
    LIGHT_SOURCE_PRESETS,
    AestheticLightEffect as CpuAestheticLightEffect,
)
from linux_studio_effects.video.effects.aesthetic_light.npu import (
    AestheticLightEffect as NpuAestheticLightEffect,
)
from linux_studio_effects.video.effects.mirror.cpu import MirrorEffect
from linux_studio_effects.video.models.video_depth_anything import (
    create_synthetic_weights,
    get_video_depth_config,
)
from linux_studio_effects.video.pixels import split_yuv422


class AestheticLightUnitTests(unittest.TestCase):
    """CPU unit tests for configurable side source presets, warm-white shading, and native formats."""

    def test_source_and_color_presets_and_validation(self) -> None:
        eff = CpuAestheticLightEffect(
            quality="fast",
            source="right",
            color_preset="golden",
            use_real_weights=False,
        )
        self.assertEqual(eff.source, "right")
        self.assertEqual(eff.color_preset, "golden")
        self.assertEqual(eff.light_pos, LIGHT_SOURCE_PRESETS["right"])
        self.assertEqual(eff.light_color, LIGHT_COLOR_PRESETS["golden"])

        eff.set_params(source="top_left", color_preset="warm_white", split_sharpness=2.1, specular_strength=0.35, bg_spill=0.15)
        self.assertEqual(eff.source, "top_left")
        self.assertEqual(eff.light_pos, LIGHT_SOURCE_PRESETS["top_left"])
        self.assertEqual(eff.light_color, LIGHT_COLOR_PRESETS["warm_white"])

        eff.set_params(light_pos=(0.2, 0.3, 0.9), light_color=(1.0, 0.85, 0.7))
        self.assertEqual(eff.source, "custom")
        self.assertEqual(eff.color_preset, "custom")

        for bad in (
            {"source": "behind"},
            {"color_preset": "ultraviolet"},
            {"shadow_color": (1.0, -0.1, 1.0)},
            {"split_sharpness": 0.0},
            {"specular_strength": -0.2},
            {"bg_spill": 1.5},
        ):
            with self.assertRaises(ValueError):
                eff.set_params(**bad)

    def test_directional_side_split_light_asymmetry(self) -> None:
        cfg = get_video_depth_config("small", input_size=140)
        weights = create_synthetic_weights(cfg, seed=5)
        eff = CpuAestheticLightEffect(
            variant="small",
            input_size=140,
            weights=weights,
            source="left",
            temporal_alpha=1.0,
            async_depth=False,
        )
        eff.setup(160, 120, 30.0)
        try:
            yy, xx = np.mgrid[0:140, 0:140].astype(np.float32) / 140.0
            r2 = (xx - 0.5) ** 2 + (yy - 0.5) ** 2
            depth_raw = np.where(r2 < 0.09, 1.0 - 2.5 * r2, 0.05).astype(np.float32)

            left_map = eff.compute_light_map(depth_raw, 120, 160)
            eff.set_params(source="right")
            right_map = eff.compute_light_map(depth_raw, 120, 160)
        finally:
            eff.teardown()

        # Left source strongly illuminates left cheek over right cheek; right source flips it.
        self.assertGreater(float(left_map[40:80, 40:80].mean()), float(left_map[40:80, 80:120].mean()))
        self.assertGreater(float(right_map[40:80, 80:120].mean()), float(right_map[40:80, 40:80].mean()))
        self.assertGreater(float(left_map[40:80, 40:80].mean()), float(right_map[40:80, 40:80].mean()))

    def test_native_formats_and_warm_chroma_tint(self) -> None:
        cfg = get_video_depth_config("small", input_size=140)
        weights = create_synthetic_weights(cfg, seed=9)
        rng = np.random.default_rng(9)

        for fmt, shape in (
            ("RGB3", (120, 160, 3)),
            ("BGR3", (120, 160, 3)),
            ("YUYV", (120, 80, 4)),
            ("UYVY", (120, 80, 4)),
        ):
            frame = rng.integers(80, 180, size=shape, dtype=np.uint8)
            if fmt in ("YUYV", "UYVY"):
                # Neutral gray YUV422 frame so warm-white tint direction is clean to check.
                y_in = np.full((120, 160), 140, dtype=np.uint8)
                u_in = np.full((120, 80), 128, dtype=np.uint8)
                v_in = np.full((120, 80), 128, dtype=np.uint8)
                from linux_studio_effects.video.pixels import pack_yuv422

                frame = pack_yuv422(y_in, u_in, v_in, fmt)

            eff = CpuAestheticLightEffect(
                variant="small",
                input_size=140,
                weights=weights,
                source="left",
                color_preset="warm_white",
                async_depth=False,
            )
            eff.setup(160, 120, 30.0)
            try:
                out = eff.process(frame, FrameContext(1, 0.0, 160, 120, 30.0, pixel_format=fmt))
                self.assertEqual(out.shape, shape, msg=fmt)
                self.assertEqual(out.dtype, np.uint8, msg=fmt)
                if fmt in ("YUYV", "UYVY"):
                    _, u_out, v_out = split_yuv422(out, fmt)
                    # Warm tint adds red-yellow (Cr/V up, Cb/U down where key light hits).
                    self.assertGreaterEqual(int(v_out.max()), 128)
                    self.assertLessEqual(int(u_out.min()), 128)

                eff.set_params(visualize_light=True)
                vis = eff.process(frame, FrameContext(2, 0.033, 160, 120, 30.0, pixel_format=fmt))
                self.assertEqual(vis.shape, shape, msg=fmt)
            finally:
                eff.teardown()

    def test_async_depth_worker_and_frame_graph(self) -> None:
        cfg = get_video_depth_config("small", input_size=140)
        weights = create_synthetic_weights(cfg, seed=11)
        yuyv = np.random.default_rng(11).integers(0, 256, size=(72, 64, 4), dtype=np.uint8)

        graph = FrameGraph()
        graph.add("mirror", MirrorEffect(device="cpu"))
        graph.add(
            "aesthetic_light",
            create_effect(
                "aesthetic_light",
                device="cpu",
                variant="small",
                input_size=140,
                weights=weights,
                async_depth=True,
                depth_interval_s=0.0,
            ),
            source="mirror",
        )
        graph.set_output("aesthetic_light")
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

    def test_gpu_backend_rejected_for_aesthetic_light(self) -> None:
        with self.assertRaises(DeviceUnavailableError):
            create_effect("aesthetic_light", device="gpu")


class AestheticLightNpuTests(unittest.TestCase):
    """Hardware tests verifying NPU1 Aesthetic Light parity with CPU."""

    def test_npu_vs_cpu_aesthetic_light_parity(self) -> None:
        if not probe_npu().available:
            self.skipTest("No AMD NPU available on this host")

        cfg = get_video_depth_config("small", input_size=140)
        weights = create_synthetic_weights(cfg, seed=19)
        rgb = np.random.default_rng(19).integers(0, 256, size=(120, 160, 3), dtype=np.uint8)

        auto_eff = create_effect("aesthetic_light", variant="small", input_size=140, weights=weights, async_depth=False)
        self.assertIsInstance(auto_eff, NpuAestheticLightEffect)
        auto_eff.teardown()

        cpu_eff = CpuAestheticLightEffect(variant="small", input_size=140, weights=weights, async_depth=False)
        npu_eff = NpuAestheticLightEffect(variant="small", input_size=140, weights=weights, async_depth=False)
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
