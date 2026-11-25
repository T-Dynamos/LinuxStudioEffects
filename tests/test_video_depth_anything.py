"""Unit and hardware verification tests for streaming Video Depth Anything (Small / Base / Large)."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from linux_studio_effects.npu import probe_npu
from linux_studio_effects.video import (
    DeviceUnavailableError,
    FrameContext,
    is_device_available,
)
from linux_studio_effects.video.effects.video_depth.cpu import VideoDepthEffect as CpuVideoDepthEffect
from linux_studio_effects.video.effects.video_depth.gpu import VideoDepthEffect as GpuVideoDepthEffect
from linux_studio_effects.video.effects.video_depth.npu import VideoDepthEffect as NpuVideoDepthEffect
from linux_studio_effects.video.models.video_depth_anything import (
    NpuMatMulExecutor,
    VideoDepthAnythingModel,
    analyze_feasibility,
    create_synthetic_weights,
    depth_to_colormap_rgb,
    format_feasibility_table,
    get_video_depth_config,
)


class VideoDepthAnythingUnitTests(unittest.TestCase):
    """Offline tests for Small/Base/Large configs, feasibility math, streaming state, and CPU forward."""

    def test_variants_and_parameter_counts(self) -> None:
        s_rep = analyze_feasibility(get_video_depth_config("small", input_size=518))
        b_rep = analyze_feasibility(get_video_depth_config("base", input_size=518))
        l_rep = analyze_feasibility(get_video_depth_config("large", input_size=518))

        # ViT backbone + DPT + 4 MotionModules
        self.assertGreater(s_rep.total_params, 25_000_000)
        self.assertLess(s_rep.total_params, 35_000_000)
        self.assertGreater(b_rep.total_params, 100_000_000)
        self.assertGreater(l_rep.total_params, 350_000_000)

        self.assertLess(s_rep.total_gflops, b_rep.total_gflops)
        self.assertLess(b_rep.total_gflops, l_rep.total_gflops)
        self.assertTrue(s_rep.npu1_sram_tile_fit)

        table = format_feasibility_table(("small", "base", "large"), (252, 518))
        self.assertIn("small", table)
        self.assertIn("base", table)
        self.assertIn("large", table)

    def test_streaming_temporal_forward_cpu_across_small_base_large(self) -> None:
        rng = np.random.default_rng(1)
        rgb0 = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
        rgb1 = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
        for variant in ("small", "base", "large"):
            cfg = get_video_depth_config(variant, input_size=140)
            weights = create_synthetic_weights(cfg, seed=1)
            model = VideoDepthAnythingModel(cfg, weights, backend="cpu")
            r0 = model.infer(rgb0)
            r1 = model.infer(rgb1)
            self.assertEqual(r0.frame_index, 0)
            self.assertEqual(r1.frame_index, 1)
            self.assertEqual(r1.depth.shape, (140, 140), msg=variant)
            self.assertEqual(r1.depth.dtype, np.float32, msg=variant)
            self.assertTrue(np.all(np.isfinite(r1.depth)))
            model.reset_temporal_state()
            r_reset = model.infer(rgb0)
            self.assertEqual(r_reset.frame_index, 0)
            model.close()

    def test_colormap_and_cpu_effect_interface(self) -> None:
        rgb = np.random.default_rng(2).integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
        depth = np.linspace(0.0, 10.0, 140 * 140, dtype=np.float32).reshape(140, 140)
        vis = depth_to_colormap_rgb(depth, out_width=160, out_height=120)
        self.assertEqual(vis.shape, (120, 160, 3))
        self.assertEqual(vis.dtype, np.uint8)

        eff = CpuVideoDepthEffect(variant="small", input_size=140, use_real_weights=False)
        eff.setup(160, 120, 30.0)
        out0 = eff.process(rgb, FrameContext(1, 0.0, 160, 120, 30.0))
        out1 = eff.process(rgb, FrameContext(2, 0.033, 160, 120, 30.0))
        eff.teardown()
        self.assertEqual(out0.shape, (120, 160, 3))
        self.assertEqual(out1.shape, (120, 160, 3))
        self.assertEqual(out1.dtype, np.uint8)

    def test_gpu_effect_checks_device_cleanly(self) -> None:
        if not is_device_available("gpu"):
            with self.assertRaises(DeviceUnavailableError):
                GpuVideoDepthEffect()


class VideoDepthAnythingNpuTests(unittest.TestCase):
    """Hardware tests verifying NPU1 bf16 MatMul and streaming Video Depth Anything parity."""

    def test_npu_matmul_padded_tokens_accuracy(self) -> None:
        if not probe_npu().available:
            self.skipTest("No AMD NPU available on this host")

        rng = np.random.default_rng(11)
        a = (rng.standard_normal((325, 384)) * 0.25).astype(np.float32)
        b = (rng.standard_normal((384, 384)) * 0.25).astype(np.float32)

        mm_exec = NpuMatMulExecutor()
        try:
            got = mm_exec.matmul(a, b, cache_b=True)
        finally:
            mm_exec.close()

        ref = a @ b
        self.assertEqual(got.shape, ref.shape)
        corr = float(np.corrcoef(got.ravel(), ref.ravel())[0, 1])
        self.assertGreater(corr, 0.999)

    def test_npu_vs_cpu_video_depth_small_streaming_parity(self) -> None:
        if not probe_npu().available:
            self.skipTest("No AMD NPU available on this host")

        cfg = get_video_depth_config("small", input_size=140)
        weights = create_synthetic_weights(cfg, seed=5)
        rng = np.random.default_rng(5)
        rgb0 = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
        rgb1 = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)

        cpu_model = VideoDepthAnythingModel(cfg, weights, backend="cpu")
        npu_model = VideoDepthAnythingModel(cfg, weights, backend="npu")
        try:
            cpu_model.infer(rgb0)
            npu_model.infer(rgb0)
            cpu_res = cpu_model.infer(rgb1)
            npu_res = npu_model.infer(rgb1)
        finally:
            cpu_model.close()
            npu_model.close()

        self.assertEqual(cpu_res.depth.shape, npu_res.depth.shape)
        self.assertIsNotNone(npu_res.npu_stats)
        assert npu_res.npu_stats is not None
        self.assertGreater(npu_res.npu_stats.dispatches, 0)

        corr = float(np.corrcoef(cpu_res.depth.ravel(), npu_res.depth.ravel())[0, 1])
        self.assertGreater(corr, 0.99)

        eff = NpuVideoDepthEffect(variant="small", input_size=140, use_real_weights=False)
        eff.setup(320, 240, 30.0)
        vis = eff.process(rgb0, FrameContext(1, 0.0, 320, 240, 30.0))
        eff.teardown()
        self.assertEqual(vis.shape, (240, 320, 3))


if __name__ == "__main__":
    unittest.main()
