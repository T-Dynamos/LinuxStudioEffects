"""Unit and hardware verification tests for Depth Anything V2 (Small / Base / Large)."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from linux_studio_effects.npu import probe_npu
from linux_studio_effects.video import (
    CpuDepthAnythingEffect,
    DeviceUnavailableError,
    FrameContext,
    GpuDepthAnythingEffect,
    NpuDepthAnythingEffect,
    is_device_available,
)
from linux_studio_effects.video.models.depth_anything_v2 import (
    DepthAnythingV2Model,
    NpuMatMulExecutor,
    analyze_feasibility,
    create_synthetic_weights,
    depth_to_colormap_rgb,
    format_feasibility_table,
    get_depth_config,
)


class DepthAnythingV2UnitTests(unittest.TestCase):
    """Offline tests for Small/Base/Large configs, feasibility math, colormap, and CPU forward."""

    def test_variants_and_parameter_counts(self) -> None:
        s_rep = analyze_feasibility(get_depth_config("small", input_size=518))
        b_rep = analyze_feasibility(get_depth_config("base", input_size=518))
        l_rep = analyze_feasibility(get_depth_config("large", input_size=518))

        self.assertAlmostEqual(s_rep.total_params / 1e6, 24.8, delta=0.5)
        self.assertAlmostEqual(b_rep.total_params / 1e6, 97.5, delta=1.0)
        # Depth-Anything-V2-Large-hf uses neck_hidden_sizes=(256,256,256,256) (~319.7M vs 335.3M in dpt.py).
        self.assertAlmostEqual(l_rep.total_params / 1e6, 319.7, delta=2.0)

        self.assertLess(s_rep.total_gflops, b_rep.total_gflops)
        self.assertLess(b_rep.total_gflops, l_rep.total_gflops)
        self.assertTrue(s_rep.npu1_sram_tile_fit)

        table = format_feasibility_table(("small", "base", "large"), (252, 518))
        self.assertIn("small", table)
        self.assertIn("base", table)
        self.assertIn("large", table)

    def test_synthetic_forward_cpu_across_small_base_large(self) -> None:
        rgb = np.random.default_rng(1).integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
        for variant in ("small", "base", "large"):
            cfg = get_depth_config(variant, input_size=140)
            weights = create_synthetic_weights(cfg, seed=1)
            model = DepthAnythingV2Model(cfg, weights, backend="cpu")
            res = model.infer(rgb)
            self.assertEqual(res.depth.shape, (140, 140), msg=variant)
            self.assertEqual(res.depth.dtype, np.float32, msg=variant)
            self.assertTrue(np.all(np.isfinite(res.depth)))
            model.close()

    def test_colormap_and_cpu_effect_interface(self) -> None:
        rgb = np.random.default_rng(2).integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
        depth = np.linspace(0.0, 10.0, 140 * 140, dtype=np.float32).reshape(140, 140)
        vis = depth_to_colormap_rgb(depth, out_width=160, out_height=120)
        self.assertEqual(vis.shape, (120, 160, 3))
        self.assertEqual(vis.dtype, np.uint8)

        eff = CpuDepthAnythingEffect(variant="small", input_size=140, use_real_weights=False)
        eff.setup(160, 120, 30.0)
        out = eff.process(rgb, FrameContext(1, 0.0, 160, 120, 30.0))
        eff.teardown()
        self.assertEqual(out.shape, (120, 160, 3))
        self.assertEqual(out.dtype, np.uint8)

    def test_gpu_effect_checks_device_cleanly(self) -> None:
        if not is_device_available("gpu"):
            with self.assertRaises(DeviceUnavailableError):
                GpuDepthAnythingEffect()


class DepthAnythingV2NpuTests(unittest.TestCase):
    """Hardware tests verifying NPU1 bf16 MatMul and full Depth Anything V2 parity."""

    def test_npu_matmul_padded_tokens_accuracy(self) -> None:
        if not probe_npu().available:
            self.skipTest("No AMD NPU available on this host")

        rng = np.random.default_rng(11)
        # Odd ViT token count (325 -> padded to 384 on 4 NPU columns)
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

    def test_npu_vs_cpu_depth_anything_v2_small_parity(self) -> None:
        if not probe_npu().available:
            self.skipTest("No AMD NPU available on this host")

        # Use a fast 140x140 (10x10+1 = 101 tokens -> 128 padded) check on synthetic or cached weights
        cfg = get_depth_config("small", input_size=140)
        weights = create_synthetic_weights(cfg, seed=5)
        rgb = np.random.default_rng(5).integers(0, 256, size=(240, 320, 3), dtype=np.uint8)

        cpu_model = DepthAnythingV2Model(cfg, weights, backend="cpu")
        npu_model = DepthAnythingV2Model(cfg, weights, backend="npu")
        try:
            cpu_res = cpu_model.infer(rgb)
            npu_res = npu_model.infer(rgb)
        finally:
            cpu_model.close()
            npu_model.close()

        self.assertEqual(cpu_res.depth.shape, npu_res.depth.shape)
        self.assertIsNotNone(npu_res.npu_stats)
        assert npu_res.npu_stats is not None
        self.assertGreater(npu_res.npu_stats.dispatches, 0)

        corr = float(np.corrcoef(cpu_res.depth.ravel(), npu_res.depth.ravel())[0, 1])
        self.assertGreater(corr, 0.99)

        # Also test NpuDepthAnythingEffect wrapper
        eff = NpuDepthAnythingEffect(variant="small", input_size=140, use_real_weights=False)
        eff.setup(320, 240, 30.0)
        vis = eff.process(rgb, FrameContext(1, 0.0, 320, 240, 30.0))
        eff.teardown()
        self.assertEqual(vis.shape, (240, 320, 3))


if __name__ == "__main__":
    unittest.main()
