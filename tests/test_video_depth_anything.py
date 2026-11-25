"""Unit and hardware verification tests for streaming Video Depth Anything and Portrait Blur."""

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
    MirrorEffect,
    create_effect,
    is_device_available,
)
from linux_studio_effects.video.effects.portrait_blur.cpu import (
    PortraitBlurEffect as CpuPortraitBlurEffect,
    _build_bilinear_plan,
)
from linux_studio_effects.video.effects.portrait_blur.npu import PortraitBlurEffect as NpuPortraitBlurEffect
from linux_studio_effects.video.models.video_depth_anything import (
    NpuMatMulExecutor,
    VideoDepthAnythingModel,
    analyze_feasibility,
    create_synthetic_weights,
    depth_to_colormap_rgb,
    format_feasibility_table,
    get_video_depth_config,
    host_ops,
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

    def test_colormap_and_portrait_blur_cpu_native_formats(self) -> None:
        rgb = np.random.default_rng(2).integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
        depth = np.linspace(0.0, 10.0, 140 * 140, dtype=np.float32).reshape(140, 140)
        vis = depth_to_colormap_rgb(depth, out_width=160, out_height=120)
        self.assertEqual(vis.shape, (120, 160, 3))
        self.assertEqual(vis.dtype, np.uint8)

        cfg = get_video_depth_config("small", input_size=140)
        weights = create_synthetic_weights(cfg, seed=2)
        rng = np.random.default_rng(3)

        for fmt, shape in (
            ("RGB3", (120, 160, 3)),
            ("BGR3", (120, 160, 3)),
            ("YUYV", (120, 80, 4)),
            ("UYVY", (120, 80, 4)),
        ):
            frame = rng.integers(0, 256, size=shape, dtype=np.uint8)
            eff = CpuPortraitBlurEffect(variant="small", input_size=140, weights=weights, blur_strength=0.8)
            eff.setup(160, 120, 30.0)
            ctx0 = FrameContext(1, 0.0, 160, 120, 30.0, pixel_format=fmt)
            ctx1 = FrameContext(2, 0.033, 160, 120, 30.0, pixel_format=fmt)
            out0 = eff.process(frame, ctx0)
            out1 = eff.process(frame, ctx1)
            self.assertEqual(out0.shape, shape, msg=fmt)
            self.assertEqual(out1.shape, shape, msg=fmt)
            self.assertEqual(out1.dtype, np.uint8, msg=fmt)
            self.assertIsNotNone(eff.last_coc, msg=fmt)
            assert eff.last_coc is not None
            self.assertEqual(eff.last_coc.shape, (120, 160), msg=fmt)
            eff.teardown()

        # Zero blur_strength returns an exact copy of the native frame without blurring.
        yuyv_frame = rng.integers(0, 256, size=(64, 48, 4), dtype=np.uint8)
        eff_zero = CpuPortraitBlurEffect(variant="small", input_size=140, weights=weights, blur_strength=0.0)
        eff_zero.setup(96, 64, 30.0)
        out_zero = eff_zero.process(yuyv_frame, FrameContext(1, 0.0, 96, 64, 30.0, pixel_format="YUYV"))
        np.testing.assert_array_equal(out_zero, yuyv_frame)
        eff_zero.teardown()

    def test_portrait_blur_cpu_frame_graph_and_async_depth(self) -> None:
        cfg = get_video_depth_config("small", input_size=140)
        weights = create_synthetic_weights(cfg, seed=7)
        rng = np.random.default_rng(7)
        yuyv = rng.integers(0, 256, size=(72, 64, 4), dtype=np.uint8)

        graph = FrameGraph()
        graph.add("mirror", MirrorEffect(device="cpu"))
        graph.add(
            "portrait_blur",
            create_effect(
                "portrait_blur",
                device="cpu",
                variant="small",
                input_size=140,
                weights=weights,
                async_depth=True,
                depth_interval_s=0.0,
            ),
            source="mirror",
        )
        graph.set_output("portrait_blur")

        graph.setup(128, 72, 30.0)
        try:
            for idx in range(4):
                out = graph.process(yuyv, FrameContext(idx + 1, idx / 30.0, 128, 72, 30.0, pixel_format="YUYV"))
                self.assertEqual(out.shape, yuyv.shape)
                self.assertEqual(out.dtype, np.uint8)
                time.sleep(0.02)
            blur_node = graph.nodes[1][2]
            self.assertGreaterEqual(blur_node.depth_updates, 1)  # type: ignore[attr-defined]
        finally:
            graph.teardown()

    def test_portrait_blur_bokeh_host_ops_match_numpy_reference(self) -> None:
        rng = np.random.default_rng(17)
        h, w = 53, 79  # Odd dimensions exercise edge-clamping in downsample2x_bg_hw4
        frame = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        w0 = rng.random((h, w), dtype=np.float32)

        got_ds = host_ops.downsample2x_bg_hw4(frame, w0)
        cur = np.empty((h, w, 4), dtype=np.float32)
        cur[:, :, :3] = frame.astype(np.float32) * w0[:, :, None]
        cur[:, :, 3] = w0
        cur_pad = np.pad(cur, ((0, h & 1), (0, w & 1), (0, 0)), mode="edge")
        ref_ds = np.float32(0.25) * (
            cur_pad[0::2, 0::2] + cur_pad[0::2, 1::2] + cur_pad[1::2, 0::2] + cur_pad[1::2, 1::2]
        )
        np.testing.assert_allclose(got_ds, ref_ds, atol=1e-4)

        small = rng.uniform(0.0, 255.0, size=(27, 40, 3)).astype(np.float32)
        fast_up = host_ops.bilinear_resize_hwc_u8_fast(small, h, w)
        if fast_up is not None:
            plan = _build_bilinear_plan(27, 40, h, w)
            tmp = small[:, plan.x0, :] * plan.wx0_3d + small[:, plan.x1, :] * plan.wx1_3d
            ref_up = np.clip(np.round(tmp[plan.y0, :, :] * plan.wy0_3d + tmp[plan.y1, :, :] * plan.wy1_3d), 0, 255).astype(np.uint8)
            self.assertLessEqual(int(np.max(np.abs(fast_up.astype(np.int32) - ref_up.astype(np.int32)))), 1)

        l1 = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        l2 = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        l3 = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        coc_u8 = np.arange(h * w, dtype=np.int32).reshape(h, w) % 256
        coc_u8 = coc_u8.astype(np.uint8)
        got_comp = host_ops.bokeh_composite_u8(frame, l1, l2, l3, coc_u8)

        c = coc_u8.astype(np.int16)
        d2 = c - 170
        t2 = d2 * 3 + (d2 >> 6)
        w_0 = np.where(c < 85, 256 - c * 3, 0)[:, :, None]
        w_1 = np.where(c < 85, c * 3, np.where(c < 170, 256 - (c - 85) * 3, 0))[:, :, None]
        w_2 = np.where((c >= 85) & (c < 170), (c - 85) * 3, np.where(c >= 170, 256 - t2, 0))[:, :, None]
        w_3 = np.where(c >= 170, t2, 0)[:, :, None]
        ref_comp = (
            (128 + frame.astype(np.int32) * w_0 + l1.astype(np.int32) * w_1 + l2.astype(np.int32) * w_2 + l3.astype(np.int32) * w_3)
            >> 8
        ).astype(np.uint8)
        np.testing.assert_array_equal(got_comp, ref_comp)

    def test_gpu_backend_rejected_for_portrait_blur(self) -> None:
        with self.assertRaises(DeviceUnavailableError):
            create_effect("portrait_blur", device="gpu")

    def test_portrait_blur_quality_presets_and_set_params(self) -> None:
        fast = CpuPortraitBlurEffect(quality="fast", use_real_weights=False)
        balanced = CpuPortraitBlurEffect(quality="balanced", use_real_weights=False)
        high = CpuPortraitBlurEffect(quality="high", use_real_weights=False)
        self.assertEqual((fast.depth_config.variant, fast.depth_config.input_size, fast.blur_passes), ("small", 196, (1, 2, 4)))
        self.assertEqual((balanced.depth_config.variant, balanced.depth_config.input_size, balanced.blur_passes), ("base", 252, (2, 4, 6)))
        self.assertEqual((high.depth_config.variant, high.depth_config.input_size, high.blur_passes), ("base", 336, (3, 6, 8)))
        self.assertTrue(balanced.async_depth)
        self.assertAlmostEqual(balanced.depth_interval_s, 0.05)

        override = CpuPortraitBlurEffect(
            quality="high",
            variant="small",
            input_size=140,
            blur_passes=(1, 3, 5),
            edge_refine_radius=3,
            edge_refine_eps=5e-4,
            focus_percentile=70.0,
            focus_region=(0.2, 0.8, 0.2, 0.8),
            depth_percentiles=(2.0, 98.0),
            mask_gamma=1.2,
            halo_suppression=12.0,
            use_real_weights=False,
        )
        self.assertEqual(override.depth_config.variant, "small")
        self.assertEqual(override.depth_config.input_size, 140)
        self.assertEqual(override.blur_passes, (1, 3, 5))
        self.assertEqual(override.edge_refine_radius, 3)

        override.set_params(
            blur_strength=0.6,
            focus_mode="manual",
            focus_depth=0.5,
            dof_range=0.25,
            falloff=0.15,
            mask_gamma=0.9,
            async_depth=False,
        )
        self.assertEqual(override.blur_strength, 0.6)
        self.assertEqual(override.focus_mode, "manual")
        self.assertEqual(override.mask_gamma, 0.9)
        self.assertFalse(override.async_depth)

        with self.assertRaises(ValueError):
            CpuPortraitBlurEffect(quality="ultra", use_real_weights=False)
        with self.assertRaises(ValueError):
            override.set_params(blur_passes=(0, 2, 4))
        with self.assertRaises(ValueError):
            override.set_params(unknown_param=1)

    def test_temporal_attn_roll_and_transpose_host_ops_match_numpy_reference(self) -> None:
        rng = np.random.default_rng(29)

        # Transpose + bf16 packing parity against sequential pad
        src = rng.standard_normal((71, 139)).astype(np.float32)
        dst_tr = np.empty((256, 128), dtype=np.uint16)
        dst_ref = np.empty((256, 128), dtype=np.uint16)
        host_ops.f32_transpose_to_bf16_pad(src, dst_tr, 71, 139, zero_tail=True)
        host_ops.f32_to_bf16_pad(np.ascontiguousarray(src.T), dst_ref, 139, 71, zero_tail=True)
        np.testing.assert_array_equal(dst_tr, dst_ref)

        # Unpad + transpose + bias + relu parity
        c_pad = rng.standard_normal((256, 128)).astype(np.float32)
        bias = rng.standard_normal((101,)).astype(np.float32)
        out_nchw = np.empty((1, 101, 13, 11), dtype=np.float32)
        host_ops.f32_unpad_transpose_bias_relu(c_pad, out_nchw.reshape(101, 143), 143, 101, bias=bias, apply_relu=True)
        ref_nchw = np.maximum(c_pad[:143, :101] + bias, 0.0).T.reshape(1, 101, 13, 11)
        np.testing.assert_allclose(out_nchw, ref_nchw, atol=1e-6)

        # Fused streaming rolling temporal attention parity across frames and head dims
        n_heads, dh, d, t_len = 8, 48, 32, 32
        ch = n_heads * dh
        k_pe = rng.standard_normal((t_len, ch)).astype(np.float32) * 0.2
        v_pe = rng.standard_normal((t_len, ch)).astype(np.float32) * 0.2
        kb1 = np.zeros((d, t_len, ch), dtype=np.float32)
        vb1 = np.zeros((d, t_len, ch), dtype=np.float32)
        kb2 = np.zeros((d, t_len, ch), dtype=np.float32)
        vb2 = np.zeros((d, t_len, ch), dtype=np.float32)
        scale = float(1.0 / np.sqrt(dh))
        for step in range(4):
            q = rng.standard_normal((d, ch)).astype(np.float32)
            kc = rng.standard_normal((d, ch)).astype(np.float32)
            vc = rng.standard_normal((d, ch)).astype(np.float32)
            has_h = step > 0
            got = host_ops.temporal_attn_roll(q, kc, vc, kb1, vb1, k_pe, v_pe, n_heads, scale, has_h)
            if not has_h:
                kb2[:, : t_len - 1, :] = kc[:, None, :]
                vb2[:, : t_len - 1, :] = vc[:, None, :]
                ref = (vc + v_pe[0]).astype(np.float32)
            else:
                kb2[:, t_len - 1, :] = kc
                vb2[:, t_len - 1, :] = vc
                kk = kb2 + k_pe[None, :, :]
                vv = vb2 + v_pe[None, :, :]
                qh = q.reshape(d, n_heads, 1, dh)
                kh = kk.reshape(d, t_len, n_heads, dh).transpose(0, 2, 3, 1)
                vh = vv.reshape(d, t_len, n_heads, dh).transpose(0, 2, 1, 3)
                att = host_ops.softmax_rows(np.matmul(qh, kh) * np.float32(scale))
                ref = np.matmul(att, vh).reshape(d, ch)
                kb2[:, 2 : t_len - 2, :] = kb2[:, 3 : t_len - 1, :]
                kb2[:, t_len - 2, :] = kc
                vb2[:, 2 : t_len - 2, :] = vb2[:, 3 : t_len - 1, :]
                vb2[:, t_len - 2, :] = vc
            np.testing.assert_allclose(got, ref, atol=1e-5)
            np.testing.assert_allclose(kb1, kb2, atol=1e-6)
            np.testing.assert_allclose(vb1, vb2, atol=1e-6)


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

        eff = NpuPortraitBlurEffect(variant="small", input_size=140, use_real_weights=False)
        eff.setup(320, 240, 30.0)
        out = eff.process(rgb0, FrameContext(1, 0.0, 320, 240, 30.0))
        eff.teardown()
        self.assertEqual(out.shape, (240, 320, 3))

    def test_npu_portrait_blur_native_formats_composite_and_async(self) -> None:
        if not probe_npu().available:
            self.skipTest("No AMD NPU available on this host")

        cfg = get_video_depth_config("small", input_size=140)
        weights = create_synthetic_weights(cfg, seed=9)
        rng = np.random.default_rng(9)

        # Verify NPU AIE2 4-column bokeh composite is bit-exact with CPU composite.
        h, w = 120, 160
        l0 = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        l1 = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        l2 = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        l3 = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        coc = rng.integers(0, 256, size=(h, w), dtype=np.uint8)

        eff_comp = NpuPortraitBlurEffect(variant="small", input_size=140, weights=weights, use_npu_composite=True)
        eff_comp.setup(w, h, 30.0)
        try:
            np.testing.assert_array_equal(
                eff_comp._composite_bokeh(l0, l1, l2, l3, coc),
                host_ops.bokeh_composite_u8(l0, l1, l2, l3, coc),
            )
        finally:
            eff_comp.teardown()

        # Verify all native formats + async_depth + context-cache survival across formats.
        for fmt, shape in (
            ("RGB3", (120, 160, 3)),
            ("BGR3", (120, 160, 3)),
            ("YUYV", (120, 80, 4)),
            ("UYVY", (120, 80, 4)),
        ):
            frame = rng.integers(0, 256, size=shape, dtype=np.uint8)
            eff = NpuPortraitBlurEffect(
                variant="small",
                input_size=140,
                weights=weights,
                async_depth=(fmt == "YUYV"),
                depth_interval_s=0.0,
            )
            eff.setup(160, 120, 30.0)
            try:
                out0 = eff.process(frame, FrameContext(1, 0.0, 160, 120, 30.0, pixel_format=fmt))
                out1 = eff.process(frame, FrameContext(2, 0.033, 160, 120, 30.0, pixel_format=fmt))
                self.assertEqual(out0.shape, shape, msg=fmt)
                self.assertEqual(out1.shape, shape, msg=fmt)
                self.assertEqual(out1.dtype, np.uint8, msg=fmt)
            finally:
                eff.teardown()

    def test_npu_portrait_blur_base_backbone_parity(self) -> None:
        if not probe_npu().available:
            self.skipTest("No AMD NPU available on this host")

        cfg = get_video_depth_config("base", input_size=252)
        weights = create_synthetic_weights(cfg, seed=13)
        rng = np.random.default_rng(13)
        rgb = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)

        cpu_eff = CpuPortraitBlurEffect(variant="base", input_size=252, weights=weights)
        npu_eff = NpuPortraitBlurEffect(variant="base", input_size=252, weights=weights)
        try:
            cpu_out = cpu_eff.process(rgb, FrameContext(1, 0.0, 320, 240, 30.0))
            npu_out = npu_eff.process(rgb, FrameContext(1, 0.0, 320, 240, 30.0))
        finally:
            cpu_eff.teardown()
            npu_eff.teardown()

        assert cpu_eff.last_result is not None and npu_eff.last_result is not None
        corr = float(np.corrcoef(cpu_eff.last_result.depth.ravel(), npu_eff.last_result.depth.ravel())[0, 1])
        self.assertGreater(corr, 0.99)
        self.assertEqual(cpu_out.shape, npu_out.shape)
        self.assertIsNotNone(npu_eff.last_result.npu_stats)
        assert npu_eff.last_result.npu_stats is not None
        # 48 backbone linear projections + high-FLOP DPT 3x3 convolutions offloaded to NPU.
        self.assertGreater(npu_eff.last_result.npu_stats.dispatches, 48)

    def test_npu_conv2d_3x3_s1_p1_parity(self) -> None:
        if not probe_npu().available:
            self.skipTest("No AMD NPU available on this host")

        rng = np.random.default_rng(31)
        x = rng.standard_normal((1, 128, 72, 72)).astype(np.float32)
        w4 = (rng.standard_normal((128, 128, 3, 3)) * 0.05).astype(np.float32)
        bias = (rng.standard_normal((128,)) * 0.1).astype(np.float32)
        w_ck_cout = np.ascontiguousarray(w4.reshape(128, -1).T)
        w_kykxc_cout = np.ascontiguousarray(w4.transpose(2, 3, 1, 0).reshape(9 * 128, 128))

        ref = np.maximum(host_ops.conv2d_3x3_s1_p1(x, w4, bias), 0.0)
        mm_exec = NpuMatMulExecutor()
        try:
            got = mm_exec.conv2d_3x3_s1_p1_nchw(x, w_kykxc_cout, w_ck_cout, bias=bias, apply_relu=True)
        finally:
            mm_exec.close()

        self.assertEqual(got.shape, ref.shape)
        corr = float(np.corrcoef(got.ravel(), ref.ravel())[0, 1])
        self.assertGreater(corr, 0.9999)


if __name__ == "__main__":
    unittest.main()
