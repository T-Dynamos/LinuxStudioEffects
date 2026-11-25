"""Unit and NPU parity tests for MODNet and BackgroundSegmentationEffect."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from linux_studio_effects.video import (
    DeviceUnavailableError,
    FrameContext,
    FrameGraph,
    create_effect,
)
from linux_studio_effects.video.effects.background_segmentation.cpu import (
    BackgroundSegmentationEffect as CpuBackgroundSegmentationEffect,
)
from linux_studio_effects.video.effects.background_segmentation.npu import (
    BackgroundSegmentationEffect as NpuBackgroundSegmentationEffect,
)
from linux_studio_effects.video.effects.mirror.cpu import MirrorEffect
from linux_studio_effects.video.models.modnet import (
    ModnetModel,
    create_synthetic_weights,
    get_modnet_config,
    host_ops,
)
from linux_studio_effects.npu import probe_npu


class ModnetUnitTests(unittest.TestCase):
    """CPU unit tests for MODNet ops, forward pass, and BackgroundSegmentationEffect modes."""

    def test_modnet_host_ops_match_numpy_reference(self) -> None:
        rng = np.random.default_rng(17)
        x = rng.standard_normal((1, 20, 34, 49)).astype(np.float32)

        # Depthwise 3x3 stride 1 and stride 2 with ReLU6
        dw_w = (rng.standard_normal((20, 9)) * 0.2).astype(np.float32)
        dw_b = (rng.standard_normal((20,)) * 0.1).astype(np.float32)
        for stride in (1, 2):
            got = host_ops.dwconv_3x3_p1(x, dw_w, dw_b, stride=stride, relu6=True)
            orig = host_ops._LIB
            host_ops._LIB = None
            try:
                ref = host_ops.dwconv_3x3_p1(x, dw_w, dw_b, stride=stride, relu6=True)
            finally:
                host_ops._LIB = orig
            np.testing.assert_allclose(got, ref, atol=1e-5)

        # Dense 3x3 direct path (c_out <= 16) + 5x5 im2col path
        w3 = (rng.standard_normal((15, 20 * 9)) * 0.1).astype(np.float32)
        b3 = (rng.standard_normal((15,)) * 0.1).astype(np.float32)
        got3 = host_ops.conv2d_nchw(x, w3, b3, kh=3, kw=3, stride=1, padding=1, relu_mode="relu")
        orig = host_ops._LIB
        host_ops._LIB = None
        try:
            ref3 = host_ops.conv2d_nchw(x, w3, b3, kh=3, kw=3, stride=1, padding=1, relu_mode="relu")
        finally:
            host_ops._LIB = orig
        np.testing.assert_allclose(got3, ref3, atol=1e-4)

        # IBNorm tail InstanceNorm2d + ReLU
        got_ibn = host_ops.ibnorm_tail_relu(x.copy(), eps=1e-5, apply_relu=True)
        orig = host_ops._LIB
        host_ops._LIB = None
        try:
            ref_ibn = host_ops.ibnorm_tail_relu(x.copy(), eps=1e-5, apply_relu=True)
        finally:
            host_ops._LIB = orig
        np.testing.assert_allclose(got_ibn, ref_ibn, atol=1e-5)

        # Alpha composite
        fg = rng.integers(0, 256, size=(34, 49, 3), dtype=np.uint8)
        bg = rng.integers(0, 256, size=(34, 49, 3), dtype=np.uint8)
        alpha = rng.random((34, 49), dtype=np.float32)
        got_c = host_ops.alpha_composite_bg_u8(fg, bg, alpha)
        orig = host_ops._LIB
        host_ops._LIB = None
        try:
            ref_c = host_ops.alpha_composite_bg_u8(fg, bg, alpha)
        finally:
            host_ops._LIB = orig
        self.assertLessEqual(int(np.max(np.abs(got_c.astype(np.int32) - ref_c.astype(np.int32)))), 1)

    def test_modnet_synthetic_forward_shape_and_bounds(self) -> None:
        cfg = get_modnet_config("webcam", max_size=128)
        weights = create_synthetic_weights(cfg, seed=5)
        model = ModnetModel(cfg, weights, backend="cpu")
        try:
            rgb = np.random.default_rng(5).integers(0, 256, size=(108, 192, 3), dtype=np.uint8)
            res = model.infer(rgb)
        finally:
            model.close()
        self.assertEqual(res.alpha.shape, (64, 128))
        self.assertEqual(res.alpha.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(res.alpha)))
        self.assertGreaterEqual(float(res.alpha.min()), 0.0)
        self.assertLessEqual(float(res.alpha.max()), 1.0)

    def test_background_segmentation_modes_and_native_formats(self) -> None:
        cfg = get_modnet_config("webcam", max_size=128)
        weights = create_synthetic_weights(cfg, seed=7)
        rng = np.random.default_rng(7)
        bg_img = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)

        # Test native formats across blur, image, color, matte modes
        for fmt, shape in (
            ("RGB3", (120, 160, 3)),
            ("BGR3", (120, 160, 3)),
            ("YUYV", (120, 80, 4)),
            ("UYVY", (120, 80, 4)),
        ):
            frame = rng.integers(0, 256, size=shape, dtype=np.uint8)
            eff = CpuBackgroundSegmentationEffect(
                mode="image",
                background_image=bg_img,
                max_size=128,
                weights=weights,
                async_matte=False,
            )
            eff.setup(160, 120, 30.0)
            try:
                out_img = eff.process(frame, FrameContext(1, 0.0, 160, 120, 30.0, pixel_format=fmt))
                self.assertEqual(out_img.shape, shape, msg=fmt)
                self.assertEqual(out_img.dtype, np.uint8, msg=fmt)
                eff.set_params(mode="blur", blur_strength=0.9)
                out_blur = eff.process(frame, FrameContext(2, 0.033, 160, 120, 30.0, pixel_format=fmt))
                self.assertEqual(out_blur.shape, shape, msg=fmt)
                eff.set_params(mode="color", background_color=(10, 200, 30))
                out_col = eff.process(frame, FrameContext(3, 0.066, 160, 120, 30.0, pixel_format=fmt))
                self.assertEqual(out_col.shape, shape, msg=fmt)
                eff.set_params(mode="matte")
                out_mat = eff.process(frame, FrameContext(4, 0.1, 160, 120, 30.0, pixel_format=fmt))
                self.assertEqual(out_mat.shape, shape, msg=fmt)
            finally:
                eff.teardown()

        # Verify zero blur_strength returns exact frame copy in blur mode
        eff_zero = CpuBackgroundSegmentationEffect(mode="blur", blur_strength=0.0, max_size=128, weights=weights, async_matte=False)
        rgb = rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
        eff_zero.setup(96, 64, 30.0)
        np.testing.assert_array_equal(eff_zero.process(rgb, FrameContext(1, 0.0, 96, 64, 30.0)), rgb)
        eff_zero.teardown()

        # Verify image path loading via PyAV on disk PNG
        import av

        with tempfile.TemporaryDirectory() as tmpdir:
            bg_path = pathlib.Path(tmpdir) / "bg.png"
            with av.open(str(bg_path), "w") as cont:
                st = cont.add_stream("png", rate=1)
                st.width, st.height = 128, 96
                st.pix_fmt = "rgb24"
                for pkt in st.encode(av.VideoFrame.from_ndarray(bg_img, format="rgb24")):
                    cont.mux(pkt)
                for pkt in st.encode():
                    cont.mux(pkt)
            eff_file = CpuBackgroundSegmentationEffect(mode="image", background_image=bg_path, max_size=128, weights=weights, async_matte=False)
            eff_file.setup(96, 64, 30.0)
            out_file = eff_file.process(rgb, FrameContext(1, 0.0, 96, 64, 30.0))
            self.assertEqual(out_file.shape, rgb.shape)
            eff_file.teardown()

    def test_background_segmentation_async_and_frame_graph(self) -> None:
        cfg = get_modnet_config("webcam", max_size=128)
        weights = create_synthetic_weights(cfg, seed=11)
        yuyv = np.random.default_rng(11).integers(0, 256, size=(72, 64, 4), dtype=np.uint8)

        graph = FrameGraph()
        graph.add("mirror", MirrorEffect(device="cpu"))
        graph.add(
            "bg",
            create_effect(
                "background_segmentation",
                device="cpu",
                max_size=128,
                weights=weights,
                async_matte=True,
                matte_interval_s=0.0,
            ),
            source="mirror",
        )
        graph.set_output("bg")
        graph.setup(128, 72, 30.0)
        try:
            for idx in range(4):
                out = graph.process(yuyv, FrameContext(idx + 1, idx / 30.0, 128, 72, 30.0, pixel_format="YUYV"))
                self.assertEqual(out.shape, yuyv.shape)
                self.assertEqual(out.dtype, np.uint8)
                time.sleep(0.02)
            node = graph.nodes[1][2]
            self.assertGreaterEqual(node.matte_updates, 1)  # type: ignore[attr-defined]
        finally:
            graph.teardown()

    def test_gpu_backend_rejected_for_background_segmentation(self) -> None:
        with self.assertRaises(DeviceUnavailableError):
            create_effect("background_segmentation", device="gpu")


class BackgroundSegmentationNpuTests(unittest.TestCase):
    """Hardware tests verifying NPU1 MODNet parity with CPU."""

    def test_npu_vs_cpu_modnet_parity(self) -> None:
        if not probe_npu().available:
            self.skipTest("No AMD NPU available on this host")

        cfg = get_modnet_config("webcam", max_size=384)
        weights = create_synthetic_weights(cfg, seed=19)
        rgb = np.random.default_rng(19).integers(0, 256, size=(240, 320, 3), dtype=np.uint8)

        cpu_eff = CpuBackgroundSegmentationEffect(max_size=384, weights=weights, async_matte=False)
        npu_eff = NpuBackgroundSegmentationEffect(max_size=384, weights=weights, async_matte=False)
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


if __name__ == "__main__":
    unittest.main()
