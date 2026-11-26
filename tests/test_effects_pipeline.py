"""Unit and live-hardware tests for the device-agnostic frame effect pipeline."""

from __future__ import annotations

import pathlib
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from linux_studio_effects.video import (
    BUILTIN_EFFECTS,
    CallableEffect,
    CameraConfig,
    ComputeDevice,
    DeviceUnavailableError,
    EffectChain,
    FrameGraph,
    FastBlurEffect,
    FrameContext,
    FrameEffect,
    MirrorEffect,
    PassthroughEffect,
    PipelineConfig,
    SpotlightEffect,
    V4L2Camera,
    VideoPipeline,
    ZoomCropEffect,
    check_camera_deps,
    create_effect,
    pick_default_loopback,
    resolve_compute_device,
)


class _LifecycleTrackerEffect(FrameEffect):
    def __init__(self) -> None:
        super().__init__(device=ComputeDevice.AUTO)
        self.setup_calls: list[tuple[int, int, float]] = []
        self.teardown_calls = 0
        self.seen_contexts: list[FrameContext | None] = []

    def setup(self, width: int, height: int, fps: float) -> None:
        super().setup(width, height, fps)
        self.setup_calls.append((width, height, fps))

    def process(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        self.seen_contexts.append(ctx)
        return frame[:, ::-1, :]

    def teardown(self) -> None:
        self.teardown_calls += 1
        super().teardown()


class EffectsUnitTests(unittest.TestCase):
    """Tests for ComputeDevice resolution, FrameEffect lifecycle, EffectChain, and CPU effects."""

    def test_compute_device_resolution(self) -> None:
        from linux_studio_effects.video import is_device_available

        self.assertEqual(resolve_compute_device("fast_blur", "auto"), "cpu")
        self.assertEqual(resolve_compute_device("fast_blur", "cpu"), "cpu")
        with self.assertRaises(DeviceUnavailableError):
            resolve_compute_device("fast_blur", "npu")
        with self.assertRaises(DeviceUnavailableError):
            resolve_compute_device("fast_blur", "gpu")
        expected_mirror = "npu" if is_device_available("npu") else "cpu"
        self.assertEqual(resolve_compute_device("mirror", "auto"), expected_mirror)

    def test_auto_effect_factory_uses_discovered_backend(self) -> None:
        effect = create_effect("mirror")
        self.assertEqual(effect.name, "mirror")
        self.assertEqual(effect.active_device.value, resolve_compute_device("mirror", "auto"))

    def test_callable_effect_and_disabled_skip(self) -> None:
        def invert(frame: np.ndarray) -> np.ndarray:
            return 255 - frame

        eff = CallableEffect(invert)
        sample = np.full((8, 10, 3), 20, dtype=np.uint8)
        out = eff(sample)
        self.assertEqual(out.dtype, np.uint8)
        np.testing.assert_array_equal(out, 235)

        eff.enabled = False
        np.testing.assert_array_equal(eff(sample), sample)

    def test_effect_chain_lifecycle_and_toggle(self) -> None:
        tracker = _LifecycleTrackerEffect()
        chain = EffectChain([tracker, lambda f: np.clip(f.astype(np.int16) + 10, 0, 255).astype(np.uint8)])
        chain.setup(16, 12, 30.0)
        self.assertEqual(tracker.setup_calls, [(16, 12, 30.0)])

        img = np.zeros((12, 16, 3), dtype=np.uint8)
        img[0, 0] = [50, 60, 70]
        ctx = FrameContext(frame_index=1, timestamp_s=0.1, width=16, height=12, fps=30.0)
        out = chain.process(img, ctx)
        self.assertEqual(len(tracker.seen_contexts), 1)
        self.assertEqual(tracker.seen_contexts[0], ctx)
        # Mirrored horizontally + +10
        np.testing.assert_array_equal(out[0, -1], [60, 70, 80])

        # Toggle tracker off: pixel stays at index 0, only +10 applies.
        self.assertFalse(chain.toggle("_LifecycleTrackerEffect"))
        out2 = chain.process(img, ctx)
        np.testing.assert_array_equal(out2[0, 0], [60, 70, 80])

        chain.teardown()
        self.assertEqual(tracker.teardown_calls, 1)

    def test_frame_graph_routes_input_from_named_effect_outputs(self) -> None:
        graph = FrameGraph()
        graph.add("mirror", MirrorEffect())
        graph.add("plus_ten", lambda frame: np.clip(frame.astype(np.int16) + 10, 0, 255).astype(np.uint8), source="mirror")
        graph.add("original_plus_twenty", lambda frame: np.clip(frame.astype(np.int16) + 20, 0, 255).astype(np.uint8))
        graph.set_output("plus_ten")

        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        frame[0, 0] = [30, 40, 50]
        out = graph.process(frame)
        np.testing.assert_array_equal(out[0, -1], [40, 50, 60])
        np.testing.assert_array_equal(out[0, 0], [10, 10, 10])

        graph.set_output("original_plus_twenty")
        np.testing.assert_array_equal(graph.process(frame)[0, 0], [50, 60, 70])

        with self.assertRaises(ValueError):
            graph.add("invalid", MirrorEffect(), source="later_node")

    def test_builtin_cpu_effects_preserve_shape_and_dtype(self) -> None:
        from linux_studio_effects.video.effects.portrait_blur.cpu import PortraitBlurEffect

        frame = np.random.default_rng(42).integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
        for name, factory in BUILTIN_EFFECTS.items():
            if name.startswith(("portrait_blur", "background_segmentation", "automatic_framing", "portrait_light", "aesthetic_light")):
                continue
            try:
                eff = factory()
            except DeviceUnavailableError:
                continue
            eff.setup(160, 120, 30.0)
            out = eff.process(frame, FrameContext(1, 0.0, 160, 120, 30.0))
            self.assertEqual(out.shape, frame.shape, msg=name)
            self.assertEqual(out.dtype, np.uint8, msg=name)
            eff.teardown()

        pb = PortraitBlurEffect(variant="small", input_size=140, use_real_weights=False)
        pb.setup(160, 120, 30.0)
        pb_out = pb.process(frame, FrameContext(1, 0.0, 160, 120, 30.0, pixel_format="RGB3"))
        self.assertEqual(pb_out.shape, frame.shape)
        self.assertEqual(pb_out.dtype, np.uint8)
        pb.teardown()

        from linux_studio_effects.video.effects.background_segmentation.cpu import BackgroundSegmentationEffect

        bs = BackgroundSegmentationEffect(max_size=128, use_real_weights=False)
        bs.setup(160, 120, 30.0)
        bs_out = bs.process(frame, FrameContext(1, 0.0, 160, 120, 30.0, pixel_format="RGB3"))
        self.assertEqual(bs_out.shape, frame.shape)
        self.assertEqual(bs_out.dtype, np.uint8)
        bs.teardown()

        from linux_studio_effects.video.effects.automatic_framing.cpu import AutomaticFramingEffect
        from linux_studio_effects.video.effects.portrait_light.cpu import PortraitLightEffect
        from linux_studio_effects.video.effects.aesthetic_light.cpu import AestheticLightEffect

        af = AutomaticFramingEffect(max_size=128, use_real_weights=False)
        af.setup(160, 120, 30.0)
        af_out = af.process(frame, FrameContext(1, 0.0, 160, 120, 30.0, pixel_format="RGB3"))
        self.assertEqual(af_out.shape, frame.shape)
        self.assertEqual(af_out.dtype, np.uint8)
        af.teardown()

        pl = PortraitLightEffect(variant="small", input_size=140, use_real_weights=False, async_depth=False)
        pl.setup(160, 120, 30.0)
        pl_out = pl.process(frame, FrameContext(1, 0.0, 160, 120, 30.0, pixel_format="RGB3"))
        self.assertEqual(pl_out.shape, frame.shape)
        self.assertEqual(pl_out.dtype, np.uint8)
        pl.teardown()

        al = AestheticLightEffect(variant="small", input_size=140, use_real_weights=False, async_depth=False)
        al.setup(160, 120, 30.0)
        al_out = al.process(frame, FrameContext(1, 0.0, 160, 120, 30.0, pixel_format="RGB3"))
        self.assertEqual(al_out.shape, frame.shape)
        self.assertEqual(al_out.dtype, np.uint8)
        al.teardown()

        self.assertIs(PassthroughEffect().process(frame), frame)
        mirrored = MirrorEffect().process(frame)
        np.testing.assert_array_equal(mirrored[:, 0], frame[:, -1])
        self.assertEqual(ZoomCropEffect(1.5).process(frame).shape, frame.shape)
        self.assertEqual(SpotlightEffect(0.5).process(frame).shape, frame.shape)
        self.assertEqual(FastBlurEffect(scale=4, passes=2).process(frame).shape, frame.shape)

    def test_builtin_cpu_effects_preserve_native_yuyv(self) -> None:
        from linux_studio_effects.video.effects.portrait_blur.cpu import PortraitBlurEffect
        from linux_studio_effects.video.pixels import split_yuv422

        width, height = 32, 24
        rng = np.random.default_rng(19)
        frame = rng.integers(0, 256, size=(height, width // 2, 4), dtype=np.uint8)
        ctx = FrameContext(1, 0.0, width, height, 30.0, pixel_format="YUYV")
        effects = (
            MirrorEffect(),
            FastBlurEffect(scale=4),
            SpotlightEffect(),
            ZoomCropEffect(),
            PortraitBlurEffect(variant="small", input_size=140, use_real_weights=False),
        )
        for effect in effects:
            effect.setup(width, height, 30.0)
            out = effect.process(frame, ctx)
            self.assertEqual(out.shape, frame.shape, msg=effect.name)
            self.assertEqual(out.dtype, frame.dtype, msg=effect.name)
            effect.teardown()

        spotlight = SpotlightEffect().process(frame, ctx)
        _, input_u, input_v = split_yuv422(frame, "YUYV")
        _, output_u, output_v = split_yuv422(spotlight, "YUYV")
        np.testing.assert_array_equal(output_u, input_u)
        np.testing.assert_array_equal(output_v, input_v)

    def test_effects_organized_by_effect_and_backend_file(self) -> None:
        from linux_studio_effects.video.effects.fast_blur.cpu import FastBlurEffect as CpuBlur
        from linux_studio_effects.video.effects.mirror.cpu import MirrorEffect as CpuMirror
        from linux_studio_effects.video.effects.passthrough.cpu import PassthroughEffect as CpuPass
        from linux_studio_effects.video.effects.portrait_blur.cpu import PortraitBlurEffect as CpuPortraitBlur
        from linux_studio_effects.video.effects.spotlight.cpu import SpotlightEffect as CpuSpot
        from linux_studio_effects.video.effects.zoom_crop.cpu import ZoomCropEffect as CpuZoom
        from linux_studio_effects.video.effects.greyscale.npu import GreyscaleEffect as NpuGrey
        from linux_studio_effects.video.effects.portrait_blur.npu import PortraitBlurEffect as NpuPortraitBlur

        self.assertIs(CpuBlur, FastBlurEffect)
        self.assertIs(CpuMirror, MirrorEffect)
        self.assertIs(CpuPass, PassthroughEffect)
        self.assertIs(CpuSpot, SpotlightEffect)
        self.assertIs(CpuZoom, ZoomCropEffect)
        self.assertEqual(CpuPortraitBlur.effect_name, "portrait_blur")
        self.assertEqual(NpuGrey.effect_name, "greyscale")
        self.assertEqual(CpuBlur().name, "fast_blur")
        self.assertEqual(CpuMirror().name, "mirror")
        self.assertEqual(CpuPass().name, "passthrough")
        self.assertEqual(CpuSpot().name, "spotlight")
        self.assertEqual(CpuZoom().name, "zoom_crop")
        self.assertEqual(CpuPortraitBlur.__name__, NpuPortraitBlur.__name__)
        self.assertEqual(CpuPortraitBlur.effect_name, NpuPortraitBlur.effect_name)
        from linux_studio_effects.video import available_effect_backends
        catalog = available_effect_backends()
        self.assertNotIn("video_depth", catalog)
        self.assertEqual(catalog["portrait_blur"], ("cpu", "npu"))
        self.assertEqual(catalog["background_segmentation"], ("cpu", "npu"))
        self.assertEqual(catalog["automatic_framing"], ("cpu", "npu"))
        self.assertEqual(catalog["portrait_light"], ("cpu", "npu"))
        self.assertEqual(catalog["aesthetic_light"], ("cpu", "npu"))
        self.assertEqual(catalog["greyscale"], ("npu",))
        self.assertEqual(catalog["mirror"], ("cpu", "npu"))


class EffectsPipelineLiveTests(unittest.TestCase):
    """Live test of FrameEffect + EffectChain inside VideoPipeline -> v4l2loopback."""

    def test_live_effect_chain_streams_to_loopback(self) -> None:
        deps = check_camera_deps()
        loopback = pick_default_loopback(list(deps.devices))
        if not deps.ok or deps.default_device is None or loopback is None:
            self.skipTest("Requires both an accessible camera and a loaded v4l2loopback device")
        mode = next((m for m in deps.default_device.modes if m.pixel_format == "YUYV"), None)
        if mode is None:
            self.skipTest("Camera has no native YUYV mode matching the loopback format")

        tracker = _LifecycleTrackerEffect()
        chain = EffectChain([tracker, SpotlightEffect(edge_dim=0.4)])
        cfg = PipelineConfig(
            input_device=deps.default_device.path,
            input_width=mode.width,
            input_height=mode.height,
            input_format="YUYV",
            input_fps=max(mode.fps_options, default=30.0),
            output_device=loopback.path,
            output_width=640,
            output_height=480,
            output_format="YUYV",
            output_fps=30.0,
        )
        pipe = VideoPipeline(cfg, effect=chain)
        holder: dict[str, object] = {}

        t = threading.Thread(target=lambda: holder.setdefault("stats", pipe.run(duration_s=2.0)), daemon=True)
        t.start()

        deadline = time.monotonic() + 2.0
        while pipe.negotiated_output is None and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.25)

        with V4L2Camera(
            CameraConfig(device=loopback.path, width=640, height=480, pixel_format="YUYV", fps=30.0)
        ) as consumer:
            frame = consumer.read_frame(decode=True, timeout_s=2.0)
            assert frame.data is not None
            self.assertEqual(frame.data.shape, (480, 640, 3))

        t.join(timeout=5.0)
        stats = holder.get("stats")
        self.assertIsNotNone(stats)
        self.assertEqual(len(tracker.setup_calls), 1)
        self.assertEqual(tracker.setup_calls[0][:2], (mode.width, mode.height))
        self.assertGreater(tracker.setup_calls[0][2], 0.0)
        self.assertEqual(tracker.teardown_calls, 1)
        self.assertGreater(len(tracker.seen_contexts), 20)


if __name__ == "__main__":
    unittest.main()
