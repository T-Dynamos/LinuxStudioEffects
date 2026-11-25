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
    pick_default_loopback,
    resolve_compute_device,
)


class _LifecycleTrackerEffect(FrameEffect):
    supported_devices = (ComputeDevice.CPU,)

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
        self.assertEqual(resolve_compute_device("auto", (ComputeDevice.CPU,)), ComputeDevice.CPU)
        self.assertEqual(resolve_compute_device(ComputeDevice.CPU, (ComputeDevice.CPU,)), ComputeDevice.CPU)
        with self.assertRaises(DeviceUnavailableError):
            resolve_compute_device(ComputeDevice.NPU, (ComputeDevice.CPU,))
        with self.assertRaises(DeviceUnavailableError):
            resolve_compute_device(ComputeDevice.GPU, (ComputeDevice.CPU,))

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

    def test_builtin_cpu_effects_preserve_shape_and_dtype(self) -> None:
        frame = np.random.default_rng(42).integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
        for name, factory in BUILTIN_EFFECTS.items():
            try:
                eff = factory()
            except DeviceUnavailableError:
                continue
            eff.setup(160, 120, 30.0)
            out = eff.process(frame, FrameContext(1, 0.0, 160, 120, 30.0))
            self.assertEqual(out.shape, frame.shape, msg=name)
            self.assertEqual(out.dtype, np.uint8, msg=name)
            eff.teardown()

        self.assertIs(PassthroughEffect().process(frame), frame)
        mirrored = MirrorEffect().process(frame)
        np.testing.assert_array_equal(mirrored[:, 0], frame[:, -1])
        self.assertEqual(ZoomCropEffect(1.5).process(frame).shape, frame.shape)
        self.assertEqual(SpotlightEffect(0.5).process(frame).shape, frame.shape)
        self.assertEqual(FastBlurEffect(scale=4, passes=2).process(frame).shape, frame.shape)

    def test_effects_organized_by_device_and_single_file(self) -> None:
        from linux_studio_effects.video.effects import gpu, npu
        from linux_studio_effects.video.effects.cpu.fast_blur import FastBlurEffect as CpuBlur
        from linux_studio_effects.video.effects.cpu.mirror import MirrorEffect as CpuMirror
        from linux_studio_effects.video.effects.cpu.passthrough import PassthroughEffect as CpuPass
        from linux_studio_effects.video.effects.cpu.spotlight import SpotlightEffect as CpuSpot
        from linux_studio_effects.video.effects.cpu.zoom_crop import ZoomCropEffect as CpuZoom
        from linux_studio_effects.video.effects.npu.greyscale import GreyscaleEffect as NpuGrey

        self.assertIs(CpuBlur, FastBlurEffect)
        self.assertIs(CpuMirror, MirrorEffect)
        self.assertIs(CpuPass, PassthroughEffect)
        self.assertIs(CpuSpot, SpotlightEffect)
        self.assertIs(CpuZoom, ZoomCropEffect)
        self.assertEqual(npu.__all__, ["GreyscaleEffect"])
        self.assertEqual(NpuGrey.supported_devices, (ComputeDevice.NPU,))
        self.assertEqual(gpu.__all__, [])


class EffectsPipelineLiveTests(unittest.TestCase):
    """Live test of FrameEffect + EffectChain inside VideoPipeline -> v4l2loopback."""

    def test_live_effect_chain_streams_to_loopback(self) -> None:
        deps = check_camera_deps()
        loopback = pick_default_loopback(list(deps.devices))
        if not deps.ok or deps.default_device is None or loopback is None:
            self.skipTest("Requires both an accessible camera and a loaded v4l2loopback device")

        tracker = _LifecycleTrackerEffect()
        chain = EffectChain([tracker, SpotlightEffect(edge_dim=0.4)])
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
        self.assertEqual(tracker.setup_calls, [(1280, 720, 30.0)])
        self.assertEqual(tracker.teardown_calls, 1)
        self.assertGreater(len(tracker.seen_contexts), 20)


if __name__ == "__main__":
    unittest.main()
