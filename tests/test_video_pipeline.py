"""Unit and live-hardware tests for the configurable input -> v4l2loopback output pipeline."""

from __future__ import annotations

import pathlib
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from linux_studio_effects.video import (
    CameraConfig,
    PipelineConfig,
    V4L2Camera,
    VideoDeviceInfo,
    VideoMode,
    VideoPipeline,
    check_camera_deps,
    decode_yuyv_to_rgb,
    encode_rgb_frame,
    encode_rgb_to_yuyv,
    pick_default_camera,
    pick_default_loopback,
    resize_rgb,
)


class VideoPipelineUnitTests(unittest.TestCase):
    """Offline tests for frame encoding, resizing, and device selection."""

    def test_rgb_yuyv_roundtrip_accuracy(self) -> None:
        rgb = np.zeros((4, 8, 3), dtype=np.uint8)
        rgb[:, 0:4] = [180, 180, 180]
        rgb[:, 4:6] = [200, 60, 60]
        rgb[:, 6:8] = [60, 180, 90]

        yuyv = encode_rgb_to_yuyv(rgb)
        self.assertEqual(yuyv.shape, (4, 4, 4))
        decoded = decode_yuyv_to_rgb(yuyv.tobytes(), width=8, height=4)
        self.assertEqual(decoded.shape, (4, 8, 3))
        max_err = int(np.max(np.abs(decoded.astype(np.int32) - rgb.astype(np.int32))))
        self.assertLessEqual(max_err, 6)

    def test_encode_rgb_and_bgr_formats(self) -> None:
        rgb = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
        np.testing.assert_array_equal(encode_rgb_frame(rgb, "RGB3"), rgb)
        bgr = encode_rgb_frame(rgb, "BGR3")
        np.testing.assert_array_equal(bgr[0, 0], [30, 20, 10])

    def test_resize_rgb_cover_and_stretch(self) -> None:
        # 1280x720 image with bright white center stripe and black sides.
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        img[:, 320:960] = 255

        same = resize_rgb(img, 1280, 720)
        self.assertIs(same, img)

        covered = resize_rgb(img, 640, 480, mode="cover")
        self.assertEqual(covered.shape, (480, 640, 3))
        # Center crop (width 960 -> 320..960 white core fills middle 2/3 of 640x480).
        self.assertEqual(int(covered[240, 320, 0]), 255)
        self.assertEqual(int(covered[240, 10, 0]), 0)

        stretched = resize_rgb(img, 320, 240, mode="stretch")
        self.assertEqual(stretched.shape, (240, 320, 3))
        self.assertEqual(int(stretched[120, 160, 0]), 255)

    def test_device_selection_separates_physical_and_loopback(self) -> None:
        phys = VideoDeviceInfo(
            path="/dev/video0",
            driver="uvcvideo",
            card="Webcam",
            bus_info="usb-1",
            capabilities=0,
            device_caps=0,
            can_capture=True,
            can_stream=True,
            is_metadata_only=False,
            rw_accessible=True,
            modes=(VideoMode("MJPG", "Motion-JPEG", 1280, 720, (30.0,)),),
        )
        loop = VideoDeviceInfo(
            path="/dev/video4",
            driver="v4l2 loopback",
            card="Dummy video device",
            bus_info="platform:v4l2loopback-000",
            capabilities=0,
            device_caps=0,
            can_capture=True,
            can_stream=True,
            is_metadata_only=False,
            rw_accessible=True,
            modes=(),
            can_output=True,
            is_loopback=True,
        )
        # Put loopback first to prove pick_default_camera ignores it.
        devices = [loop, phys]
        picked_in = pick_default_camera(devices)
        picked_out = pick_default_loopback(devices)
        self.assertIsNotNone(picked_in)
        self.assertIsNotNone(picked_out)
        assert picked_in is not None and picked_out is not None
        self.assertEqual(picked_in.path, "/dev/video0")
        self.assertEqual(picked_out.path, "/dev/video4")


class VideoPipelineLiveTests(unittest.TestCase):
    """Live integration test: physical camera -> configurable resize/fps -> v4l2loopback consumer."""

    def test_live_input_to_loopback_output_with_res_and_fps_config(self) -> None:
        deps = check_camera_deps()
        loopback = pick_default_loopback(list(deps.devices))
        if not deps.ok or deps.default_device is None or loopback is None:
            self.skipTest("Requires both an accessible camera and a loaded v4l2loopback device")

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
            output_fps=15.0,
            resize_mode="cover",
        )
        pipe = VideoPipeline(cfg)
        result_holder: dict[str, object] = {}

        def _producer() -> None:
            result_holder["stats"] = pipe.run(duration_s=2.2)

        thread = threading.Thread(target=_producer, daemon=True)
        thread.start()

        # Wait briefly until the producer negotiates and writes initial frames.
        deadline = time.monotonic() + 2.0
        while pipe.negotiated_output is None and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.25)

        received_shapes: list[tuple[int, ...]] = []
        with V4L2Camera(
            CameraConfig(
                device=loopback.path,
                width=640,
                height=480,
                pixel_format="YUYV",
                fps=15.0,
            )
        ) as consumer:
            for _ in range(5):
                frame = consumer.read_frame(decode=True, timeout_s=2.0)
                assert frame.data is not None
                received_shapes.append(tuple(frame.data.shape))

        thread.join(timeout=5.0)
        stats = result_holder.get("stats")
        self.assertIsNotNone(stats)
        assert stats is not None

        self.assertTrue(all(s == (480, 640, 3) for s in received_shapes))
        self.assertGreaterEqual(stats.frames_in, 40)
        self.assertGreaterEqual(stats.frames_out, 20)
        self.assertLessEqual(stats.output_fps, 20.0)
        # Ensure the loop does not spin and saturate a CPU core.
        self.assertLess(stats.cpu_percent, 80.0)


if __name__ == "__main__":
    unittest.main()
