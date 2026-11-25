"""Unit and live-hardware tests for camera capability summarization."""

from __future__ import annotations

import ctypes
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from linux_studio_effects.video import (
    ActiveStreamFormat,
    VideoControlInfo,
    VideoDeviceInfo,
    VideoMode,
    check_camera_deps,
    format_all_camera_summaries,
    format_camera_summary,
    summarize_all_cameras,
    summarize_camera_capabilities,
    summarize_device_capabilities,
)
from linux_studio_effects.video.v4l2 import (
    V4L2_CAP_EXT_PIX_FORMAT,
    V4L2_CAP_META_CAPTURE,
    V4L2_CAP_STREAMING,
    V4L2_CAP_VIDEO_CAPTURE,
    v4l2_control,
    v4l2_queryctrl,
)


def _sample_devices() -> list[VideoDeviceInfo]:
    color_cam = VideoDeviceInfo(
        path="/dev/video0",
        driver="uvcvideo",
        card="Integrated Camera: Integrated C",
        bus_info="usb-0000:c3:00.4-1",
        capabilities=0x84A00001,
        device_caps=V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING | V4L2_CAP_EXT_PIX_FORMAT,
        can_capture=True,
        can_stream=True,
        is_metadata_only=False,
        rw_accessible=True,
        driver_version="7.3.5",
        active_format=ActiveStreamFormat(
            width=1280,
            height=720,
            pixel_format="MJPG",
            fps=30.0,
            bytesperline=0,
            sizeimage=1843200,
        ),
        modes=(
            VideoMode("MJPG", "Motion-JPEG", 640, 480, (30.0,)),
            VideoMode("MJPG", "Motion-JPEG", 1280, 720, (30.0,)),
            VideoMode("MJPG", "Motion-JPEG", 1920, 1080, (30.0,)),
            VideoMode("YUYV", "YUYV 4:2:2", 640, 480, (30.0,)),
            VideoMode("YUYV", "YUYV 4:2:2", 1280, 720, (10.0,)),
        ),
        controls=(
            VideoControlInfo(
                ctrl_id=0x00980900,
                name="Brightness",
                ctrl_type="int",
                minimum=0,
                maximum=255,
                step=1,
                default_value=128,
                current_value=128,
                flags=0,
            ),
            VideoControlInfo(
                ctrl_id=0x009A0910,
                name="Privacy",
                ctrl_type="bool",
                minimum=0,
                maximum=1,
                step=1,
                default_value=0,
                current_value=0,
                flags=0x0004,
                read_only=True,
            ),
        ),
    )
    meta_node = VideoDeviceInfo(
        path="/dev/video1",
        driver="uvcvideo",
        card="Integrated Camera: Integrated C",
        bus_info="usb-0000:c3:00.4-1",
        capabilities=0x84A00001,
        device_caps=V4L2_CAP_META_CAPTURE | V4L2_CAP_STREAMING | V4L2_CAP_EXT_PIX_FORMAT,
        can_capture=False,
        can_stream=True,
        is_metadata_only=True,
        rw_accessible=True,
        driver_version="7.3.5",
        modes=(),
    )
    ir_cam = VideoDeviceInfo(
        path="/dev/video2",
        driver="uvcvideo",
        card="Integrated Camera: Integrated I",
        bus_info="usb-0000:c3:00.4-1",
        capabilities=0x84A00001,
        device_caps=V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING | V4L2_CAP_EXT_PIX_FORMAT,
        can_capture=True,
        can_stream=True,
        is_metadata_only=False,
        rw_accessible=True,
        driver_version="7.3.5",
        modes=(VideoMode("GREY", "8-bit Greyscale", 640, 360, (15.0,)),),
    )
    return [color_cam, meta_node, ir_cam]


class CameraCapabilitiesTests(unittest.TestCase):
    """Tests for camera capability summarization and formatting."""

    def test_v4l2_control_struct_sizes(self) -> None:
        self.assertEqual(ctypes.sizeof(v4l2_queryctrl), 68)
        self.assertEqual(ctypes.sizeof(v4l2_control), 8)

    def test_summarize_color_camera_capabilities(self) -> None:
        devs = _sample_devices()
        summary = summarize_device_capabilities(devs[0], default_path="/dev/video0")

        self.assertEqual(summary.device, "/dev/video0")
        self.assertEqual(summary.role, "color_camera")
        self.assertTrue(summary.is_default)
        self.assertEqual(summary.driver_version, "7.3.5")
        self.assertIn("VIDEO_CAPTURE", summary.capability_flags)
        self.assertIn("STREAMING", summary.capability_flags)
        self.assertEqual(summary.max_resolution, (1920, 1080))
        self.assertEqual(summary.max_fps, 30.0)
        self.assertEqual(len(summary.pixel_formats), 2)
        self.assertIsNotNone(summary.recommended_mode)
        assert summary.recommended_mode is not None
        self.assertEqual(
            (summary.recommended_mode.pixel_format, summary.recommended_mode.width, summary.recommended_mode.height),
            ("MJPG", 1280, 720),
        )
        self.assertEqual(len(summary.controls), 2)

        serialized = json.dumps(summary.to_dict())
        self.assertIn("Integrated Camera: Integrated C", serialized)

        rendered = format_camera_summary(summary)
        self.assertIn("[DEFAULT]", rendered)
        self.assertIn("Brightness=128", rendered)
        self.assertIn("Privacy=0", rendered)

    def test_summarize_all_cameras_filters_metadata(self) -> None:
        devs = _sample_devices()
        all_summaries = summarize_all_cameras(devs, include_metadata=True)
        self.assertEqual([s.role for s in all_summaries], ["color_camera", "metadata", "ir_or_depth_camera"])

        capture_only = summarize_all_cameras(devs, include_metadata=False)
        self.assertEqual([s.device for s in capture_only], ["/dev/video0", "/dev/video2"])

        combined_text = format_all_camera_summaries(capture_only)
        self.assertIn("/dev/video0 (color_camera) [DEFAULT]", combined_text)
        self.assertIn("/dev/video2 (ir_or_depth_camera)", combined_text)

    def test_summarize_device_with_error(self) -> None:
        broken = VideoDeviceInfo(
            path="/dev/video99",
            driver="",
            card="",
            bus_info="",
            capabilities=0,
            device_caps=0,
            can_capture=False,
            can_stream=False,
            is_metadata_only=False,
            rw_accessible=False,
            modes=(),
            error="No such file or directory",
        )
        summary = summarize_device_capabilities(broken)
        self.assertEqual(summary.role, "other")
        self.assertIsNone(summary.recommended_mode)
        self.assertIn("Error", format_camera_summary(summary))

    def test_live_camera_capabilities_summary(self) -> None:
        report = check_camera_deps()
        if not report.ok or report.default_device is None:
            self.skipTest("No accessible V4L2 camera available on this host")

        summary = summarize_camera_capabilities()
        self.assertEqual(summary.device, report.default_device.path)
        self.assertTrue(summary.is_default)
        self.assertTrue(summary.can_capture)
        self.assertTrue(summary.can_stream)
        self.assertGreater(len(summary.pixel_formats), 0)
        self.assertIsNotNone(summary.max_resolution)
        self.assertGreater(summary.max_fps, 0.0)
        self.assertIsNotNone(summary.recommended_mode)
        self.assertIsNotNone(summary.active_format)

        all_capture = summarize_all_cameras(include_metadata=False)
        self.assertTrue(any(s.is_default for s in all_capture))


if __name__ == "__main__":
    unittest.main()
