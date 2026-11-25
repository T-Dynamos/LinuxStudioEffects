"""Tests and live verification for the V4L2 camera capture & FPS pipeline."""

from __future__ import annotations

import ctypes
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from linux_studio_effects.video import (
    CameraConfig,
    V4L2Camera,
    benchmark_camera,
    check_camera_deps,
    decode_yuyv_to_rgb,
    format_benchmark_result,
    format_deps_report,
)
from linux_studio_effects.video.v4l2 import (
    v4l2_buffer,
    v4l2_capability,
    v4l2_fmtdesc,
    v4l2_format,
    v4l2_frmivalenum,
    v4l2_frmsizeenum,
    v4l2_requestbuffers,
    v4l2_streamparm,
)


class CameraPipelineTests(unittest.TestCase):
    """Verify V4L2 ABI layout, decoders, dependency probing, and live camera capture."""

    def test_v4l2_struct_sizes_match_linux_x86_64(self) -> None:
        self.assertEqual(ctypes.sizeof(v4l2_capability), 104)
        self.assertEqual(ctypes.sizeof(v4l2_fmtdesc), 64)
        self.assertEqual(ctypes.sizeof(v4l2_frmsizeenum), 44)
        self.assertEqual(ctypes.sizeof(v4l2_frmivalenum), 52)
        self.assertEqual(ctypes.sizeof(v4l2_format), 208)
        self.assertEqual(ctypes.sizeof(v4l2_streamparm), 204)
        self.assertEqual(ctypes.sizeof(v4l2_requestbuffers), 20)
        self.assertEqual(ctypes.sizeof(v4l2_buffer), 88)

    def test_yuyv_decoder_neutral_luma(self) -> None:
        # Y0=100, U=128, Y1=200, V=128 -> neutral gray pixels (100,100,100) and (200,200,200)
        raw = bytes([100, 128, 200, 128] * 2)
        rgb = decode_yuyv_to_rgb(raw, width=4, height=1)
        self.assertEqual(rgb.shape, (1, 4, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        np.testing.assert_array_equal(rgb[0, 0], [100, 100, 100])
        np.testing.assert_array_equal(rgb[0, 1], [200, 200, 200])

    def test_live_camera_capture_and_fps(self) -> None:
        report = check_camera_deps()
        if not report.ok or report.default_device is None:
            self.skipTest("No accessible V4L2 camera available on this host")

        dev = report.default_device
        # Pick 1280x720 MJPG if advertised, otherwise fall back to first mode.
        preferred = next(
            (m for m in dev.modes if m.pixel_format == "MJPG" and m.width == 1280 and m.height == 720),
            dev.modes[0],
        )
        target_fps = preferred.fps_options[0] if preferred.fps_options else 30.0
        cfg = CameraConfig(
            device=dev.path,
            width=preferred.width,
            height=preferred.height,
            pixel_format=preferred.pixel_format,
            fps=target_fps,
        )

        with V4L2Camera(cfg) as cam:
            frame = cam.read_frame(decode=True)
            self.assertIsNotNone(frame.data)
            assert frame.data is not None
            self.assertEqual(frame.data.shape, (cam.negotiated.height, cam.negotiated.width, 3))
            self.assertEqual(frame.data.dtype, np.uint8)
            self.assertGreater(frame.raw_size, 0)

        bench = benchmark_camera(cfg, duration_s=1.5, warmup_frames=3, decode=True)
        self.assertGreater(bench.frames, 10)
        self.assertGreater(bench.wall_fps, 5.0)


def main() -> int:
    report = check_camera_deps()
    print(format_deps_report(report))
    if not report.ok:
        return 1

    print()
    res = benchmark_camera(duration_s=3.0, warmup_frames=5, decode=True)
    print(format_benchmark_result(res))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--unittest":
        sys.argv.pop(1)
        unittest.main()
    raise SystemExit(main())
