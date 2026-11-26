"""Tests for the LinuxStudioEffects Textual TUI and effect lag telemetry."""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import importlib.util
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from linux_studio_effects.npu import probe_npu
from linux_studio_effects.tui import LiveControllerEffect, StudioEffectsApp, _ensure_loopback_device, _select_camera_mode
from linux_studio_effects.video import FrameContext, VideoDeviceInfo, VideoMode, create_effect
from linux_studio_effects.video.v4l2 import v4l2_loopback_config


def _make_app_args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "effect": "mirror",
        "device": "cpu",
        "quality": "fast",
        "variant": "small",
        "input_size": 140,
        "mirror": False,
        "in_size": (160, 120),
        "fps": 30.0,
        "pixel_format": None,
        "input_device": "",
        "output_device": "/dev/video4",
        "duration": 0.0,
        "image": None,
        "frames": 5,
        "sync": True,
        "interval": 0.05,
        "local_only": True,
        "synthetic_weights": True,
        "real_weights": False,
        "blur_strength": 0.6,
        "dof_range": 0.25,
        "light_source": "left",
        "light_color_preset": "warm_white",
        "light_intensity": 2.25,
        "bg_mode": "blur",
        "max_zoom": 2.0,
        "cpu_composite": False,
        "no_stream": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TuiTests(unittest.TestCase):
    """Tests for `lse` / `lse-ui` / `python -m linux_studio_effects` TUI and lag telemetry."""

    def test_cli_removed_and_tui_module_entrypoint(self) -> None:
        self.assertIsNone(importlib.util.find_spec("linux_studio_effects.cli"))
        proc = subprocess.run(
            [sys.executable, "-m", "linux_studio_effects", "--help"],
            cwd=str(pathlib.Path(__file__).resolve().parents[1]),
            env={"PYTHONPATH": "src", "PATH": ""},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("lse-ui", proc.stdout)
        self.assertIn("Textual interactive UI", proc.stdout)

    def test_camera_mode_scoring_prefers_1080p30_and_downgrades_cleanly(self) -> None:
        # Exact 1080p@30 MJPG beats 1080p@5 YUYV, 720p@30, and 4K@30
        modes_full = (
            VideoMode("YUYV", "YUYV", 1920, 1080, (5.0,)),
            VideoMode("MJPG", "MJPG", 1920, 1080, (30.0,)),
            VideoMode("MJPG", "MJPG", 1280, 720, (30.0,)),
            VideoMode("MJPG", "MJPG", 3840, 2160, (30.0,)),
        )
        picked = _select_camera_mode(modes_full, (1920, 1080), 30.0)
        self.assertIsNotNone(picked)
        assert picked is not None
        self.assertEqual((picked[0].pixel_format, picked[0].width, picked[0].height, picked[1]), ("MJPG", 1920, 1080, 30.0))

        # Without 1080p@30, downgrades resolution to 720p@30 rather than taking 1080p@10 or 4K@30
        modes_downgrade = (
            VideoMode("YUYV", "YUYV", 1920, 1080, (10.0,)),
            VideoMode("MJPG", "MJPG", 640, 480, (30.0,)),
            VideoMode("MJPG", "MJPG", 1280, 720, (30.0,)),
            VideoMode("MJPG", "MJPG", 3840, 2160, (30.0,)),
        )
        picked_dg = _select_camera_mode(modes_downgrade, (1920, 1080), 30.0)
        self.assertIsNotNone(picked_dg)
        assert picked_dg is not None
        self.assertEqual((picked_dg[0].pixel_format, picked_dg[0].width, picked_dg[0].height, picked_dg[1]), ("MJPG", 1280, 720, 30.0))

        # Native raw format wins tie against MJPG at 1080p@30
        modes_raw_tie = (
            VideoMode("MJPG", "MJPG", 1920, 1080, (30.0,)),
            VideoMode("YUYV", "YUYV", 1920, 1080, (30.0,)),
        )
        picked_raw = _select_camera_mode(modes_raw_tie, (1920, 1080), 30.0)
        self.assertIsNotNone(picked_raw)
        assert picked_raw is not None
        self.assertEqual(picked_raw[0].pixel_format, "YUYV")

    def test_rootless_loopback_resolution_and_ioctl_config(self) -> None:
        self.assertEqual(ctypes.sizeof(v4l2_loopback_config), 72)

        existing = VideoDeviceInfo(
            path="/dev/video9",
            driver="v4l2 loopback",
            card="LinuxStudioEffects",
            bus_info="",
            capabilities=0,
            device_caps=0,
            can_capture=False,
            can_stream=True,
            is_metadata_only=False,
            rw_accessible=True,
            modes=(),
            can_output=True,
        )
        with mock.patch("linux_studio_effects.tui.pick_default_loopback", return_value=existing):
            dev, cleanup = _ensure_loopback_device("")
        self.assertEqual(dev, "/dev/video9")
        self.assertIsNone(cleanup)

        # When neither an existing loopback nor user-writable /dev/v4l2loopback exists,
        # fail fast with instructions instead of invoking sudo inside the TUI.
        with (
            mock.patch("linux_studio_effects.tui.pick_default_loopback", return_value=None),
            mock.patch("linux_studio_effects.tui.os.geteuid", return_value=1000),
            mock.patch("linux_studio_effects.tui.os.access", return_value=False),
            mock.patch("linux_studio_effects.tui.subprocess.run") as sub_run,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                _ensure_loopback_device("")
            self.assertIn("Rootless virtual camera", str(ctx.exception))
            sub_run.assert_not_called()

        # When /dev/v4l2loopback is writable by the user, create without root or subprocesses.
        fake_rm = mock.Mock()
        with (
            mock.patch("linux_studio_effects.tui.pick_default_loopback", return_value=None),
            mock.patch("linux_studio_effects.tui.os.geteuid", return_value=1000),
            mock.patch("linux_studio_effects.tui.pathlib.Path.exists", return_value=True),
            mock.patch("linux_studio_effects.tui.os.access", return_value=True),
            mock.patch("linux_studio_effects.tui.create_loopback_device", return_value=("/dev/video12", fake_rm)) as mk_lb,
            mock.patch("linux_studio_effects.tui.subprocess.run") as sub_run,
        ):
            dev2, rm2 = _ensure_loopback_device("")
        self.assertEqual(dev2, "/dev/video12")
        self.assertIs(rm2, fake_rm)
        mk_lb.assert_called_once()
        sub_run.assert_not_called()

    def test_async_effect_lag_telemetry_tracks_frame_and_time_delay(self) -> None:
        frame = np.full((120, 160, 3), 120, dtype=np.uint8)

        blur = create_effect(
            "portrait_blur",
            device="cpu",
            variant="small",
            input_size=140,
            use_real_weights=False,
            async_depth=True,
            depth_interval_s=0.25,
        )
        ctrl = LiveControllerEffect(blur)
        ctrl.setup(160, 120, 30.0)
        try:
            # Frame 1 seeds depth synchronously (lag_frames == 0); rapid follow-up frames reuse older depth.
            ctrl.process(frame, FrameContext(1, 0.0, 160, 120, 30.0, pixel_format="RGB3"))
            time.sleep(0.03)
            for idx in range(2, 6):
                ctrl.process(frame, FrameContext(idx, idx / 30.0, 160, 120, 30.0, pixel_format="RGB3"))
            snap = ctrl.snapshot_stats()
            self.assertGreaterEqual(snap["effect_lag_frames"], 1)
            self.assertGreater(snap["effect_lag_ms"], 15.0)
            self.assertGreater(snap["ema_lag_ms"], 0.0)
        finally:
            ctrl.teardown()

        seg = create_effect(
            "background_segmentation",
            device="cpu",
            use_real_weights=False,
            async_matte=True,
            matte_interval_s=0.25,
        )
        seg_ctrl = LiveControllerEffect(seg)
        seg_ctrl.setup(160, 120, 30.0)
        try:
            seg_ctrl.process(frame, FrameContext(1, 0.0, 160, 120, 30.0, pixel_format="RGB3"))
            time.sleep(0.03)
            for idx in range(2, 5):
                seg_ctrl.process(frame, FrameContext(idx, idx / 30.0, 160, 120, 30.0, pixel_format="RGB3"))
            seg_snap = seg_ctrl.snapshot_stats()
            self.assertGreaterEqual(seg_snap["effect_lag_frames"], 1)
            self.assertGreater(seg_snap["effect_lag_ms"], 15.0)
        finally:
            seg_ctrl.teardown()

    def test_textual_tui_headless_controls_and_lag_banner(self) -> None:
        args = _make_app_args(effect="mirror", device="cpu", frames=5)

        async def _drive() -> tuple[StudioEffectsApp, str]:
            app = StudioEffectsApp(args)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.35)
                await pilot.press("m")
                await pilot.click("#src_right")
                await pilot.click("#col_golden")
                await pilot.click("#blur_inc")
                await pilot.pause(0.35)
                banner = str(app.query_one("#stats_banner").render())
            return app, banner

        app, banner_txt = asyncio.run(_drive())
        self.assertTrue(app._mirror)
        self.assertEqual(app._light_source, "right")
        self.assertEqual(app._color_preset, "golden")
        self.assertAlmostEqual(app._blur_strength, 0.7, places=2)
        self.assertIsNotNone(app._controller)
        self.assertIn("video_lag=", banner_txt)
        self.assertIn("Effect lag vs video:", banner_txt)

    def test_textual_tui_image_and_optional_npu_async(self) -> None:
        import av

        with tempfile.TemporaryDirectory() as tmpdir:
            in_p = pathlib.Path(tmpdir) / "in.png"
            img = np.full((120, 160, 3), 130, dtype=np.uint8)
            with av.open(str(in_p), mode="w") as oc:
                st = oc.add_stream("png", rate=1)
                st.width, st.height, st.pix_fmt = 160, 120, "rgb24"
                for pkt in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                    oc.mux(pkt)
                for pkt in st.encode():
                    oc.mux(pkt)

            dev = "npu" if probe_npu().available else "cpu"
            args = _make_app_args(
                effect="aesthetic_light",
                device=dev,
                image=str(in_p),
                sync=False,
                interval=0.1,
            )

            async def _drive() -> tuple[StudioEffectsApp, str]:
                app = StudioEffectsApp(args)
                async with app.run_test(size=(120, 40)) as pilot:
                    await pilot.pause(1.2)
                    banner = str(app.query_one("#stats_banner").render())
                return app, banner

            app, banner_txt = asyncio.run(_drive())
            self.assertIsNotNone(app._controller)
            self.assertIn(f"aesthetic_light:{dev}", banner_txt)
            self.assertIn("Effect lag vs video:", banner_txt)


if __name__ == "__main__":
    unittest.main()
