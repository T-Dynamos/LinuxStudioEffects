"""Interactive Textual TUI control panel for live LinuxStudioEffects."""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Callable
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Footer, Header, RadioButton, RadioSet, RichLog, Static, Switch

from .npu import probe_npu
from .video import (
    ComputeDevice,
    FrameContext,
    FrameEffect,
    PipelineConfig,
    PipelineStats,
    VideoPipeline,
    available_effect_backends,
    create_effect,
    pick_default_camera,
    pick_default_loopback,
    query_video_device,
)
from .video.pixels import NATIVE_PIXEL_FORMATS
from .video.v4l2 import create_loopback_device


def _parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Expected WIDTHxHEIGHT, e.g. 1280x720") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Width and height must be positive")
    return width, height


def _fmt_elapsed(seconds: float) -> str:
    secs = max(0, int(seconds))
    mins, rem = divmod(secs, 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:02d}:{mins:02d}:{rem:02d}" if hrs else f"{mins:02d}:{rem:02d}"


def _score_camera_mode(
    mode: Any,
    target_size: tuple[int, int] = (1920, 1080),
    target_fps: float = 30.0,
) -> tuple[Any, ...]:
    """Score a camera mode: prefer exact target_size @ target_fps, else downgrade cleanly."""
    tw, th = target_size
    target_pixels = tw * th
    pixels = mode.width * mode.height
    max_fps = max(mode.fps_options, default=0.0)
    meets_fps = max_fps >= (target_fps - 0.5)
    within_budget = pixels <= target_pixels
    exact_res = (mode.width, mode.height) == (tw, th)
    effective_pixels = pixels if within_budget else target_pixels - (pixels - target_pixels)
    aspect_diff = abs((mode.width / max(mode.height, 1)) - (tw / max(th, 1)))
    capped_fps = round(min(max_fps, target_fps), 1)
    is_native_raw = mode.pixel_format != "MJPG"
    return (
        int(exact_res and meets_fps),
        int(meets_fps),
        int(within_budget),
        capped_fps,
        effective_pixels,
        -round(aspect_diff, 3),
        int(is_native_raw),
        max_fps,
    )


def _select_camera_mode(
    modes: Any,
    target_size: tuple[int, int] = (1920, 1080),
    target_fps: float = 30.0,
) -> tuple[Any, float] | None:
    """Pick the highest-scoring raw or MJPG camera mode and negotiated target FPS."""
    compatible = [
        m
        for m in modes
        if m.pixel_format in NATIVE_PIXEL_FORMATS or m.pixel_format == "MJPG"
    ]
    if not compatible:
        return None
    best = max(compatible, key=lambda m: _score_camera_mode(m, target_size, target_fps))
    max_fps = max(best.fps_options, default=target_fps)
    chosen_fps = target_fps if max_fps >= (target_fps - 0.5) or max_fps <= 0.0 else max_fps
    return best, chosen_fps


_ROOTLESS_V4L2_HELP = (
    "Rootless virtual camera requires either a pre-created v4l2loopback device or user access to "
    "`/dev/v4l2loopback`. One-time setup:\n"
    "  sudo modprobe v4l2loopback\n"
    "  echo 'KERNEL==\"v4l2loopback\", TAG+=\"uaccess\"' | sudo tee /etc/udev/rules.d/99-v4l2loopback.rules\n"
    "  echo 'SUBSYSTEM==\"video4linux\", KERNEL==\"video*\", DRIVERS==\"v4l2loopback\", TAG+=\"uaccess\"' | sudo tee -a /etc/udev/rules.d/99-v4l2loopback.rules\n"
    "  sudo udevadm control --reload-rules && sudo udevadm trigger /dev/v4l2loopback"
)


def _ensure_loopback_device(
    device: str,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[str, Callable[[], None] | None]:
    """Resolve an existing accessible loopback device or create one without sudo."""

    def require_accessible(path_str: str) -> None:
        if not os.access(path_str, os.R_OK | os.W_OK):
            raise RuntimeError(
                f"{path_str} exists but is not readable/writable; ensure udev/video permissions allow non-root access."
            )

    if not device:
        existing = pick_default_loopback()
        if existing is not None:
            return existing.path, None
        video_nr = -1
        target_hint = "auto-assigned /dev/videoN"
    else:
        device_path = pathlib.Path(device)
        if device_path.exists():
            require_accessible(device)
            return device, None
        number = device_path.name.removeprefix("video")
        if device_path.parent != pathlib.Path("/dev") or not device_path.name.startswith("video") or not number.isdigit():
            raise RuntimeError(f"Cannot create {device}: expected /dev/videoN")
        video_nr = int(number)
        target_hint = device

    ctrl_path = pathlib.Path("/dev/v4l2loopback")
    cleanup: Callable[[], None] | None = None
    created_path: str | None = None

    if ctrl_path.exists() and os.access(ctrl_path, os.R_OK | os.W_OK):
        if log_fn is not None:
            log_fn(f"[yellow]Creating virtual camera ({target_hint}) without root...[/]")
        created_path, cleanup = create_loopback_device(video_nr=video_nr, card_label="LinuxStudioEffects")
    elif os.geteuid() == 0:
        modprobe = shutil.which("modprobe") or "/usr/bin/modprobe"
        nr_args = [f"video_nr={video_nr}"] if video_nr >= 0 else []
        res = subprocess.run(
            [modprobe, "v4l2loopback", *nr_args, "card_label=LinuxStudioEffects", "exclusive_caps=1"],
            check=False,
        )
        if res.returncode != 0:
            raise RuntimeError(f"modprobe v4l2loopback failed ({res.returncode})")
        if video_nr >= 0:
            created_path = f"/dev/video{video_nr}"
        cleanup = lambda: subprocess.run([modprobe, "-r", "v4l2loopback"], check=False)
    else:
        raise RuntimeError(_ROOTLESS_V4L2_HELP)

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        resolved = created_path or (existing.path if (existing := pick_default_loopback()) is not None else None)
        if resolved and pathlib.Path(resolved).exists() and os.access(resolved, os.R_OK | os.W_OK):
            if log_fn is not None:
                log_fn(f"[green]Ready virtual camera {resolved}[/]")
            return resolved, cleanup
        time.sleep(0.05)

    if cleanup is not None:
        try:
            cleanup()
        except Exception:
            pass
    raise RuntimeError(
        f"Loopback node created ({created_path or target_hint}) is not user-accessible without root. "
        "Add a udev rule granting `TAG+=\"uaccess\"` on v4l2loopback video nodes."
    )

_EFFECT_CHOICES = (
    "portrait_blur",
    "aesthetic_light",
    "portrait_light",
    "background_segmentation",
    "automatic_framing",
    "mirror",
    "passthrough",
)

_LIGHT_SOURCES = ("left", "right", "top_left", "top_right", "bottom_left", "bottom_right")
_COLOR_PRESETS = ("warm_white", "golden", "candle", "soft_white", "cool_fill")
_BG_MODES = ("blur", "color", "matte")


class LiveControllerEffect(FrameEffect):
    """Thread-safe hot-swappable effect wrapper driving live TUI parameter and effect changes."""

    def __init__(self, primary: FrameEffect, mirror_enabled: bool = False) -> None:
        self._lock = threading.RLock()
        self._primary = primary
        self._mirror: FrameEffect = create_effect("mirror", device="cpu")
        self._mirror.enabled = bool(mirror_enabled)
        self._pending_teardown: list[FrameEffect] = []
        self._dims: tuple[int, int, float] | None = None
        super().__init__(name=primary.name, device=ComputeDevice.CPU, enabled=True)
        self.requested_device = primary.requested_device
        self.active_device = primary.active_device
        self.render_times_ms: deque[float] = deque(maxlen=120)
        self.first_frame_ms: float | None = None
        self.ema_render_ms: float = 0.0
        self.effect_lag_ms: float = 0.0
        self.ema_effect_lag_ms: float = 0.0
        self.effect_lag_frames: int = 0
        self.total_frames: int = 0

    def setup(self, width: int, height: int, fps: float) -> None:
        with self._lock:
            self._dims = (width, height, fps)
            self._mirror.setup(width, height, fps)
            self._primary.setup(width, height, fps)

    def swap_primary(self, new_effect: FrameEffect) -> None:
        with self._lock:
            old = self._primary
            if self._dims is not None:
                new_effect.setup(*self._dims)
            self._primary = new_effect
            self.name = new_effect.name
            self.requested_device = new_effect.requested_device
            self.active_device = new_effect.active_device
            self.render_times_ms.clear()
            self.first_frame_ms = None
            self.ema_render_ms = 0.0
            self.effect_lag_ms = 0.0
            self.ema_effect_lag_ms = 0.0
            self.effect_lag_frames = 0
            old.teardown()

    def set_mirror(self, enabled: bool) -> None:
        with self._lock:
            self._mirror.enabled = bool(enabled)

    def update_params(self, **params: object) -> None:
        with self._lock:
            fn = getattr(self._primary, "set_params", None)
            if callable(fn):
                fn(**params)

    @staticmethod
    def _read_effect_lag(eff: FrameEffect, fallback_ms: float) -> tuple[float, int]:
        for ms_attr, fr_attr in (
            ("depth_lag_ms", "depth_lag_frames"),
            ("matte_lag_ms", "matte_lag_frames"),
            ("tracker_lag_ms", "tracker_lag_frames"),
        ):
            if hasattr(eff, ms_attr):
                return float(getattr(eff, ms_attr)), int(getattr(eff, fr_attr, 0))
        return fallback_ms, 0

    def snapshot_stats(self) -> dict[str, Any]:
        with self._lock:
            eff = self._primary
            samples = list(self.render_times_ms)
            med = float(np.median(samples)) if samples else (self.first_frame_ms or 0.0)
            p95 = float(np.percentile(samples, 95)) if samples else med
            updates = int(
                getattr(
                    eff,
                    "depth_updates",
                    getattr(eff, "matte_updates", getattr(eff, "tracker_updates", 0)),
                )
            )
            res = getattr(eff, "last_result", None)
            model_ms = float(res.total_ms) if res is not None and hasattr(res, "total_ms") else None
            return {
                "effect": eff.name,
                "device": eff.active_device.value,
                "mirror": bool(self._mirror.enabled),
                "ema_ms": self.ema_render_ms,
                "med_ms": med,
                "p95_ms": p95,
                "frames": self.total_frames,
                "model_ms": model_ms,
                "updates": updates,
                "effect_lag_ms": self.effect_lag_ms,
                "ema_lag_ms": self.ema_effect_lag_ms,
                "effect_lag_frames": self.effect_lag_frames,
            }

    def process(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        t0 = time.perf_counter()
        with self._lock:
            cur = self._mirror.process(frame, ctx) if self._mirror.enabled else frame
            out = self._primary.process(cur, ctx)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            lag_ms, lag_frames = self._read_effect_lag(self._primary, dt_ms)
        self.total_frames += 1
        self.effect_lag_ms = lag_ms
        self.effect_lag_frames = lag_frames
        self.ema_effect_lag_ms = lag_ms if self.ema_effect_lag_ms == 0.0 else (0.2 * lag_ms + 0.8 * self.ema_effect_lag_ms)
        if self.first_frame_ms is None:
            self.first_frame_ms = dt_ms
        else:
            self.render_times_ms.append(dt_ms)
            self.ema_render_ms = dt_ms if self.ema_render_ms == 0.0 else (0.2 * dt_ms + 0.8 * self.ema_render_ms)
        return out

    def teardown(self) -> None:
        with self._lock:
            self._primary.teardown()
            self._mirror.teardown()
            self._dims = None


class StudioEffectsApp(App[None]):
    """Interactive Textual studio control UI for LinuxStudioEffects."""

    TITLE = "LinuxStudioEffects Studio UI"
    SUB_TITLE = "NPU-Accelerated Camera Effects"

    CSS = """
    Screen {
        layout: vertical;
    }
    #stats_banner {
        height: 4;
        padding: 0 2;
        border: round $accent;
        background: $surface;
        content-align: left middle;
    }
    #main_row {
        height: 1fr;
    }
    #left_panel {
        width: 34;
        border: round $panel;
        padding: 1;
    }
    #controls_panel {
        width: 1fr;
        border: round $panel;
        padding: 1 2;
    }
    .section_title {
        text-style: bold;
        color: $accent;
        margin-top: 1;
    }
    .btn_row {
        height: auto;
        margin-bottom: 1;
    }
    .btn_row Button {
        margin-right: 1;
        min-width: 10;
    }
    #mirror_row {
        height: 3;
        align: left middle;
    }
    #log_panel {
        height: 7;
        border: round $panel;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("m", "toggle_mirror", "Mirror"),
        Binding("1", "select_effect('portrait_blur')", "Blur"),
        Binding("2", "select_effect('aesthetic_light')", "Aesthetic"),
        Binding("3", "select_effect('portrait_light')", "PortraitLight"),
        Binding("4", "select_effect('background_segmentation')", "BGSeg"),
        Binding("5", "select_effect('automatic_framing')", "Framing"),
    ]

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self._effect_name: str = args.effect
        self._device: str = args.device
        self._quality: str = args.quality
        self._mirror: bool = bool(args.mirror)
        self._light_source: str = getattr(args, "light_source", "left")
        self._color_preset: str = getattr(args, "light_color_preset", "warm_white")
        self._intensity: float = float(args.light_intensity) if args.light_intensity is not None else 2.25
        self._blur_strength: float = float(args.blur_strength)
        self._dof_range: float = float(args.dof_range)
        self._bg_mode: str = getattr(args, "bg_mode", "blur")
        self._max_zoom: float = float(getattr(args, "max_zoom", 2.0))

        self._stop_event = threading.Event()
        self._pipeline: VideoPipeline | None = None
        self._controller: LiveControllerEffect | None = None
        self._last_pipe_stats: PipelineStats | None = None
        self._stream_start_t = time.perf_counter()
        self._stats_timer: Any = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(Text.from_markup("[bold cyan]● READY[/] Starting pipeline..."), id="stats_banner")
        with Horizontal(id="main_row"):
            with VerticalScroll(id="left_panel"):
                yield Static("Effect", classes="section_title")
                with RadioSet(id="effect_radio"):
                    for name in _EFFECT_CHOICES:
                        yield RadioButton(name, id=f"eff_{name}", value=(name == self._effect_name))
                yield Static("Backend", classes="section_title")
                with RadioSet(id="device_radio"):
                    for dev in ("auto", "npu", "cpu"):
                        yield RadioButton(dev.upper(), id=f"dev_{dev}", value=(dev == self._device))
                yield Static("Quality", classes="section_title")
                with RadioSet(id="quality_radio"):
                    for q in ("fast", "balanced", "high"):
                        yield RadioButton(q, id=f"qual_{q}", value=(q == self._quality))
                with Horizontal(id="mirror_row"):
                    yield Static("Mirror: ")
                    yield Switch(value=self._mirror, id="mirror_switch")
            with VerticalScroll(id="controls_panel"):
                yield Static("Aesthetic / Portrait Light", classes="section_title")
                with Horizontal(classes="btn_row"):
                    for src in _LIGHT_SOURCES:
                        yield Button(src, id=f"src_{src}", variant="primary" if src == self._light_source else "default")
                with Horizontal(classes="btn_row"):
                    for col in _COLOR_PRESETS:
                        yield Button(col, id=f"col_{col}", variant="primary" if col == self._color_preset else "default")
                with Horizontal(classes="btn_row"):
                    yield Button("Int -", id="int_dec")
                    yield Button("Int +", id="int_inc")
                    yield Button("Blur -", id="blur_dec")
                    yield Button("Blur +", id="blur_inc")
                    yield Button("DoF -", id="dof_dec")
                    yield Button("DoF +", id="dof_inc")
                yield Static("Background Segmentation & Framing", classes="section_title")
                with Horizontal(classes="btn_row"):
                    for m in _BG_MODES:
                        yield Button(f"bg:{m}", id=f"bg_{m}", variant="primary" if m == self._bg_mode else "default")
                    yield Button("Zoom -", id="zoom_dec")
                    yield Button("Zoom +", id="zoom_inc")
                yield Static(Text.from_markup(self._params_summary()), id="params_readout")
        yield RichLog(id="log_panel", markup=True, highlight=False)
        yield Footer()

    def _params_summary(self) -> str:
        return (
            f"Active: [bold]{self._effect_name}[/] ({self._device}, {self._quality}, mirror={self._mirror}) | "
            f"light={self._color_preset}@{self._light_source} int={self._intensity:.2f} | "
            f"blur={self._blur_strength:.2f} dof={self._dof_range:.2f} | bg={self._bg_mode} zoom<={self._max_zoom:.2f}"
        )

    def _build_current_effect(self) -> FrameEffect:
        is_offline = bool(self.args.image or self.args.frames)
        use_real = False if self.args.synthetic_weights else (self.args.real_weights or not (self.args.frames and not self.args.image))
        if not is_offline and not self.args.synthetic_weights:
            use_real = True
        use_async = not self.args.sync
        opts: dict[str, Any] = {}
        eff = self._effect_name
        if eff in ("portrait_blur", "portrait_light", "aesthetic_light"):
            opts.update(
                quality=self._quality,
                variant=getattr(self.args, "variant", None),
                input_size=getattr(self.args, "input_size", None),
                use_real_weights=use_real,
                local_files_only=self.args.local_only,
                async_depth=use_async,
                depth_interval_s=self.args.interval,
            )
            if eff == "portrait_blur":
                opts.update(blur_strength=self._blur_strength, dof_range=self._dof_range)
                if self._device in ("auto", "npu") and self.args.cpu_composite:
                    opts["use_npu_composite"] = False
            elif eff == "portrait_light":
                opts.update(intensity=self._intensity)
            else:
                opts.update(source=self._light_source, color_preset=self._color_preset, intensity=self._intensity)
        elif eff in ("background_segmentation", "automatic_framing"):
            opts.update(
                quality=self._quality,
                use_real_weights=use_real,
                local_files_only=self.args.local_only,
            )
            if eff == "background_segmentation":
                opts.update(mode=self._bg_mode, blur_strength=self._blur_strength, async_matte=use_async)
            else:
                opts.update(max_zoom=self._max_zoom, async_tracking=use_async)
        return create_effect(eff, device=self._device, **opts)

    def _log(self, msg: str) -> None:
        if not self.is_running:
            return
        logs = self.query("#log_panel")
        if logs:
            logs.first(RichLog).write(msg)

    def on_mount(self) -> None:
        npu = probe_npu()
        self._log(f"[cyan]NPU probe:[/] available={npu.available} ({npu.reason or 'ready'})")
        self._stats_timer = self.set_interval(0.25, self._refresh_stats)
        if not getattr(self.args, "no_stream", False):
            self._run_stream_worker()

    def _refresh_stats(self) -> None:
        if not self.is_running or self._stop_event.is_set() or self._controller is None:
            return
        banners = self.query("#stats_banner")
        if not banners:
            return
        banner = banners.first(Static)
        st = self._controller.snapshot_stats()
        elapsed = _fmt_elapsed(time.perf_counter() - self._stream_start_t)
        ps = self._last_pipe_stats
        video_lag_ms = ps.avg_lag_ms if ps is not None else st["ema_ms"]
        fps_txt = (
            f"in={ps.input_fps:4.1f} out={ps.output_fps:4.1f} fps | drop={ps.dropped_in} | cpu={ps.cpu_percent:4.1f}%"
            if ps is not None
            else f"frames={st['frames']}"
        )
        m_txt = f"{st['model_ms']:.0f}ms" if st["model_ms"] is not None else "inline"
        lag_frames = st["effect_lag_frames"]
        lag_ms = st["effect_lag_ms"]
        line1 = (
            f"[bold green]● LIVE[/] [bold]{st['effect']}:{st['device']}[/]  "
            f"{fps_txt}  |  video_lag=[cyan]{video_lag_ms:.1f}ms[/] "
            f"(render=[yellow]{st['ema_ms']:.1f}ms[/], p95={st['p95_ms']:.1f}ms)  |  {elapsed}"
        )
        line2 = (
            f"[dim]Effect lag vs video:[/] [bold yellow]{lag_ms:.0f}ms[/] ([cyan]{lag_frames}f[/], ema={st['ema_lag_ms']:.0f}ms)  |  "
            f"[dim]Model:[/] [magenta]{m_txt}[/] (updates={st['updates']})  |  mirror={st['mirror']}"
        )
        banner.update(Text.from_markup(f"{line1}\n{line2}"))

    @work(thread=True, exclusive=True)
    def _run_stream_worker(self) -> None:
        try:
            eff = self._build_current_effect()
        except Exception as exc:
            self.call_from_thread(self._log, f"[red]Init error:[/] {exc}")
            return
        ctrl = LiveControllerEffect(eff, mirror_enabled=self._mirror)
        self._controller = ctrl
        self.call_from_thread(self._log, f"[green]Active effect:[/] {ctrl.name} on {ctrl.active_device.value}")

        # Offline image / synthetic mode or live V4L2 fallback if no camera exists
        use_image = bool(self.args.image)
        camera_info = None if (use_image or self.args.frames) else (
            query_video_device(self.args.input_device) if self.args.input_device else pick_default_camera()
        )
        if camera_info is not None:
            self._run_live_camera_loop(camera_info, ctrl)
            return
        self._run_offline_loop(ctrl)

    def _run_live_camera_loop(self, camera_info: Any, ctrl: LiveControllerEffect) -> None:
        picked = _select_camera_mode(camera_info.modes, self.args.in_size, self.args.fps)
        if picked is None:
            self.call_from_thread(self._log, "[yellow]No raw/MJPG camera mode found; switching to offline frames.[/]")
            self._run_offline_loop(ctrl)
            return
        mode, chosen_fps = picked
        out_fmt = self.args.pixel_format or ("YUYV" if mode.pixel_format == "MJPG" else mode.pixel_format)
        try:
            out_device, cleanup = _ensure_loopback_device(
                self.args.output_device,
                log_fn=lambda m: self.call_from_thread(self._log, m),
            )
        except Exception as exc:
            self.call_from_thread(self._log, f"[red]Loopback error:[/] {exc}")
            ctrl.teardown()
            return

        cfg = PipelineConfig(
            input_device=camera_info.path,
            input_width=mode.width,
            input_height=mode.height,
            input_format=mode.pixel_format,
            output_device=out_device,
            output_width=mode.width,
            output_height=mode.height,
            output_format=out_fmt,
            input_fps=chosen_fps,
            output_fps=chosen_fps,
        )
        pipe = VideoPipeline(cfg, effect=ctrl)
        self._pipeline = pipe
        self._stream_start_t = time.perf_counter()
        self.call_from_thread(
            self._log,
            f"[cyan]Streaming[/] {camera_info.path} ({mode.width}x{mode.height} {mode.pixel_format} @ {chosen_fps:g}fps -> {out_fmt}) -> {out_device}",
        )
        try:
            pipe.run(
                duration_s=self.args.duration or None,
                stats_interval_s=0.25,
                on_stats=lambda st: setattr(self, "_last_pipe_stats", st),
            )
        except Exception as exc:
            self.call_from_thread(self._log, f"[red]Pipeline error:[/] {exc}")
        finally:
            if cleanup is not None:
                try:
                    cleanup()
                except Exception:
                    pass

    def _run_offline_loop(self, ctrl: LiveControllerEffect) -> None:
        w, h = self.args.in_size
        fmt = (self.args.pixel_format or "RGB3").upper()
        if self.args.image:
            import av

            with av.open(str(self.args.image)) as cont:
                rgb = next(cont.decode(video=0)).to_ndarray(format="rgb24")
            if (rgb.shape[1], rgb.shape[0]) != (w, h):
                rgb = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(width=w, height=h).to_ndarray(format="rgb24")
            frame = np.ascontiguousarray(rgb, dtype=np.uint8)
            fmt = "RGB3"
        else:
            yy, xx = np.indices((h, w), dtype=np.uint16)
            frame = np.stack([(xx * 2) % 256, (yy * 2) % 256, (xx + yy) % 256], axis=2).astype(np.uint8)
            fmt = "RGB3"

        fps = float(self.args.fps or 30.0)
        ctrl.setup(w, h, fps)
        self._stream_start_t = time.perf_counter()
        idx = 0
        period = 1.0 / fps
        next_t = time.perf_counter()
        self.call_from_thread(self._log, f"[cyan]Offline stream[/] {w}x{h} {fmt} @ {fps:g} fps")
        try:
            while not self._stop_event.is_set():
                now = time.perf_counter()
                if next_t > now:
                    self._stop_event.wait(min(next_t - now, 0.05))
                    if self._stop_event.is_set():
                        break
                ctrl.process(frame, FrameContext(idx + 1, idx / fps, w, h, fps, pixel_format=fmt))
                idx += 1
                next_t += period
        finally:
            ctrl.teardown()

    def _rebuild_effect(self) -> None:
        self.query_one("#params_readout", Static).update(Text.from_markup(self._params_summary()))
        if self._controller is None:
            return
        try:
            new_eff = self._build_current_effect()
            self._controller.swap_primary(new_eff)
            self._log(f"[green]Switched to[/] {new_eff.name}:{new_eff.active_device.value}")
        except Exception as exc:
            self._log(f"[red]Failed switch:[/] {exc}")

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        rid = event.radio_set.id
        bid = event.pressed.id or ""
        if rid == "effect_radio" and bid.startswith("eff_"):
            self._effect_name = bid.removeprefix("eff_")
        elif rid == "device_radio" and bid.startswith("dev_"):
            self._device = bid.removeprefix("dev_")
        elif rid == "quality_radio" and bid.startswith("qual_"):
            self._quality = bid.removeprefix("qual_")
        else:
            return
        self._rebuild_effect()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "mirror_switch":
            self._mirror = bool(event.value)
            if self._controller is not None:
                self._controller.set_mirror(self._mirror)
            self.query_one("#params_readout", Static).update(Text.from_markup(self._params_summary()))
            self._log(f"Mirror -> {self._mirror}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid.startswith("src_"):
            self._light_source = bid.removeprefix("src_")
            if self._controller and self._effect_name == "aesthetic_light":
                self._controller.update_params(source=self._light_source)
                self._log(f"Light source -> {self._light_source}")
        elif bid.startswith("col_"):
            self._color_preset = bid.removeprefix("col_")
            if self._controller and self._effect_name == "aesthetic_light":
                self._controller.update_params(color_preset=self._color_preset)
                self._log(f"Light color -> {self._color_preset}")
        elif bid in ("int_dec", "int_inc"):
            self._intensity = round(max(0.2, min(4.0, self._intensity + (-0.2 if bid == "int_dec" else 0.2))), 2)
            if self._controller and self._effect_name in ("aesthetic_light", "portrait_light"):
                self._controller.update_params(intensity=self._intensity)
        elif bid in ("blur_dec", "blur_inc"):
            self._blur_strength = round(max(0.0, min(1.0, self._blur_strength + (-0.1 if bid == "blur_dec" else 0.1))), 2)
            if self._controller and self._effect_name in ("portrait_blur", "background_segmentation"):
                self._controller.update_params(blur_strength=self._blur_strength)
        elif bid in ("dof_dec", "dof_inc"):
            self._dof_range = round(max(0.05, min(0.8, self._dof_range + (-0.05 if bid == "dof_dec" else 0.05))), 2)
            if self._controller and self._effect_name == "portrait_blur":
                self._controller.update_params(dof_range=self._dof_range)
        elif bid.startswith("bg_"):
            self._bg_mode = bid.removeprefix("bg_")
            if self._controller and self._effect_name == "background_segmentation":
                self._controller.update_params(mode=self._bg_mode)
                self._log(f"Segmentation mode -> {self._bg_mode}")
        elif bid in ("zoom_dec", "zoom_inc"):
            self._max_zoom = round(max(1.0, min(3.0, self._max_zoom + (-0.2 if bid == "zoom_dec" else 0.2))), 2)
            if self._controller and self._effect_name == "automatic_framing":
                self._controller.update_params(max_zoom=self._max_zoom)
        self.query_one("#params_readout", Static).update(Text.from_markup(self._params_summary()))

    def action_toggle_mirror(self) -> None:
        sw = self.query_one("#mirror_switch", Switch)
        sw.value = not sw.value

    def action_select_effect(self, name: str) -> None:
        btn = self.query_one(f"#eff_{name}", RadioButton)
        btn.value = True

    def on_unmount(self) -> None:
        self._stop_event.set()
        if self._stats_timer is not None:
            self._stats_timer.stop()
            self._stats_timer = None
        if self._pipeline is not None:
            self._pipeline.stop()


def main(argv: list[str] | None = None) -> int:
    catalog = available_effect_backends()
    parser = argparse.ArgumentParser(prog="lse-ui", description="Textual interactive UI for LinuxStudioEffects")
    parser.add_argument("--effect", choices=sorted(catalog), default="portrait_blur")
    parser.add_argument("--device", choices=("auto", "cpu", "npu"), default="auto")
    parser.add_argument("--quality", choices=("fast", "balanced", "high"), default="balanced")
    parser.add_argument("--variant", choices=("small", "base", "large"))
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--in-size", type=_parse_size, default=(1920, 1080))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--pixel-format", choices=NATIVE_PIXEL_FORMATS)
    parser.add_argument("--input-device", default="")
    parser.add_argument("--output-device", default="")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--image")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--synthetic-weights", action="store_true")
    parser.add_argument("--real-weights", action="store_true")
    parser.add_argument("--blur-strength", type=float, default=1.0)
    parser.add_argument("--dof-range", type=float, default=0.25)
    parser.add_argument("--light-source", default="left")
    parser.add_argument("--light-color-preset", default="warm_white")
    parser.add_argument("--light-intensity", type=float)
    parser.add_argument("--bg-mode", default="blur")
    parser.add_argument("--max-zoom", type=float, default=2.0)
    parser.add_argument("--cpu-composite", action="store_true")
    args = parser.parse_args(argv)
    StudioEffectsApp(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
