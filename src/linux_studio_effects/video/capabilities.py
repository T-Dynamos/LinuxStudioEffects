"""Structured camera capability summaries for V4L2 video devices."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

from .v4l2 import (
    ActiveStreamFormat,
    VideoControlInfo,
    VideoDeviceInfo,
    VideoMode,
    decode_capability_flags,
    list_video_devices,
    pick_default_camera,
    query_video_device,
)

_STUDIO_FORMAT_PRIORITY = ("MJPG", "YUYV", "NV12", "UYVY", "RGB3", "BGR3", "GREY")


@dataclass(frozen=True)
class PixelFormatSummary:
    """Aggregated resolution and framerate bounds for one pixel format."""

    pixel_format: str
    description: str
    resolutions: tuple[str, ...]
    max_width: int
    max_height: int
    max_fps: float
    mode_count: int


@dataclass(frozen=True)
class RecommendedMode:
    """Recommended capture mode for real-time studio processing."""

    pixel_format: str
    width: int
    height: int
    fps: float
    reason: str


@dataclass(frozen=True)
class CameraCapabilitiesSummary:
    """Comprehensive summary of a V4L2 video node's capabilities and modes."""

    device: str
    card: str
    driver: str
    driver_version: str
    bus_info: str
    role: str
    is_default: bool
    rw_accessible: bool
    can_capture: bool
    can_stream: bool
    capability_flags: tuple[str, ...]
    pixel_formats: tuple[PixelFormatSummary, ...]
    max_resolution: tuple[int, int] | None
    max_fps: float
    recommended_mode: RecommendedMode | None
    active_format: ActiveStreamFormat | None
    controls: tuple[VideoControlInfo, ...]
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation of this summary."""
        return asdict(self)


def _classify_role(info: VideoDeviceInfo) -> str:
    if info.is_loopback:
        return "virtual_camera"
    if info.is_metadata_only:
        return "metadata"
    if not info.can_capture:
        return "other"
    if info.is_color_camera:
        return "color_camera"
    if any(m.pixel_format in ("GREY", "Y16", "Z16") for m in info.modes):
        return "ir_or_depth_camera"
    return "capture_device"


def _summarize_pixel_formats(modes: tuple[VideoMode, ...]) -> tuple[PixelFormatSummary, ...]:
    grouped: dict[str, list[VideoMode]] = {}
    for mode in modes:
        grouped.setdefault(mode.pixel_format, []).append(mode)

    summaries: list[PixelFormatSummary] = []
    for fmt, fmt_modes in grouped.items():
        # Sort resolutions ascending by pixel count for deterministic display.
        ordered = sorted(fmt_modes, key=lambda m: (m.width * m.height, m.width, m.height))
        res_labels: list[str] = []
        max_w = 0
        max_h = 0
        max_fps = 0.0
        for m in ordered:
            fps_str = "/".join(f"{f:g}" for f in m.fps_options) or "?"
            res_labels.append(f"{m.width}x{m.height}@{fps_str}fps")
            if m.width * m.height > max_w * max_h:
                max_w, max_h = m.width, m.height
            if m.fps_options:
                max_fps = max(max_fps, max(m.fps_options))
        summaries.append(
            PixelFormatSummary(
                pixel_format=fmt,
                description=ordered[0].description,
                resolutions=tuple(res_labels),
                max_width=max_w,
                max_height=max_h,
                max_fps=max_fps,
                mode_count=len(ordered),
            )
        )
    return tuple(summaries)


def _choose_recommended_mode(info: VideoDeviceInfo) -> RecommendedMode | None:
    if not info.can_capture or not info.can_stream or not info.modes:
        return None

    def _fmt_rank(fmt: str) -> int:
        return _STUDIO_FORMAT_PRIORITY.index(fmt) if fmt in _STUDIO_FORMAT_PRIORITY else len(_STUDIO_FORMAT_PRIORITY)

    candidates: list[tuple[tuple[int, int, float, int], VideoMode, float, str]] = []
    for m in info.modes:
        fps = max(m.fps_options) if m.fps_options else 0.0
        pixels = m.width * m.height
        is_rt = int(fps >= 29.0)
        is_720p = int(m.width == 1280 and m.height == 720 and m.pixel_format == "MJPG" and fps >= 29.0)
        is_hd = int(m.height >= 720)
        score = (is_720p, is_rt, -_fmt_rank(m.pixel_format), is_hd, pixels, fps)
        if is_720p:
            reason = "Optimal 720p@30fps MJPG baseline for real-time studio effects"
        elif is_rt:
            reason = f"Highest real-time ({fps:g} fps) mode in {m.pixel_format}"
        else:
            reason = f"Best available capture mode ({m.pixel_format}, {fps:g} fps)"
        candidates.append((score, m, fps, reason))

    _, best, best_fps, reason = max(candidates, key=lambda item: item[0])
    return RecommendedMode(
        pixel_format=best.pixel_format,
        width=best.width,
        height=best.height,
        fps=best_fps,
        reason=reason,
    )


def summarize_device_capabilities(
    device: VideoDeviceInfo | str,
    *,
    default_path: str | None = None,
) -> CameraCapabilitiesSummary:
    """Summarize capabilities, formats, controls, and recommended mode for one device."""
    info = query_video_device(device) if isinstance(device, str) else device
    fmt_summaries = _summarize_pixel_formats(info.modes)

    max_res: tuple[int, int] | None = None
    max_fps = 0.0
    for mode in info.modes:
        if max_res is None or (mode.width * mode.height) > (max_res[0] * max_res[1]):
            max_res = (mode.width, mode.height)
        if mode.fps_options:
            max_fps = max(max_fps, max(mode.fps_options))

    return CameraCapabilitiesSummary(
        device=info.path,
        card=info.card,
        driver=info.driver,
        driver_version=info.driver_version,
        bus_info=info.bus_info,
        role=_classify_role(info),
        is_default=(default_path is not None and info.path == default_path),
        rw_accessible=info.rw_accessible,
        can_capture=info.can_capture,
        can_stream=info.can_stream,
        capability_flags=decode_capability_flags(info.device_caps or info.capabilities),
        pixel_formats=fmt_summaries,
        max_resolution=max_res,
        max_fps=max_fps,
        recommended_mode=_choose_recommended_mode(info),
        active_format=info.active_format,
        controls=info.controls,
        error=info.error,
    )


def summarize_camera_capabilities(device: str | None = None) -> CameraCapabilitiesSummary:
    """Summarize capabilities for a specific device path or the auto-detected default camera."""
    devices = list_video_devices()
    default_dev = pick_default_camera(devices)
    default_path = default_dev.path if default_dev else None

    if device:
        matched = next((d for d in devices if d.path == device), None)
        target: VideoDeviceInfo | str = matched if matched is not None else device
        return summarize_device_capabilities(target, default_path=default_path)

    if default_dev is None:
        raise RuntimeError("No accessible V4L2 capture camera found")
    return summarize_device_capabilities(default_dev, default_path=default_path)


def summarize_all_cameras(
    devices: list[VideoDeviceInfo] | None = None,
    *,
    include_metadata: bool = True,
) -> list[CameraCapabilitiesSummary]:
    """Summarize capabilities across all discovered /dev/video* devices."""
    devs = devices if devices is not None else list_video_devices()
    default_dev = pick_default_camera(devs)
    default_path = default_dev.path if default_dev else None

    summaries = [summarize_device_capabilities(d, default_path=default_path) for d in devs]
    if not include_metadata:
        summaries = [s for s in summaries if s.role != "metadata"]
    return summaries


def format_camera_summary(summary: CameraCapabilitiesSummary) -> str:
    """Render a single CameraCapabilitiesSummary into a readable multi-line report."""
    default_tag = " [DEFAULT]" if summary.is_default else ""
    drv = f"{summary.driver} {summary.driver_version}".strip() or "unknown"
    lines = [
        f"{summary.device} ({summary.role}){default_tag}: {summary.card or 'unknown'}",
        f"  Driver / Bus : {drv} | {summary.bus_info or 'n/a'} | rw={summary.rw_accessible}",
    ]
    if summary.error:
        lines.append(f"  Error        : {summary.error}")
        return "\n".join(lines)

    if summary.capability_flags:
        lines.append(f"  Capabilities : {', '.join(summary.capability_flags)}")

    if summary.active_format:
        af = summary.active_format
        fps_part = f" @ {af.fps:g}fps" if af.fps else ""
        lines.append(f"  Active Format: {af.width}x{af.height} {af.pixel_format}{fps_part}")

    if summary.max_resolution:
        lines.append(
            f"  Peak Limits  : {summary.max_resolution[0]}x{summary.max_resolution[1]}, "
            f"max_fps={summary.max_fps:g}"
        )

    if summary.recommended_mode:
        rm = summary.recommended_mode
        lines.append(
            f"  Recommended  : {rm.width}x{rm.height} {rm.pixel_format} @ {rm.fps:g}fps ({rm.reason})"
        )

    for pf in summary.pixel_formats:
        lines.append(f"  Format {pf.pixel_format:4s}  : {', '.join(pf.resolutions)}")

    if summary.controls:
        ctrl_strs: list[str] = []
        for c in summary.controls:
            val = "?" if c.current_value is None else str(c.current_value)
            state = " [inactive]" if c.inactive else (" [ro]" if c.read_only else "")
            ctrl_strs.append(f"{c.name}={val} ({c.minimum}..{c.maximum}, def={c.default_value}){state}")
        lines.append(f"  Controls ({len(summary.controls)}): " + "; ".join(ctrl_strs))

    return "\n".join(lines)


def format_all_camera_summaries(summaries: list[CameraCapabilitiesSummary]) -> str:
    """Render a list of CameraCapabilitiesSummary objects into a combined report."""
    if not summaries:
        return "No V4L2 video devices found."
    return "\n\n".join(format_camera_summary(s) for s in summaries)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize V4L2 camera capabilities.")
    parser.add_argument("--device", default="", help="Summarize only a specific /dev/videoX node")
    parser.add_argument("--capture-only", action="store_true", help="Exclude metadata-only /dev/video* nodes")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON output")
    args = parser.parse_args(argv)

    if args.device:
        summary = summarize_camera_capabilities(args.device)
        if args.as_json:
            print(json.dumps(summary.to_dict(), indent=2))
        else:
            print(format_camera_summary(summary))
        return 1 if summary.error else 0

    summaries = summarize_all_cameras(include_metadata=not args.capture_only)
    if args.as_json:
        print(json.dumps([s.to_dict() for s in summaries], indent=2))
    else:
        print(format_all_camera_summaries(summaries))
    return 0 if summaries else 1


if __name__ == "__main__":
    raise SystemExit(main())
