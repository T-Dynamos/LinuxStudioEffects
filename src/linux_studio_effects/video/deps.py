"""Camera pipeline dependency checks and V4L2 device diagnostics."""

from __future__ import annotations

import ctypes
import pathlib
import shutil
import subprocess
from dataclasses import dataclass

from .decoder import has_pyav
from .v4l2 import VideoDeviceInfo, list_video_devices, pick_default_camera


@dataclass(frozen=True)
class CheckItem:
    """Single dependency or capability probe result."""

    name: str
    ok: bool
    required: bool
    detail: str
    remedy: str = ""


@dataclass(frozen=True)
class CameraDepsReport:
    """Aggregated camera capture dependency and device discovery report."""

    ok: bool
    items: tuple[CheckItem, ...]
    devices: tuple[VideoDeviceInfo, ...]
    default_device: VideoDeviceInfo | None


def _check_lib(names: tuple[str, ...]) -> str | None:
    for name in names:
        try:
            ctypes.CDLL(name)
            return name
        except OSError:
            continue
    return None


def _check_v4l2loopback() -> tuple[bool, str]:
    if pathlib.Path("/sys/module/v4l2loopback").exists():
        return True, "kernel module loaded"
    modinfo = shutil.which("modinfo")
    if modinfo:
        res = subprocess.run([modinfo, "-n", "v4l2loopback"], capture_output=True, text=True, check=False)
        if res.returncode == 0:
            return True, f"installed ({res.stdout.strip()}) but not loaded"
    return False, "not installed (optional; needed later for virtual webcam output)"


def check_camera_deps() -> CameraDepsReport:
    """Verify V4L2 camera devices, permissions, decoders, and optional system tools."""
    devices = list_video_devices()
    capture_devs = [d for d in devices if d.can_capture and d.can_stream and not d.is_loopback]
    rw_capture_devs = [d for d in capture_devs if d.rw_accessible]
    default_dev = pick_default_camera(devices)

    items: list[CheckItem] = []

    if devices:
        items.append(
            CheckItem(
                name="video_nodes",
                ok=True,
                required=True,
                detail=f"found {len(devices)} node(s): {', '.join(d.path for d in devices)}",
            )
        )
    else:
        items.append(
            CheckItem(
                name="video_nodes",
                ok=False,
                required=True,
                detail="no /dev/video* devices found",
                remedy="Connect a webcam or ensure the uvcvideo kernel module is loaded.",
            )
        )

    if rw_capture_devs:
        desc = ", ".join(f"{d.path} ({d.card})" for d in rw_capture_devs)
        items.append(
            CheckItem(
                name="v4l2_capture",
                ok=True,
                required=True,
                detail=f"R/W streaming capture ready: {desc}",
            )
        )
    elif capture_devs:
        items.append(
            CheckItem(
                name="v4l2_capture",
                ok=False,
                required=True,
                detail=f"found capture device(s) ({', '.join(d.path for d in capture_devs)}) without R/W permission",
                remedy="Add your user to the 'video' group (sudo usermod -aG video $USER) and re-login.",
            )
        )
    else:
        items.append(
            CheckItem(
                name="v4l2_capture",
                ok=False,
                required=True,
                detail="no streaming V4L2 Video Capture device found",
                remedy="Check 'v4l2-ctl --list-devices' and ensure a webcam is attached.",
            )
        )

    av_ok = has_pyav()
    items.append(
        CheckItem(
            name="pyav",
            ok=av_ok,
            required=False,
            detail="PyAV/FFmpeg available for MJPG decoding and pixel format conversion" if av_ok else "PyAV is not installed",
            remedy="" if av_ok else "Install the project dependencies (av).",
        )
    )

    v4l2_ctl = shutil.which("v4l2-ctl")
    items.append(
        CheckItem(
            name="v4l2_ctl",
            ok=v4l2_ctl is not None,
            required=False,
            detail=v4l2_ctl or "not found (optional CLI diagnostic)",
        )
    )

    pw_lib = _check_lib(("libpipewire-0.3.so.0", "libpipewire-0.3.so"))
    pw_cli = shutil.which("pw-cli")
    items.append(
        CheckItem(
            name="pipewire",
            ok=pw_lib is not None,
            required=False,
            detail=f"{pw_lib}" + (f" ({pw_cli})" if pw_cli else "") if pw_lib else "libpipewire-0.3 not found",
        )
    )

    loop_ok, loop_detail = _check_v4l2loopback()
    items.append(
        CheckItem(
            name="v4l2loopback",
            ok=loop_ok,
            required=False,
            detail=loop_detail,
        )
    )

    all_required_ok = all(item.ok for item in items if item.required)
    return CameraDepsReport(
        ok=all_required_ok,
        items=tuple(items),
        devices=tuple(devices),
        default_device=default_dev,
    )


def format_deps_report(report: CameraDepsReport) -> str:
    """Format a CameraDepsReport into a human-readable diagnostic summary."""
    lines = ["=== Camera & V4L2 Dependency Check ==="]
    for item in report.items:
        if item.ok:
            tag = "[OK]  "
        elif item.required:
            tag = "[FAIL]"
        else:
            tag = "[OPT] "
        lines.append(f"  {tag} {item.name:14s}: {item.detail}")
        if not item.ok and item.remedy:
            lines.append(f"         remedy: {item.remedy}")

    if report.devices:
        lines.append("\n=== V4L2 Devices ===")
        for dev in report.devices:
            role = "capture" if dev.can_capture else ("metadata" if dev.is_metadata_only else "other")
            default_mark = " *default*" if report.default_device and dev.path == report.default_device.path else ""
            lines.append(
                f"  {dev.path} [{role}]{default_mark}: {dev.card or 'unknown'} "
                f"(driver={dev.driver or '?'}, rw={dev.rw_accessible})"
            )
            if dev.modes:
                by_fmt: dict[str, list[str]] = {}
                for m in dev.modes:
                    fps_str = "/".join(f"{f:g}" for f in m.fps_options) or "?"
                    by_fmt.setdefault(m.pixel_format, []).append(f"{m.width}x{m.height}@{fps_str}fps")
                for fmt, entries in by_fmt.items():
                    lines.append(f"    {fmt}: {', '.join(entries)}")
    return "\n".join(lines)
