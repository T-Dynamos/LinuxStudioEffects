"""V4L2 loopback virtual camera output sink and RGB frame adapters."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .v4l2 import (
    V4L2_BUF_TYPE_VIDEO_OUTPUT,
    V4L2_CAP_VIDEO_OUTPUT,
    V4L2_FIELD_NONE,
    VIDIOC_S_FMT,
    VIDIOC_S_PARM,
    fourcc_to_str,
    pick_default_loopback,
    str_to_fourcc,
    v4l2_capability,
    v4l2_format,
    v4l2_streamparm,
    VIDIOC_QUERYCAP,
    V4L2_CAP_DEVICE_CAPS,
    xioctl,
)

SUPPORTED_OUTPUT_FORMATS = ("YUYV", "UYVY", "RGB3", "BGR3")


@dataclass(frozen=True)
class VirtualCameraConfig:
    """Requested or negotiated V4L2 loopback output configuration."""

    device: str = ""
    width: int = 1280
    height: int = 720
    pixel_format: str = "YUYV"
    fps: float = 30.0
    bytesperline: int = 0
    sizeimage: int = 0
    colorspace: int = 0
    ycbcr_enc: int = 0
    quantization: int = 0
    xfer_func: int = 0


def resize_rgb(frame: np.ndarray, out_width: int, out_height: int, *, mode: str = "cover") -> np.ndarray:
    """Resize an (H, W, 3) uint8 RGB frame to (out_height, out_width, 3).

    Args:
        frame: Input RGB frame of shape ``(H, W, 3)``.
        out_width: Target frame width.
        out_height: Target frame height.
        mode: ``"cover"`` (center-crop to preserve aspect ratio, then scale) or ``"stretch"``.
    """
    in_h, in_w = frame.shape[:2]
    if in_h == out_height and in_w == out_width:
        return frame

    src = frame
    if mode == "cover":
        target_ratio = out_width / out_height
        src_ratio = in_w / in_h
        if src_ratio > target_ratio:
            crop_w = max(1, int(round(in_h * target_ratio)))
            x0 = (in_w - crop_w) // 2
            src = frame[:, x0 : x0 + crop_w]
        elif src_ratio < target_ratio:
            crop_h = max(1, int(round(in_w / target_ratio)))
            y0 = (in_h - crop_h) // 2
            src = frame[y0 : y0 + crop_h, :]
    elif mode != "stretch":
        raise ValueError(f"Unsupported resize mode: {mode!r} (expected 'cover' or 'stretch')")

    src_h, src_w = src.shape[:2]
    y_idx = (np.arange(out_height, dtype=np.int64) * src_h) // out_height
    x_idx = (np.arange(out_width, dtype=np.int64) * src_w) // out_width
    return np.ascontiguousarray(src[y_idx[:, None], x_idx], dtype=np.uint8)


class _YuyvScratch:
    """Reusable int16/uint8 buffers to avoid per-frame allocations during RGB->YUYV."""

    def __init__(self, height: int, width: int) -> None:
        self.shape = (height, width)
        self.y = np.empty((height, width), dtype=np.int16)
        self.ytmp = np.empty((height, width), dtype=np.int16)
        self.chroma = np.empty((height, width // 2), dtype=np.int16)
        self.ctmp = np.empty((height, width // 2), dtype=np.int16)
        self.out = np.empty((height, width // 2, 4), dtype=np.uint8)


def encode_rgb_to_yuyv(rgb: np.ndarray, *, scratch: _YuyvScratch | None = None) -> np.ndarray:
    """Convert an (H, W, 3) uint8 RGB image (even W) to packed YUYV bytes array."""
    h, w, c = rgb.shape
    if c != 3:
        raise ValueError(f"Expected 3-channel RGB image, got shape {rgb.shape}")
    if w % 2 != 0:
        raise ValueError(f"YUYV requires even width, got {w}")

    s = scratch if (scratch is not None and scratch.shape == (h, w)) else _YuyvScratch(h, w)

    # Full-range BT.601 scaled by 128 so max intermediate (255*128 + 16384 = 32704) fits in int16.
    np.multiply(rgb[:, :, 0], 38, out=s.y, dtype=np.int16)
    np.multiply(rgb[:, :, 1], 75, out=s.ytmp, dtype=np.int16)
    s.y += s.ytmp
    np.multiply(rgb[:, :, 2], 15, out=s.ytmp, dtype=np.int16)
    s.y += s.ytmp
    s.y += 64
    s.y >>= 7
    s.out[:, :, 0] = s.y[:, 0::2]
    s.out[:, :, 2] = s.y[:, 1::2]

    r0 = rgb[:, 0::2, 0]
    g0 = rgb[:, 0::2, 1]
    b0 = rgb[:, 0::2, 2]

    np.multiply(r0, -22, out=s.chroma, dtype=np.int16)
    np.multiply(g0, 42, out=s.ctmp, dtype=np.int16)
    s.chroma -= s.ctmp
    np.multiply(b0, 64, out=s.ctmp, dtype=np.int16)
    s.chroma += s.ctmp
    s.chroma += 16384
    s.chroma >>= 7
    s.out[:, :, 1] = s.chroma

    np.multiply(r0, 64, out=s.chroma, dtype=np.int16)
    np.multiply(g0, 53, out=s.ctmp, dtype=np.int16)
    s.chroma -= s.ctmp
    np.multiply(b0, 11, out=s.ctmp, dtype=np.int16)
    s.chroma -= s.ctmp
    s.chroma += 16384
    s.chroma >>= 7
    s.out[:, :, 3] = s.chroma
    return s.out


def encode_rgb_frame(
    rgb: np.ndarray,
    pixel_format: str,
    *,
    scratch: _YuyvScratch | None = None,
) -> np.ndarray:
    """Encode an (H, W, 3) uint8 RGB array into the requested V4L2 output pixel format."""
    fmt = pixel_format.upper()
    if fmt == "YUYV":
        return encode_rgb_to_yuyv(rgb, scratch=scratch)
    if fmt == "RGB3":
        return np.ascontiguousarray(rgb, dtype=np.uint8)
    if fmt == "BGR3":
        return np.ascontiguousarray(rgb[:, :, ::-1], dtype=np.uint8)
    raise ValueError(f"Unsupported virtual camera pixel format: {pixel_format!r}")


class VirtualCameraOutput:
    """Writes pixel-format-preserving frames to a v4l2loopback device."""

    def __init__(self, config: VirtualCameraConfig | None = None) -> None:
        self._requested = config or VirtualCameraConfig()
        self._negotiated: VirtualCameraConfig | None = None
        self._fd: int | None = None

    @property
    def negotiated(self) -> VirtualCameraConfig:
        """Return actual negotiated virtual camera output parameters."""
        if self._negotiated is None:
            raise RuntimeError("Virtual camera output is not open")
        return self._negotiated

    def open(self) -> VirtualCameraConfig:
        """Open the v4l2loopback device and configure output format and frame rate."""
        if self._fd is not None:
            return self.negotiated

        dev_path = self._requested.device
        if not dev_path:
            loopback = pick_default_loopback()
            if loopback is None:
                raise RuntimeError(
                    "No accessible v4l2loopback device found. "
                    "Load it with: sudo modprobe v4l2loopback devices=1 card_label='LinuxStudioEffects' exclusive_caps=1"
                )
            dev_path = loopback.path

        fmt_name = self._requested.pixel_format.upper()
        if fmt_name not in SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(
                f"Unsupported output format {fmt_name!r}; choose one of {SUPPORTED_OUTPUT_FORMATS}"
            )
        if fmt_name in {"YUYV", "UYVY"} and self._requested.width % 2 != 0:
            raise ValueError(f"{fmt_name} output width must be even, got {self._requested.width}")

        fd = os.open(dev_path, os.O_RDWR)
        try:
            cap = v4l2_capability()
            xioctl(fd, VIDIOC_QUERYCAP, cap)
            effective_caps = cap.device_caps if (cap.capabilities & V4L2_CAP_DEVICE_CAPS) else cap.capabilities
            if not (effective_caps & V4L2_CAP_VIDEO_OUTPUT):
                raise RuntimeError(f"{dev_path} does not support V4L2_CAP_VIDEO_OUTPUT (not a loopback device)")

            bpp = 2 if fmt_name in {"YUYV", "UYVY"} else 3
            fmt = v4l2_format()
            fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT
            fmt.fmt.pix.width = self._requested.width
            fmt.fmt.pix.height = self._requested.height
            fmt.fmt.pix.pixelformat = str_to_fourcc(fmt_name)
            fmt.fmt.pix.field = V4L2_FIELD_NONE
            fmt.fmt.pix.bytesperline = self._requested.width * bpp
            fmt.fmt.pix.sizeimage = self._requested.width * self._requested.height * bpp
            fmt.fmt.pix.colorspace = self._requested.colorspace
            fmt.fmt.pix.ycbcr_enc = self._requested.ycbcr_enc
            fmt.fmt.pix.quantization = self._requested.quantization
            fmt.fmt.pix.xfer_func = self._requested.xfer_func
            xioctl(fd, VIDIOC_S_FMT, fmt)
            actual_format = fourcc_to_str(fmt.fmt.pix.pixelformat)
            if actual_format != fmt_name:
                raise RuntimeError(
                    f"{dev_path} negotiated {actual_format} instead of requested {fmt_name}; "
                    "format conversion is disabled"
                )
            if (fmt.fmt.pix.width, fmt.fmt.pix.height) != (self._requested.width, self._requested.height):
                raise RuntimeError(
                    f"{dev_path} negotiated {fmt.fmt.pix.width}x{fmt.fmt.pix.height} instead of requested "
                    f"{self._requested.width}x{self._requested.height}; resolution changes are disabled"
                )

            actual_fps = self._configure_fps(fd, self._requested.fps)

            self._fd = fd
            self._negotiated = VirtualCameraConfig(
                device=dev_path,
                width=fmt.fmt.pix.width,
                height=fmt.fmt.pix.height,
                pixel_format=fourcc_to_str(fmt.fmt.pix.pixelformat),
                fps=actual_fps,
                bytesperline=fmt.fmt.pix.bytesperline,
                sizeimage=fmt.fmt.pix.sizeimage,
                colorspace=fmt.fmt.pix.colorspace,
                ycbcr_enc=fmt.fmt.pix.ycbcr_enc,
                quantization=fmt.fmt.pix.quantization,
                xfer_func=fmt.fmt.pix.xfer_func,
            )
            return self._negotiated
        except Exception:
            os.close(fd)
            raise

    def _configure_fps(self, fd: int, target_fps: float) -> float:
        if target_fps <= 0:
            return target_fps
        parm = v4l2_streamparm()
        parm.type = V4L2_BUF_TYPE_VIDEO_OUTPUT
        parm.parm.capture.timeperframe.numerator = 1000
        parm.parm.capture.timeperframe.denominator = max(1, int(round(target_fps * 1000)))
        try:
            xioctl(fd, VIDIOC_S_PARM, parm)
            num = parm.parm.capture.timeperframe.numerator
            den = parm.parm.capture.timeperframe.denominator
            if num > 0 and den > 0:
                return round(den / num, 3)
        except OSError:
            pass
        return target_fps

    def write_frame(self, frame: np.ndarray) -> int:
        """Write a frame directly in the negotiated pixel format, without conversion."""
        if self._fd is None or self._negotiated is None:
            raise RuntimeError("Virtual camera output is not open")

        fmt = self._negotiated.pixel_format.upper()
        expected_shape = (
            (self._negotiated.height, self._negotiated.width // 2, 4)
            if fmt in {"YUYV", "UYVY"}
            else (self._negotiated.height, self._negotiated.width, 3)
        )
        if frame.shape != expected_shape:
            raise ValueError(f"Expected {fmt} frame shape {expected_shape}, got {frame.shape}")

        packed = np.ascontiguousarray(frame)
        row_bytes = expected_shape[1] * expected_shape[2]
        stride = self._negotiated.bytesperline or row_bytes
        if stride < row_bytes:
            raise RuntimeError(f"Negotiated {fmt} row stride {stride} is smaller than frame row size {row_bytes}")
        if stride != row_bytes:
            padded = np.zeros((expected_shape[0], stride), dtype=np.uint8)
            padded[:, :row_bytes] = packed.reshape(expected_shape[0], row_bytes)
            packed = padded
        with memoryview(packed) as view:
            written = os.write(self._fd, view)
        return written

    def close(self) -> None:
        """Close the virtual camera file descriptor."""
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
            self._negotiated = None

    def __enter__(self) -> VirtualCameraOutput:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
