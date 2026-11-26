"""Pixel-format-preserving frame helpers for the streaming video path."""

from __future__ import annotations

import numpy as np

NATIVE_PIXEL_FORMATS = ("YUYV", "UYVY", "RGB3", "BGR3")
PACKED_YUV422_FORMATS = frozenset(("YUYV", "UYVY"))


def is_yuv422(pixel_format: str) -> bool:
    return pixel_format.upper() in PACKED_YUV422_FORMATS


def native_frame(
    raw: bytes | bytearray | memoryview,
    width: int,
    height: int,
    pixel_format: str,
    *,
    bytesperline: int = 0,
) -> np.ndarray:
    """Copy an uncompressed V4L2 frame into a packed, format-preserving array."""
    fmt = pixel_format.upper()
    channels = 2 if fmt in PACKED_YUV422_FORMATS else 3 if fmt in {"RGB3", "BGR3"} else 0
    if channels == 0:
        raise ValueError(f"No native frame layout for pixel format {pixel_format!r}")
    row_bytes = bytesperline or width * channels
    if row_bytes < width * channels:
        raise ValueError(f"Invalid bytesperline={row_bytes} for {width} pixels of {fmt}")
    source = np.frombuffer(raw, dtype=np.uint8)
    required = (height - 1) * row_bytes + width * channels
    if source.size < required:
        raise ValueError(f"Truncated {fmt} frame: got {source.size} bytes, need at least {required}")
    packed = np.empty((height, width * channels), dtype=np.uint8)
    for row in range(height):
        offset = row * row_bytes
        packed[row] = source[offset : offset + width * channels]
    if fmt in PACKED_YUV422_FORMATS:
        if width % 2:
            raise ValueError(f"{fmt} requires an even frame width, got {width}")
        return packed.reshape(height, width // 2, 4)
    return packed.reshape(height, width, 3)


def split_yuv422(frame: np.ndarray, pixel_format: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return full-resolution luma and half-width U/V planes from packed 4:2:2."""
    if frame.ndim != 3 or frame.shape[2] != 4:
        raise ValueError(f"Expected packed YUV422 pairs, got {frame.shape}")
    fmt = pixel_format.upper()
    if fmt == "YUYV":
        y0, u, y1, v = (frame[:, :, i] for i in range(4))
    elif fmt == "UYVY":
        u, y0, v, y1 = (frame[:, :, i] for i in range(4))
    else:
        raise ValueError(f"Expected YUYV or UYVY, got {pixel_format!r}")
    y = np.empty((frame.shape[0], frame.shape[1] * 2), dtype=np.uint8)
    y[:, 0::2] = y0
    y[:, 1::2] = y1
    return y, u, v


def pack_yuv422(y: np.ndarray, u: np.ndarray, v: np.ndarray, pixel_format: str) -> np.ndarray:
    """Pack luma and half-width chroma planes into YUYV or UYVY pairs."""
    if y.ndim != 2 or y.shape[1] % 2:
        raise ValueError(f"YUV422 luma must have an even width, got {y.shape}")
    expected = (y.shape[0], y.shape[1] // 2)
    if u.shape != expected or v.shape != expected:
        raise ValueError(f"Expected U/V planes shaped {expected}, got {u.shape} and {v.shape}")
    out = np.empty((*expected, 4), dtype=np.uint8)
    fmt = pixel_format.upper()
    if fmt == "YUYV":
        out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3] = y[:, 0::2], u, y[:, 1::2], v
    elif fmt == "UYVY":
        out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3] = u, y[:, 0::2], v, y[:, 1::2]
    else:
        raise ValueError(f"Expected YUYV or UYVY, got {pixel_format!r}")
    return out


def yuv422_to_channels(frame: np.ndarray, pixel_format: str) -> np.ndarray:
    """Expose Y, U, V planes at luma resolution for channel-wise spatial effects."""
    from .models.video_depth_anything.host_ops import yuv422_to_channels_u8_fast

    fmt = pixel_format.upper()
    if fmt not in PACKED_YUV422_FORMATS:
        raise ValueError(f"Expected YUYV or UYVY, got {pixel_format!r}")
    fast = yuv422_to_channels_u8_fast(frame, fmt)
    if fast is not None:
        return fast
    y, u, v = split_yuv422(frame, fmt)
    return np.stack((y, np.repeat(u, 2, axis=1), np.repeat(v, 2, axis=1)), axis=2)


def frame_to_channels(frame: np.ndarray, pixel_format: str) -> np.ndarray:
    """Expose three spatial planes without converting YUV values to RGB."""
    return yuv422_to_channels(frame, pixel_format) if is_yuv422(pixel_format) else frame


def channels_to_yuv422(frame: np.ndarray, pixel_format: str) -> np.ndarray:
    """Repack Y/U/V channels, averaging each horizontal chroma pair."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected full-resolution YUV channels, got {frame.shape}")
    from .models.video_depth_anything.host_ops import channels_to_yuv422_u8_fast

    fmt = pixel_format.upper()
    if fmt not in PACKED_YUV422_FORMATS:
        raise ValueError(f"Expected YUYV or UYVY, got {pixel_format!r}")
    if not (frame.shape[1] & 1):
        fast = channels_to_yuv422_u8_fast(frame, fmt)
        if fast is not None:
            return fast
    return pack_yuv422(
        frame[:, :, 0],
        ((frame[:, 0::2, 1].astype(np.uint16) + frame[:, 1::2, 1]) // 2).astype(np.uint8),
        ((frame[:, 0::2, 2].astype(np.uint16) + frame[:, 1::2, 2]) // 2).astype(np.uint8),
        fmt,
    )


def channels_to_frame(frame: np.ndarray, pixel_format: str) -> np.ndarray:
    """Restore the original packed layout after a channel-wise frame operation."""
    return channels_to_yuv422(frame, pixel_format) if is_yuv422(pixel_format) else np.ascontiguousarray(frame)


def model_luma_input(frame: np.ndarray, pixel_format: str) -> np.ndarray:
    """Make a three-channel model input from luma without converting YUV color."""
    if is_yuv422(pixel_format):
        y, _, _ = split_yuv422(frame, pixel_format)
        return np.repeat(y[:, :, None], 3, axis=2)
    if pixel_format.upper() == "BGR3":
        return np.ascontiguousarray(frame[:, :, ::-1])
    return frame


def resize_native(
    frame: np.ndarray,
    out_width: int,
    out_height: int,
    pixel_format: str,
    *,
    mode: str = "cover",
) -> np.ndarray:
    """Nearest-neighbor resize in the original pixel format, without color conversion."""
    fmt = pixel_format.upper()
    yuv = is_yuv422(fmt)
    in_h, in_w = frame.shape[:2]
    if yuv:
        in_w *= 2
        if out_width % 2:
            raise ValueError(f"{fmt} output width must be even, got {out_width}")
    if in_w == out_width and in_h == out_height:
        return frame

    x0, y0, crop_w, crop_h = 0, 0, in_w, in_h
    if mode == "cover":
        target_ratio = out_width / out_height
        source_ratio = in_w / in_h
        if source_ratio > target_ratio:
            crop_w = max(2 if yuv else 1, int(round(in_h * target_ratio)))
            if yuv:
                crop_w -= crop_w % 2
            x0 = ((in_w - crop_w) // 2)
            if yuv:
                x0 -= x0 % 2
        elif source_ratio < target_ratio:
            crop_h = max(1, int(round(in_w / target_ratio)))
            y0 = (in_h - crop_h) // 2
    elif mode != "stretch":
        raise ValueError(f"Unsupported resize mode: {mode!r} (expected 'cover' or 'stretch')")

    y_idx = y0 + (np.arange(out_height, dtype=np.int64) * crop_h) // out_height
    if yuv:
        crop_groups = crop_w // 2
        x_idx = (x0 // 2) + (np.arange(out_width // 2, dtype=np.int64) * crop_groups) // (out_width // 2)
    else:
        x_idx = x0 + (np.arange(out_width, dtype=np.int64) * crop_w) // out_width
    return np.ascontiguousarray(frame[y_idx[:, None], x_idx])
