"""PyAV-backed decoding and pixel-format conversion for camera frames."""

from __future__ import annotations

import numpy as np

from .pixels import NATIVE_PIXEL_FORMATS, native_frame

_AV_FORMATS = {"YUYV": "yuyv422", "UYVY": "uyvy422", "RGB3": "rgb24", "BGR3": "bgr24"}
_YUV422_FORMATS = {"YUYV", "UYVY"}


def has_pyav() -> bool:
    """Return whether PyAV is importable."""
    try:
        import av  # noqa: F401
    except ImportError:
        return False
    return True


def _frame_array(frame, target_format: str) -> np.ndarray:
    """Copy a PyAV frame plane into the public tightly packed ndarray layout."""
    plane = frame.planes[0]
    bpp = 2 if target_format in {"YUYV", "UYVY"} else 3
    row_bytes = frame.width * bpp
    rows = np.frombuffer(plane, dtype=np.uint8).reshape(frame.height, plane.line_size)
    packed = np.ascontiguousarray(rows[:, :row_bytes])
    if target_format in {"YUYV", "UYVY"}:
        return packed.reshape(frame.height, frame.width // 2, 4)
    return packed.reshape(frame.height, frame.width, 3)


def _native_to_av(source: np.ndarray, width: int, height: int, source_format: str, av):
    frame = av.VideoFrame(width, height, format=_AV_FORMATS[source_format])
    plane = frame.planes[0]
    row_bytes = width * (2 if source_format in _YUV422_FORMATS else 3)
    dst = np.frombuffer(plane, dtype=np.uint8).reshape(height, plane.line_size)[:, :row_bytes]
    np.copyto(dst, np.asarray(source, dtype=np.uint8).reshape(height, row_bytes))
    # The standalone helper has no V4L2 range metadata; treat packed values
    # as full-range, matching FFmpeg's JPEG/full-range convention.
    frame.color_range = 2
    frame.colorspace = 5  # ITU-R BT.601
    return frame


def pyav_crop_resize_native(
    source: np.ndarray,
    x0: int,
    y0: int,
    crop_w: int,
    crop_h: int,
    out_h: int,
    out_w: int,
    pixel_format: str,
) -> np.ndarray | None:
    """Fast PyAV/libswscale crop+scale for upscaling native YUYV/UYVY/RGB3/BGR3 frames."""
    fmt = pixel_format.upper()
    if fmt not in _AV_FORMATS or crop_w < 4 or crop_h < 4 or out_h <= 0 or out_w <= 0:
        return None
    try:
        import av
    except ImportError:
        return None

    yuv = fmt in _YUV422_FORMATS
    if yuv:
        if out_w & 1:
            return None
        in_pair_w = int(source.shape[1])
        x0_p = max(0, min(in_pair_w - 1, int(x0) // 2))
        cw_p = max(2, min(in_pair_w - x0_p, (int(crop_w) + (int(x0) & 1) + 1) // 2))
        src_w_even = cw_p * 2
        row_bytes = cw_p * 4
        out_row_bytes = out_w * 2
        out_shape: tuple[int, ...] = (out_h, out_w // 2, 4)
        crop_2d = source[y0 : y0 + crop_h, x0_p : x0_p + cw_p, :].reshape(crop_h, row_bytes)
    else:
        src_w_even = int(crop_w)
        row_bytes = src_w_even * 3
        out_row_bytes = out_w * 3
        out_shape = (out_h, out_w, 3)
        crop_2d = source[y0 : y0 + crop_h, x0 : x0 + crop_w, :].reshape(crop_h, row_bytes)

    av_fmt = _AV_FORMATS[fmt]
    src_frame = av.VideoFrame(src_w_even, crop_h, format=av_fmt)
    src_frame.color_range = 2
    src_frame.colorspace = 5
    spl = src_frame.planes[0]
    np.copyto(np.frombuffer(spl, dtype=np.uint8).reshape(crop_h, spl.line_size)[:, :row_bytes], crop_2d)

    scaled = src_frame.reformat(width=out_w, height=out_h, format=av_fmt)
    rpl = scaled.planes[0]
    out = np.empty(out_shape, dtype=np.uint8)
    np.copyto(out.reshape(out_h, out_row_bytes), np.frombuffer(rpl, dtype=np.uint8).reshape(out_h, rpl.line_size)[:, :out_row_bytes])
    return out


class FrameDecoder:
    """Decode MJPEG packets and use FFmpeg/PyAV for frame reformatting."""

    def __init__(self, *, fast_dct: bool = True) -> None:
        self._fast_dct = bool(fast_dct)
        self._codec = None

    def _av(self):
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("PyAV is required to decode MJPG or convert camera pixel formats; install av") from exc
        return av

    def __enter__(self) -> FrameDecoder:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def decode_native(
        self,
        raw: bytes | bytearray | memoryview,
        width: int,
        height: int,
        pixel_format: str,
        *,
        bytesperline: int = 0,
        output_format: str | None = None,
    ) -> np.ndarray:
        """Return a native packed frame, decoding MJPG through PyAV as needed."""
        source_format = pixel_format.upper()
        target_format = (output_format or source_format).upper()
        if target_format not in NATIVE_PIXEL_FORMATS:
            raise ValueError(f"Unsupported output pixel format {target_format!r}; choose from {NATIVE_PIXEL_FORMATS}")
        if source_format == "MJPG":
            av = self._av()
            if self._codec is None:
                self._codec = av.CodecContext.create("mjpeg", "r")
                self._codec.thread_count = 1
                if self._fast_dct:
                    if hasattr(av.codec.context, "Flags2") and hasattr(av.codec.context.Flags2, "fast"):
                        self._codec.flags2 |= av.codec.context.Flags2.fast
                    if hasattr(av.codec.context, "Flags") and hasattr(av.codec.context.Flags, "low_delay"):
                        self._codec.flags |= av.codec.context.Flags.low_delay
            frames = self._codec.decode(av.Packet(bytes(raw)))
            if not frames:
                raise ValueError("PyAV did not decode a frame from the MJPG packet")
            frame = frames[0]
            if (frame.width, frame.height) != (width, height):
                raise ValueError(f"MJPG frame is {frame.width}x{frame.height}; expected {width}x{height}")
            reformatted = frame.reformat(width=width, height=height, format=_AV_FORMATS[target_format])
            return _frame_array(reformatted, target_format)
        if source_format not in NATIVE_PIXEL_FORMATS:
            raise ValueError(f"Unsupported camera pixel format {source_format!r}")
        source = native_frame(raw, width, height, source_format, bytesperline=bytesperline)
        if source_format == target_format:
            return source
        av = self._av()
        src_frame = _native_to_av(source, width, height, source_format, av)
        return _frame_array(src_frame.reformat(width=width, height=height, format=_AV_FORMATS[target_format]), target_format)

    def decode(self, raw, width: int, height: int, pixel_format: str) -> np.ndarray:
        """Decode to RGB3 for the legacy decoder API."""
        return self.decode_native(raw, width, height, pixel_format, output_format="RGB3")

    def close(self) -> None:
        self._codec = None


def decode_yuyv_to_rgb(raw: bytes | bytearray | memoryview, width: int, height: int) -> np.ndarray:
    """Convert packed YUYV bytes to RGB using PyAV/FFmpeg."""
    decoder = FrameDecoder()
    try:
        return decoder.decode_native(raw, width, height, "YUYV", output_format="RGB3")
    finally:
        decoder.close()
