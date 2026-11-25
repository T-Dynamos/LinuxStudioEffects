"""PyAV-backed decoding and pixel-format conversion for camera frames."""

from __future__ import annotations

import numpy as np

from .pixels import NATIVE_PIXEL_FORMATS, native_frame

_AV_FORMATS = {"YUYV": "yuyv422", "UYVY": "uyvy422", "RGB3": "rgb24", "BGR3": "bgr24"}


def has_pyav() -> bool:
    """Return whether PyAV is importable."""
    try:
        import av  # noqa: F401
    except ImportError:
        return False
    return True


def _frame_array(frame, target_format: str) -> np.ndarray:
    """Copy a PyAV frame plane into the public tightly packed ndarray layout."""
    if target_format != "UYVY":
        arr = frame.to_ndarray()
        if target_format in {"YUYV", "UYVY"}:
            return np.ascontiguousarray(arr).reshape(frame.height, frame.width // 2, 4)
        return np.ascontiguousarray(arr).reshape(frame.height, frame.width, 3)
    plane = frame.planes[0]
    row_bytes = frame.width * 2
    data = bytes(plane)
    rows = np.frombuffer(data, dtype=np.uint8).reshape(frame.height, plane.line_size)
    return np.ascontiguousarray(rows[:, :row_bytes]).reshape(frame.height, frame.width // 2, 4)


def _native_to_av(source: np.ndarray, width: int, height: int, source_format: str, av):
    if source_format in {"YUYV", "UYVY"}:
        frame = av.VideoFrame(width, height, format=_AV_FORMATS[source_format])
        plane = frame.planes[0]
        packed = source.reshape(height, width * 2)
        padded = np.zeros((height, plane.line_size), dtype=np.uint8)
        padded[:, : width * 2] = packed
        plane.update(padded.tobytes())
        # The standalone helper has no V4L2 range metadata; treat packed values
        # as full-range, matching FFmpeg's JPEG/full-range convention.
        frame.color_range = 2
        frame.colorspace = 5  # ITU-R BT.601
        return frame
    frame = av.VideoFrame.from_ndarray(source, format=_AV_FORMATS[source_format])
    frame.color_range = 2
    frame.colorspace = 5
    return frame


class FrameDecoder:
    """Decode MJPEG packets and use FFmpeg/PyAV for frame reformatting."""

    def __init__(self, *, fast_dct: bool = True) -> None:
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
