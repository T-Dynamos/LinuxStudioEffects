"""Fast frame decoding from V4L2 pixel formats (MJPG, YUYV, GREY) to RGB NumPy arrays."""

from __future__ import annotations

import ctypes
import ctypes.util
import threading

import numpy as np

# TurboJPEG constants from <turbojpeg.h>
_TJPF_RGB = 0
_TJFLAG_FASTDCT = 2048

_LIBTURBOJPEG_CANDIDATES = (
    "libturbojpeg.so.0",
    "libturbojpeg.so",
)


def _load_turbojpeg() -> ctypes.CDLL | None:
    for name in _LIBTURBOJPEG_CANDIDATES:
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    else:
        found = ctypes.util.find_library("turbojpeg")
        if not found:
            return None
        try:
            lib = ctypes.CDLL(found)
        except OSError:
            return None

    lib.tjInitDecompress.argtypes = []
    lib.tjInitDecompress.restype = ctypes.c_void_p

    lib.tjDestroy.argtypes = [ctypes.c_void_p]
    lib.tjDestroy.restype = ctypes.c_int

    lib.tjDecompressHeader3.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.tjDecompressHeader3.restype = ctypes.c_int

    lib.tjDecompress2.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.tjDecompress2.restype = ctypes.c_int

    lib.tjGetErrorStr2.argtypes = [ctypes.c_void_p]
    lib.tjGetErrorStr2.restype = ctypes.c_char_p
    return lib


_TJ_LIB = _load_turbojpeg()


def has_turbojpeg() -> bool:
    """Return True if libturbojpeg can be loaded and initialized."""
    if _TJ_LIB is None:
        return False
    handle = _TJ_LIB.tjInitDecompress()
    if not handle:
        return False
    _TJ_LIB.tjDestroy(handle)
    return True


def decode_yuyv_to_rgb(raw: bytes | bytearray | memoryview, width: int, height: int) -> np.ndarray:
    """Decode packed YUYV (4:2:2) bytes into an (H, W, 3) uint8 RGB NumPy array."""
    expected = width * height * 2
    arr = np.frombuffer(raw, dtype=np.uint8, count=expected).reshape(height, width // 2, 4)

    y0 = arr[:, :, 0].astype(np.int32)
    u = arr[:, :, 1].astype(np.int32) - 128
    y1 = arr[:, :, 2].astype(np.int32)
    v = arr[:, :, 3].astype(np.int32) - 128

    # Integer BT.601 full-range approximation (scaled by 256)
    r_off = (359 * v) >> 8
    g_off = (-88 * u - 183 * v) >> 8
    b_off = (454 * u) >> 8

    out = np.empty((height, width, 3), dtype=np.uint8)
    out[:, 0::2, 0] = np.clip(y0 + r_off, 0, 255)
    out[:, 0::2, 1] = np.clip(y0 + g_off, 0, 255)
    out[:, 0::2, 2] = np.clip(y0 + b_off, 0, 255)
    out[:, 1::2, 0] = np.clip(y1 + r_off, 0, 255)
    out[:, 1::2, 1] = np.clip(y1 + g_off, 0, 255)
    out[:, 1::2, 2] = np.clip(y1 + b_off, 0, 255)
    return out


class FrameDecoder:
    """Stateful decoder converting raw V4L2 payloads to RGB uint8 NumPy frames."""

    def __init__(self, *, fast_dct: bool = True) -> None:
        self._fast_dct = fast_dct
        self._lock = threading.Lock()
        self._tj_handle: int | None = None
        if _TJ_LIB is not None:
            handle = _TJ_LIB.tjInitDecompress()
            if handle:
                self._tj_handle = handle

    def close(self) -> None:
        with self._lock:
            if self._tj_handle is not None and _TJ_LIB is not None:
                _TJ_LIB.tjDestroy(self._tj_handle)
                self._tj_handle = None

    def __enter__(self) -> FrameDecoder:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def decode(self, raw: bytes | bytearray | memoryview, width: int, height: int, pixel_format: str) -> np.ndarray:
        """Decode a single raw frame into an (H, W, 3) uint8 RGB array."""
        fmt = pixel_format.upper()
        if fmt == "MJPG":
            return self._decode_mjpeg(raw, width, height)
        if fmt == "YUYV":
            return decode_yuyv_to_rgb(raw, width, height)
        if fmt == "GREY":
            gray = np.frombuffer(raw, dtype=np.uint8, count=width * height).reshape(height, width)
            return np.repeat(gray[:, :, np.newaxis], 3, axis=2)
        raise ValueError(f"Unsupported V4L2 pixel format for decoding: {pixel_format!r}")

    def _decode_mjpeg(self, raw: bytes | bytearray | memoryview, width: int, height: int) -> np.ndarray:
        if _TJ_LIB is None or not self._tj_handle:
            raise RuntimeError("libturbojpeg.so.0 is required to decode MJPG frames")

        nbytes = len(raw)
        if isinstance(raw, bytes):
            src_arr = (ctypes.c_uint8 * nbytes).from_buffer_copy(raw)
        else:
            try:
                src_arr = (ctypes.c_uint8 * nbytes).from_buffer(raw)
            except TypeError:
                src_arr = (ctypes.c_uint8 * nbytes).from_buffer_copy(raw)

        out = np.empty((height, width, 3), dtype=np.uint8)
        dst_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        flags = _TJFLAG_FASTDCT if self._fast_dct else 0

        with self._lock:
            rc = _TJ_LIB.tjDecompress2(
                self._tj_handle,
                src_arr,
                nbytes,
                dst_ptr,
                width,
                width * 3,
                height,
                _TJPF_RGB,
                flags,
            )
            if rc != 0:
                err_msg = _TJ_LIB.tjGetErrorStr2(self._tj_handle)
                raise RuntimeError(f"TurboJPEG decode failed: {err_msg.decode('utf-8', errors='replace')}")
        return out
