"""AVX-512 / AVX-512-BF16 host kernels with auto-compilation and pure-NumPy fallback."""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np

_SRC_PATH = Path(__file__).with_name("_host_ops.c")
_REQUIRED_CPU_FLAGS = frozenset({"avx512f", "avx512_bf16", "avx512bw", "avx512vl", "fma"})
_COMPILE_FLAGS = [
    "-O3",
    "-mavx512f",
    "-mavx512bf16",
    "-mavx512bw",
    "-mavx512vl",
    "-mfma",
    "-shared",
    "-fPIC",
]

_C_FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
_C_U16_PTR = ctypes.POINTER(ctypes.c_uint16)
_C_U8_PTR = ctypes.POINTER(ctypes.c_uint8)

_TLS = threading.local()


def _cpu_supports_avx512_bf16() -> bool:
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    for line in cpuinfo.splitlines():
        if line.startswith("flags"):
            flags = set(line.split(":", 1)[1].split())
            return _REQUIRED_CPU_FLAGS.issubset(flags)
    return False


def _get_cache_dir() -> Path:
    candidates = [
        Path.home() / ".cache" / "linux_studio_effects",
        Path(tempfile.gettempdir()) / "linux_studio_effects",
    ]
    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            if os.access(path, os.W_OK | os.X_OK):
                return path
        except OSError:
            continue
    return Path(tempfile.gettempdir())


def _load_c_lib() -> ctypes.CDLL | None:
    if not _SRC_PATH.is_file() or not _cpu_supports_avx512_bf16():
        return None

    try:
        src_bytes = _SRC_PATH.read_bytes()
    except OSError:
        return None

    src_hash = hashlib.sha256(src_bytes).hexdigest()[:16]
    cache_dir = _get_cache_dir()
    so_path = cache_dir / f"_host_ops_{src_hash}.so"

    if not so_path.is_file():
        gcc = shutil.which("gcc")
        if gcc is None:
            return None
        tmp_so = cache_dir / f"_host_ops_{src_hash}.{os.getpid()}.tmp.so"
        try:
            cmd = [gcc, *_COMPILE_FLAGS, "-o", str(tmp_so), str(_SRC_PATH), "-lm"]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0 or not tmp_so.is_file():
                tmp_so.unlink(missing_ok=True)
                return None
            os.replace(tmp_so, so_path)
        except OSError:
            tmp_so.unlink(missing_ok=True)
            return None

    try:
        lib = ctypes.CDLL(str(so_path))
    except OSError:
        return None

    lib.f32_to_bf16_pad.argtypes = [
        _C_FLOAT_PTR,
        ctypes.c_int64,
        _C_U16_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int,
    ]
    lib.f32_to_bf16_pad.restype = None

    lib.bf16_to_f32_unpad.argtypes = [
        _C_U16_PTR,
        ctypes.c_int64,
        _C_FLOAT_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    lib.bf16_to_f32_unpad.restype = None

    lib.bilinear_resize_nchw_f32.argtypes = [
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.bilinear_resize_nchw_f32.restype = ctypes.c_int

    lib.layer_norm_f32.argtypes = [
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
    ]
    lib.layer_norm_f32.restype = None

    lib.softmax_rows_f32.argtypes = [
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    lib.softmax_rows_f32.restype = None

    lib.gelu_inplace_f32.argtypes = [
        _C_FLOAT_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    lib.gelu_inplace_f32.restype = None

    lib.group_norm_nchw_f32.argtypes = [
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
    ]
    lib.group_norm_nchw_f32.restype = None

    lib.im2col_3x3_s1_p1_f32.argtypes = [
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    lib.im2col_3x3_s1_p1_f32.restype = None

    lib.downsample2x_bg_hw4_f32.argtypes = [
        _C_U8_PTR,
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.downsample2x_bg_hw4_f32.restype = None

    lib.bilinear_resize_hwc_f32_to_u8.argtypes = [
        _C_FLOAT_PTR,
        _C_U8_PTR,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.bilinear_resize_hwc_f32_to_u8.restype = ctypes.c_int

    lib.bokeh_composite_u8.argtypes = [
        _C_U8_PTR,
        _C_U8_PTR,
        _C_U8_PTR,
        _C_U8_PTR,
        _C_U8_PTR,
        _C_U8_PTR,
        ctypes.c_int64,
    ]
    lib.bokeh_composite_u8.restype = None

    return lib


_LIB: ctypes.CDLL | None = _load_c_lib()


def has_avx512_host_ops() -> bool:
    """Return True when the compiled AVX-512 / AVX-512-BF16 C backend is active."""
    return _LIB is not None


def _ptr_f32(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return arr.ctypes.data_as(_C_FLOAT_PTR)


def _ptr_u16(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_uint16):
    return arr.ctypes.data_as(_C_U16_PTR)


def _ptr_u8(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_uint8):
    return arr.ctypes.data_as(_C_U8_PTR)


def _get_im2col_scratch(rows: int, cols: int) -> np.ndarray:
    needed = rows * cols
    buf: np.ndarray | None = getattr(_TLS, "im2col_buf", None)
    if buf is None or buf.size < needed:
        buf = np.empty((needed,), dtype=np.float32)
        _TLS.im2col_buf = buf
    return buf[:needed].reshape(rows, cols)


def f32_to_bf16_pad(
    src_f32: np.ndarray,
    dst_u16: np.ndarray,
    rows: int,
    cols: int,
    zero_tail: bool = True,
) -> None:
    """Convert float32 2D array slice [:rows, :cols] to bfloat16 uint16 buffer with optional tail zeroing."""
    src = np.asarray(src_f32, dtype=np.float32)
    dst = dst_u16.view(np.uint16) if dst_u16.dtype != np.uint16 else dst_u16
    if src.ndim != 2 or dst.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got src={src.ndim}D dst={dst.ndim}D")
    dst_rows, dst_cols = dst.shape
    if rows < 0 or cols < 0 or rows > src.shape[0] or cols > src.shape[1]:
        raise ValueError(f"Source slice ({rows}, {cols}) out of bounds for {src.shape}")
    if rows > dst_rows or cols > dst_cols:
        raise ValueError(f"Destination shape {dst.shape} smaller than ({rows}, {cols})")

    if _LIB is not None and dst.flags["WRITEABLE"] and dst.strides[1] == 2 and dst.strides[0] % 2 == 0:
        if src.strides[1] != 4 or src.strides[0] % 4 != 0 or src.strides[0] < cols * 4:
            src = np.ascontiguousarray(src[:rows, :cols], dtype=np.float32)
        _LIB.f32_to_bf16_pad(
            _ptr_f32(src),
            src.strides[0] // 4,
            _ptr_u16(dst),
            dst.strides[0] // 2,
            dst_rows,
            dst_cols,
            rows,
            cols,
            1 if zero_tail else 0,
        )
        return

    sub = np.ascontiguousarray(src[:rows, :cols], dtype=np.float32)
    u32 = sub.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    dst[:rows, :cols] = ((u32 + rounding_bias) >> np.uint32(16)).astype(np.uint16)
    if dst_cols > cols:
        dst[:rows, cols:] = 0
    if zero_tail and dst_rows > rows:
        dst[rows:, :] = 0


def bf16_to_f32_unpad(
    src_u16: np.ndarray,
    dst_f32: np.ndarray,
    rows: int,
    cols: int,
) -> None:
    """Convert bfloat16 uint16 buffer [:rows, :cols] into float32 destination array."""
    src = src_u16.view(np.uint16) if src_u16.dtype != np.uint16 else src_u16
    if src.ndim != 2 or dst_f32.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got src={src.ndim}D dst={dst_f32.ndim}D")
    if dst_f32.dtype != np.float32:
        raise TypeError(f"dst_f32 must have dtype float32, got {dst_f32.dtype}")
    if rows < 0 or cols < 0 or rows > min(src.shape[0], dst_f32.shape[0]) or cols > min(src.shape[1], dst_f32.shape[1]):
        raise ValueError(f"Slice ({rows}, {cols}) out of bounds for src={src.shape}, dst={dst_f32.shape}")

    if (
        _LIB is not None
        and dst_f32.flags["WRITEABLE"]
        and src.strides[1] == 2
        and src.strides[0] % 2 == 0
        and dst_f32.strides[1] == 4
        and dst_f32.strides[0] % 4 == 0
    ):
        _LIB.bf16_to_f32_unpad(
            _ptr_u16(src),
            src.strides[0] // 2,
            _ptr_f32(dst_f32),
            dst_f32.strides[0] // 4,
            rows,
            cols,
        )
        return

    dst_f32[:rows, :cols] = (src[:rows, :cols].astype(np.uint32) << np.uint32(16)).view(np.float32)


def bilinear_resize_nchw(
    x: np.ndarray,
    out_h: int,
    out_w: int,
    align_corners: bool = True,
) -> np.ndarray:
    """Two-pass separable bilinear interpolation for NCHW float32 tensors."""
    if x.ndim != 4:
        raise ValueError(f"Expected 4D NCHW tensor, got shape {x.shape}")
    n, c, in_h, in_w = x.shape
    if in_h == out_h and in_w == out_w:
        return np.ascontiguousarray(x, dtype=np.float32)

    x_c = np.ascontiguousarray(x, dtype=np.float32)
    if _LIB is not None:
        out = np.empty((n, c, out_h, out_w), dtype=np.float32)
        rc = _LIB.bilinear_resize_nchw_f32(
            _ptr_f32(x_c),
            _ptr_f32(out),
            n * c,
            in_h,
            in_w,
            out_h,
            out_w,
            1 if align_corners else 0,
        )
        if rc == 0:
            return out

    if out_w == 1:
        x_coords = np.zeros((1,), dtype=np.float32)
    elif align_corners:
        x_coords = np.linspace(0.0, in_w - 1, out_w, dtype=np.float32)
    else:
        x_coords = np.clip((np.arange(out_w, dtype=np.float32) + 0.5) * (in_w / out_w) - 0.5, 0.0, in_w - 1)

    if out_h == 1:
        y_coords = np.zeros((1,), dtype=np.float32)
    elif align_corners:
        y_coords = np.linspace(0.0, in_h - 1, out_h, dtype=np.float32)
    else:
        y_coords = np.clip((np.arange(out_h, dtype=np.float32) + 0.5) * (in_h / out_h) - 0.5, 0.0, in_h - 1)

    x0 = np.floor(x_coords).astype(np.int64)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wx = (x_coords - x0)[None, None, None, :]

    y0 = np.floor(y_coords).astype(np.int64)
    y1 = np.minimum(y0 + 1, in_h - 1)
    wy = (y_coords - y0)[None, None, :, None]

    tmp = x_c[:, :, :, x0] * (1.0 - wx) + x_c[:, :, :, x1] * wx
    return (tmp[:, :, y0, :] * (1.0 - wy) + tmp[:, :, y1, :] * wy).astype(np.float32, copy=False)


def layer_norm(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """LayerNorm over the last dimension for float32 tensors."""
    orig_shape = x.shape
    d = orig_shape[-1]
    m = x.size // d
    x_c = np.ascontiguousarray(x, dtype=np.float32)
    w_c = np.ascontiguousarray(weight, dtype=np.float32)
    b_c = np.ascontiguousarray(bias, dtype=np.float32)

    if _LIB is not None:
        out = np.empty(orig_shape, dtype=np.float32)
        _LIB.layer_norm_f32(_ptr_f32(x_c), _ptr_f32(w_c), _ptr_f32(b_c), _ptr_f32(out), m, d, float(eps))
        return out

    mean = np.mean(x_c, axis=-1, keepdims=True)
    diff = x_c - mean
    var = np.mean(diff * diff, axis=-1, keepdims=True)
    return ((diff / np.sqrt(var + eps)) * w_c + b_c).astype(np.float32, copy=False)


def softmax_rows(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last dimension for float32 tensors."""
    orig_shape = x.shape
    d = orig_shape[-1]
    m = x.size // d
    x_c = np.ascontiguousarray(x, dtype=np.float32)

    if _LIB is not None:
        out = np.empty(orig_shape, dtype=np.float32)
        _LIB.softmax_rows_f32(_ptr_f32(x_c), _ptr_f32(out), m, d)
        return out

    mx = np.max(x_c, axis=-1, keepdims=True)
    e = np.exp(x_c - mx)
    return (e / np.sum(e, axis=-1, keepdims=True)).astype(np.float32, copy=False)


def gelu_inplace(x: np.ndarray) -> np.ndarray:
    """In-place tanh-approximation GELU for float32 arrays (supports 2D row-strided views)."""
    if x.dtype != np.float32:
        raise TypeError(f"gelu_inplace requires float32 array, got {x.dtype}")
    if not x.flags["WRITEABLE"]:
        raise ValueError("gelu_inplace requires a writeable array")

    if _LIB is not None:
        if x.flags["C_CONTIGUOUS"]:
            _LIB.gelu_inplace_f32(_ptr_f32(x), 1, x.size, x.size)
            return x
        if x.ndim == 2 and x.strides[1] == 4 and x.strides[0] >= x.shape[1] * 4 and x.strides[0] % 4 == 0:
            _LIB.gelu_inplace_f32(_ptr_f32(x), x.shape[0], x.shape[1], x.strides[0] // 4)
            return x
        tmp = np.ascontiguousarray(x, dtype=np.float32)
        _LIB.gelu_inplace_f32(_ptr_f32(tmp), 1, tmp.size, tmp.size)
        x[...] = tmp
        return x

    x[:] = (
        np.float32(0.5)
        * x
        * (np.float32(1.0) + np.tanh(np.float32(0.79788456) * (x + np.float32(0.044715) * (x * x * x))))
    ).astype(np.float32, copy=False)
    return x


def group_norm_nchw(
    x: np.ndarray,
    num_groups: int,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """GroupNorm for NCHW float32 tensors."""
    if x.ndim != 4:
        raise ValueError(f"Expected 4D NCHW tensor, got shape {x.shape}")
    n, c, h, w = x.shape
    if c % num_groups != 0:
        raise ValueError(f"Channels ({c}) must be divisible by num_groups ({num_groups})")

    x_c = np.ascontiguousarray(x, dtype=np.float32)
    w_c = np.ascontiguousarray(weight, dtype=np.float32)
    b_c = np.ascontiguousarray(bias, dtype=np.float32)

    if _LIB is not None:
        out = np.empty((n, c, h, w), dtype=np.float32)
        _LIB.group_norm_nchw_f32(
            _ptr_f32(x_c),
            _ptr_f32(w_c),
            _ptr_f32(b_c),
            _ptr_f32(out),
            n,
            c,
            h,
            w,
            num_groups,
            float(eps),
        )
        return out

    g_arr = x_c.reshape(n, num_groups, c // num_groups, h * w)
    mean = np.mean(g_arr, axis=(2, 3), keepdims=True)
    diff = g_arr - mean
    var = np.mean(diff * diff, axis=(2, 3), keepdims=True)
    normed = (diff / np.sqrt(var + eps)).reshape(n, c, h, w)
    return (normed * w_c[None, :, None, None] + b_c[None, :, None, None]).astype(np.float32, copy=False)


def im2col_3x3_s1_p1(x: np.ndarray, out_cols: np.ndarray | None = None) -> np.ndarray:
    """Pack N=1 NCHW float32 tensor into contiguous (C_in * 9, H * W) im2col matrix for 3x3 s=1 p=1 conv."""
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError(f"Expected (1, C, H, W) tensor, got {x.shape}")
    _, c_in, h, w = x.shape
    x_c = np.ascontiguousarray(x, dtype=np.float32)

    expected_shape = (c_in * 9, h * w)
    if out_cols is None:
        out_cols = np.empty(expected_shape, dtype=np.float32)
    else:
        if out_cols.shape != expected_shape:
            raise ValueError(f"out_cols shape {out_cols.shape} != expected {expected_shape}")
        if out_cols.dtype != np.float32 or not out_cols.flags["C_CONTIGUOUS"]:
            raise ValueError("out_cols must be a C-contiguous float32 array")

    if _LIB is not None:
        _LIB.im2col_3x3_s1_p1_f32(_ptr_f32(x_c), _ptr_f32(out_cols), c_in, h, w)
        return out_cols

    x_pad = np.pad(x_c, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
    _, s_c, s_h, s_w = x_pad.strides
    patches = np.lib.stride_tricks.as_strided(
        x_pad,
        shape=(c_in, 3, 3, h, w),
        strides=(s_c, s_h, s_w, s_h, s_w),
    ).reshape(c_in * 9, h * w)
    np.copyto(out_cols, patches)
    return out_cols


def conv2d_3x3_s1_p1(
    x: np.ndarray,
    w_2d: np.ndarray,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """Fast 3x3 stride=1 padding=1 Conv2d for NCHW float32 tensors using C im2col + BLAS SGEMM."""
    if x.ndim != 4:
        raise ValueError(f"Expected 4D NCHW tensor, got {x.shape}")
    n, c_in, h, w = x.shape
    if w_2d.ndim == 4:
        c_out = w_2d.shape[0]
        w_mat = np.ascontiguousarray(w_2d.reshape(c_out, c_in * 9), dtype=np.float32)
    else:
        c_out = w_2d.shape[0]
        w_mat = np.ascontiguousarray(w_2d, dtype=np.float32)

    if n == 1:
        cols = _get_im2col_scratch(c_in * 9, h * w)
        im2col_3x3_s1_p1(x, cols)
        out = (w_mat @ cols).reshape(1, c_out, h, w)
    else:
        out = np.empty((n, c_out, h, w), dtype=np.float32)
        cols = _get_im2col_scratch(c_in * 9, h * w)
        for idx in range(n):
            im2col_3x3_s1_p1(x[idx : idx + 1], cols)
            out[idx] = (w_mat @ cols).reshape(c_out, h, w)

    if bias is not None:
        out += np.ascontiguousarray(bias, dtype=np.float32).reshape(1, c_out, 1, 1)
    return out


def downsample2x_bg_hw4(frame_u8: np.ndarray, w0: np.ndarray) -> np.ndarray:
    """Premultiply uint8 (H, W, 3) by background mask w0 and downsample 2x into (ceil(H/2), ceil(W/2), 4)."""
    h, w, c = frame_u8.shape
    if c != 3:
        raise ValueError(f"Expected (H, W, 3) frame, got {frame_u8.shape}")
    out_h, out_w = (h + 1) // 2, (w + 1) // 2
    if _LIB is not None:
        f_c = np.ascontiguousarray(frame_u8, dtype=np.uint8)
        w_c = np.ascontiguousarray(w0, dtype=np.float32)
        out = np.empty((out_h, out_w, 4), dtype=np.float32)
        _LIB.downsample2x_bg_hw4_f32(_ptr_u8(f_c), _ptr_f32(w_c), _ptr_f32(out), h, w)
        return out

    cur = np.empty((h, w, 4), dtype=np.float32)
    cur[:, :, :3] = frame_u8.astype(np.float32) * w0[:, :, None]
    cur[:, :, 3] = w0
    pad_h = h & 1
    pad_w = w & 1
    if pad_h or pad_w:
        cur = np.pad(cur, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return np.float32(0.25) * (cur[0::2, 0::2] + cur[0::2, 1::2] + cur[1::2, 0::2] + cur[1::2, 1::2])


def bilinear_resize_hwc_u8_fast(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray | None:
    """Fast C/AVX-512 bilinear resize of float32 (H, W, 3) to uint8 (out_h, out_w, 3), or None if unavailable."""
    if _LIB is None:
        return None
    in_h, in_w, c = arr.shape
    if c != 3:
        raise ValueError(f"Expected (H, W, 3) array, got {arr.shape}")
    arr_c = np.ascontiguousarray(arr, dtype=np.float32)
    out = np.empty((out_h, out_w, 3), dtype=np.uint8)
    rc = _LIB.bilinear_resize_hwc_f32_to_u8(_ptr_f32(arr_c), _ptr_u8(out), in_h, in_w, out_h, out_w)
    return out if rc == 0 else None


def bokeh_composite_u8(
    l0: np.ndarray,
    l1: np.ndarray,
    l2: np.ndarray,
    l3: np.ndarray,
    coc_u8: np.ndarray,
) -> np.ndarray:
    """Fixed-point 4-level portrait bokeh blend matching the AIE2 NPU kernel bit-for-bit."""
    if _LIB is not None:
        l0_c = np.ascontiguousarray(l0, dtype=np.uint8)
        l1_c = np.ascontiguousarray(l1, dtype=np.uint8)
        l2_c = np.ascontiguousarray(l2, dtype=np.uint8)
        l3_c = np.ascontiguousarray(l3, dtype=np.uint8)
        coc_c = np.ascontiguousarray(coc_u8, dtype=np.uint8)
        out = np.empty_like(l0_c)
        _LIB.bokeh_composite_u8(
            _ptr_u8(l0_c),
            _ptr_u8(l1_c),
            _ptr_u8(l2_c),
            _ptr_u8(l3_c),
            _ptr_u8(coc_c),
            _ptr_u8(out),
            coc_c.size,
        )
        return out

    # uint16 is exact because sum(w_i) == 256 and max(l_i) == 255 -> max sum == 65408 <= 65535.
    c = coc_u8.astype(np.uint16)
    d2 = np.maximum(c, 170) - 170
    t2 = d2 * 3 + (d2 >> 6)
    w0 = np.where(c < 85, 256 - c * 3, 0)[:, :, None].astype(np.uint16)
    w1 = np.where(c < 85, c * 3, np.where(c < 170, 256 - (np.maximum(c, 85) - 85) * 3, 0))[:, :, None].astype(np.uint16)
    w2 = np.where((c >= 85) & (c < 170), (np.maximum(c, 85) - 85) * 3, np.where(c >= 170, 256 - t2, 0))[:, :, None].astype(np.uint16)
    w3 = np.where(c >= 170, t2, 0)[:, :, None].astype(np.uint16)
    acc = l0.astype(np.uint16) * w0
    acc += l1.astype(np.uint16) * w1
    acc += l2.astype(np.uint16) * w2
    acc += l3.astype(np.uint16) * w3
    acc += 128
    acc >>= 8
    return np.ascontiguousarray(acc.astype(np.uint8))
