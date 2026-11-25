"""AVX-512 C host kernels for MODNet inference and alpha compositing with pure-NumPy fallbacks."""

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

from ..video_depth_anything import host_ops as vda_host_ops

_SRC_PATH = Path(__file__).with_name("_modnet_ops.c")
_REQUIRED_CPU_FLAGS = frozenset({"avx512f", "avx512bw", "avx512vl", "fma"})
_COMPILE_FLAGS = ["-O3", "-mavx512f", "-mavx512bw", "-mavx512vl", "-mfma", "-shared", "-fPIC"]

_C_FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
_C_U8_PTR = ctypes.POINTER(ctypes.c_uint8)
_TLS = threading.local()


def _cpu_supports_avx512() -> bool:
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
    if not _SRC_PATH.is_file() or not _cpu_supports_avx512():
        return None
    try:
        src_bytes = _SRC_PATH.read_bytes()
    except OSError:
        return None

    src_hash = hashlib.sha256(src_bytes).hexdigest()[:16]
    cache_dir = _get_cache_dir()
    so_path = cache_dir / f"_modnet_ops_{src_hash}.so"
    if not so_path.is_file():
        gcc = shutil.which("gcc")
        if gcc is None:
            return None
        tmp_so = cache_dir / f"_modnet_ops_{src_hash}.{os.getpid()}.tmp.so"
        try:
            proc = subprocess.run([gcc, *_COMPILE_FLAGS, "-o", str(tmp_so), str(_SRC_PATH), "-lm"], capture_output=True, text=True, check=False)
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

    lib.dwconv_3x3_p1_f32.argtypes = [
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int,
    ]
    lib.dwconv_3x3_p1_f32.restype = None

    lib.im2col_kxk_p_f32.argtypes = [
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    lib.im2col_kxk_p_f32.restype = None

    lib.conv2d_3x3_s1_p1_direct_f32.argtypes = [
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int,
    ]
    lib.conv2d_3x3_s1_p1_direct_f32.restype = None

    lib.add_bias_act_f32.argtypes = [
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int,
    ]
    lib.add_bias_act_f32.restype = None

    lib.ibnorm_tail_relu_f32.argtypes = [
        _C_FLOAT_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_int,
    ]
    lib.ibnorm_tail_relu_f32.restype = None

    lib.alpha_composite_bg_u8.argtypes = [
        _C_U8_PTR,
        _C_U8_PTR,
        _C_FLOAT_PTR,
        _C_U8_PTR,
        ctypes.c_int64,
    ]
    lib.alpha_composite_bg_u8.restype = None
    return lib


_LIB: ctypes.CDLL | None = _load_c_lib()


def _ptr_f32(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return arr.ctypes.data_as(_C_FLOAT_PTR)


def _ptr_u8(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_uint8):
    return arr.ctypes.data_as(_C_U8_PTR)


def _get_im2col_buf(rows: int, cols: int) -> np.ndarray:
    needed = rows * cols
    buf: np.ndarray | None = getattr(_TLS, "im2col_buf", None)
    if buf is None or buf.size < needed:
        buf = np.empty((needed,), dtype=np.float32)
        _TLS.im2col_buf = buf
    return buf[:needed].reshape(rows, cols)


def dwconv_3x3_p1(
    x: np.ndarray,
    w_c_3x3: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    stride: int = 1,
    relu6: bool = True,
) -> np.ndarray:
    """Depthwise 3x3 padding=1 convolution for (1, C, H, W) float32 tensors."""
    _, c, h, w = x.shape
    oh = (h + 2 - 3) // stride + 1
    ow = (w + 2 - 3) // stride + 1
    x_c = np.ascontiguousarray(x, dtype=np.float32)
    w_c = np.ascontiguousarray(w_c_3x3, dtype=np.float32)
    b_c = np.ascontiguousarray(bias, dtype=np.float32) if bias is not None else None

    if _LIB is not None:
        out = np.empty((1, c, oh, ow), dtype=np.float32)
        _LIB.dwconv_3x3_p1_f32(
            _ptr_f32(x_c),
            _ptr_f32(w_c),
            _ptr_f32(b_c) if b_c is not None else ctypes.cast(0, _C_FLOAT_PTR),
            _ptr_f32(out),
            c,
            h,
            w,
            stride,
            1 if relu6 else 0,
        )
        return out

    x_pad = np.pad(x_c[0], ((0, 0), (1, 1), (1, 1)), mode="constant")
    w_flat = w_c.reshape(c, 3, 3)
    out = np.zeros((1, c, oh, ow), dtype=np.float32)
    for ky in range(3):
        for kx in range(3):
            out[0] += x_pad[:, ky : ky + oh * stride : stride, kx : kx + ow * stride : stride] * w_flat[:, ky, kx][:, None, None]
    if b_c is not None:
        out += b_c.reshape(1, c, 1, 1)
    if relu6:
        np.clip(out, 0.0, 6.0, out=out)
    return out


def im2col_kxk_p(x: np.ndarray, *, kh: int, kw: int, stride: int = 1, padding: int = 0) -> np.ndarray:
    """Pack (1, C_in, H, W) float32 into contiguous (C_in * kh * kw, out_h * out_w)."""
    _, c_in, h, w = x.shape
    oh = (h + 2 * padding - kh) // stride + 1
    ow = (w + 2 * padding - kw) // stride + 1
    k_dim = c_in * kh * kw
    x_c = np.ascontiguousarray(x, dtype=np.float32)
    cols = _get_im2col_buf(k_dim, oh * ow)
    if _LIB is not None:
        _LIB.im2col_kxk_p_f32(_ptr_f32(x_c), _ptr_f32(cols), c_in, h, w, kh, stride, padding)
    else:
        x_pad = np.pad(x_c, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant") if padding > 0 else x_c
        _, s_c, s_h, s_w = x_pad.strides
        patches = np.lib.stride_tricks.as_strided(
            x_pad,
            shape=(c_in, kh, kw, oh, ow),
            strides=(s_c, s_h, s_w, s_h * stride, s_w * stride),
        ).reshape(k_dim, oh * ow)
        np.copyto(cols, patches)
    return cols


def conv2d_nchw(
    x: np.ndarray,
    w_2d: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    kh: int = 1,
    kw: int = 1,
    stride: int = 1,
    padding: int = 0,
    relu_mode: str | None = None,
) -> np.ndarray:
    """Dense Conv2d for N=1 NCHW float32 tensors with fused bias and ReLU/ReLU6."""
    _, c_in, h, w = x.shape
    c_out = w_2d.shape[0]
    w_mat = np.ascontiguousarray(w_2d, dtype=np.float32)
    x_c = np.ascontiguousarray(x, dtype=np.float32)
    b_c = np.ascontiguousarray(bias, dtype=np.float32) if bias is not None else None

    act_id = 1 if relu_mode == "relu" else 2 if relu_mode == "relu6" else 0
    if kh == 1 and kw == 1 and stride == 1 and padding == 0:
        oh, ow = h, w
        out = (w_mat @ x_c.reshape(c_in, h * w)).reshape(1, c_out, oh, ow)
    elif _LIB is not None and kh == 3 and kw == 3 and stride == 1 and padding == 1 and c_out <= 16 and w >= 16:
        out = np.empty((1, c_out, h, w), dtype=np.float32)
        _LIB.conv2d_3x3_s1_p1_direct_f32(
            _ptr_f32(x_c),
            _ptr_f32(w_mat),
            _ptr_f32(b_c) if b_c is not None else ctypes.cast(0, _C_FLOAT_PTR),
            _ptr_f32(out),
            c_in,
            c_out,
            h,
            w,
            act_id,
        )
        return out
    else:
        if kh != kw:
            raise ValueError(f"Expected square kernel, got ({kh}, {kw})")
        oh = (h + 2 * padding - kh) // stride + 1
        ow = (w + 2 * padding - kw) // stride + 1
        k_dim = c_in * kh * kw
        cols = _get_im2col_buf(k_dim, oh * ow)
        if _LIB is not None:
            _LIB.im2col_kxk_p_f32(_ptr_f32(x_c), _ptr_f32(cols), c_in, h, w, kh, stride, padding)
        else:
            x_pad = np.pad(x_c, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant") if padding > 0 else x_c
            _, s_c, s_h, s_w = x_pad.strides
            patches = np.lib.stride_tricks.as_strided(
                x_pad,
                shape=(c_in, kh, kw, oh, ow),
                strides=(s_c, s_h, s_w, s_h * stride, s_w * stride),
            ).reshape(k_dim, oh * ow)
            np.copyto(cols, patches)
        out = (w_mat @ cols).reshape(1, c_out, oh, ow)

    if _LIB is not None and (b_c is not None or act_id != 0):
        _LIB.add_bias_act_f32(
            _ptr_f32(out),
            _ptr_f32(b_c) if b_c is not None else ctypes.cast(0, _C_FLOAT_PTR),
            c_out,
            oh * ow,
            act_id,
        )
    else:
        if b_c is not None:
            out = out + b_c.reshape(1, c_out, 1, 1)
        if relu_mode == "relu":
            out = np.maximum(out, np.float32(0.0))
        elif relu_mode == "relu6":
            out = np.clip(out, np.float32(0.0), np.float32(6.0))
    return out


def ibnorm_tail_relu(x: np.ndarray, *, eps: float = 1e-5, apply_relu: bool = True) -> np.ndarray:
    """Apply InstanceNorm2d to second half of channels (first half BN folded) + ReLU."""
    _, c, h, w = x.shape
    x_c = np.ascontiguousarray(x, dtype=np.float32)
    if _LIB is not None:
        _LIB.ibnorm_tail_relu_f32(_ptr_f32(x_c), c, h * w, float(eps), 1 if apply_relu else 0)
        return x_c

    c_bn = c // 2
    tail = x_c[:, c_bn:, :, :].reshape(1, c - c_bn, h * w)
    mean = np.mean(tail, axis=2, keepdims=True)
    diff = tail - mean
    var = np.mean(diff * diff, axis=2, keepdims=True)
    x_c[:, c_bn:, :, :] = (diff / np.sqrt(var + np.float32(eps))).reshape(1, c - c_bn, h, w)
    if apply_relu:
        np.maximum(x_c, np.float32(0.0), out=x_c)
    return x_c


def bilinear_resize_align_false(x: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """PyTorch F.interpolate(mode='bilinear', align_corners=False) for NCHW float32."""
    return vda_host_ops.bilinear_resize_nchw(x, out_h, out_w, align_corners=False)


def alpha_composite_bg_u8(fg_rgb: np.ndarray, bg_rgb: np.ndarray, alpha_hw: np.ndarray) -> np.ndarray:
    """Composite uint8 (H, W, 3) foreground over background using float32 alpha matte in [0, 1]."""
    fg_c = np.ascontiguousarray(fg_rgb, dtype=np.uint8)
    bg_c = np.ascontiguousarray(bg_rgb, dtype=np.uint8)
    a_c = np.ascontiguousarray(alpha_hw, dtype=np.float32)
    if _LIB is not None:
        out = np.empty_like(fg_c)
        _LIB.alpha_composite_bg_u8(_ptr_u8(fg_c), _ptr_u8(bg_c), _ptr_f32(a_c), _ptr_u8(out), a_c.size)
        return out

    a = np.clip(a_c, 0.0, 1.0)[:, :, None]
    out_f = fg_c.astype(np.float32) * a + bg_c.astype(np.float32) * (np.float32(1.0) - a)
    return np.clip(np.round(out_f), 0.0, 255.0).astype(np.uint8)


def _make_bilinear_1d(in_len: int, out_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if in_len == 1 or out_len == 1:
        idx0 = np.zeros((out_len,), dtype=np.intp)
        return idx0, idx0, np.zeros((out_len,), dtype=np.float32)
    coords = np.clip(
        (np.arange(out_len, dtype=np.float32) + np.float32(0.5)) * np.float32(in_len / out_len) - np.float32(0.5),
        0.0,
        float(in_len - 1),
    )
    i0 = np.floor(coords).astype(np.intp)
    i1 = np.minimum(i0 + 1, in_len - 1)
    w = (coords - i0).astype(np.float32)
    return i0, i1, w


def resize_hwc_u8(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize float32 or uint8 (H, W, 3) array cleanly to uint8 (out_h, out_w, 3)."""
    in_h, in_w, c = arr.shape
    if c != 3:
        raise ValueError(f"Expected (H, W, 3) array, got {arr.shape}")
    if in_h == out_h and in_w == out_w:
        if arr.dtype == np.uint8:
            return np.ascontiguousarray(arr, dtype=np.uint8)
        return np.clip(np.round(arr), 0, 255).astype(np.uint8)
    arr_f32 = np.ascontiguousarray(arr, dtype=np.float32)
    fast = vda_host_ops.bilinear_resize_hwc_u8_fast(arr_f32, out_h, out_w)
    if fast is not None:
        return fast
    y0, y1, wy = _make_bilinear_1d(in_h, out_h)
    x0, x1, wx = _make_bilinear_1d(in_w, out_w)
    wx1 = wx[None, :, None]
    wx0 = np.float32(1.0) - wx1
    wy1 = wy[:, None, None]
    wy0 = np.float32(1.0) - wy1
    tmp = arr_f32[:, x0, :] * wx0 + arr_f32[:, x1, :] * wx1
    out = tmp[y0, :, :] * wy0 + tmp[y1, :, :] * wy1
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
