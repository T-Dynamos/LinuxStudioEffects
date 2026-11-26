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
    "-pthread",
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

    src_hash = hashlib.sha256(src_bytes + " ".join(_COMPILE_FLAGS).encode("utf-8")).hexdigest()[:16]
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

    lib.f32_unpad_bias_gelu.argtypes = [
        _C_FLOAT_PTR,
        ctypes.c_int64,
        _C_FLOAT_PTR,
        ctypes.c_int64,
        _C_FLOAT_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int,
    ]
    lib.f32_unpad_bias_gelu.restype = None

    lib.bf16_unpad_bias_gelu.argtypes = [
        _C_U16_PTR,
        ctypes.c_int64,
        _C_FLOAT_PTR,
        ctypes.c_int64,
        _C_FLOAT_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int,
    ]
    lib.bf16_unpad_bias_gelu.restype = None

    lib.f32_transpose_to_bf16_pad.argtypes = [
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
    lib.f32_transpose_to_bf16_pad.restype = None

    lib.f32_unpad_transpose_bias_relu.argtypes = [
        _C_FLOAT_PTR,
        ctypes.c_int64,
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int,
    ]
    lib.f32_unpad_transpose_bias_relu.restype = None

    lib.bf16_unpad_transpose_bias_relu.argtypes = [
        _C_U16_PTR,
        ctypes.c_int64,
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int,
    ]
    lib.bf16_unpad_transpose_bias_relu.restype = None

    lib.temporal_attn_roll_f32.argtypes = [
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_int,
    ]
    lib.temporal_attn_roll_f32.restype = None

    lib.temporal_attn_roll_ring_f32.argtypes = [
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.temporal_attn_roll_ring_f32.restype = None

    lib.im2col_3x3_s1_p1_kykxc_bf16_pad.argtypes = [
        _C_FLOAT_PTR,
        _C_U16_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int,
    ]
    lib.im2col_3x3_s1_p1_kykxc_bf16_pad.restype = ctypes.c_int

    lib.gaussian5_hw4_f32.argtypes = [
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.gaussian5_hw4_f32.restype = ctypes.c_int

    lib.downsample2x_hw4_f32.argtypes = [
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.downsample2x_hw4_f32.restype = None

    lib.box_filter_2d_f32.argtypes = [
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.box_filter_2d_f32.restype = ctypes.c_int

    lib.extract_luma_and_downsample_f32.argtypes = [
        _C_U8_PTR,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
    ]
    lib.extract_luma_and_downsample_f32.restype = ctypes.c_int

    lib.guided_upsample_coc_f32.argtypes = [
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]
    lib.guided_upsample_coc_f32.restype = ctypes.c_int

    lib.preprocess_rgb_u8_to_nchw_f32.argtypes = [
        _C_U8_PTR,
        _C_FLOAT_PTR,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.preprocess_rgb_u8_to_nchw_f32.restype = ctypes.c_int

    lib.fused_mha_qkv_f32.argtypes = [
        _C_FLOAT_PTR,
        _C_FLOAT_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
    ]
    lib.fused_mha_qkv_f32.restype = ctypes.c_int

    lib.pack_bokeh_npu_inputs_soa32_u8.argtypes = [
        _C_U8_PTR,
        _C_U8_PTR,
        _C_U8_PTR,
        _C_U8_PTR,
        _C_U8_PTR,
        _C_U8_PTR,
        _C_U8_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    lib.pack_bokeh_npu_inputs_soa32_u8.restype = ctypes.c_int

    lib.unpremul_resize_hw4_to_rgb_u8_f32.argtypes = [
        _C_FLOAT_PTR,
        _C_U8_PTR,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.unpremul_resize_hw4_to_rgb_u8_f32.restype = ctypes.c_int

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


def f32_unpad_bias_gelu(
    src_f32: np.ndarray,
    dst_f32: np.ndarray,
    rows: int,
    cols: int,
    *,
    bias: np.ndarray | None = None,
    apply_gelu: bool = False,
) -> None:
    """Copy float32 [:rows, :cols] from padded NPU output with optional bias add and GELU."""
    if _LIB is not None and dst_f32.flags["WRITEABLE"] and src_f32.strides[1] == 4 and dst_f32.strides[1] == 4:
        b_c = np.ascontiguousarray(bias, dtype=np.float32) if bias is not None else None
        _LIB.f32_unpad_bias_gelu(
            _ptr_f32(src_f32),
            src_f32.strides[0] // 4,
            _ptr_f32(dst_f32),
            dst_f32.strides[0] // 4,
            _ptr_f32(b_c) if b_c is not None else ctypes.cast(0, _C_FLOAT_PTR),
            rows,
            cols,
            1 if apply_gelu else 0,
        )
        return

    sub = src_f32[:rows, :cols]
    if bias is not None:
        sub = sub + bias[:cols]
    dst_f32[:rows, :cols] = sub
    if apply_gelu:
        gelu_inplace(dst_f32[:rows, :cols])


def bf16_unpad_bias_gelu(
    src_u16: np.ndarray,
    dst_f32: np.ndarray,
    rows: int,
    cols: int,
    *,
    bias: np.ndarray | None = None,
    apply_gelu: bool = False,
) -> None:
    """Convert bfloat16 uint16 [:rows, :cols] from padded NPU output to float32 with optional bias add and GELU."""
    src = src_u16.view(np.uint16) if src_u16.dtype != np.uint16 else src_u16
    if (
        _LIB is not None
        and dst_f32.flags["WRITEABLE"]
        and src.strides[1] == 2
        and src.strides[0] % 2 == 0
        and dst_f32.strides[1] == 4
        and dst_f32.strides[0] % 4 == 0
    ):
        b_c = np.ascontiguousarray(bias, dtype=np.float32) if bias is not None else None
        _LIB.bf16_unpad_bias_gelu(
            _ptr_u16(src),
            src.strides[0] // 2,
            _ptr_f32(dst_f32),
            dst_f32.strides[0] // 4,
            _ptr_f32(b_c) if b_c is not None else ctypes.cast(0, _C_FLOAT_PTR),
            rows,
            cols,
            1 if apply_gelu else 0,
        )
        return

    sub = (src[:rows, :cols].astype(np.uint32) << np.uint32(16)).view(np.float32)
    if bias is not None:
        sub = sub + bias[:cols]
    dst_f32[:rows, :cols] = sub
    if apply_gelu:
        gelu_inplace(dst_f32[:rows, :cols])


def f32_transpose_to_bf16_pad(
    src_f32: np.ndarray,
    dst_u16: np.ndarray,
    src_rows: int,
    src_cols: int,
    zero_tail: bool = True,
) -> None:
    """Transpose float32 [:src_rows, :src_cols] -> dst[:src_cols, :src_rows] as padded bfloat16 uint16."""
    src = np.asarray(src_f32, dtype=np.float32)
    dst = dst_u16.view(np.uint16) if dst_u16.dtype != np.uint16 else dst_u16
    if src.ndim != 2 or dst.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got src={src.ndim}D dst={dst.ndim}D")
    dst_rows, dst_cols = dst.shape
    if src_rows < 0 or src_cols < 0 or src_rows > src.shape[0] or src_cols > src.shape[1]:
        raise ValueError(f"Source slice ({src_rows}, {src_cols}) out of bounds for {src.shape}")
    if src_cols > dst_rows or src_rows > dst_cols:
        raise ValueError(f"Destination shape {dst.shape} smaller than transposed ({src_cols}, {src_rows})")

    if _LIB is not None and dst.flags["WRITEABLE"] and dst.strides[1] == 2 and dst.strides[0] % 2 == 0:
        if src.strides[1] != 4 or src.strides[0] % 4 != 0 or src.strides[0] < src_cols * 4:
            src = np.ascontiguousarray(src[:src_rows, :src_cols], dtype=np.float32)
        _LIB.f32_transpose_to_bf16_pad(
            _ptr_f32(src),
            src.strides[0] // 4,
            _ptr_u16(dst),
            dst.strides[0] // 2,
            dst_rows,
            dst_cols,
            src_rows,
            src_cols,
            1 if zero_tail else 0,
        )
        return

    sub = np.ascontiguousarray(src[:src_rows, :src_cols].T, dtype=np.float32)
    u32 = sub.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    dst[:src_cols, :src_rows] = ((u32 + rounding_bias) >> np.uint32(16)).astype(np.uint16)
    if dst_cols > src_rows:
        dst[:src_cols, src_rows:] = 0
    if zero_tail and dst_rows > src_cols:
        dst[src_cols:, :] = 0


def f32_unpad_transpose_bias_relu(
    src_f32: np.ndarray,
    dst_c_hw: np.ndarray,
    m_hw: int,
    c_out: int,
    *,
    bias: np.ndarray | None = None,
    apply_relu: bool = False,
) -> None:
    """Transpose (M, C_out) float32 NPU output into contiguous (C_out, M) with optional bias and ReLU."""
    if dst_c_hw.dtype != np.float32 or not dst_c_hw.flags["C_CONTIGUOUS"] or dst_c_hw.size != c_out * m_hw:
        raise ValueError("dst_c_hw must be C-contiguous float32 matching (c_out, m_hw)")

    if _LIB is not None and dst_c_hw.flags["WRITEABLE"] and src_f32.strides[1] == 4:
        b_c = np.ascontiguousarray(bias, dtype=np.float32) if bias is not None else None
        _LIB.f32_unpad_transpose_bias_relu(
            _ptr_f32(src_f32),
            src_f32.strides[0] // 4,
            _ptr_f32(dst_c_hw),
            _ptr_f32(b_c) if b_c is not None else ctypes.cast(0, _C_FLOAT_PTR),
            m_hw,
            c_out,
            1 if apply_relu else 0,
        )
        return

    sub = src_f32[:m_hw, :c_out]
    if bias is not None:
        sub = sub + bias[:c_out]
    if apply_relu:
        sub = np.maximum(sub, np.float32(0.0))
    np.copyto(dst_c_hw.reshape(c_out, m_hw), sub.T)


def bf16_unpad_transpose_bias_relu(
    src_u16: np.ndarray,
    dst_c_hw: np.ndarray,
    m_hw: int,
    c_out: int,
    *,
    bias: np.ndarray | None = None,
    apply_relu: bool = False,
) -> None:
    """Convert and transpose (M, C_out) bfloat16 uint16 NPU output into contiguous (C_out, M) float32."""
    src = src_u16.view(np.uint16) if src_u16.dtype != np.uint16 else src_u16
    if dst_c_hw.dtype != np.float32 or not dst_c_hw.flags["C_CONTIGUOUS"] or dst_c_hw.size != c_out * m_hw:
        raise ValueError("dst_c_hw must be C-contiguous float32 matching (c_out, m_hw)")

    if _LIB is not None and dst_c_hw.flags["WRITEABLE"] and src.strides[1] == 2 and src.strides[0] % 2 == 0:
        b_c = np.ascontiguousarray(bias, dtype=np.float32) if bias is not None else None
        _LIB.bf16_unpad_transpose_bias_relu(
            _ptr_u16(src),
            src.strides[0] // 2,
            _ptr_f32(dst_c_hw),
            _ptr_f32(b_c) if b_c is not None else ctypes.cast(0, _C_FLOAT_PTR),
            m_hw,
            c_out,
            1 if apply_relu else 0,
        )
        return

    sub = (src[:m_hw, :c_out].astype(np.uint32) << np.uint32(16)).view(np.float32)
    if bias is not None:
        sub = sub + bias[:c_out]
    if apply_relu:
        sub = np.maximum(sub, np.float32(0.0))
    np.copyto(dst_c_hw.reshape(c_out, m_hw), sub.T)


def im2col_3x3_s1_p1_kykxc_bf16_pad(
    x_nchw: np.ndarray,
    dst_u16: np.ndarray,
    zero_tail: bool = True,
) -> None:
    """Pack (1, C_in, H, W) float32 directly into (H*W, 9*C_in) padded bf16 uint16 in (ky, kx, C_in) order."""
    if x_nchw.ndim != 4 or x_nchw.shape[0] != 1:
        raise ValueError(f"Expected (1, C, H, W) tensor, got {x_nchw.shape}")
    _, c_in, h, w = x_nchw.shape
    dst = dst_u16.view(np.uint16) if dst_u16.dtype != np.uint16 else dst_u16
    dst_rows, dst_cols = dst.shape
    hw = h * w
    k_elems = 9 * c_in
    if hw > dst_rows or k_elems > dst_cols:
        raise ValueError(f"Destination {dst.shape} smaller than ({hw}, {k_elems})")

    x_c = np.ascontiguousarray(x_nchw, dtype=np.float32)
    if _LIB is not None and dst.flags["WRITEABLE"] and dst.strides[1] == 2 and dst.strides[0] % 2 == 0:
        rc = _LIB.im2col_3x3_s1_p1_kykxc_bf16_pad(
            _ptr_f32(x_c),
            _ptr_u16(dst),
            dst.strides[0] // 2,
            dst_rows,
            dst_cols,
            c_in,
            h,
            w,
            1 if zero_tail else 0,
        )
        if rc == 0:
            return

    x_pad = np.pad(x_c[0], ((0, 0), (1, 1), (1, 1)), mode="constant")
    hwc_pad = np.transpose(x_pad, (1, 2, 0))
    patches = np.empty((hw, k_elems), dtype=np.float32)
    for ky in range(3):
        for kx in range(3):
            sl = hwc_pad[ky : ky + h, kx : kx + w, :].reshape(hw, c_in)
            off = (ky * 3 + kx) * c_in
            patches[:, off : off + c_in] = sl
    f32_to_bf16_pad(patches, dst, hw, k_elems, zero_tail=zero_tail)


def temporal_attn_roll(
    q: np.ndarray,
    k_cur: np.ndarray,
    v_cur: np.ndarray,
    k_buf: np.ndarray,
    v_buf: np.ndarray,
    k_pe: np.ndarray,
    v_pe: np.ndarray,
    n_heads: int,
    scale: float,
    has_history: bool,
) -> np.ndarray:
    """Fused rolling-window temporal multi-head self-attention for (d, ch) inputs."""
    d, ch = k_cur.shape
    t_len = k_buf.shape[1]
    dh = ch // n_heads

    if (
        _LIB is not None
        and q.flags["C_CONTIGUOUS"]
        and k_cur.flags["C_CONTIGUOUS"]
        and v_cur.flags["C_CONTIGUOUS"]
        and k_buf.flags["C_CONTIGUOUS"]
        and v_buf.flags["C_CONTIGUOUS"]
        and k_pe.flags["C_CONTIGUOUS"]
        and v_pe.flags["C_CONTIGUOUS"]
    ):
        out = np.empty((d, ch), dtype=np.float32)
        _LIB.temporal_attn_roll_f32(
            _ptr_f32(q),
            _ptr_f32(k_cur),
            _ptr_f32(v_cur),
            _ptr_f32(k_buf),
            _ptr_f32(v_buf),
            _ptr_f32(k_pe),
            _ptr_f32(v_pe),
            _ptr_f32(out),
            d,
            t_len,
            n_heads,
            dh,
            float(scale),
            1 if has_history else 0,
        )
        return out

    if not has_history:
        k_buf[:, : t_len - 1, :] = k_cur[:, None, :]
        v_buf[:, : t_len - 1, :] = v_cur[:, None, :]
        return (v_cur + v_pe[0]).astype(np.float32, copy=False)

    k_buf[:, t_len - 1, :] = k_cur
    v_buf[:, t_len - 1, :] = v_cur
    k = k_buf + k_pe[None, :, :]
    v = v_buf + v_pe[None, :, :]
    qh = q.reshape(d, n_heads, 1, dh)
    kh = k.reshape(d, t_len, n_heads, dh).transpose(0, 2, 3, 1)
    vh = v.reshape(d, t_len, n_heads, dh).transpose(0, 2, 1, 3)
    attn = softmax_rows(np.matmul(qh, kh) * np.float32(scale))
    out = np.ascontiguousarray(np.matmul(attn, vh).reshape(d, ch), dtype=np.float32)
    if t_len >= 4:
        k_buf[:, 2 : t_len - 2, :] = k_buf[:, 3 : t_len - 1, :]
        k_buf[:, t_len - 2, :] = k_cur
        v_buf[:, 2 : t_len - 2, :] = v_buf[:, 3 : t_len - 1, :]
        v_buf[:, t_len - 2, :] = v_cur
    return out


def temporal_attn_roll_ring(
    q: np.ndarray,
    k_cur: np.ndarray,
    v_cur: np.ndarray,
    k_buf: np.ndarray,
    v_buf: np.ndarray,
    k_pe: np.ndarray,
    v_pe: np.ndarray,
    n_heads: int,
    scale: float,
    has_history: bool,
    cursor: int = 2,
) -> np.ndarray:
    """Zero-copy ring-buffer rolling-window temporal multi-head self-attention for (d, ch) inputs."""
    d, ch = k_cur.shape
    t_len = k_buf.shape[1]
    dh = ch // n_heads

    if (
        _LIB is not None
        and q.flags["C_CONTIGUOUS"]
        and k_cur.flags["C_CONTIGUOUS"]
        and v_cur.flags["C_CONTIGUOUS"]
        and k_buf.flags["C_CONTIGUOUS"]
        and v_buf.flags["C_CONTIGUOUS"]
        and k_pe.flags["C_CONTIGUOUS"]
        and v_pe.flags["C_CONTIGUOUS"]
    ):
        out = np.empty((d, ch), dtype=np.float32)
        _LIB.temporal_attn_roll_ring_f32(
            _ptr_f32(q),
            _ptr_f32(k_cur),
            _ptr_f32(v_cur),
            _ptr_f32(k_buf),
            _ptr_f32(v_buf),
            _ptr_f32(k_pe),
            _ptr_f32(v_pe),
            _ptr_f32(out),
            d,
            t_len,
            n_heads,
            dh,
            float(scale),
            1 if has_history else 0,
            int(cursor),
        )
        return out

    if not has_history:
        k_buf[:, : t_len - 1, :] = k_cur[:, None, :]
        v_buf[:, : t_len - 1, :] = v_cur[:, None, :]
        return (v_cur + v_pe[0]).astype(np.float32, copy=False)

    k = np.empty((d, t_len, ch), dtype=np.float32)
    v = np.empty((d, t_len, ch), dtype=np.float32)
    if t_len >= 4:
        ring = t_len - 3
        base_off = (cursor - 2) % ring
        idx = [0, 1] + [2 + (base_off + i) % ring for i in range(ring)]
        k[:, : t_len - 1, :] = k_buf[:, idx, :] + k_pe[None, : t_len - 1, :]
        v[:, : t_len - 1, :] = v_buf[:, idx, :] + v_pe[None, : t_len - 1, :]
    else:
        base_off = 0
        k[:, : t_len - 1, :] = k_buf[:, : t_len - 1, :] + k_pe[None, : t_len - 1, :]
        v[:, : t_len - 1, :] = v_buf[:, : t_len - 1, :] + v_pe[None, : t_len - 1, :]

    k[:, t_len - 1, :] = k_cur + k_pe[t_len - 1]
    v[:, t_len - 1, :] = v_cur + v_pe[t_len - 1]
    qh = q.reshape(d, n_heads, 1, dh)
    kh = k.reshape(d, t_len, n_heads, dh).transpose(0, 2, 3, 1)
    vh = v.reshape(d, t_len, n_heads, dh).transpose(0, 2, 1, 3)
    attn = softmax_rows(np.matmul(qh, kh) * np.float32(scale))
    out = np.ascontiguousarray(np.matmul(attn, vh).reshape(d, ch), dtype=np.float32)
    if t_len >= 4:
        write_slot = 2 + base_off
        k_buf[:, write_slot, :] = k_cur
        v_buf[:, write_slot, :] = v_cur
    return out


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


def gaussian5_hw4(arr: np.ndarray, passes: int = 1) -> np.ndarray:
    """Separable 5-tap [1,4,6,4,1]/16 Gaussian blur on (H, W, 4) float32 with edge clamping."""
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"Expected (H, W, 4) array, got {arr.shape}")
    h, w, _ = arr.shape
    arr_c = np.ascontiguousarray(arr, dtype=np.float32)
    if passes <= 0:
        return arr_c.copy()
    if _LIB is not None:
        out = np.empty((h, w, 4), dtype=np.float32)
        rc = _LIB.gaussian5_hw4_f32(_ptr_f32(arr_c), _ptr_f32(out), h, w, int(passes))
        if rc == 0:
            return out

    cur = arr_c
    scale = np.float32(1.0 / 16.0)
    for _ in range(passes):
        px = np.pad(cur, ((0, 0), (2, 2), (0, 0)), mode="edge")
        cur = (
            px[:, :-4]
            + np.float32(4.0) * px[:, 1:-3]
            + np.float32(6.0) * px[:, 2:-2]
            + np.float32(4.0) * px[:, 3:-1]
            + px[:, 4:]
        ) * scale
        py = np.pad(cur, ((2, 2), (0, 0), (0, 0)), mode="edge")
        cur = (
            py[:-4, :]
            + np.float32(4.0) * py[1:-3, :]
            + np.float32(6.0) * py[2:-2, :]
            + np.float32(4.0) * py[3:-1, :]
            + py[4:, :]
        ) * scale
    return cur


def downsample2x_hw4(arr: np.ndarray) -> np.ndarray:
    """Downsample (H, W, 4) float32 to (ceil(H/2), ceil(W/2), 4) with edge clamping."""
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"Expected (H, W, 4) array, got {arr.shape}")
    h, w, _ = arr.shape
    out_h, out_w = (h + 1) // 2, (w + 1) // 2
    arr_c = np.ascontiguousarray(arr, dtype=np.float32)
    if _LIB is not None:
        out = np.empty((out_h, out_w, 4), dtype=np.float32)
        _LIB.downsample2x_hw4_f32(_ptr_f32(arr_c), _ptr_f32(out), h, w)
        return out

    pad_h = h & 1
    pad_w = w & 1
    if pad_h or pad_w:
        arr_c = np.pad(arr_c, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return np.float32(0.25) * (arr_c[0::2, 0::2] + arr_c[0::2, 1::2] + arr_c[1::2, 0::2] + arr_c[1::2, 1::2])


def box_filter_2d(arr: np.ndarray, radius: int) -> np.ndarray:
    """Separable O(H*W) box filter on 2D float32 array with edge clamping."""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}")
    h, w = arr.shape
    arr_c = np.ascontiguousarray(arr, dtype=np.float32)
    rad = int(radius)
    if rad <= 0:
        return arr_c.copy()
    if _LIB is not None:
        out = np.empty((h, w), dtype=np.float32)
        rc = _LIB.box_filter_2d_f32(_ptr_f32(arr_c), _ptr_f32(out), h, w, rad)
        if rc == 0:
            return out

    win = float(2 * rad + 1)
    px = np.pad(arr_c, ((0, 0), (rad, rad)), mode="edge")
    cs_x = np.empty((h, w + 2 * rad + 1), dtype=np.float32)
    cs_x[:, 0] = 0.0
    np.cumsum(px, axis=1, dtype=np.float32, out=cs_x[:, 1:])
    blur_x = (cs_x[:, 2 * rad + 1 :] - cs_x[:, :w]) / np.float32(win)

    py = np.pad(blur_x, ((rad, rad), (0, 0)), mode="edge")
    cs_y = np.empty((h + 2 * rad + 1, w), dtype=np.float32)
    cs_y[0, :] = 0.0
    np.cumsum(py, axis=0, dtype=np.float32, out=cs_y[1:, :])
    return (cs_y[2 * rad + 1 :, :] - cs_y[:h, :]) / np.float32(win)


_LUMA_FMT_CODES = {
    "RGB3": 0,
    "BGR3": 1,
    "YUYV": 2,
    "UYVY": 3,
}


def extract_luma_and_downsample(
    frame: np.ndarray,
    pixel_format: str,
    out_h: int,
    out_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract normalized [0, 1] luma_full (H, W) and bilinear-downsampled luma_low (out_h, out_w)."""
    fmt = pixel_format.upper()
    if fmt not in _LUMA_FMT_CODES:
        raise ValueError(f"Unsupported pixel format {pixel_format!r}")
    fmt_code = _LUMA_FMT_CODES[fmt]
    if fmt_code >= 2:
        if frame.ndim != 3 or frame.shape[2] != 4:
            raise ValueError(f"Expected packed {fmt} pairs, got {frame.shape}")
        in_h, pair_w, _ = frame.shape
        in_w = pair_w * 2
    else:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected {fmt} frame shaped (H, W, 3), got {frame.shape}")
        in_h, in_w, _ = frame.shape

    frame_c = np.ascontiguousarray(frame, dtype=np.uint8)
    if _LIB is not None:
        luma_full = np.empty((in_h, in_w), dtype=np.float32)
        luma_low = np.empty((out_h, out_w), dtype=np.float32)
        rc = _LIB.extract_luma_and_downsample_f32(
            _ptr_u8(frame_c),
            fmt_code,
            in_h,
            in_w,
            out_h,
            out_w,
            _ptr_f32(luma_full),
            _ptr_f32(luma_low),
        )
        if rc == 0:
            return luma_full, luma_low

    if fmt == "YUYV":
        y = np.empty((in_h, in_w), dtype=np.float32)
        y[:, 0::2] = frame_c[:, :, 0]
        y[:, 1::2] = frame_c[:, :, 2]
        luma_full = (y * np.float32(1.0 / 255.0)).astype(np.float32, copy=False)
    elif fmt == "UYVY":
        y = np.empty((in_h, in_w), dtype=np.float32)
        y[:, 0::2] = frame_c[:, :, 1]
        y[:, 1::2] = frame_c[:, :, 3]
        luma_full = (y * np.float32(1.0 / 255.0)).astype(np.float32, copy=False)
    elif fmt == "BGR3":
        luma_full = (
            frame_c[:, :, 0] * np.float32(0.114 / 255.0)
            + frame_c[:, :, 1] * np.float32(0.587 / 255.0)
            + frame_c[:, :, 2] * np.float32(0.299 / 255.0)
        ).astype(np.float32, copy=False)
    else:
        luma_full = (
            frame_c[:, :, 0] * np.float32(0.299 / 255.0)
            + frame_c[:, :, 1] * np.float32(0.587 / 255.0)
            + frame_c[:, :, 2] * np.float32(0.114 / 255.0)
        ).astype(np.float32, copy=False)

    luma_low = bilinear_resize_nchw(luma_full[None, None, :, :], out_h, out_w, align_corners=False)[0, 0]
    return luma_full, luma_low


def guided_upsample_coc(
    luma_full: np.ndarray,
    luma_low: np.ndarray,
    coc_low: np.ndarray,
    radius: int,
    eps: float,
) -> np.ndarray:
    """Guided filter CoC refinement on (S, S) with fused full-res bilinear upsample and clip."""
    if luma_full.ndim != 2 or luma_low.ndim != 2 or coc_low.ndim != 2:
        raise ValueError("Expected 2D arrays for guided_upsample_coc")
    if luma_low.shape != coc_low.shape:
        raise ValueError(f"luma_low {luma_low.shape} != coc_low {coc_low.shape}")
    full_h, full_w = luma_full.shape
    low_h, low_w = luma_low.shape

    lf_c = np.ascontiguousarray(luma_full, dtype=np.float32)
    ll_c = np.ascontiguousarray(luma_low, dtype=np.float32)
    cl_c = np.ascontiguousarray(coc_low, dtype=np.float32)

    if _LIB is not None:
        coc_full = np.empty((full_h, full_w), dtype=np.float32)
        rc = _LIB.guided_upsample_coc_f32(
            _ptr_f32(lf_c),
            _ptr_f32(ll_c),
            _ptr_f32(cl_c),
            _ptr_f32(coc_full),
            full_h,
            full_w,
            low_h,
            low_w,
            int(radius),
            float(eps),
        )
        if rc == 0:
            return coc_full

    eps_f = np.float32(eps)
    mean_i = box_filter_2d(ll_c, radius)
    mean_p = box_filter_2d(cl_c, radius)
    corr_i = box_filter_2d(ll_c * ll_c, radius)
    corr_ip = box_filter_2d(ll_c * cl_c, radius)
    var_i = np.maximum(corr_i - mean_i * mean_i, np.float32(0.0))
    cov_ip = corr_ip - mean_i * mean_p
    a_low = cov_ip / (var_i + eps_f)
    b_low = mean_p - a_low * mean_i
    mean_a = bilinear_resize_nchw(box_filter_2d(a_low, radius)[None, None, :, :], full_h, full_w, align_corners=False)[0, 0]
    mean_b = bilinear_resize_nchw(box_filter_2d(b_low, radius)[None, None, :, :], full_h, full_w, align_corners=False)[0, 0]
    return np.clip(mean_a * lf_c + mean_b, np.float32(0.0), np.float32(1.0)).astype(np.float32, copy=False)


def preprocess_rgb_u8_to_nchw(
    rgb_u8: np.ndarray,
    out_h: int,
    out_w: int | None = None,
) -> np.ndarray:
    """Fused uint8 (H, W, 3) -> bilinear resize (align_corners=False) + ImageNet normalize to (1, 3, out_h, out_w)."""
    if out_w is None:
        out_w = out_h
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB array, got {rgb_u8.shape}")
    in_h, in_w, _ = rgb_u8.shape
    rgb_c = np.ascontiguousarray(rgb_u8, dtype=np.uint8)
    out = np.empty((1, 3, out_h, out_w), dtype=np.float32)

    if _LIB is not None:
        rc = _LIB.preprocess_rgb_u8_to_nchw_f32(
            _ptr_u8(rgb_c),
            _ptr_f32(out),
            in_h,
            in_w,
            out_h,
            out_w,
        )
        if rc == 0:
            return out

    x = (rgb_c.astype(np.float32) * np.float32(1.0 / 255.0)).transpose(2, 0, 1)[None, :, :, :]
    if in_h != out_h or in_w != out_w:
        x = bilinear_resize_nchw(x, out_h, out_w, align_corners=False)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    return ((x - mean) / std).astype(np.float32, copy=False)


def fused_mha_qkv(
    qkv: np.ndarray,
    n_heads: int,
    head_dim: int,
    scale: float,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Fused multi-head self-attention directly from packed (n_tok, 3 * D) QKV float32 tensor."""
    if qkv.ndim != 2:
        raise ValueError(f"Expected 2D (n_tok, 3*D) array, got {qkv.shape}")
    n_tok, three_d = qkv.shape
    d_total = n_heads * head_dim
    if three_d != 3 * d_total:
        raise ValueError(f"Expected 3*D={3 * d_total} columns, got {three_d}")

    qkv_c = np.ascontiguousarray(qkv, dtype=np.float32)
    if out is None:
        out = np.empty((n_tok, d_total), dtype=np.float32)
    else:
        if out.shape != (n_tok, d_total) or out.dtype != np.float32 or not out.flags["C_CONTIGUOUS"]:
            raise ValueError(f"out must be C-contiguous float32 with shape {(n_tok, d_total)}")

    if _LIB is not None:
        rc = _LIB.fused_mha_qkv_f32(
            _ptr_f32(qkv_c),
            _ptr_f32(out),
            n_tok,
            n_heads,
            head_dim,
            float(scale),
        )
        if rc == 0:
            return out

    qkv_h = qkv_c.reshape(n_tok, 3, n_heads, head_dim)
    q = qkv_h[:, 0].transpose(1, 0, 2)
    k = qkv_h[:, 1].transpose(1, 2, 0)
    v = qkv_h[:, 2].transpose(1, 0, 2)
    attn = softmax_rows(np.matmul(q, k) * np.float32(scale))
    res = np.matmul(attn, v).transpose(1, 0, 2).reshape(n_tok, d_total)
    np.copyto(out, res)
    return out


def pack_bokeh_npu_inputs_soa32(
    l0: np.ndarray,
    coc_u8: np.ndarray,
    l1: np.ndarray,
    l2: np.ndarray,
    l3: np.ndarray,
    dst_a: np.ndarray,
    dst_b: np.ndarray,
    padded_pixels: int,
) -> None:
    """Pack uint8 bokeh pyramid inputs into SoA-32 blocks for vectorized AIE2 NPU execution."""
    num_pixels = int(coc_u8.size)
    if _LIB is not None:
        l0_c = np.ascontiguousarray(l0, dtype=np.uint8)
        coc_c = np.ascontiguousarray(coc_u8, dtype=np.uint8)
        l1_c = np.ascontiguousarray(l1, dtype=np.uint8)
        l2_c = np.ascontiguousarray(l2, dtype=np.uint8)
        l3_c = np.ascontiguousarray(l3, dtype=np.uint8)
        rc = _LIB.pack_bokeh_npu_inputs_soa32_u8(
            _ptr_u8(l0_c),
            _ptr_u8(coc_c),
            _ptr_u8(l1_c),
            _ptr_u8(l2_c),
            _ptr_u8(l3_c),
            _ptr_u8(dst_a),
            _ptr_u8(dst_b),
            num_pixels,
            int(padded_pixels),
        )
        if rc == 0:
            return

    chunks = padded_pixels // 32
    va = dst_a.reshape(chunks, 4, 32)
    vb = dst_b.reshape(chunks, 9, 32)
    full = num_pixels // 32
    n_full = full * 32
    if full > 0:
        f0 = l0.reshape(num_pixels, 3)[:n_full].reshape(full, 32, 3).transpose(0, 2, 1)
        f1 = l1.reshape(num_pixels, 3)[:n_full].reshape(full, 32, 3).transpose(0, 2, 1)
        f2 = l2.reshape(num_pixels, 3)[:n_full].reshape(full, 32, 3).transpose(0, 2, 1)
        f3 = l3.reshape(num_pixels, 3)[:n_full].reshape(full, 32, 3).transpose(0, 2, 1)
        va[:full, :3, :] = f0
        va[:full, 3, :] = coc_u8.reshape(num_pixels)[:n_full].reshape(full, 32)
        vb[:full, 0:3, :] = f1
        vb[:full, 3:6, :] = f2
        vb[:full, 6:9, :] = f3
    if padded_pixels > n_full:
        va[full:] = 0
        vb[full:] = 0
        rem = num_pixels - n_full
        if rem > 0:
            va[full, :3, :rem] = l0.reshape(num_pixels, 3)[n_full:].T
            va[full, 3, :rem] = coc_u8.reshape(num_pixels)[n_full:]
            vb[full, 0:3, :rem] = l1.reshape(num_pixels, 3)[n_full:].T
            vb[full, 3:6, :rem] = l2.reshape(num_pixels, 3)[n_full:].T
            vb[full, 6:9, :rem] = l3.reshape(num_pixels, 3)[n_full:].T


def unpremul_resize_hw4_to_rgb_u8(src_hw4: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Unpremultiply (H, W, 4) [R*w, G*w, B*w, w] and bilinearly resize to uint8 (out_h, out_w, 3)."""
    in_h, in_w, c = src_hw4.shape
    if c != 4:
        raise ValueError(f"Expected (H, W, 4) array, got {src_hw4.shape}")
    src_c = np.ascontiguousarray(src_hw4, dtype=np.float32)
    out = np.empty((out_h, out_w, 3), dtype=np.uint8)
    if _LIB is not None:
        rc = _LIB.unpremul_resize_hw4_to_rgb_u8_f32(_ptr_f32(src_c), _ptr_u8(out), in_h, in_w, out_h, out_w)
        if rc == 0:
            return out

    rgb_f32 = src_c[:, :, :3] / np.maximum(src_c[:, :, 3:4], np.float32(1e-5))
    arr_nchw = np.ascontiguousarray(rgb_f32.transpose(2, 0, 1))[None, :, :, :]
    resized = bilinear_resize_nchw(arr_nchw, out_h, out_w, align_corners=False)[0].transpose(1, 2, 0)
    return np.clip(np.round(resized), 0.0, 255.0).astype(np.uint8)
