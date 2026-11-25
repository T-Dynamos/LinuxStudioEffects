"""Reusable NPU1 (AIE2) 16-core DispatchTime bfloat16->float32 MatMul executor on MLIR-AIE / IRON."""

from dataclasses import dataclass
import threading
import time
from typing import Any

import numpy as np
from ml_dtypes import bfloat16

from . import host_ops

# Global re-entrant lock serializing NPU load/dispatch across threads. The selected
# MLIR-AIE host runtime owns the device context shared by depth and portrait blur.
NPU_DISPATCH_LOCK = threading.RLock()

_NROWS = 4
_NCOLS = 4
_NCORES = _NROWS * _NCOLS
_TM = 32
_TK = 64
_TN = 32
_M_ALIGN = _NROWS * _TM  # 128
_N_ALIGN = _NCOLS * _TN  # 128

# Shared host-visible NPU scratchpad capacities for the single DispatchTime xclbin.
_MAX_A_ELEMS = 16 * 1024 * 1024  # 32 MiB bf16
_MAX_B_ELEMS = 9 * 1024 * 1024   # 18 MiB bf16
_MAX_C_ELEMS = 8 * 1024 * 1024   # 32 MiB f32
_MAX_EXACT_IO_CACHE = 32


def _align_up(val: int, mult: int) -> int:
    return ((val + mult - 1) // mult) * mult


@dataclass
class NpuExecStats:
    """Cumulative NPU kernel dispatch and host-sync timings for one forward pass."""

    dispatches: int = 0
    npu_wall_s: float = 0.0
    host_copy_s: float = 0.0


def _build_whole_array_design(max_a: int, max_b: int, max_c: int) -> Any:
    """Compile a single 16-core DispatchTime[M, K, N] bf16->f32 MatMul design."""
    import aie.iron as iron
    import aie.iron.kernels as kernels
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import (
        Buffer,
        DispatchTime,
        In,
        ObjectFifo,
        Out,
        Program,
        Runtime,
        TaskGroup,
        Worker,
        WorkerRuntimeBarrier,
        require,
    )
    from aie.iron.controlflow import range_

    iron.ensure_current_device()
    dev = iron.get_current_device()
    if dev is None or dev.cols < _NCOLS:
        raise RuntimeError(f"Expected an AMD NPU with at least {_NCOLS} columns, got {dev!r}")

    @iron.jit
    def _whole_array_bf16_f32(
        A: In,
        B: In,
        C: Out,
        *,
        M: DispatchTime[np.int32],
        K: DispatchTime[np.int32],
        N: DispatchTime[np.int32],
    ):
        fifo_depth = 2
        matmul_kernel = kernels.mm(
            dim_m=_TM,
            dim_k=_TK,
            dim_n=_TN,
            input_dtype=bfloat16,
            output_dtype=np.float32,
            round_conv_even=True,
        )
        zero_kernel = kernels.zero(_TM * _TN, np.float32)
        dims = matmul_kernel.stream_dims

        A_l2_ty = np.ndarray[(_TM * _TK,), np.dtype[bfloat16]]
        B_l2_ty = np.ndarray[(_TK * _TN,), np.dtype[bfloat16]]
        C_l2_ty = np.ndarray[(_TM * _TN * _NROWS,), np.dtype[np.float32]]
        A_l1_ty = np.ndarray[(_TM, _TK), np.dtype[bfloat16]]
        B_l1_ty = np.ndarray[(_TK, _TN), np.dtype[bfloat16]]
        C_l1_ty = np.ndarray[(_TM, _TN), np.dtype[np.float32]]
        rtp_ty = np.ndarray[(2,), np.dtype[np.int32]]

        A_l3l2_fifos: list[ObjectFifo] = []
        A_l2l1_fifos: list[ObjectFifo] = []
        B_l3l2_fifos: list[ObjectFifo] = []
        B_l2l1_fifos: list[ObjectFifo] = []
        C_l1l2_fifos: list[list[ObjectFifo]] = [[] for _ in range(_NROWS)]
        C_l2l3_fifos: list[ObjectFifo] = []

        for row in range(_NROWS):
            a_l3l2 = ObjectFifo(A_l2_ty, name=f"A_L3L2_{row}", depth=fifo_depth)
            A_l3l2_fifos.append(a_l3l2)
            A_l2l1_fifos.append(
                a_l3l2.cons().forward(obj_type=A_l1_ty, name=f"A_L2L1_{row}", to_stream=dims.A)
            )

        for col in range(_NCOLS):
            b_l3l2 = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
            B_l3l2_fifos.append(b_l3l2)
            B_l2l1_fifos.append(
                b_l3l2.cons().forward(obj_type=B_l1_ty, name=f"B_L2L1_{col}", to_stream=dims.B)
            )
            c_l2l3 = ObjectFifo(C_l2_ty, name=f"C_L2L3_{col}", depth=fifo_depth, to_stream=dims.C)
            C_l2l3_fifos.append(c_l2l3)
            c_tmp_fifos = c_l2l3.prod().join(
                [_TM * _TN * i for i in range(_NROWS)],
                obj_types=[C_l1_ty] * _NROWS,
                names=[f"C_L1L2_{col}_{i}" for i in range(_NROWS)],
                depths=[fifo_depth] * _NROWS,
            )
            for row in range(_NROWS):
                C_l1l2_fifos[row].append(c_tmp_fifos[row])

        rtps = [
            [Buffer(rtp_ty, name=f"rtp_{row}_{col}", use_write_rtp=True) for col in range(_NCOLS)]
            for row in range(_NROWS)
        ]
        barriers = [[WorkerRuntimeBarrier() for _ in range(_NCOLS)] for _ in range(_NROWS)]

        def core_fn(in_a, in_b, out_c, zero, matmul, rtp, barrier):
            barrier.wait_for_value(1)
            k_iters = rtp[0]
            n_tiles = rtp[1]
            barrier.release_with_value(1)
            for _ in range_(n_tiles):
                elem_out = out_c.acquire(1)
                zero(elem_out)
                for _ in range_(k_iters):
                    elem_in_a = in_a.acquire(1)
                    elem_in_b = in_b.acquire(1)
                    matmul(elem_in_a, elem_in_b, elem_out)
                    in_a.release(1)
                    in_b.release(1)
                out_c.release(1)

        workers = Worker.grid(
            _NROWS,
            _NCOLS,
            lambda row, col: Worker(
                core_fn,
                [
                    A_l2l1_fifos[row].cons(),
                    B_l2l1_fifos[col].cons(),
                    C_l1l2_fifos[row][col].prod(),
                    zero_kernel,
                    matmul_kernel,
                    rtps[row][col],
                    barriers[row][col],
                ],
                stack_size=0xD00,
            ),
        )
        flat_workers = [w for row in workers for w in row]

        A_prods = [f.prod() for f in A_l3l2_fifos]
        B_prods = [f.prod() for f in B_l3l2_fifos]
        C_conses = [f.cons() for f in C_l2l3_fifos]

        def sequence(A_h, B_h, C_h, M_d, K_d, N_d, A_hs, B_hs, C_hs):
            require(M_d > 0, "M must be positive")
            require(K_d > 0, "K must be positive")
            require(N_d > 0, "N must be positive")
            require(M_d % _M_ALIGN == 0, "M must be a multiple of 128")
            require(K_d % _TK == 0, "K must be a multiple of 64")
            require(N_d % _N_ALIGN == 0, "N must be a multiple of 128")

            for row in range(_NROWS):
                for col in range(_NCOLS):
                    rtps[row][col][0] = K_d // _TK
                    rtps[row][col][1] = (M_d // _TM) * (N_d // _TN) // _NCORES
                    barriers[row][col].set(1)

            A_tiles = (
                TensorAccessPattern.full((M_d, K_d))
                .tile((_TM, _TK))
                .repeat(N_d // _TN // _NCOLS)
                .permute((1, 0, 2, 3, 4))
            )
            B_tiles = TensorAccessPattern.full((K_d, N_d)).tile((_TK, _TN)).permute((1, 0, 2, 3))
            C_tiles = TensorAccessPattern.full((M_d, N_d)).tile((_TM * _NROWS, _TN))

            for row, tg in TaskGroup.pipelined(M_d // _TM // _NROWS, depth=4):
                for col in range(_NCOLS):
                    C_hs[col].drain(C_h, tap=C_tiles[row, col::_NCOLS], wait=True, group=tg)
                    A_hs[col].fill(A_h, tap=A_tiles[row * _NROWS + col], group=tg)
                    B_hs[col].fill(B_h, tap=B_tiles[col::_NCOLS], group=tg)

        rt = Runtime(sequence, [A, B, C, M, K, N, A_prods, B_prods, C_conses])
        return Program(dev, rt, workers=flat_workers).resolve_program()

    return _whole_array_bf16_f32.specialize(
        A=np.ndarray[(max_a,), np.dtype[bfloat16]],
        B=np.ndarray[(max_b,), np.dtype[bfloat16]],
        C=np.ndarray[(max_c,), np.dtype[np.float32]],
    )


class NpuMatMulExecutor:
    """Manages a single 16-core DispatchTime IRON MatMul xclbin with cached per-shape instructions."""

    def __init__(self) -> None:
        self._design: Any = None
        self._kernel: Any = None
        self._handle: Any = None
        self._runtime: Any = None
        self._t_a: Any = None
        self._t_b_scratch: Any = None
        self._t_c: Any = None
        self._insts_cache: dict[tuple[int, int, int], np.ndarray] = {}
        self._a_views: dict[tuple[int, int], Any] = {}
        self._c_views: dict[tuple[int, int], Any] = {}
        self._a_exact_bufs: dict[tuple[int, int], Any] = {}
        self._c_exact_bufs: dict[tuple[int, int], Any] = {}
        self._b_views: dict[tuple[int, int], Any] = {}
        self._weight_cache: dict[tuple[int, int, int], tuple[np.ndarray, Any]] = {}
        self.stats = NpuExecStats()

    def reset_stats(self) -> None:
        self.stats = NpuExecStats()

    def _ensure_initialized(self) -> None:
        if self._kernel is not None:
            return
        import aie.iron as iron
        from aie.utils import DefaultNPURuntime

        self._runtime = DefaultNPURuntime
        self._t_a = iron.zeros((_MAX_A_ELEMS,), dtype=bfloat16, device="npu")
        self._t_b_scratch = iron.zeros((_MAX_B_ELEMS,), dtype=bfloat16, device="npu")
        self._t_c = iron.zeros((_MAX_C_ELEMS,), dtype=np.float32, device="npu")
        self._design = _build_whole_array_design(_MAX_A_ELEMS, _MAX_B_ELEMS, _MAX_C_ELEMS)
        # Warmup once on minimal aligned dims to populate _kernel_cache and verify XRT load.
        self._design(self._t_a, self._t_b_scratch, self._t_c, M=_M_ALIGN, K=_TK, N=_N_ALIGN)
        self._kernel = next(iter(self._design._kernel_cache.values()))
        self._handle = self._runtime.load(self._kernel)

    def _get_a_view(self, m_pad: int, k_pad: int) -> Any:
        key = (m_pad, k_pad)
        v = self._a_exact_bufs.get(key)
        if v is not None:
            return v
        if len(self._a_exact_bufs) < _MAX_EXACT_IO_CACHE:
            import aie.iron as iron

            v = iron.zeros((m_pad, k_pad), dtype=bfloat16, device="npu")
            self._a_exact_bufs[key] = v
            return v
        v = self._a_views.get(key)
        if v is None:
            v = self._t_a.subview(0, (m_pad, k_pad))
            self._a_views[key] = v
        return v

    def _get_c_view(self, m_pad: int, n_pad: int) -> Any:
        key = (m_pad, n_pad)
        v = self._c_exact_bufs.get(key)
        if v is not None:
            return v
        if len(self._c_exact_bufs) < _MAX_EXACT_IO_CACHE:
            import aie.iron as iron

            v = iron.zeros((m_pad, n_pad), dtype=np.float32, device="npu")
            self._c_exact_bufs[key] = v
            return v
        v = self._c_views.get(key)
        if v is None:
            v = self._t_c.subview(0, (m_pad, n_pad))
            self._c_views[key] = v
        return v

    def _get_b_view(self, k_pad: int, n_pad: int) -> Any:
        key = (k_pad, n_pad)
        v = self._b_views.get(key)
        if v is None:
            v = self._t_b_scratch.subview(0, (k_pad, n_pad))
            self._b_views[key] = v
        return v

    def _get_insts(self, m_pad: int, k_pad: int, n_pad: int) -> np.ndarray:
        key = (m_pad, k_pad, n_pad)
        insts = self._insts_cache.get(key)
        if insts is None:
            insts = self._kernel.instructions(M=m_pad, K=k_pad, N=n_pad)
            self._insts_cache[key] = insts
        return insts

    def _run_kernel(self, args: list[Any], insts: np.ndarray) -> None:
        from aie.utils.hostruntime.hostruntime import HostRuntimeError

        if self._handle is None or not getattr(self._handle, "_is_valid", False):
            self._handle = self._runtime.load(self._kernel)
        try:
            self._runtime.run(self._handle, args, dispatch_insts=insts, only_if_loaded=True)
        except HostRuntimeError:
            self._handle = self._runtime.load(self._kernel)
            self._runtime.run(self._handle, args, dispatch_insts=insts, only_if_loaded=True)

    def _get_weight_tensor(self, b_mat: np.ndarray, k_pad: int, n_pad: int) -> Any:
        import aie.iron as iron

        key = (id(b_mat), k_pad, n_pad)
        cached = self._weight_cache.get(key)
        if cached is not None:
            cached_ref, t_b = cached
            if cached_ref is b_mat:
                return t_b

        k, n = b_mat.shape
        b_pad = np.zeros((k_pad, n_pad), dtype=bfloat16)
        b_pad[:k, :n] = b_mat.astype(bfloat16)
        t_b = iron.tensor(b_pad.reshape(-1), device="npu")
        # Keep a reference to b_mat so its id() cannot be recycled onto a new temporary array.
        self._weight_cache[key] = (b_mat, t_b)
        return t_b

    def matmul(
        self,
        a: np.ndarray,
        b: np.ndarray,
        *,
        bias: np.ndarray | None = None,
        activation: str | None = None,
        cache_b: bool = True,
        transpose_a: bool = False,
        out_nchw_hw: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Compute (M, K) @ (K, N) [+ bias] [with optional activation] on 16-core NPU1, returning float32.

        When transpose_a=True, `a` has shape (K, M). When out_nchw_hw=(H, W) is given,
        the result is returned directly as contiguous NCHW (1, N, H, W).
        """
        if transpose_a:
            k, m = a.shape
        else:
            m, k = a.shape
        bk, n = b.shape
        if k != bk:
            raise ValueError(f"Inner dimensions mismatch: a={a.shape} (transpose_a={transpose_a}), b={b.shape}")

        if activation not in (None, "gelu", "relu"):
            raise ValueError(f"Unsupported activation {activation!r}; expected None, 'gelu', or 'relu'")
        apply_gelu = activation == "gelu"
        apply_relu = activation == "relu"

        if out_nchw_hw is not None:
            oh, ow = out_nchw_hw
            if oh * ow != m:
                raise ValueError(f"out_nchw_hw {out_nchw_hw} does not match M={m}")
            if apply_gelu:
                raise ValueError("GELU activation is not supported with out_nchw_hw")

        bias_1d: np.ndarray | None = None
        if bias is not None:
            if bias.ndim == 1:
                bias_1d = bias
            elif bias.ndim == 2 and bias.shape[0] == 1:
                bias_1d = bias[0]
            else:
                raise ValueError(f"Bias shape mismatch: expected ({n},) or (1, {n}), got {bias.shape}")
            if bias_1d.shape[0] != n:
                raise ValueError(f"Bias length mismatch: expected {n}, got {bias_1d.shape[0]}")

        k_pad = _align_up(k, _TK)
        n_pad = _align_up(n, _N_ALIGN)

        max_m_fit = min(_MAX_A_ELEMS // k_pad, _MAX_C_ELEMS // n_pad)
        max_chunk_m = (max_m_fit // _M_ALIGN) * _M_ALIGN
        if k_pad * n_pad > _MAX_B_ELEMS or max_chunk_m < _M_ALIGN:
            a_mat = a.T if transpose_a else a
            y = a_mat.astype(np.float32, copy=False) @ b.astype(np.float32, copy=False)
            if bias_1d is not None:
                y = y + bias_1d
            if apply_gelu:
                return host_ops.gelu_inplace(np.array(y, dtype=np.float32, copy=True))
            if apply_relu:
                y = np.maximum(y, np.float32(0.0))
            if out_nchw_hw is not None:
                return np.ascontiguousarray(y.T.reshape(1, n, out_nchw_hw[0], out_nchw_hw[1]), dtype=np.float32)
            return np.ascontiguousarray(y, dtype=np.float32)

        single_chunk_nchw = out_nchw_hw is not None and m <= max_chunk_m
        if single_chunk_nchw:
            out = np.empty((1, n, out_nchw_hw[0], out_nchw_hw[1]), dtype=np.float32)
        else:
            out = np.empty((m, n), dtype=np.float32)

        t0_prep = time.perf_counter()
        if cache_b:
            t_b = self._get_weight_tensor(b, k_pad, n_pad)
        else:
            with NPU_DISPATCH_LOCK:
                self._ensure_initialized()
                t_b = self._get_b_view(k_pad, n_pad)
                with t_b.overwrite() as dst_b_2d:
                    if b.dtype == np.float32 and host_ops.has_avx512_host_ops():
                        host_ops.f32_to_bf16_pad(b, dst_b_2d, k, n, zero_tail=(k_pad > k))
                    else:
                        dst_b_2d.fill(0)
                        dst_b_2d[:k, :n] = b.astype(bfloat16)
        self.stats.host_copy_s += time.perf_counter() - t0_prep

        use_fast_bf16 = a.dtype == np.float32 and host_ops.has_avx512_host_ops()
        row = 0
        while row < m:
            rem = m - row
            chunk_m = min(rem, max_chunk_m)
            m_pad = _align_up(chunk_m, _M_ALIGN)

            with NPU_DISPATCH_LOCK:
                self._ensure_initialized()
                insts = self._get_insts(m_pad, k_pad, n_pad)
                a_view = self._get_a_view(m_pad, k_pad)
                c_view_t = self._get_c_view(m_pad, n_pad)

                t0 = time.perf_counter()
                with a_view.overwrite() as dst_a_2d:
                    if transpose_a:
                        a_chunk = a if (chunk_m == m and a.dtype == np.float32 and a.flags["C_CONTIGUOUS"]) else np.ascontiguousarray(a[:, row : row + chunk_m], dtype=np.float32)
                        host_ops.f32_transpose_to_bf16_pad(
                            a_chunk,
                            dst_a_2d,
                            k,
                            chunk_m,
                            zero_tail=(m_pad > chunk_m),
                        )
                    elif use_fast_bf16:
                        host_ops.f32_to_bf16_pad(
                            a[row : row + chunk_m],
                            dst_a_2d,
                            chunk_m,
                            k,
                            zero_tail=(m_pad > chunk_m),
                        )
                    else:
                        if m_pad > chunk_m or k_pad > k:
                            dst_a_2d.fill(0)
                        dst_a_2d[:chunk_m, :k] = a[row : row + chunk_m].astype(bfloat16)
                self.stats.host_copy_s += time.perf_counter() - t0

                t1 = time.perf_counter()
                self._run_kernel([a_view, t_b, c_view_t], insts)
                self.stats.npu_wall_s += time.perf_counter() - t1
                self.stats.dispatches += 1

                t2 = time.perf_counter()
                c_view = c_view_t.numpy()
                if single_chunk_nchw:
                    host_ops.f32_unpad_transpose_bias_relu(
                        c_view,
                        out.reshape(n, m),
                        chunk_m,
                        n,
                        bias=bias_1d,
                        apply_relu=apply_relu,
                    )
                else:
                    host_ops.f32_unpad_bias_gelu(
                        c_view,
                        out[row : row + chunk_m],
                        chunk_m,
                        n,
                        bias=bias_1d,
                        apply_gelu=apply_gelu,
                    )
                    if apply_relu:
                        np.maximum(out[row : row + chunk_m], np.float32(0.0), out=out[row : row + chunk_m])
                self.stats.host_copy_s += time.perf_counter() - t2

            row += chunk_m
        if out_nchw_hw is not None and not single_chunk_nchw:
            return np.ascontiguousarray(out.T.reshape(1, n, out_nchw_hw[0], out_nchw_hw[1]), dtype=np.float32)
        return out

    def conv2d_3x3_s1_p1_nchw(
        self,
        x_nchw: np.ndarray,
        w_kykxc_cout: np.ndarray,
        w_ck_cout: np.ndarray,
        *,
        bias: np.ndarray | None = None,
        apply_relu: bool = False,
    ) -> np.ndarray:
        """Compute 3x3 stride=1 pad=1 Conv2d on NPU1 using (ky, kx, C_in) packed patches directly."""
        _, c_in, h, w = x_nchw.shape
        m = h * w
        k = 9 * c_in
        bk, n = w_kykxc_cout.shape
        if bk != k:
            raise ValueError(f"Weight inner dim {bk} != 9*c_in={k}")

        k_pad = _align_up(k, _TK)
        n_pad = _align_up(n, _N_ALIGN)
        max_m_fit = min(_MAX_A_ELEMS // k_pad, _MAX_C_ELEMS // n_pad)
        max_chunk_m = (max_m_fit // _M_ALIGN) * _M_ALIGN
        if k_pad * n_pad > _MAX_B_ELEMS or m > max_chunk_m or max_chunk_m < _M_ALIGN:
            cols = host_ops.im2col_3x3_s1_p1(x_nchw)
            return self.matmul(
                cols,
                w_ck_cout,
                bias=bias,
                activation="relu" if apply_relu else None,
                cache_b=True,
                transpose_a=True,
                out_nchw_hw=(h, w),
            )

        bias_1d = None if bias is None else np.ascontiguousarray(bias.reshape(-1), dtype=np.float32)
        m_pad = _align_up(m, _M_ALIGN)
        out = np.empty((1, n, h, w), dtype=np.float32)

        t0_prep = time.perf_counter()
        t_b = self._get_weight_tensor(w_kykxc_cout, k_pad, n_pad)
        self.stats.host_copy_s += time.perf_counter() - t0_prep

        with NPU_DISPATCH_LOCK:
            self._ensure_initialized()
            insts = self._get_insts(m_pad, k_pad, n_pad)
            a_view = self._get_a_view(m_pad, k_pad)
            c_view_t = self._get_c_view(m_pad, n_pad)

            t0 = time.perf_counter()
            with a_view.overwrite() as dst_a_2d:
                host_ops.im2col_3x3_s1_p1_kykxc_bf16_pad(x_nchw, dst_a_2d, zero_tail=(m_pad > m))
            self.stats.host_copy_s += time.perf_counter() - t0

            t1 = time.perf_counter()
            self._run_kernel([a_view, t_b, c_view_t], insts)
            self.stats.npu_wall_s += time.perf_counter() - t1
            self.stats.dispatches += 1

            t2 = time.perf_counter()
            c_view = c_view_t.numpy()
            host_ops.f32_unpad_transpose_bias_relu(
                c_view,
                out.reshape(n, m),
                m,
                n,
                bias=bias_1d,
                apply_relu=apply_relu,
            )
            self.stats.host_copy_s += time.perf_counter() - t2

        return out

    def close(self) -> None:
        self._weight_cache.clear()
        self._insts_cache.clear()
        self._a_exact_bufs.clear()
        self._c_exact_bufs.clear()
        self._a_views.clear()
        self._c_views.clear()
        self._b_views.clear()
        self._t_a = None
        self._t_b_scratch = None
        self._t_c = None
        self._handle = None
        self._kernel = None
        self._design = None
        self._runtime = None
