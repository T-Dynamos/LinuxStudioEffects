"""Reusable NPU1 (AIE2) 16-core DispatchTime bfloat16 MatMul executor on MLIR-AIE / IRON."""

from contextlib import contextmanager
from dataclasses import dataclass
import os
import threading
import time
from typing import Any, Iterator

import numpy as np
from ml_dtypes import bfloat16

from . import host_ops

# Global re-entrant lock serializing NPU load/dispatch across threads. The selected
# MLIR-AIE host runtime owns the device context shared by depth and portrait blur.
NPU_DISPATCH_LOCK = threading.RLock()

_WORKER_LOCAL = threading.local()
_RENDER_COND = threading.Condition()
_RENDER_WAITERS = 0


def set_npu_worker_thread(is_worker: bool = True) -> None:
    """Mark/unmark the calling thread as a low-priority background NPU depth worker."""
    _WORKER_LOCAL.is_worker = bool(is_worker)


@contextmanager
def render_priority() -> Iterator[None]:
    """Signal background depth workers to yield NPU access during a live render burst."""
    global _RENDER_WAITERS
    with _RENDER_COND:
        _RENDER_WAITERS += 1
    try:
        yield
    finally:
        with _RENDER_COND:
            _RENDER_WAITERS = max(0, _RENDER_WAITERS - 1)
            if _RENDER_WAITERS == 0:
                _RENDER_COND.notify_all()


def wait_for_npu_render_priority() -> None:
    """Called by background NPU workers before locking so live 30 FPS rendering never stalls."""
    if not getattr(_WORKER_LOCAL, "is_worker", False):
        return
    if _RENDER_WAITERS <= 0:
        return
    with _RENDER_COND:
        while _RENDER_WAITERS > 0:
            _RENDER_COND.wait(timeout=0.02)

_NROWS = 4
_NCOLS = 4
_NCORES = _NROWS * _NCOLS
_TM = 32
_TK = 64
_TN = 32
_M_ALIGN = _NROWS * _TM  # 128
_N_ALIGN = _NCOLS * _TN  # 128
_M_ALIGN_NARROW = _NCORES * _TM  # 512
_N_ALIGN_NARROW = _TN  # 32
_NARROW_N_THRESHOLD = 96

# Shared host-visible NPU scratchpad capacities for DispatchTime xclbins.
_MAX_A_ELEMS = 16 * 1024 * 1024  # 32 MiB bf16
_MAX_B_ELEMS = 9 * 1024 * 1024   # 18 MiB bf16
_MAX_C_ELEMS = 8 * 1024 * 1024   # 32 MiB f32 (or 16 MiB bf16)
_MAX_EXACT_IO_CACHE = 32


def _align_up(val: int, mult: int) -> int:
    return ((val + mult - 1) // mult) * mult


@dataclass
class NpuExecStats:
    """Cumulative NPU kernel dispatch and host-sync timings for one forward pass."""

    dispatches: int = 0
    npu_wall_s: float = 0.0
    host_copy_s: float = 0.0


def _build_matmul_design(
    max_a: int,
    max_b: int,
    max_c: int,
    *,
    narrow_n: bool = False,
    output_bf16: bool = False,
) -> Any:
    """Compile a 16-core DispatchTime[M, K, N] bf16 MatMul design (whole-array or narrow-N)."""
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
    from aie.iron.device.tile import Tile

    iron.ensure_current_device()
    dev = iron.get_current_device()
    n_core_rows = len(dev.core_rows) if isinstance(dev.core_rows, (list, tuple)) else int(dev.core_rows)
    if dev is None or dev.cols < _NCOLS or n_core_rows < _NROWS:
        raise RuntimeError(
            f"Expected an AMD NPU with at least {_NROWS}x{_NCOLS} cores, got {dev!r}"
        )

    c_out_dtype: type = bfloat16 if output_bf16 else np.float32

    @iron.jit
    def _bf16_matmul_design(
        A: In,
        B: In,
        C: Out,
        *,
        M: DispatchTime[np.int32],
        K: DispatchTime[np.int32],
        N: DispatchTime[np.int32],
    ):
        fifo_depth = 2
        # Always accumulate partial products in fp32 inside L1; in bf16 output
        # mode, convert once after all K iterations so precision is preserved.
        matmul_kernel = kernels.mm(
            dim_m=_TM,
            dim_k=_TK,
            dim_n=_TN,
            input_dtype=bfloat16,
            output_dtype=np.float32,
            round_conv_even=True,
        )
        zero_kernel = kernels.zero(_TM * _TN, np.float32)
        cast_kernel = kernels.convert_copy(_TM * _TN) if output_bf16 else None
        dims = matmul_kernel.stream_dims

        A_l1_ty = np.ndarray[(_TM, _TK), np.dtype[bfloat16]]
        B_l1_ty = np.ndarray[(_TK, _TN), np.dtype[bfloat16]]
        C_l1_ty = np.ndarray[(_TM, _TN), np.dtype[c_out_dtype]]
        B_l2_ty = np.ndarray[(_TK * _TN,), np.dtype[bfloat16]]
        C_l2_ty = np.ndarray[(_TM * _TN * _NROWS,), np.dtype[c_out_dtype]]
        rtp_ty = np.ndarray[(2,), np.dtype[np.int32]]

        A_l3l2_fifos: list[ObjectFifo] = []
        B_l3l2_fifos: list[ObjectFifo] = []
        B_l2l1_fifos: list[ObjectFifo] = []
        C_l1l2_fifos: list[list[ObjectFifo]] = [[] for _ in range(_NROWS)]
        C_l2l3_fifos: list[ObjectFifo] = []

        if narrow_n:
            A_l2_ty = np.ndarray[(_TM * _NROWS * _TK,), np.dtype[bfloat16]]
            A_l2l1_cols: list[list[ObjectFifo]] = []
            for col in range(_NCOLS):
                mem_tile = Tile(col, 1)
                a_l3l2 = ObjectFifo(A_l2_ty, name=f"A_L3L2_{col}", depth=fifo_depth)
                A_l3l2_fifos.append(a_l3l2)
                A_l2l1_cols.append(
                    a_l3l2.cons().split(
                        [_TM * _TK * i for i in range(_NROWS)],
                        tile=mem_tile,
                        obj_types=[A_l1_ty] * _NROWS,
                        names=[f"A_L2L1_{col}_{i}" for i in range(_NROWS)],
                        depths=[fifo_depth] * _NROWS,
                        to_stream=[dims.A] * _NROWS,
                    )
                )
        else:
            A_l2_ty = np.ndarray[(_TM * _TK,), np.dtype[bfloat16]]
            A_l2l1_rows: list[ObjectFifo] = []
            for row in range(_NROWS):
                a_l3l2 = ObjectFifo(A_l2_ty, name=f"A_L3L2_{row}", depth=fifo_depth)
                A_l3l2_fifos.append(a_l3l2)
                A_l2l1_rows.append(
                    a_l3l2.cons().forward(obj_type=A_l1_ty, name=f"A_L2L1_{row}", to_stream=dims.A)
                )

        for col in range(_NCOLS):
            mem_tile = Tile(col, 1) if narrow_n else None
            b_l3l2 = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
            B_l3l2_fifos.append(b_l3l2)
            b_fwd_kwargs: dict[str, Any] = {
                "obj_type": B_l1_ty,
                "name": f"B_L2L1_{col}",
                "to_stream": dims.B,
            }
            if mem_tile is not None:
                b_fwd_kwargs["tile"] = mem_tile
            B_l2l1_fifos.append(b_l3l2.cons().forward(**b_fwd_kwargs))

            c_l2l3 = ObjectFifo(C_l2_ty, name=f"C_L2L3_{col}", depth=fifo_depth, to_stream=dims.C)
            C_l2l3_fifos.append(c_l2l3)
            c_join_kwargs: dict[str, Any] = {
                "obj_types": [C_l1_ty] * _NROWS,
                "names": [f"C_L1L2_{col}_{i}" for i in range(_NROWS)],
                "depths": [fifo_depth] * _NROWS,
            }
            if mem_tile is not None:
                c_join_kwargs["tile"] = mem_tile
            c_tmp_fifos = c_l2l3.prod().join(
                [_TM * _TN * i for i in range(_NROWS)],
                **c_join_kwargs,
            )
            for row in range(_NROWS):
                C_l1l2_fifos[row].append(c_tmp_fifos[row])

        rtps = [
            [Buffer(rtp_ty, name=f"rtp_{row}_{col}", use_write_rtp=True) for col in range(_NCOLS)]
            for row in range(_NROWS)
        ]
        barriers = [[WorkerRuntimeBarrier() for _ in range(_NCOLS)] for _ in range(_NROWS)]

        if output_bf16:
            c_acc_ty = np.ndarray[(_TM * _TN,), np.dtype[np.float32]]
            c_accs = [
                [Buffer(c_acc_ty, name=f"c_acc_{row}_{col}") for col in range(_NCOLS)]
                for row in range(_NROWS)
            ]

            def core_fn(in_a, in_b, out_c, c_acc, zero, matmul, cast, rtp, barrier):
                barrier.wait_for_value(1)
                k_iters = rtp[0]
                n_tiles = rtp[1]
                barrier.release_with_value(1)
                for _ in range_(n_tiles):
                    zero(c_acc)
                    for _ in range_(k_iters):
                        elem_in_a = in_a.acquire(1)
                        elem_in_b = in_b.acquire(1)
                        matmul(elem_in_a, elem_in_b, c_acc)
                        in_a.release(1)
                        in_b.release(1)
                    elem_out = out_c.acquire(1)
                    cast(c_acc, elem_out, _TM * _TN)
                    out_c.release(1)
        else:

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

        def _make_worker(row: int, col: int) -> Worker:
            a_cons = A_l2l1_cols[col][row].cons() if narrow_n else A_l2l1_rows[row].cons()
            fn_args: list[Any] = [a_cons, B_l2l1_fifos[col].cons(), C_l1l2_fifos[row][col].prod()]
            if output_bf16:
                fn_args.extend([c_accs[row][col], zero_kernel, matmul_kernel, cast_kernel])
            else:
                fn_args.extend([zero_kernel, matmul_kernel])
            fn_args.extend([rtps[row][col], barriers[row][col]])
            worker_kwargs: dict[str, Any] = {"stack_size": 0xD00}
            if narrow_n:
                worker_kwargs["tile"] = Tile(col, row + 2)
            return Worker(core_fn, fn_args, **worker_kwargs)

        workers = Worker.grid(_NROWS, _NCOLS, _make_worker)
        flat_workers = [w for row in workers for w in row]

        if narrow_n:
            A_prods = [f.prod(tile=Tile(col, 0)) for col, f in enumerate(A_l3l2_fifos)]
            B_prods = [f.prod(tile=Tile(col, 0)) for col, f in enumerate(B_l3l2_fifos)]
            C_conses = [f.cons(tile=Tile(col, 0)) for col, f in enumerate(C_l2l3_fifos)]

            def sequence(A_h, B_h, C_h, M_d, K_d, N_d, A_hs, B_hs, C_hs):
                require(M_d > 0, "M must be positive")
                require(K_d > 0, "K must be positive")
                require(N_d > 0, "N must be positive")
                require(M_d % _M_ALIGN_NARROW == 0, "M must be a multiple of 512")
                require(K_d % _TK == 0, "K must be a multiple of 64")
                require(N_d % _N_ALIGN_NARROW == 0, "N must be a multiple of 32")

                for row in range(_NROWS):
                    for col in range(_NCOLS):
                        rtps[row][col][0] = K_d // _TK
                        rtps[row][col][1] = (M_d // _M_ALIGN_NARROW) * (N_d // _TN)
                        barriers[row][col].set(1)

                A_tiles = (
                    TensorAccessPattern.full((M_d, K_d))
                    .tile((_TM * _NROWS, _TK))
                    .repeat(N_d // _TN)
                    .permute((1, 0, 2, 3, 4))
                )
                B_tiles = TensorAccessPattern.full((K_d, N_d)).tile((_TK, _TN)).permute((1, 0, 2, 3))
                C_tiles = TensorAccessPattern.full((M_d, N_d)).tile((_TM * _NROWS, _TN))

                for m_idx, tg in TaskGroup.pipelined(M_d // _M_ALIGN_NARROW, depth=4):
                    for col in range(_NCOLS):
                        C_hs[col].drain(C_h, tap=C_tiles[m_idx * _NCOLS + col], wait=True, group=tg)
                        A_hs[col].fill(A_h, tap=A_tiles[m_idx * _NCOLS + col], wait=True, group=tg)
                        B_hs[col].fill(B_h, tap=B_tiles, wait=True, group=tg)
        else:
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
                        A_hs[col].fill(A_h, tap=A_tiles[row * _NROWS + col], wait=True, group=tg)
                        B_hs[col].fill(B_h, tap=B_tiles[col::_NCOLS], wait=True, group=tg)

        rt = Runtime(sequence, [A, B, C, M, K, N, A_prods, B_prods, C_conses])
        return Program(dev, rt, workers=flat_workers).resolve_program()

    return _bf16_matmul_design.specialize(
        A=np.ndarray[(max_a,), np.dtype[bfloat16]],
        B=np.ndarray[(max_b,), np.dtype[bfloat16]],
        C=np.ndarray[(max_c,), np.dtype[c_out_dtype]],
    )


def _build_whole_array_design(
    max_a: int,
    max_b: int,
    max_c: int,
    *,
    output_bf16: bool = False,
) -> Any:
    """Compile a single 16-core DispatchTime[M, K, N] bf16->(f32|bf16) MatMul design."""
    return _build_matmul_design(max_a, max_b, max_c, narrow_n=False, output_bf16=output_bf16)


def _build_narrow_n_design(
    max_a: int,
    max_b: int,
    max_c: int,
    *,
    output_bf16: bool = False,
) -> Any:
    """Compile a 16-core M-partitioned DispatchTime[M, K, N] design with N_align=32."""
    return _build_matmul_design(max_a, max_b, max_c, narrow_n=True, output_bf16=output_bf16)


class NpuMatMulExecutor:
    """Manages 16-core DispatchTime IRON MatMul xclbins with cached instruction BOs."""

    def __init__(self, *, use_bf16_out: bool | None = None) -> None:
        if use_bf16_out is None:
            env = os.environ.get("LSE_NPU_BF16_OUT", "").strip().lower()
            use_bf16_out = env in ("1", "true", "yes")
        self._use_bf16_out = bool(use_bf16_out)
        self._c_dtype: type = bfloat16 if self._use_bf16_out else np.float32
        self._out_tag: str = "bf16" if self._use_bf16_out else "f32"

        self._design: Any = None
        self._kernel: Any = None
        self._handle: Any = None
        self._narrow_design: Any = None
        self._narrow_kernel: Any = None
        self._narrow_handle: Any = None
        self._runtime: Any = None
        self._t_a: Any = None
        self._t_b_scratch: Any = None
        self._t_c: Any = None
        self._insts_cache: dict[tuple[tuple[str, str], int, int, int], np.ndarray] = {}
        self._insts_bo_cache: dict[tuple[tuple[str, str], int, int, int], tuple[Any, int, Any]] = {}
        self._a_views: dict[tuple[int, int], Any] = {}
        self._c_views: dict[tuple[int, int], Any] = {}
        self._a_exact_bufs: dict[tuple[int, int], Any] = {}
        self._c_exact_bufs: dict[tuple[int, int], Any] = {}
        self._b_views: dict[tuple[int, int], Any] = {}
        self._weight_cache: dict[tuple[int, int, int], tuple[np.ndarray, Any]] = {}
        self._conv_strip_a_u16: np.ndarray | None = None
        self._conv_strip_out_f32: np.ndarray | None = None
        self._bf16_tmp_f32: np.ndarray | None = None
        self.stats = NpuExecStats()

    def reset_stats(self) -> None:
        self.stats = NpuExecStats()

    def _ensure_buffers(self) -> None:
        if self._t_a is not None:
            return
        import aie.iron as iron
        from aie.utils import DefaultNPURuntime

        self._runtime = DefaultNPURuntime
        self._t_a = iron.zeros((_MAX_A_ELEMS,), dtype=bfloat16, device="npu")
        self._t_b_scratch = iron.zeros((_MAX_B_ELEMS,), dtype=bfloat16, device="npu")
        self._t_c = iron.zeros((_MAX_C_ELEMS,), dtype=self._c_dtype, device="npu")

    def _ensure_initialized(self) -> None:
        self._ensure_buffers()
        if self._kernel is not None:
            return
        self._design = _build_whole_array_design(
            _MAX_A_ELEMS,
            _MAX_B_ELEMS,
            _MAX_C_ELEMS,
            output_bf16=self._use_bf16_out,
        )
        # Warmup once on minimal aligned dims to populate _kernel_cache and verify XRT load.
        self._design(self._t_a, self._t_b_scratch, self._t_c, M=_M_ALIGN, K=_TK, N=_N_ALIGN)
        self._kernel = next(iter(self._design._kernel_cache.values()))
        self._handle = self._runtime.load(self._kernel)

    def _ensure_narrow_initialized(self) -> None:
        self._ensure_buffers()
        if self._narrow_kernel is not None:
            return
        self._narrow_design = _build_narrow_n_design(
            _MAX_A_ELEMS,
            _MAX_B_ELEMS,
            _MAX_C_ELEMS,
            output_bf16=self._use_bf16_out,
        )
        self._narrow_design(
            self._t_a,
            self._t_b_scratch,
            self._t_c,
            M=_M_ALIGN_NARROW,
            K=_TK,
            N=_N_ALIGN_NARROW,
        )
        self._narrow_kernel = next(iter(self._narrow_design._kernel_cache.values()))
        self._narrow_handle = self._runtime.load(self._narrow_kernel)

    def _variant_key(self, is_narrow: bool) -> tuple[str, str]:
        return ("narrow" if is_narrow else "whole", self._out_tag)

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

            v = iron.zeros((m_pad, n_pad), dtype=self._c_dtype, device="npu")
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

    def _get_insts(self, is_narrow: bool, m_pad: int, k_pad: int, n_pad: int) -> np.ndarray:
        variant_key = self._variant_key(is_narrow)
        key = (variant_key, m_pad, k_pad, n_pad)
        insts = self._insts_cache.get(key)
        if insts is None:
            kernel = self._narrow_kernel if is_narrow else self._kernel
            insts = kernel.instructions(M=m_pad, K=k_pad, N=n_pad)
            self._insts_cache[key] = insts
        return insts

    def _get_handle(self, is_narrow: bool) -> Any:
        return self._narrow_handle if is_narrow else self._handle

    def _reload_handle(self, is_narrow: bool) -> Any:
        self._insts_bo_cache.clear()
        if is_narrow:
            self._narrow_handle = self._runtime.load(self._narrow_kernel)
            return self._narrow_handle
        self._handle = self._runtime.load(self._kernel)
        return self._handle

    def _can_direct_dispatch(self, handle: Any) -> bool:
        return (
            getattr(self._runtime, "_device", None) is not None
            and getattr(handle, "kernel_run", None) is not None
            and hasattr(getattr(handle, "kernel", None), "group_id")
        )

    def _dispatch_direct(
        self,
        handle: Any,
        bo_key: tuple[tuple[str, str], int, int, int],
        args: list[Any],
        insts: np.ndarray,
    ) -> None:
        import pyxrt
        from aie.utils.hostruntime.hostruntime import HostRuntimeError
        from aie.utils.hostruntime.tensor_class import TensorRegion
        from aie.utils.hostruntime.xrtruntime.tensor import XrtTransport

        for arg in args:
            owner = arg.tensor if isinstance(arg, TensorRegion) else arg
            owner.to("npu")
        buffers = [
            (
                arg.tensor.storage.binding_handle(arg.storage_offset, arg.nbytes)
                if isinstance(arg, TensorRegion)
                else arg.buffer_object()
            )
            for arg in args
        ]

        if getattr(handle, "num_args", None) is not None:
            declared_bo_count = handle.num_args - 3
            if len(buffers) > declared_bo_count:
                raise HostRuntimeError(
                    f"The xclbin declares {declared_bo_count} host buffer "
                    f"argument(s) but {len(buffers)} were passed."
                )

        if hasattr(self._runtime, "_dispatched_context"):
            ctx = getattr(handle, "context", None)
            if ctx is not None and self._runtime._dispatched_context is not ctx:
                self._runtime._dispatched_context = ctx

        bo_entry = self._insts_bo_cache.get(bo_key)
        if bo_entry is None:
            insts_u8 = np.ascontiguousarray(insts).view(np.uint8).reshape(-1)
            insts_bytes = insts_u8.nbytes
            transport = XrtTransport(
                self._runtime._device,
                insts_bytes,
                pyxrt.bo.cacheable,
                handle.kernel.group_id(1),
            )
            transport.host_bytes[:] = insts_u8
            transport.to_device(0, insts_bytes)
            insts_bo = transport.handle(0, insts_bytes)
            bo_entry = (transport, insts_bytes, insts_bo)
            self._insts_bo_cache[bo_key] = bo_entry
        else:
            _, insts_bytes, insts_bo = bo_entry

        r, _ = handle.kernel_run.dispatch([3, insts_bo, insts_bytes, *buffers])
        if r != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise HostRuntimeError(f"Kernel returned {str(r)}")

    def _run_kernel(
        self,
        is_narrow: bool,
        m_pad: int,
        k_pad: int,
        n_pad: int,
        args: list[Any],
        insts: np.ndarray,
    ) -> None:
        from aie.utils.hostruntime.hostruntime import HostRuntimeError

        handle = self._get_handle(is_narrow)
        if handle is None or not getattr(handle, "_is_valid", True):
            handle = self._reload_handle(is_narrow)

        if self._can_direct_dispatch(handle):
            bo_key = (self._variant_key(is_narrow), m_pad, k_pad, n_pad)
            try:
                self._dispatch_direct(handle, bo_key, args, insts)
            except (HostRuntimeError, RuntimeError):
                handle = self._reload_handle(is_narrow)
                self._dispatch_direct(handle, bo_key, args, insts)
            return

        try:
            self._runtime.run(handle, args, dispatch_insts=insts, only_if_loaded=True)
        except HostRuntimeError:
            handle = self._reload_handle(is_narrow)
            self._runtime.run(handle, args, dispatch_insts=insts, only_if_loaded=True)

    def _get_bf16_tmp_f32(self, rows: int, cols: int) -> np.ndarray:
        needed = rows * cols
        buf = self._bf16_tmp_f32
        if buf is None or buf.size < needed:
            buf = np.empty((needed,), dtype=np.float32)
            self._bf16_tmp_f32 = buf
        return buf[:needed].reshape(rows, cols)

    def _get_conv_strip_a_u16(self, rows: int, cols: int) -> np.ndarray:
        needed = rows * cols
        buf = self._conv_strip_a_u16
        if buf is None or buf.size < needed:
            buf = np.empty((needed,), dtype=np.uint16)
            self._conv_strip_a_u16 = buf
        return buf[:needed].reshape(rows, cols)

    def _get_conv_strip_out_f32(self, c_out: int, m_hw: int) -> np.ndarray:
        needed = c_out * m_hw
        buf = self._conv_strip_out_f32
        if buf is None or buf.size < needed:
            buf = np.empty((needed,), dtype=np.float32)
            self._conv_strip_out_f32 = buf
        return buf[:needed].reshape(c_out, m_hw)

    def _write_c_rowmajor(
        self,
        c_view: np.ndarray,
        dst_f32: np.ndarray,
        rows: int,
        cols: int,
        *,
        bias: np.ndarray | None,
        apply_gelu: bool,
        apply_relu: bool,
    ) -> None:
        if self._use_bf16_out:
            bf16_fn = getattr(host_ops, "bf16_unpad_bias_gelu", None)
            if bf16_fn is not None:
                bf16_fn(c_view, dst_f32, rows, cols, bias=bias, apply_gelu=apply_gelu)
            else:
                host_ops.bf16_to_f32_unpad(c_view, dst_f32, rows, cols)
                if bias is not None:
                    dst_f32[:rows, :cols] += bias[:cols]
                if apply_gelu:
                    host_ops.gelu_inplace(dst_f32[:rows, :cols])
        else:
            host_ops.f32_unpad_bias_gelu(
                c_view,
                dst_f32,
                rows,
                cols,
                bias=bias,
                apply_gelu=apply_gelu,
            )
        if apply_relu:
            np.maximum(dst_f32[:rows, :cols], np.float32(0.0), out=dst_f32[:rows, :cols])

    def _write_c_nchw(
        self,
        c_view: np.ndarray,
        dst_c_hw: np.ndarray,
        m_hw: int,
        c_out: int,
        *,
        bias: np.ndarray | None,
        apply_relu: bool,
    ) -> None:
        if self._use_bf16_out:
            bf16_t_fn = getattr(host_ops, "bf16_unpad_transpose_bias_relu", None)
            if bf16_t_fn is not None:
                bf16_t_fn(c_view, dst_c_hw, m_hw, c_out, bias=bias, apply_relu=apply_relu)
                return
            tmp_f32 = self._get_bf16_tmp_f32(m_hw, c_out)
            host_ops.bf16_to_f32_unpad(c_view, tmp_f32, m_hw, c_out)
            host_ops.f32_unpad_transpose_bias_relu(
                tmp_f32,
                dst_c_hw,
                m_hw,
                c_out,
                bias=bias,
                apply_relu=apply_relu,
            )
            return

        host_ops.f32_unpad_transpose_bias_relu(
            c_view,
            dst_c_hw,
            m_hw,
            c_out,
            bias=bias,
            apply_relu=apply_relu,
        )

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

        is_narrow = n <= _NARROW_N_THRESHOLD
        m_align = _M_ALIGN_NARROW if is_narrow else _M_ALIGN
        n_align = _N_ALIGN_NARROW if is_narrow else _N_ALIGN

        k_pad = _align_up(k, _TK)
        n_pad = _align_up(n, n_align)

        max_m_fit = min(_MAX_A_ELEMS // k_pad, _MAX_C_ELEMS // n_pad)
        max_chunk_m = (max_m_fit // m_align) * m_align
        if k_pad * n_pad > _MAX_B_ELEMS or max_chunk_m < m_align:
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
                self._ensure_buffers()
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
            m_pad = _align_up(chunk_m, m_align)

            wait_for_npu_render_priority()
            with NPU_DISPATCH_LOCK:
                if is_narrow:
                    self._ensure_narrow_initialized()
                else:
                    self._ensure_initialized()
                insts = self._get_insts(is_narrow, m_pad, k_pad, n_pad)
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
                self._run_kernel(is_narrow, m_pad, k_pad, n_pad, [a_view, t_b, c_view_t], insts)
                self.stats.npu_wall_s += time.perf_counter() - t1
                self.stats.dispatches += 1

                t2 = time.perf_counter()
                c_view = c_view_t.numpy()
                if single_chunk_nchw:
                    self._write_c_nchw(
                        c_view,
                        out.reshape(n, m),
                        chunk_m,
                        n,
                        bias=bias_1d,
                        apply_relu=apply_relu,
                    )
                else:
                    self._write_c_rowmajor(
                        c_view,
                        out[row : row + chunk_m],
                        chunk_m,
                        n,
                        bias=bias_1d,
                        apply_gelu=apply_gelu,
                        apply_relu=apply_relu,
                    )
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

        is_narrow = n <= _NARROW_N_THRESHOLD
        m_align = _M_ALIGN_NARROW if is_narrow else _M_ALIGN
        n_align = _N_ALIGN_NARROW if is_narrow else _N_ALIGN

        k_pad = _align_up(k, _TK)
        n_pad = _align_up(n, n_align)
        max_m_fit = min(_MAX_A_ELEMS // k_pad, _MAX_C_ELEMS // n_pad)
        max_chunk_m = (max_m_fit // m_align) * m_align
        rows_per_chunk = max_chunk_m // w if w > 0 else 0
        if k_pad * n_pad > _MAX_B_ELEMS or max_chunk_m < m_align or (m > max_chunk_m and rows_per_chunk < 1):
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
        out = np.empty((1, n, h, w), dtype=np.float32)

        t0_prep = time.perf_counter()
        t_b = self._get_weight_tensor(w_kykxc_cout, k_pad, n_pad)
        self.stats.host_copy_s += time.perf_counter() - t0_prep

        if m <= max_chunk_m:
            m_pad = _align_up(m, m_align)
            wait_for_npu_render_priority()
            with NPU_DISPATCH_LOCK:
                if is_narrow:
                    self._ensure_narrow_initialized()
                else:
                    self._ensure_initialized()
                insts = self._get_insts(is_narrow, m_pad, k_pad, n_pad)
                a_view = self._get_a_view(m_pad, k_pad)
                c_view_t = self._get_c_view(m_pad, n_pad)

                t0 = time.perf_counter()
                with a_view.overwrite() as dst_a_2d:
                    host_ops.im2col_3x3_s1_p1_kykxc_bf16_pad(x_nchw, dst_a_2d, zero_tail=(m_pad > m))
                self.stats.host_copy_s += time.perf_counter() - t0

                t1 = time.perf_counter()
                self._run_kernel(is_narrow, m_pad, k_pad, n_pad, [a_view, t_b, c_view_t], insts)
                self.stats.npu_wall_s += time.perf_counter() - t1
                self.stats.dispatches += 1

                t2 = time.perf_counter()
                c_view = c_view_t.numpy()
                self._write_c_nchw(
                    c_view,
                    out.reshape(n, m),
                    m,
                    n,
                    bias=bias_1d,
                    apply_relu=apply_relu,
                )
                self.stats.host_copy_s += time.perf_counter() - t2
            return out

        # Spatial row-chunking for large feature maps (e.g. 252x252): preserve exact p=1 top/bottom halo.
        y0 = 0
        while y0 < h:
            y1 = min(h, y0 + rows_per_chunk)
            strip_h = y1 - y0
            chunk_m = strip_h * w
            m_pad = _align_up(chunk_m, m_align)
            top_halo = 1 if y0 > 0 else 0
            bot_halo = 1 if y1 < h else 0
            halo_h = strip_h + top_halo + bot_halo
            halo_m = halo_h * w
            x_halo = np.ascontiguousarray(x_nchw[:, :, y0 - top_halo : y1 + bot_halo, :], dtype=np.float32)

            wait_for_npu_render_priority()
            with NPU_DISPATCH_LOCK:
                if is_narrow:
                    self._ensure_narrow_initialized()
                else:
                    self._ensure_initialized()
                insts = self._get_insts(is_narrow, m_pad, k_pad, n_pad)
                a_view = self._get_a_view(m_pad, k_pad)
                c_view_t = self._get_c_view(m_pad, n_pad)

                t0 = time.perf_counter()
                with a_view.overwrite() as dst_a_2d:
                    dst_u16 = dst_a_2d.view(np.uint16)
                    if top_halo == 0 and halo_m <= m_pad:
                        host_ops.im2col_3x3_s1_p1_kykxc_bf16_pad(x_halo, dst_u16, zero_tail=False)
                        if m_pad > chunk_m:
                            dst_u16[chunk_m:] = 0
                    else:
                        tmp_u16 = self._get_conv_strip_a_u16(halo_m, k_pad)
                        host_ops.im2col_3x3_s1_p1_kykxc_bf16_pad(x_halo, tmp_u16, zero_tail=False)
                        np.copyto(dst_u16[:chunk_m], tmp_u16[top_halo * w : top_halo * w + chunk_m])
                        if m_pad > chunk_m:
                            dst_u16[chunk_m:] = 0
                self.stats.host_copy_s += time.perf_counter() - t0

                t1 = time.perf_counter()
                self._run_kernel(is_narrow, m_pad, k_pad, n_pad, [a_view, t_b, c_view_t], insts)
                self.stats.npu_wall_s += time.perf_counter() - t1
                self.stats.dispatches += 1

                t2 = time.perf_counter()
                c_view = c_view_t.numpy()
                strip_out = self._get_conv_strip_out_f32(n, chunk_m)
                self._write_c_nchw(
                    c_view,
                    strip_out,
                    chunk_m,
                    n,
                    bias=bias_1d,
                    apply_relu=apply_relu,
                )
                out[0, :, y0:y1, :] = strip_out.reshape(n, strip_h, w)
                self.stats.host_copy_s += time.perf_counter() - t2

            y0 = y1

        return out

    def close(self) -> None:
        self._weight_cache.clear()
        self._insts_bo_cache.clear()
        self._insts_cache.clear()
        self._a_exact_bufs.clear()
        self._c_exact_bufs.clear()
        self._a_views.clear()
        self._c_views.clear()
        self._b_views.clear()
        self._conv_strip_a_u16 = None
        self._conv_strip_out_f32 = None
        self._bf16_tmp_f32 = None
        self._t_a = None
        self._t_b_scratch = None
        self._t_c = None
        self._handle = None
        self._kernel = None
        self._design = None
        self._narrow_handle = None
        self._narrow_kernel = None
        self._narrow_design = None
        self._runtime = None
