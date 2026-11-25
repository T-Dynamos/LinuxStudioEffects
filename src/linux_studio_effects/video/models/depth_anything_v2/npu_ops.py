"""Reusable NPU1 (AIE2) bfloat16 MatMul executor built on MLIR-AIE / IRON."""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from ml_dtypes import bfloat16

_NCOLS = 4
_TM = 32
_TK = 64
_TN = 64
_MAX_M_PER_CALL = 512  # m_tiles <= 4 per col -> 4+4+1 = 9 Shim BDs (<= 16 limit)


def _align_up(val: int, mult: int) -> int:
    return ((val + mult - 1) // mult) * mult


@dataclass
class NpuExecStats:
    """Cumulative NPU kernel dispatch and host-sync timings for one forward pass."""

    dispatches: int = 0
    npu_wall_s: float = 0.0
    host_copy_s: float = 0.0


def _compile_matmul_4col(m: int, k: int, n: int) -> Any:
    import aie.iron as iron
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.kernels.linalg import mm

    iron.ensure_current_device()

    @iron.jit
    def _bf16_matmul(
        a_in: In,
        b_in: In,
        c_out: Out,
        *,
        dim_m: CompileTime[int] = m,
        dim_k: CompileTime[int] = k,
        dim_n: CompileTime[int] = n,
    ):
        a_full_ty = np.ndarray[(dim_m, dim_k), np.dtype[bfloat16]]
        b_full_ty = np.ndarray[(dim_k, dim_n), np.dtype[bfloat16]]
        c_full_ty = np.ndarray[(dim_m, dim_n), np.dtype[bfloat16]]

        a_tile_ty = np.ndarray[(_TM, _TK), np.dtype[bfloat16]]
        b_tile_ty = np.ndarray[(_TK, _TN), np.dtype[bfloat16]]
        c_tile_ty = np.ndarray[(_TM, _TN), np.dtype[bfloat16]]

        matmul_k = mm(_TM, _TK, _TN, input_dtype=bfloat16, output_dtype=bfloat16, round_conv_even=True)
        sd = matmul_k.stream_dims

        inA_l2 = [ObjectFifo(a_tile_ty, name=f"inA_l2_{i}") for i in range(_NCOLS)]
        inA_l1 = [inA_l2[i].cons().forward(to_stream=sd.A, name=f"inA_l1_{i}") for i in range(_NCOLS)]

        inB_l2 = [ObjectFifo(b_tile_ty, name=f"inB_l2_{i}") for i in range(_NCOLS)]
        inB_l1 = [inB_l2[i].cons().forward(to_stream=sd.B, name=f"inB_l1_{i}") for i in range(_NCOLS)]

        outC_l1 = [ObjectFifo(c_tile_ty, name=f"outC_l1_{i}") for i in range(_NCOLS)]
        outC_l2 = [outC_l1[i].cons().forward(to_stream=sd.C, name=f"outC_l2_{i}") for i in range(_NCOLS)]

        m_per_col = dim_m // _NCOLS
        m_tiles = m_per_col // _TM
        k_tiles = dim_k // _TK
        n_tiles = dim_n // _TN

        def _make_worker(idx: int) -> Worker:
            def core_fn(of_a, of_b, of_c, fn, zero_fn):
                for _ in range_(m_tiles * n_tiles):
                    c = of_c.acquire(1)
                    zero_fn(c)
                    for _ in range_(k_tiles):
                        a = of_a.acquire(1)
                        b = of_b.acquire(1)
                        fn(a, b, c)
                        of_a.release(1)
                        of_b.release(1)
                    of_c.release(1)

            return Worker(
                core_fn,
                [inA_l1[idx].cons(), inB_l1[idx].cons(), outC_l1[idx].prod(), matmul_k, matmul_k.zero],
            )

        workers = [_make_worker(i) for i in range(_NCOLS)]

        def sequence(A_h, B_h, C_h, *fifos):
            a_prods = fifos[:_NCOLS]
            b_prods = fifos[_NCOLS : 2 * _NCOLS]
            c_conss = fifos[2 * _NCOLS :]
            for col in range(_NCOLS):
                r0 = col * m_per_col
                for mt in range(m_tiles):
                    row_off = r0 + mt * _TM
                    tap_A = TensorAccessPattern(
                        tensor_dims=(dim_m, dim_k),
                        offset=row_off * dim_k,
                        sizes=[n_tiles, k_tiles, _TM, _TK],
                        strides=[0, _TK, dim_k, 1],
                    )
                    tap_B = TensorAccessPattern(
                        tensor_dims=(dim_k, dim_n),
                        offset=0,
                        sizes=[n_tiles, k_tiles, _TK, _TN],
                        strides=[_TN, _TK * dim_n, dim_n, 1],
                    )
                    a_prods[col].fill(A_h, tap=tap_A)
                    b_prods[col].fill(B_h, tap=tap_B)
                tap_C = TensorAccessPattern(
                    tensor_dims=(dim_m, dim_n),
                    offset=r0 * dim_n,
                    sizes=[m_tiles, n_tiles, _TM, _TN],
                    strides=[_TM * dim_n, _TN, dim_n, 1],
                )
                c_conss[col].drain(C_h, tap=tap_C, wait=True)

        rt = Runtime(
            sequence,
            [
                a_full_ty,
                b_full_ty,
                c_full_ty,
                *[f.prod() for f in inA_l2],
                *[f.prod() for f in inB_l2],
                *[f.cons() for f in outC_l2],
            ],
        )
        return Program(iron.get_current_device(), rt, workers=workers).resolve_program()

    return _bf16_matmul


class NpuMatMulExecutor:
    """Manages compiled IRON bf16 MatMul kernels and persistent NPU weight/IO buffers."""

    def __init__(self) -> None:
        self._kernels: dict[tuple[int, int, int], Any] = {}
        self._io_cache: dict[tuple[int, int, int], tuple[Any, Any, Any]] = {}
        self._weight_cache: dict[tuple[int, int, int], Any] = {}
        self.stats = NpuExecStats()

    def reset_stats(self) -> None:
        self.stats = NpuExecStats()

    def _get_kernel_and_io(self, m: int, k: int, n: int) -> tuple[Any, Any, Any, Any]:
        import aie.iron as iron

        key = (m, k, n)
        if key not in self._kernels:
            self._kernels[key] = _compile_matmul_4col(m, k, n)
        if key not in self._io_cache:
            t_a = iron.zeros((m, k), dtype=bfloat16, device="npu")
            t_b = iron.zeros((k, n), dtype=bfloat16, device="npu")
            t_c = iron.zeros((m, n), dtype=bfloat16, device="npu")
            self._io_cache[key] = (t_a, t_b, t_c)
        t_a, t_b, t_c = self._io_cache[key]
        return self._kernels[key], t_a, t_b, t_c

    def _get_weight_tensor(self, b_mat: np.ndarray, k_pad: int, n_pad: int) -> Any:
        import aie.iron as iron

        key = (id(b_mat), k_pad, n_pad)
        cached = self._weight_cache.get(key)
        if cached is not None:
            return cached

        k, n = b_mat.shape
        b_pad = np.zeros((k_pad, n_pad), dtype=bfloat16)
        b_pad[:k, :n] = b_mat.astype(bfloat16)
        t_b = iron.tensor(b_pad, device="npu")
        self._weight_cache[key] = t_b
        return t_b

    def matmul(self, a: np.ndarray, b: np.ndarray, *, cache_b: bool = True) -> np.ndarray:
        """Compute (M, K) @ (K, N) on NPU1 in bfloat16, returning float32."""
        m, k = a.shape
        bk, n = b.shape
        if k != bk:
            raise ValueError(f"Inner dimensions mismatch: {a.shape} @ {b.shape}")

        k_pad = _align_up(k, _TK)
        n_pad = _align_up(n, _TN)

        out = np.empty((m, n), dtype=np.float32)
        t_b_cached = self._get_weight_tensor(b, k_pad, n_pad) if cache_b else None

        row = 0
        while row < m:
            rem = m - row
            chunk_m = min(rem, _MAX_M_PER_CALL)
            m_pad = _align_up(chunk_m, _NCOLS * _TM)

            kernel, t_a, t_b_tmp, t_c = self._get_kernel_and_io(m_pad, k_pad, n_pad)

            t0 = time.perf_counter()
            a_pad = np.zeros((m_pad, k_pad), dtype=bfloat16)
            a_pad[:chunk_m, :k] = a[row : row + chunk_m].astype(bfloat16)
            with t_a.overwrite() as dst_a:
                np.copyto(dst_a, a_pad)

            if t_b_cached is not None:
                t_b = t_b_cached
            else:
                b_pad = np.zeros((k_pad, n_pad), dtype=bfloat16)
                b_pad[:k, :n] = b.astype(bfloat16)
                with t_b_tmp.overwrite() as dst_b:
                    np.copyto(dst_b, b_pad)
                t_b = t_b_tmp
            self.stats.host_copy_s += time.perf_counter() - t0

            t1 = time.perf_counter()
            kernel(t_a, t_b, t_c, dim_m=m_pad, dim_k=k_pad, dim_n=n_pad)
            self.stats.npu_wall_s += time.perf_counter() - t1
            self.stats.dispatches += 1

            t2 = time.perf_counter()
            out[row : row + chunk_m] = t_c.numpy()[:chunk_m, :n].astype(np.float32)
            self.stats.host_copy_s += time.perf_counter() - t2

            row += chunk_m
        return out

    def close(self) -> None:
        self._weight_cache.clear()
        self._io_cache.clear()
        self._kernels.clear()
