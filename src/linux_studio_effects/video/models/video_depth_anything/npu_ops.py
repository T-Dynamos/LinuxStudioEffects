"""Reusable NPU1 (AIE2) bfloat16 MatMul executor built on MLIR-AIE / IRON."""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ml_dtypes import bfloat16

from . import host_ops

# Global re-entrant lock serializing NPU load/dispatch across threads. The selected
# MLIR-AIE host runtime owns the device context shared by depth and portrait blur.
NPU_DISPATCH_LOCK = threading.RLock()

_NCOLS = 4
_TM = 32
_TK = 64
_TN = 64
_MAX_M_PER_CALL = 512  # m_tiles <= 4 per col -> max 10 Shim BDs per col (<= 16 limit)

_EPILOGUE_BIAS_CPP = r"""
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {

// f32 acc + f32 bias -> bf16 out (4x4 micro-tile layout, 32x64 tile).
void epilogue_f32_bias_bf16(float *__restrict acc_in,
                            float *__restrict bias_in,
                            bfloat16 *__restrict c_out) {
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (int g = 0; g < 8; g++) {
    aie::vector<float, 8> b8 = aie::load_v<8>(bias_in + g * 8);
    auto [z0, z1] = aie::interleave_zip(b8, b8, 4);
    aie::vector<float, 16> b16_pair[2] = {aie::concat(z0, z0), aie::concat(z1, z1)};
    for (int q = 0; q < 2; q++) {
      int nb = g * 2 + q;
      aie::vector<float, 16> b16 = b16_pair[q];
      for (int mb = 0; mb < 8; mb++) {
        int off = (mb * 16 + nb) * 16;
        aie::vector<float, 16> y = aie::add(aie::load_v<16>(acc_in + off), b16);
        aie::accum<accfloat, 16> acc(y);
        aie::store_v(c_out + off, acc.to_vector<bfloat16>());
      }
    }
  }
  aie::set_rounding(saved);
}

} // extern "C"
"""

_EPILOGUE_GELU_CPP = r"""
#include <aie_api/aie.hpp>
#include <lut_based_ops.h>
#include <stdint.h>
#include "lut_based_ops.cpp"

// Apply tanh-GELU across 4 vectors of 16 elements, batching LUT lookups to hide latency.
static inline __attribute__((always_inline)) void
gelu_batch4_f32_to_bf16(const aie::vector<float, 16> (&xf)[4],
                        bfloat16 *__restrict c_out, int stride_elems) {
  const float sqrt_2_over_pi = 0.79788456f;
  const float kBeta = 0.044715f;
  aie::accum<accfloat, 16> c_const(aie::broadcast<float, 16>(sqrt_2_over_pi));
  auto d_const = aie::broadcast<bfloat16, 16>(sqrt_2_over_pi * kBeta);
  auto v05 = aie::broadcast<bfloat16, 16>(0.5f);

  aie::vector<bfloat16, 16> xl[4], u[4], t[4];
  for (int k = 0; k < 4; k++) {
    aie::accum<accfloat, 16> acc(xf[k]);
    xl[k] = aie::max(acc.to_vector<bfloat16>(), bfloat16(-8.0f));
    aie::vector<bfloat16, 16> x2 = aie::mul(xl[k], xl[k]).to_vector<bfloat16>();
    aie::vector<bfloat16, 16> p = aie::mac(c_const, x2, d_const).to_vector<bfloat16>();
    u[k] = aie::mul(xl[k], p).to_vector<bfloat16>();
  }
  for (int k = 0; k < 4; k++) {
    t[k] = getTanhBf16(u[k]);
  }
  for (int k = 0; k < 4; k++) {
    aie::accum<accfloat, 16> hx_acc = aie::mul(xl[k], v05);
    aie::vector<bfloat16, 16> hx = hx_acc.to_vector<bfloat16>();
    aie::store_v(c_out + k * stride_elems, aie::mac(hx_acc, t[k], hx).to_vector<bfloat16>());
  }
}

extern "C" {

// f32 acc + f32 bias + tanh-GELU -> bf16 out (4x4 micro-tile layout, batched 4x LUT).
void epilogue_f32_bias_gelu_bf16(float *__restrict acc_in,
                                 float *__restrict bias_in,
                                 bfloat16 *__restrict c_out) {
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (int g = 0; g < 8; g++) {
    aie::vector<float, 8> b8 = aie::load_v<8>(bias_in + g * 8);
    auto [z0, z1] = aie::interleave_zip(b8, b8, 4);
    aie::vector<float, 16> b16_pair[2] = {aie::concat(z0, z0), aie::concat(z1, z1)};
    for (int q = 0; q < 2; q++) {
      int nb = g * 2 + q;
      aie::vector<float, 16> b16 = b16_pair[q];
      for (int mb_base = 0; mb_base < 8; mb_base += 4) {
        int base_off = (mb_base * 16 + nb) * 16;
        aie::vector<float, 16> xf[4];
        for (int k = 0; k < 4; k++) {
          xf[k] = aie::add(aie::load_v<16>(acc_in + base_off + k * 256), b16);
        }
        gelu_batch4_f32_to_bf16(xf, c_out + base_off, 256);
      }
    }
  }
  aie::set_rounding(saved);
}

// f32 acc + tanh-GELU -> bf16 out (elementwise across 2048-elem tile, batched 4x LUT).
void epilogue_f32_gelu_bf16(float *__restrict acc_in,
                            bfloat16 *__restrict c_out) {
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (int base = 0; base < 2048; base += 64) {
    aie::vector<float, 16> xf[4];
    for (int k = 0; k < 4; k++) {
      xf[k] = aie::load_v<16>(acc_in + base + k * 16);
    }
    gelu_batch4_f32_to_bf16(xf, c_out + base, 16);
  }
  aie::set_rounding(saved);
}

} // extern "C"
"""


def _align_up(val: int, mult: int) -> int:
    return ((val + mult - 1) // mult) * mult


@dataclass
class NpuExecStats:
    """Cumulative NPU kernel dispatch and host-sync timings for one forward pass."""

    dispatches: int = 0
    npu_wall_s: float = 0.0
    host_copy_s: float = 0.0


def _make_epilogue_kernel(*, use_bias: bool, gelu: bool) -> Any:
    from aie.iron import ExternalFunction
    from aie.iron.kernels._common import _detect_arch
    from aie.utils import config
    from aie.utils.compile.utils import AIE_API_PREFIX_FLAGS

    acc_tile_ty = np.ndarray[(_TM * _TN,), np.dtype[np.float32]]
    out_tile_ty = np.ndarray[(_TM * _TN,), np.dtype[bfloat16]]
    bias_tile_ty = np.ndarray[(_TN,), np.dtype[np.float32]]

    if gelu and use_bias:
        name = "epilogue_f32_bias_gelu_bf16"
        source = _EPILOGUE_GELU_CPP
        arg_types = [acc_tile_ty, bias_tile_ty, out_tile_ty]
    elif gelu:
        name = "epilogue_f32_gelu_bf16"
        source = _EPILOGUE_GELU_CPP
        arg_types = [acc_tile_ty, out_tile_ty]
    else:
        name = "epilogue_f32_bias_bf16"
        source = _EPILOGUE_BIAS_CPP
        arg_types = [acc_tile_ty, bias_tile_ty, out_tile_ty]

    include_dirs = [config.cxx_header_path()]
    if gelu:
        arch_dir = _detect_arch().upper()
        include_dirs.extend(
            [
                str(Path(config.aie_kernels_dir())),
                str(Path(config.aie_runtime_lib_dir()) / arch_dir),
            ]
        )

    return ExternalFunction(
        name,
        source_string=source,
        arg_types=arg_types,
        include_dirs=include_dirs,
        compile_flags=list(AIE_API_PREFIX_FLAGS),
    )


def _build_fused_epilogue_program(
    dim_m: int,
    dim_k: int,
    dim_n: int,
    *,
    use_bias: bool,
    gelu: bool,
) -> Any:
    import aie.iron as iron
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import ObjectFifo, Program, Runtime, Transport, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import Tile
    from aie.iron.kernels.linalg import mm

    a_full_ty = np.ndarray[(dim_m, dim_k), np.dtype[bfloat16]]
    b_full_ty = np.ndarray[(dim_k, dim_n), np.dtype[bfloat16]]
    bias_full_ty = np.ndarray[(dim_n,), np.dtype[np.float32]]
    c_full_ty = np.ndarray[(dim_m, dim_n), np.dtype[bfloat16]]

    a_tile_ty = np.ndarray[(_TM, _TK), np.dtype[bfloat16]]
    b_tile_ty = np.ndarray[(_TK, _TN), np.dtype[bfloat16]]
    bias_tile_ty = np.ndarray[(_TN,), np.dtype[np.float32]]
    c_acc_ty = np.ndarray[(_TM, _TN), np.dtype[np.float32]]
    c_tile_ty = np.ndarray[(_TM, _TN), np.dtype[bfloat16]]

    matmul_k = mm(_TM, _TK, _TN, input_dtype=bfloat16, output_dtype=np.float32, round_conv_even=True)
    sd = matmul_k.stream_dims
    ep_fn = _make_epilogue_kernel(use_bias=use_bias, gelu=gelu)

    inA_l2 = [ObjectFifo(a_tile_ty, name=f"inA_l2_{i}") for i in range(_NCOLS)]
    inA_l1 = [inA_l2[i].cons().forward(to_stream=sd.A, name=f"inA_l1_{i}") for i in range(_NCOLS)]

    # Broadcast B across all 4 columns to halve Shim channel usage and free a channel for bias.
    inB_l2 = ObjectFifo(b_tile_ty, name="inB_l2")
    inB_l1 = inB_l2.cons().forward(to_stream=sd.B, name="inB_l1")

    inBias_l2 = ObjectFifo(bias_tile_ty, name="inBias_l2") if use_bias else None

    # Pair each MatMul core (row 2) with an adjacent Epilogue core (row 3) over shared L1.
    acc_l1 = [
        ObjectFifo(c_acc_ty, depth=2, transport=Transport.shared_mem(), name=f"acc_l1_{i}")
        for i in range(_NCOLS)
    ]

    outC_l1 = [ObjectFifo(c_tile_ty, name=f"outC_l1_{i}") for i in range(_NCOLS)]
    outC_l2 = [outC_l1[i].cons().forward(to_stream=sd.C, name=f"outC_l2_{i}") for i in range(_NCOLS)]

    m_per_col = dim_m // _NCOLS
    m_tiles = m_per_col // _TM
    k_tiles = dim_k // _TK
    n_tiles = dim_n // _TN

    def _make_mm_worker(idx: int) -> Worker:
        def core_fn(of_a, of_b, of_acc, fn, zero_fn):
            for _ in range_(m_tiles * n_tiles):
                acc = of_acc.acquire(1)
                zero_fn(acc)
                for _ in range_(k_tiles):
                    a = of_a.acquire(1)
                    b = of_b.acquire(1)
                    fn(a, b, acc)
                    of_a.release(1)
                    of_b.release(1)
                of_acc.release(1)

        return Worker(
            core_fn,
            [inA_l1[idx].cons(), inB_l1.cons(), acc_l1[idx].prod(), matmul_k, matmul_k.zero],
            tile=Tile(idx, 2),
        )

    def _make_ep_worker(idx: int) -> Worker:
        if use_bias:
            assert inBias_l2 is not None

            def core_fn_bias(of_acc, of_bias, of_c, ep_k):
                for _ in range_(m_tiles * n_tiles):
                    acc = of_acc.acquire(1)
                    bias = of_bias.acquire(1)
                    c = of_c.acquire(1)
                    ep_k(acc, bias, c)
                    of_acc.release(1)
                    of_bias.release(1)
                    of_c.release(1)

            return Worker(
                core_fn_bias,
                [acc_l1[idx].cons(), inBias_l2.cons(), outC_l1[idx].prod(), ep_fn],
                tile=Tile(idx, 3),
            )

        def core_fn_nobias(of_acc, of_c, ep_k):
            for _ in range_(m_tiles * n_tiles):
                acc = of_acc.acquire(1)
                c = of_c.acquire(1)
                ep_k(acc, c)
                of_acc.release(1)
                of_c.release(1)

        return Worker(
            core_fn_nobias,
            [acc_l1[idx].cons(), outC_l1[idx].prod(), ep_fn],
            tile=Tile(idx, 3),
        )

    workers = [_make_mm_worker(i) for i in range(_NCOLS)] + [_make_ep_worker(i) for i in range(_NCOLS)]

    def _emit_ab_and_drain(A_h, B_h, C_h, b_prod, a_prods, c_conss):
        tap_B = TensorAccessPattern(
            tensor_dims=(dim_k, dim_n),
            offset=0,
            sizes=[n_tiles, k_tiles, _TK, _TN],
            strides=[_TN, _TK * dim_n, dim_n, 1],
        )
        for _ in range(m_tiles):
            b_prod.fill(B_h, tap=tap_B)
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
                a_prods[col].fill(A_h, tap=tap_A)
            tap_C = TensorAccessPattern(
                tensor_dims=(dim_m, dim_n),
                offset=r0 * dim_n,
                sizes=[m_tiles, n_tiles, _TM, _TN],
                strides=[_TM * dim_n, _TN, dim_n, 1],
            )
            c_conss[col].drain(C_h, tap=tap_C, wait=True)

    if use_bias:
        assert inBias_l2 is not None

        def sequence_bias(A_h, B_h, Bias_h, C_h, b_prod, bias_prod, *fifos):
            tap_Bias = TensorAccessPattern(
                tensor_dims=(dim_n,),
                offset=0,
                sizes=[m_tiles, n_tiles, _TN],
                strides=[0, _TN, 1],
            )
            bias_prod.fill(Bias_h, tap=tap_Bias)
            _emit_ab_and_drain(A_h, B_h, C_h, b_prod, fifos[:_NCOLS], fifos[_NCOLS:])

        rt = Runtime(
            sequence_bias,
            [
                a_full_ty,
                b_full_ty,
                bias_full_ty,
                c_full_ty,
                inB_l2.prod(),
                inBias_l2.prod(),
                *[f.prod() for f in inA_l2],
                *[f.cons() for f in outC_l2],
            ],
        )
    else:

        def sequence_nobias(A_h, B_h, C_h, b_prod, *fifos):
            _emit_ab_and_drain(A_h, B_h, C_h, b_prod, fifos[:_NCOLS], fifos[_NCOLS:])

        rt = Runtime(
            sequence_nobias,
            [
                a_full_ty,
                b_full_ty,
                c_full_ty,
                inB_l2.prod(),
                *[f.prod() for f in inA_l2],
                *[f.cons() for f in outC_l2],
            ],
        )

    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def _compile_matmul_4col(
    m: int,
    k: int,
    n: int,
    *,
    use_bias: bool = False,
    gelu: bool = False,
) -> Any:
    import aie.iron as iron
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.kernels.linalg import mm

    iron.ensure_current_device()

    if not use_bias and not gelu:

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

    if use_bias:

        @iron.jit
        def _bf16_matmul_bias_epilogue(
            a_in: In,
            b_in: In,
            bias_in: In,
            c_out: Out,
            *,
            dim_m: CompileTime[int] = m,
            dim_k: CompileTime[int] = k,
            dim_n: CompileTime[int] = n,
        ):
            return _build_fused_epilogue_program(dim_m, dim_k, dim_n, use_bias=True, gelu=gelu)

        return _bf16_matmul_bias_epilogue

    @iron.jit
    def _bf16_matmul_gelu_epilogue(
        a_in: In,
        b_in: In,
        c_out: Out,
        *,
        dim_m: CompileTime[int] = m,
        dim_k: CompileTime[int] = k,
        dim_n: CompileTime[int] = n,
    ):
        return _build_fused_epilogue_program(dim_m, dim_k, dim_n, use_bias=False, gelu=True)

    return _bf16_matmul_gelu_epilogue


class NpuMatMulExecutor:
    """Manages compiled IRON bf16 MatMul handles and persistent NPU weight/IO buffers."""

    # Bound live handles to 4 so QKV, OutProj, FC1+Bias+GELU, and FC2+Bias
    # stay resident together under the XRT runtime.
    _MAX_HANDLE_CACHE = 4

    def __init__(self) -> None:
        self._designs: dict[tuple[int, int, int, bool, bool], Any] = {}
        self._handles: OrderedDict[tuple[int, int, int, bool, bool], Any] = OrderedDict()
        self._a_cache: dict[tuple[int, int], Any] = {}
        self._c_cache: dict[tuple[int, int], Any] = {}
        self._tmp_b_cache: dict[tuple[int, int], Any] = {}
        self._tmp_bias_cache: dict[int, Any] = {}
        self._last_a_dims: dict[tuple[int, int], tuple[int, int]] = {}
        self._weight_cache: dict[tuple[int, int, int], tuple[np.ndarray, Any]] = {}
        self._bias_cache: dict[tuple[int, int], tuple[np.ndarray, Any]] = {}
        self._runtime: Any = None
        self.stats = NpuExecStats()

    def reset_stats(self) -> None:
        self.stats = NpuExecStats()

    def _get_a_buf(self, m_pad: int, k_pad: int) -> Any:
        import aie.iron as iron

        key = (m_pad, k_pad)
        t_a = self._a_cache.get(key)
        if t_a is None:
            t_a = iron.zeros((m_pad, k_pad), dtype=bfloat16, device="npu")
            self._a_cache[key] = t_a
        return t_a

    def _get_c_buf(self, m_pad: int, n_pad: int) -> Any:
        import aie.iron as iron

        key = (m_pad, n_pad)
        t_c = self._c_cache.get(key)
        if t_c is None:
            t_c = iron.zeros((m_pad, n_pad), dtype=bfloat16, device="npu")
            self._c_cache[key] = t_c
        return t_c

    def _get_tmp_b_buf(self, k_pad: int, n_pad: int) -> Any:
        import aie.iron as iron

        key = (k_pad, n_pad)
        t_b = self._tmp_b_cache.get(key)
        if t_b is None:
            t_b = iron.zeros((k_pad, n_pad), dtype=bfloat16, device="npu")
            self._tmp_b_cache[key] = t_b
        return t_b

    def _get_tmp_bias_buf(self, n_pad: int) -> Any:
        import aie.iron as iron

        t_bias = self._tmp_bias_cache.get(n_pad)
        if t_bias is None:
            t_bias = iron.zeros((n_pad,), dtype=np.float32, device="npu")
            self._tmp_bias_cache[n_pad] = t_bias
        return t_bias

    def _get_exec(
        self,
        m: int,
        k: int,
        n: int,
        *,
        use_bias: bool,
        gelu: bool,
        t_a: Any,
        t_b: Any,
        t_bias: Any | None,
        t_c: Any,
    ) -> tuple[Any, Any]:
        from aie.utils import DefaultNPURuntime

        if self._runtime is None:
            self._runtime = DefaultNPURuntime

        key = (m, k, n, use_bias, gelu)
        handle = self._handles.get(key)
        if handle is not None:
            self._handles.move_to_end(key)
            return self._runtime, handle

        while len(self._handles) >= self._MAX_HANDLE_CACHE:
            _, old_handle = self._handles.popitem(last=False)
            del old_handle

        if key not in self._designs:
            design = _compile_matmul_4col(m, k, n, use_bias=use_bias, gelu=gelu)
            # Warmup once so CallableDesign compiles and populates _kernel_cache.
            if use_bias:
                design(t_a, t_b, t_bias, t_c, dim_m=m, dim_k=k, dim_n=n)
            else:
                design(t_a, t_b, t_c, dim_m=m, dim_k=k, dim_n=n)
            self._designs[key] = design

        design = self._designs[key]
        npu_kernel = next(iter(design._kernel_cache.values()))
        handle = self._runtime.load(npu_kernel)
        self._handles[key] = handle
        return self._runtime, handle

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
        t_b = iron.tensor(b_pad, device="npu")
        # Keep a reference to b_mat so its id() cannot be recycled onto a new temporary array.
        self._weight_cache[key] = (b_mat, t_b)
        return t_b

    def _get_bias_tensor(self, bias_key_obj: np.ndarray, bias_1d: np.ndarray, n_pad: int) -> Any:
        import aie.iron as iron

        key = (id(bias_key_obj), n_pad)
        cached = self._bias_cache.get(key)
        if cached is not None:
            cached_ref, t_bias = cached
            if cached_ref is bias_key_obj:
                return t_bias

        n = bias_1d.shape[0]
        bias_pad = np.zeros((n_pad,), dtype=np.float32)
        bias_pad[:n] = bias_1d.astype(np.float32, copy=False)
        t_bias = iron.tensor(bias_pad, device="npu")
        # Keep a reference to bias_key_obj so its id() cannot be recycled onto a new temporary array.
        self._bias_cache[key] = (bias_key_obj, t_bias)
        return t_bias

    def matmul(
        self,
        a: np.ndarray,
        b: np.ndarray,
        *,
        bias: np.ndarray | None = None,
        activation: str | None = None,
        cache_b: bool = True,
    ) -> np.ndarray:
        """Compute (M, K) @ (K, N) [+ bias] [with optional activation] on NPU1, returning float32."""
        m, k = a.shape
        bk, n = b.shape
        if k != bk:
            raise ValueError(f"Inner dimensions mismatch: {a.shape} @ {b.shape}")

        if activation is not None and activation != "gelu":
            raise ValueError(f"Unsupported activation {activation!r}; expected None or 'gelu'")
        gelu = activation == "gelu"

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
        use_bias = bias_1d is not None

        k_pad = _align_up(k, _TK)
        n_pad = _align_up(n, _TN)

        out = np.empty((m, n), dtype=np.float32)

        t0_prep = time.perf_counter()
        if cache_b:
            t_b = self._get_weight_tensor(b, k_pad, n_pad)
            t_bias = self._get_bias_tensor(bias, bias_1d, n_pad) if bias is not None and bias_1d is not None else None
        else:
            t_b = self._get_tmp_b_buf(k_pad, n_pad)
            with t_b.overwrite() as dst_b:
                dst_b.fill(0)
                dst_b[:k, :n] = b.astype(bfloat16)
            if bias_1d is not None:
                t_bias = self._get_tmp_bias_buf(n_pad)
                with t_bias.overwrite() as dst_bias:
                    dst_bias.fill(0)
                    dst_bias[:n] = bias_1d.astype(np.float32, copy=False)
            else:
                t_bias = None
        self.stats.host_copy_s += time.perf_counter() - t0_prep

        row = 0
        while row < m:
            rem = m - row
            chunk_m = min(rem, _MAX_M_PER_CALL)
            # When m > _MAX_M_PER_CALL, pad the tail chunk to _MAX_M_PER_CALL as well
            # so multi-chunk layers (e.g. 518x518 M=1370) reuse the same 4 NPU handles.
            m_pad = _MAX_M_PER_CALL if m > _MAX_M_PER_CALL else _align_up(chunk_m, _NCOLS * _TM)

            a_key = (m_pad, k_pad)
            t_a = self._get_a_buf(m_pad, k_pad)
            t_c = self._get_c_buf(m_pad, n_pad)

            with NPU_DISPATCH_LOCK:
                rt, handle = self._get_exec(
                    m_pad,
                    k_pad,
                    n_pad,
                    use_bias=use_bias,
                    gelu=gelu,
                    t_a=t_a,
                    t_b=t_b,
                    t_bias=t_bias,
                    t_c=t_c,
                )

                t0 = time.perf_counter()
                cur_dims = (chunk_m, k)
                prev_dims = self._last_a_dims.get(a_key)
                use_fast_bf16 = a.dtype == np.float32 and host_ops.has_avx512_host_ops()
                with t_a.overwrite() as dst_a:
                    if use_fast_bf16:
                        host_ops.f32_to_bf16_pad(
                            a[row : row + chunk_m],
                            dst_a,
                            chunk_m,
                            k,
                            zero_tail=(prev_dims is not None and prev_dims != cur_dims),
                        )
                    else:
                        if prev_dims is not None and prev_dims != cur_dims:
                            dst_a.fill(0)
                        dst_a[:chunk_m, :k] = a[row : row + chunk_m].astype(bfloat16)
                self._last_a_dims[a_key] = cur_dims
                self.stats.host_copy_s += time.perf_counter() - t0

                t1 = time.perf_counter()
                if use_bias:
                    rt.run(handle, [t_a, t_b, t_bias, t_c])
                else:
                    rt.run(handle, [t_a, t_b, t_c])
                self.stats.npu_wall_s += time.perf_counter() - t1
                self.stats.dispatches += 1

                t2 = time.perf_counter()
                c_view = t_c.numpy()
                if host_ops.has_avx512_host_ops():
                    host_ops.bf16_to_f32_unpad(c_view, out[row : row + chunk_m], chunk_m, n)
                else:
                    out[row : row + chunk_m] = c_view[:chunk_m, :n].astype(np.float32)
                self.stats.host_copy_s += time.perf_counter() - t2

            row += chunk_m
        return out

    def close(self) -> None:
        self._weight_cache.clear()
        self._bias_cache.clear()
        self._tmp_b_cache.clear()
        self._tmp_bias_cache.clear()
        self._last_a_dims.clear()
        self._a_cache.clear()
        self._c_cache.clear()
        self._handles.clear()
        self._designs.clear()
        self._runtime = None
