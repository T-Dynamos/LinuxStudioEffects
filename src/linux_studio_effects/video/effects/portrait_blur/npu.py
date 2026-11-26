"""AMD Ryzen AI NPU depth-aware portrait bokeh blur backed by Video Depth Anything."""

from functools import lru_cache
from typing import Any

import numpy as np
from ml_dtypes import bfloat16

from ...models.video_depth_anything import host_ops
from ...models.video_depth_anything.npu_ops import NPU_DISPATCH_LOCK, render_priority
from ..base import ComputeDevice
from .cpu import PortraitBlurEffect as _CpuPortraitBlurEffect

_BOKEH_NCOLS = 4
_BOKEH_TILE_PIXELS = 32
_BOKEH_PIXELS_ALIGN = _BOKEH_NCOLS * _BOKEH_TILE_PIXELS
_MAX_BOKEH_PIXELS = 4096 * 2160

_BLUR_NCOLS = 4
_BLUR_TILE_W = 16
_BLUR_TILE_ELEMS = _BLUR_TILE_W * 4  # 64 bf16 (128 bytes)
_MAX_BLUR_ELEMS = 16 * 1024 * 1024  # 32 MiB bf16

_BOKEH_BLEND4_C_SRC = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" void lse_bokeh_blend4(
    const uint8_t *__restrict in_a,
    const uint8_t *__restrict in_b,
    uint8_t *__restrict out,
    int32_t num_pixels
) {
    const ::aie::vector<int16_t, 32> c85 = ::aie::broadcast<int16_t, 32>(85);
    const ::aie::vector<int16_t, 32> c170 = ::aie::broadcast<int16_t, 32>(170);
    const ::aie::vector<int16_t, 32> c256 = ::aie::broadcast<int16_t, 32>(256);
    const ::aie::vector<int16_t, 32> c0 = ::aie::broadcast<int16_t, 32>(0);
    const ::aie::vector<int16_t, 32> c3 = ::aie::broadcast<int16_t, 32>(3);
    ::aie::accum<acc32, 32> rnd;
    rnd.from_vector(::aie::broadcast<int32_t, 32>(128));

    const int steps = num_pixels / 32;
    _Pragma("clang loop unroll(disable)")
    for (int s = 0; s < steps; ++s) {
        const uint8_t *__restrict pa = in_a + s * 128;
        const uint8_t *__restrict pb = in_b + s * 288;
        const ::aie::vector<int16_t, 32> r0  = ::aie::unpack(::aie::load_v<32>(pa + 0)).template cast_to<int16_t>();
        const ::aie::vector<int16_t, 32> g0  = ::aie::unpack(::aie::load_v<32>(pa + 32)).template cast_to<int16_t>();
        const ::aie::vector<int16_t, 32> b0  = ::aie::unpack(::aie::load_v<32>(pa + 64)).template cast_to<int16_t>();
        const ::aie::vector<int16_t, 32> coc = ::aie::unpack(::aie::load_v<32>(pa + 96)).template cast_to<int16_t>();
        const ::aie::vector<int16_t, 32> r1  = ::aie::unpack(::aie::load_v<32>(pb + 0)).template cast_to<int16_t>();
        const ::aie::vector<int16_t, 32> g1  = ::aie::unpack(::aie::load_v<32>(pb + 32)).template cast_to<int16_t>();
        const ::aie::vector<int16_t, 32> b1  = ::aie::unpack(::aie::load_v<32>(pb + 64)).template cast_to<int16_t>();
        const ::aie::vector<int16_t, 32> r2  = ::aie::unpack(::aie::load_v<32>(pb + 96)).template cast_to<int16_t>();
        const ::aie::vector<int16_t, 32> g2  = ::aie::unpack(::aie::load_v<32>(pb + 128)).template cast_to<int16_t>();
        const ::aie::vector<int16_t, 32> b2  = ::aie::unpack(::aie::load_v<32>(pb + 160)).template cast_to<int16_t>();
        const ::aie::vector<int16_t, 32> r3  = ::aie::unpack(::aie::load_v<32>(pb + 192)).template cast_to<int16_t>();
        const ::aie::vector<int16_t, 32> g3  = ::aie::unpack(::aie::load_v<32>(pb + 224)).template cast_to<int16_t>();
        const ::aie::vector<int16_t, 32> b3  = ::aie::unpack(::aie::load_v<32>(pb + 256)).template cast_to<int16_t>();

        auto m0 = ::aie::lt(coc, c85);
        auto m2 = ::aie::ge(coc, c170);
        auto m1 = (~m0) & (~m2);

        ::aie::vector<int16_t, 32> d1 = ::aie::sub(coc, c85);
        ::aie::vector<int16_t, 32> d2 = ::aie::sub(coc, c170);

        ::aie::vector<int16_t, 32> t0 = ::aie::mul(coc, c3).template to_vector<int16_t>(0);
        ::aie::vector<int16_t, 32> t1 = ::aie::mul(d1, c3).template to_vector<int16_t>(0);
        ::aie::vector<int16_t, 32> t2 = ::aie::add(
            ::aie::mul(d2, c3).template to_vector<int16_t>(0),
            ::aie::downshift(d2, 6)
        );

        ::aie::vector<int16_t, 32> w0 = ::aie::select(c0, ::aie::sub(c256, t0), m0);
        ::aie::vector<int16_t, 32> w1 = ::aie::select(::aie::select(c0, ::aie::sub(c256, t1), m1), t0, m0);
        ::aie::vector<int16_t, 32> w2 = ::aie::select(::aie::select(c0, ::aie::sub(c256, t2), m2), t1, m1);
        ::aie::vector<int16_t, 32> w3 = ::aie::select(c0, t2, m2);

        ::aie::accum<acc32, 32> ar = ::aie::mac(::aie::mac(::aie::mac(::aie::mac(rnd, r0, w0), r1, w1), r2, w2), r3, w3);
        ::aie::accum<acc32, 32> ag = ::aie::mac(::aie::mac(::aie::mac(::aie::mac(rnd, g0, w0), g1, w1), g2, w2), g3, w3);
        ::aie::accum<acc32, 32> ab = ::aie::mac(::aie::mac(::aie::mac(::aie::mac(rnd, b0, w0), b1, w1), b2, w2), b3, w3);

        ::aie::vector<uint8_t, 32> orv = ar.template to_vector<uint8_t>(8);
        ::aie::vector<uint8_t, 32> ogv = ag.template to_vector<uint8_t>(8);
        ::aie::vector<uint8_t, 32> obv = ab.template to_vector<uint8_t>(8);

        uint8_t *__restrict q = out + s * 96;
        for (int k = 0; k < 32; ++k) {
            q[k * 3 + 0] = orv[k];
            q[k * 3 + 1] = ogv[k];
            q[k * 3 + 2] = obv[k];
        }
    }
}
"""

_GAUSSIAN5_HW4_BF16_C_SRC = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" void lse_gaussian5_hw4_bf16(
    const bfloat16 *__restrict in_tile,
    bfloat16 *__restrict out_tile,
    int32_t mode,
    int32_t op,
    int32_t in_idx,
    int32_t out_idx,
    int32_t inner_cnt,
    int32_t valid_tail
) {
    // 9-row circular buffer in core L1 (1,152 bytes, 64B-aligned).
    // Modes 0 & 2 use ring[0..2] for strips and ring[3..4] for edge clamp vectors.
    // Mode 1 uses ring[0..4] (5-tap vertical); Mode 3 uses ring[0..8] (9-tap binomial vertical).
    alignas(64) static bfloat16 ring[9][64];

    ::aie::set_rounding(::aie::rounding_mode::conv_even);

    // op: 0 = push input only, 1 = push input + emit output, 2 = emit output only.
    const bool is_horiz = (mode == 0 || mode == 2);
    if (op != 2) {
        const int mod_n = is_horiz ? 3 : ((mode == 3) ? 9 : 5);
        const int slot = in_idx % mod_n;
        bfloat16 *__restrict dst = ring[slot];
        const ::aie::vector<bfloat16, 64> v_in = ::aie::load_v<64>(in_tile);
        ::aie::store_v(dst, v_in);
        if (is_horiz && in_idx == inner_cnt - 1 && valid_tail < 16) {
            const int last_off = (valid_tail - 1) * 4;
            const bfloat16 c0 = dst[last_off + 0];
            const bfloat16 c1 = dst[last_off + 1];
            const bfloat16 c2 = dst[last_off + 2];
            const bfloat16 c3 = dst[last_off + 3];
            for (int p = valid_tail; p < 16; ++p) {
                dst[p * 4 + 0] = c0;
                dst[p * 4 + 1] = c1;
                dst[p * 4 + 2] = c2;
                dst[p * 4 + 3] = c3;
            }
        }
    }

    if (op == 0) {
        return;
    }

    if (mode == 0 || mode == 1) {
        const ::aie::vector<bfloat16, 64> wv0 = ::aie::broadcast<bfloat16, 64>(bfloat16(0.0625f));
        const ::aie::vector<bfloat16, 64> wv1 = ::aie::broadcast<bfloat16, 64>(bfloat16(0.25f));
        const ::aie::vector<bfloat16, 64> wv2 = ::aie::broadcast<bfloat16, 64>(bfloat16(0.375f));

        if (mode == 0) {
            bfloat16 *__restrict p_cur = ring[out_idx % 3];
            const ::aie::vector<bfloat16, 64> v_cur = ::aie::load_v<64>(p_cur);

            ::aie::vector<bfloat16, 64> v_prev;
            if (out_idx == 0) {
                bfloat16 *__restrict pad_prev = ring[3];
                for (int ch = 0; ch < 4; ++ch) {
                    const bfloat16 v0 = p_cur[ch];
                    pad_prev[56 + ch] = v0;
                    pad_prev[60 + ch] = v0;
                }
                v_prev = ::aie::load_v<64>(pad_prev);
            } else {
                v_prev = ::aie::load_v<64>(ring[(out_idx - 1) % 3]);
            }

            ::aie::vector<bfloat16, 64> v_next;
            if (out_idx == inner_cnt - 1) {
                bfloat16 *__restrict pad_next = ring[4];
                const int last_off = (valid_tail - 1) * 4;
                for (int ch = 0; ch < 4; ++ch) {
                    const bfloat16 vlast = p_cur[last_off + ch];
                    pad_next[0 + ch] = vlast;
                    pad_next[4 + ch] = vlast;
                }
                v_next = ::aie::load_v<64>(pad_next);
            } else {
                v_next = ::aie::load_v<64>(ring[(out_idx + 1) % 3]);
            }

            const ::aie::vector<bfloat16, 64> v_m2 = ::aie::shuffle_up_fill(v_cur, v_prev, 8);
            const ::aie::vector<bfloat16, 64> v_m1 = ::aie::shuffle_up_fill(v_cur, v_prev, 4);
            const ::aie::vector<bfloat16, 64> v_p1 = ::aie::shuffle_down_fill(v_cur, v_next, 4);
            const ::aie::vector<bfloat16, 64> v_p2 = ::aie::shuffle_down_fill(v_cur, v_next, 8);

            ::aie::accum<accfloat, 64> acc = ::aie::mul(v_m2, wv0);
            acc = ::aie::mac(acc, v_m1, wv1);
            acc = ::aie::mac(acc, v_cur, wv2);
            acc = ::aie::mac(acc, v_p1, wv1);
            acc = ::aie::mac(acc, v_p2, wv0);
            ::aie::store_v(out_tile, acc.to_vector<bfloat16>());
        } else {
            const int y_m2 = (out_idx >= 2) ? (out_idx - 2) : 0;
            const int y_m1 = (out_idx >= 1) ? (out_idx - 1) : 0;
            const int y_0  = out_idx;
            const int y_p1 = (out_idx + 1 < inner_cnt) ? (out_idx + 1) : (inner_cnt - 1);
            const int y_p2 = (out_idx + 2 < inner_cnt) ? (out_idx + 2) : (inner_cnt - 1);

            ::aie::accum<accfloat, 64> acc = ::aie::mul(::aie::load_v<64>(ring[y_m2 % 5]), wv0);
            acc = ::aie::mac(acc, ::aie::load_v<64>(ring[y_m1 % 5]), wv1);
            acc = ::aie::mac(acc, ::aie::load_v<64>(ring[y_0  % 5]), wv2);
            acc = ::aie::mac(acc, ::aie::load_v<64>(ring[y_p1 % 5]), wv1);
            acc = ::aie::mac(acc, ::aie::load_v<64>(ring[y_p2 % 5]), wv0);
            ::aie::store_v(out_tile, acc.to_vector<bfloat16>());
        }
        return;
    }

    // 2-pass equivalent 9-tap binomial filter: [1, 8, 28, 56, 70, 56, 28, 8, 1] / 256.
    const ::aie::vector<bfloat16, 64> bw0 = ::aie::broadcast<bfloat16, 64>(bfloat16(0.00390625f));
    const ::aie::vector<bfloat16, 64> bw1 = ::aie::broadcast<bfloat16, 64>(bfloat16(0.03125f));
    const ::aie::vector<bfloat16, 64> bw2 = ::aie::broadcast<bfloat16, 64>(bfloat16(0.109375f));
    const ::aie::vector<bfloat16, 64> bw3 = ::aie::broadcast<bfloat16, 64>(bfloat16(0.21875f));
    const ::aie::vector<bfloat16, 64> bw4 = ::aie::broadcast<bfloat16, 64>(bfloat16(0.2734375f));

    if (mode == 2) {
        bfloat16 *__restrict p_cur = ring[out_idx % 3];
        const ::aie::vector<bfloat16, 64> v_cur = ::aie::load_v<64>(p_cur);

        ::aie::vector<bfloat16, 64> v_prev;
        if (out_idx == 0) {
            bfloat16 *__restrict pad_prev = ring[3];
            for (int ch = 0; ch < 4; ++ch) {
                const bfloat16 v0 = p_cur[ch];
                pad_prev[48 + ch] = v0;
                pad_prev[52 + ch] = v0;
                pad_prev[56 + ch] = v0;
                pad_prev[60 + ch] = v0;
            }
            v_prev = ::aie::load_v<64>(pad_prev);
        } else {
            v_prev = ::aie::load_v<64>(ring[(out_idx - 1) % 3]);
        }

        ::aie::vector<bfloat16, 64> v_next;
        if (out_idx == inner_cnt - 1) {
            bfloat16 *__restrict pad_next = ring[4];
            const int last_off = (valid_tail - 1) * 4;
            for (int ch = 0; ch < 4; ++ch) {
                const bfloat16 vlast = p_cur[last_off + ch];
                pad_next[0 + ch]  = vlast;
                pad_next[4 + ch]  = vlast;
                pad_next[8 + ch]  = vlast;
                pad_next[12 + ch] = vlast;
            }
            v_next = ::aie::load_v<64>(pad_next);
        } else {
            v_next = ::aie::load_v<64>(ring[(out_idx + 1) % 3]);
        }

        ::aie::accum<accfloat, 64> acc = ::aie::mul(::aie::shuffle_up_fill(v_cur, v_prev, 16), bw0);
        acc = ::aie::mac(acc, ::aie::shuffle_up_fill(v_cur, v_prev, 12), bw1);
        acc = ::aie::mac(acc, ::aie::shuffle_up_fill(v_cur, v_prev, 8), bw2);
        acc = ::aie::mac(acc, ::aie::shuffle_up_fill(v_cur, v_prev, 4), bw3);
        acc = ::aie::mac(acc, v_cur, bw4);
        acc = ::aie::mac(acc, ::aie::shuffle_down_fill(v_cur, v_next, 4), bw3);
        acc = ::aie::mac(acc, ::aie::shuffle_down_fill(v_cur, v_next, 8), bw2);
        acc = ::aie::mac(acc, ::aie::shuffle_down_fill(v_cur, v_next, 12), bw1);
        acc = ::aie::mac(acc, ::aie::shuffle_down_fill(v_cur, v_next, 16), bw0);
        ::aie::store_v(out_tile, acc.to_vector<bfloat16>());
    } else {
        const int y_m4 = (out_idx >= 4) ? (out_idx - 4) : 0;
        const int y_m3 = (out_idx >= 3) ? (out_idx - 3) : 0;
        const int y_m2 = (out_idx >= 2) ? (out_idx - 2) : 0;
        const int y_m1 = (out_idx >= 1) ? (out_idx - 1) : 0;
        const int y_0  = out_idx;
        const int y_p1 = (out_idx + 1 < inner_cnt) ? (out_idx + 1) : (inner_cnt - 1);
        const int y_p2 = (out_idx + 2 < inner_cnt) ? (out_idx + 2) : (inner_cnt - 1);
        const int y_p3 = (out_idx + 3 < inner_cnt) ? (out_idx + 3) : (inner_cnt - 1);
        const int y_p4 = (out_idx + 4 < inner_cnt) ? (out_idx + 4) : (inner_cnt - 1);

        ::aie::accum<accfloat, 64> acc = ::aie::mul(::aie::load_v<64>(ring[y_m4 % 9]), bw0);
        acc = ::aie::mac(acc, ::aie::load_v<64>(ring[y_m3 % 9]), bw1);
        acc = ::aie::mac(acc, ::aie::load_v<64>(ring[y_m2 % 9]), bw2);
        acc = ::aie::mac(acc, ::aie::load_v<64>(ring[y_m1 % 9]), bw3);
        acc = ::aie::mac(acc, ::aie::load_v<64>(ring[y_0  % 9]), bw4);
        acc = ::aie::mac(acc, ::aie::load_v<64>(ring[y_p1 % 9]), bw3);
        acc = ::aie::mac(acc, ::aie::load_v<64>(ring[y_p2 % 9]), bw2);
        acc = ::aie::mac(acc, ::aie::load_v<64>(ring[y_p3 % 9]), bw1);
        acc = ::aie::mac(acc, ::aie::load_v<64>(ring[y_p4 % 9]), bw0);
        ::aie::store_v(out_tile, acc.to_vector<bfloat16>());
    }
}
"""


def _align_up(val: int, mult: int) -> int:
    return ((val + mult - 1) // mult) * mult


def select_npu_bokeh_tiling(num_pixels: int) -> tuple[int, int] | None:
    """Return (4, 32) for 4-column AIE2 bokeh blend; any positive pixel count is supported via tail padding."""
    if num_pixels <= 0:
        return None
    return _BOKEH_NCOLS, _BOKEH_TILE_PIXELS


def _compile_dispatch_kernel(design: Any) -> Any:
    """Compile a specialized DispatchTime CallableDesign into an NPUKernel without dummy buffer warmup."""
    from aie.utils.npukernel import NPUKernel

    compilable = design.compilable
    xclbin_path, inst_path = compilable.compile()
    expected_sizes = compilable.expected_tensor_sizes
    num_host_bos = len(expected_sizes) if expected_sizes is not None else None
    return NPUKernel(
        xclbin_path,
        inst_path,
        kernel_name="MLIR_AIE",
        num_host_bos=num_host_bos,
        dispatch_params=compilable.dispatch_params,
        dispatch_lib_path=compilable.get_dispatch_lib_path(),
    )


def _can_direct_dispatch(runtime: Any, handle: Any) -> bool:
    return (
        getattr(runtime, "_device", None) is not None
        and getattr(handle, "kernel_run", None) is not None
        and hasattr(getattr(handle, "kernel", None), "group_id")
    )


def _dispatch_direct(
    runtime: Any,
    handle: Any,
    insts_bo_cache: dict[tuple[int, ...], tuple[Any, int, Any]],
    bo_key: tuple[int, ...],
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

    if hasattr(runtime, "_dispatched_context"):
        ctx = getattr(handle, "context", None)
        if ctx is not None and runtime._dispatched_context is not ctx:
            runtime._dispatched_context = ctx

    bo_entry = insts_bo_cache.get(bo_key)
    if bo_entry is None:
        insts_u8 = np.ascontiguousarray(insts).view(np.uint8).reshape(-1)
        insts_bytes = insts_u8.nbytes
        transport = XrtTransport(
            runtime._device,
            insts_bytes,
            pyxrt.bo.cacheable,
            handle.kernel.group_id(1),
        )
        transport.host_bytes[:] = insts_u8
        transport.to_device(0, insts_bytes)
        insts_bo = transport.handle(0, insts_bytes)
        bo_entry = (transport, insts_bytes, insts_bo)
        insts_bo_cache[bo_key] = bo_entry
    else:
        _, insts_bytes, insts_bo = bo_entry

    r, _ = handle.kernel_run.dispatch([3, insts_bo, insts_bytes, *buffers])
    if r != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
        raise HostRuntimeError(f"Kernel returned {str(r)}")


def _dispatch_npu_kernel(
    runtime: Any,
    kernel_obj: Any,
    handle: Any,
    insts_bo_cache: dict[tuple[int, ...], tuple[Any, int, Any]],
    bo_key: tuple[int, ...],
    args: list[Any],
    insts: np.ndarray,
) -> Any:
    """Execute one NPU dispatch with cached XrtTransport instruction BO, reloading handle once on failure."""
    from aie.utils.hostruntime.hostruntime import HostRuntimeError

    if handle is None or not getattr(handle, "_is_valid", False):
        insts_bo_cache.clear()
        handle = runtime.load(kernel_obj)

    if _can_direct_dispatch(runtime, handle):
        try:
            _dispatch_direct(runtime, handle, insts_bo_cache, bo_key, args, insts)
        except (HostRuntimeError, RuntimeError):
            insts_bo_cache.clear()
            handle = runtime.load(kernel_obj)
            _dispatch_direct(runtime, handle, insts_bo_cache, bo_key, args, insts)
        return handle

    try:
        runtime.run(handle, args, dispatch_insts=insts, only_if_loaded=True)
    except HostRuntimeError:
        insts_bo_cache.clear()
        handle = runtime.load(kernel_obj)
        runtime.run(handle, args, dispatch_insts=insts, only_if_loaded=True)
    return handle


@lru_cache(maxsize=4)
def _build_npu_composite_kernel(max_pixels: int) -> Any:
    """Compile and cache a 4-column DispatchTime IRON @jit AIE2 4-level portrait bokeh composite kernel."""
    import aie.iron as iron
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import (
        Buffer,
        DispatchTime,
        ExternalFunction,
        In,
        ObjectFifo,
        Out,
        Program,
        Runtime,
        Worker,
        WorkerRuntimeBarrier,
        require,
    )
    from aie.iron.controlflow import range_

    iron.ensure_current_device()
    dev = iron.get_current_device()
    if dev is None or dev.cols < _BOKEH_NCOLS:
        raise RuntimeError(f"Expected an AMD NPU with at least {_BOKEH_NCOLS} columns, got {dev!r}")

    @iron.jit
    def _bokeh_blend_kernel(
        in_a: In,
        in_b: In,
        out: Out,
        *,
        k_chunks: DispatchTime[np.int32],
    ):
        k_workers = _BOKEH_NCOLS
        k_tile_pixels = _BOKEH_TILE_PIXELS

        tile_a_ty = np.ndarray[(k_tile_pixels * 4,), np.dtype[np.uint8]]
        tile_b_ty = np.ndarray[(k_tile_pixels * 9,), np.dtype[np.uint8]]
        tile_o_ty = np.ndarray[(k_tile_pixels * 3,), np.dtype[np.uint8]]
        rtp_ty = np.ndarray[(2,), np.dtype[np.int32]]

        ext_fn = ExternalFunction(
            "lse_bokeh_blend4",
            source_string=_BOKEH_BLEND4_C_SRC,
            arg_types=[tile_a_ty, tile_b_ty, tile_o_ty, np.int32],
        )

        in_as = [ObjectFifo(tile_a_ty, name=f"bokeh_a{i}", depth=2) for i in range(k_workers)]
        in_bs = [ObjectFifo(tile_b_ty, name=f"bokeh_b{i}", depth=2) for i in range(k_workers)]
        outs = [ObjectFifo(tile_o_ty, name=f"bokeh_o{i}", depth=2) for i in range(k_workers)]
        rtps = [Buffer(rtp_ty, name=f"bokeh_rtp_{i}", use_write_rtp=True) for i in range(k_workers)]
        barriers = [WorkerRuntimeBarrier() for _ in range(k_workers)]

        def _make_worker(idx: int) -> Worker:
            def core_fn(fa, fb, fo, fn, rtp, barrier):
                barrier.wait_for_value(1)
                chunks = rtp[0]
                barrier.release_with_value(1)
                for _ in range_(chunks):
                    a = fa.acquire(1)
                    b = fb.acquire(1)
                    o = fo.acquire(1)
                    fn(a, b, o, k_tile_pixels)
                    fa.release(1)
                    fb.release(1)
                    fo.release(1)

            return Worker(
                core_fn,
                [in_as[idx].cons(), in_bs[idx].cons(), outs[idx].prod(), ext_fn, rtps[idx], barriers[idx]],
                stack_size=0xD00,
                dynamic_objfifo_lowering=True,
            )

        workers = [_make_worker(i) for i in range(k_workers)]
        a_prods = [f.prod() for f in in_as]
        b_prods = [f.prod() for f in in_bs]
        o_conss = [f.cons() for f in outs]

        def sequence(ha, hb, ho, k_chunks_d, ha_fifos, hb_fifos, ho_fifos):
            require(k_chunks_d > 0, "k_chunks must be positive")
            pa = k_chunks_d * (k_tile_pixels * 4)
            pb = k_chunks_d * (k_tile_pixels * 9)
            po = k_chunks_d * (k_tile_pixels * 3)
            require(pa * k_workers <= max_pixels * 4, "in_a exceeds max_pixels")
            require(pb * k_workers <= max_pixels * 9, "in_b exceeds max_pixels")
            require(po * k_workers <= max_pixels * 3, "out exceeds max_pixels")

            for i in range(k_workers):
                rtps[i][0] = k_chunks_d
                barriers[i].set(1)

            a_taps = TensorAccessPattern((max_pixels * 4,), offset=0, sizes=(k_workers, pa), strides=(pa, 1))
            b_taps = TensorAccessPattern((max_pixels * 9,), offset=0, sizes=(k_workers, pb), strides=(pb, 1))
            o_taps = TensorAccessPattern((max_pixels * 3,), offset=0, sizes=(k_workers, po), strides=(po, 1))

            for i in range(k_workers):
                ha_fifos[i].fill(ha, tap=a_taps[i])
                hb_fifos[i].fill(hb, tap=b_taps[i])
            for i in range(k_workers):
                ho_fifos[i].drain(ho, tap=o_taps[i], wait=True)

        rt = Runtime(sequence, [in_a, in_b, out, k_chunks, a_prods, b_prods, o_conss])
        return Program(dev, rt, workers=workers).resolve_program()

    specialized = _bokeh_blend_kernel.specialize(
        in_a=np.ndarray[(max_pixels * 4,), np.dtype[np.uint8]],
        in_b=np.ndarray[(max_pixels * 9,), np.dtype[np.uint8]],
        out=np.ndarray[(max_pixels * 3,), np.dtype[np.uint8]],
    )
    return _compile_dispatch_kernel(specialized)


@lru_cache(maxsize=4)
def _build_npu_blur_kernel(max_elems: int) -> Any:
    """Compile and cache a 4-column DispatchTime IRON @jit 2-mode separable Gaussian5 kernel."""
    import aie.iron as iron
    from aie.dialects import arith
    from aie.extras import types as T
    from aie.extras.dialects.arith import constant
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import (
        Buffer,
        DispatchTime,
        ExternalFunction,
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
    if dev is None or dev.cols < _BLUR_NCOLS:
        raise RuntimeError(f"Expected an AMD NPU with at least {_BLUR_NCOLS} columns, got {dev!r}")

    @iron.jit
    def _gaussian5_blur_kernel(
        buf_in: In,
        buf_out: Out,
        *,
        mode: DispatchTime[np.int32],
        outer_cnt: DispatchTime[np.int32],
        inner_cnt: DispatchTime[np.int32],
        outer_stride: DispatchTime[np.int32],
        inner_stride: DispatchTime[np.int32],
        col_stride: DispatchTime[np.int32],
        valid_tail: DispatchTime[np.int32],
    ):
        tile_ty = np.ndarray[(_BLUR_TILE_ELEMS,), np.dtype[bfloat16]]
        rtp_ty = np.ndarray[(4,), np.dtype[np.int32]]

        ext_fn = ExternalFunction(
            "lse_gaussian5_hw4_bf16",
            source_string=_GAUSSIAN5_HW4_BF16_C_SRC,
            arg_types=[tile_ty, tile_ty, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32],
        )

        in_fifos = [ObjectFifo(tile_ty, name=f"blur_in_{c}", depth=2) for c in range(_BLUR_NCOLS)]
        out_fifos = [ObjectFifo(tile_ty, name=f"blur_out_{c}", depth=2) for c in range(_BLUR_NCOLS)]
        dummies = [Buffer(tile_ty, name=f"blur_dummy_{c}") for c in range(_BLUR_NCOLS)]
        rtps = [Buffer(rtp_ty, name=f"blur_rtp_{c}", use_write_rtp=True) for c in range(_BLUR_NCOLS)]
        barriers = [WorkerRuntimeBarrier() for _ in range(_BLUR_NCOLS)]

        def _make_worker(col: int) -> Worker:
            def core_fn(in_f, out_f, dummy_tile, fn, rtp, barrier):
                barrier.wait_for_value(1)
                w_mode = rtp[0]
                w_outer = rtp[1]
                w_inner = rtp[2]
                w_tail = rtp[3]
                barrier.release_with_value(1)

                c0 = constant(0, type=T.i32())
                c1 = constant(1, type=T.i32())
                c2 = constant(2, type=T.i32())
                c3 = constant(3, type=T.i32())
                c4 = constant(4, type=T.i32())
                is_horiz = arith.cmpi(arith.CmpIPredicate.eq, arith.andi(w_mode, c1), c0)
                is_m3 = arith.cmpi(arith.CmpIPredicate.eq, w_mode, c3)
                v_look = arith.select(is_m3, c4, c2)
                lookahead = arith.select(is_horiz, c1, v_look)
                inner_m1 = arith.subi(w_inner, c1)
                p_lt = arith.cmpi(arith.CmpIPredicate.slt, lookahead, inner_m1)
                p = arith.select(p_lt, lookahead, inner_m1)
                base_tail = arith.subi(w_inner, p)

                for _ in range_(w_outer):
                    for i in range_(p):
                        i_i32 = arith.index_cast(T.i32(), i)
                        in_elem = in_f.acquire(1)
                        fn(in_elem, dummy_tile, w_mode, c0, i_i32, c0, w_inner, w_tail)
                        in_f.release(1)
                    for i in range_(p, w_inner):
                        i_i32 = arith.index_cast(T.i32(), i)
                        out_i32 = arith.subi(i_i32, p)
                        in_elem = in_f.acquire(1)
                        out_elem = out_f.acquire(1)
                        fn(in_elem, out_elem, w_mode, c1, i_i32, out_i32, w_inner, w_tail)
                        in_f.release(1)
                        out_f.release(1)
                    for i in range_(p):
                        i_i32 = arith.index_cast(T.i32(), i)
                        out_i32 = arith.addi(base_tail, i_i32)
                        out_elem = out_f.acquire(1)
                        fn(dummy_tile, out_elem, w_mode, c2, c0, out_i32, w_inner, w_tail)
                        out_f.release(1)

            return Worker(
                core_fn,
                [
                    in_fifos[col].cons(),
                    out_fifos[col].prod(),
                    dummies[col],
                    ext_fn,
                    rtps[col],
                    barriers[col],
                ],
                stack_size=0xD00,
                dynamic_objfifo_lowering=True,
            )

        workers = [_make_worker(c) for c in range(_BLUR_NCOLS)]
        in_prods = [f.prod() for f in in_fifos]
        out_conss = [f.cons() for f in out_fifos]

        def sequence(
            buf_in_h,
            buf_out_h,
            mode_d,
            outer_cnt_d,
            inner_cnt_d,
            outer_stride_d,
            inner_stride_d,
            col_stride_d,
            valid_tail_d,
            in_hs,
            out_hs,
        ):
            require(outer_cnt_d > 0, "outer_cnt must be positive")
            require(inner_cnt_d > 0, "inner_cnt must be positive")
            require(outer_stride_d > 0, "outer_stride must be positive")
            require(inner_stride_d > 0, "inner_stride must be positive")
            require(col_stride_d > 0, "col_stride must be positive")
            require(valid_tail_d > 0, "valid_tail must be positive")
            require(valid_tail_d <= _BLUR_TILE_W, "valid_tail must be <= TILE_W")

            for col in range(_BLUR_NCOLS):
                rtps[col][0] = mode_d
                rtps[col][1] = outer_cnt_d
                rtps[col][2] = inner_cnt_d
                rtps[col][3] = valid_tail_d
                barriers[col].set(1)

            col_taps = [
                TensorAccessPattern(
                    (max_elems,),
                    offset=col * col_stride_d,
                    sizes=(outer_cnt_d, inner_cnt_d, _BLUR_TILE_ELEMS),
                    strides=(outer_stride_d, inner_stride_d, 1),
                )
                for col in range(_BLUR_NCOLS)
            ]

            for step, tg in TaskGroup.pipelined(outer_cnt_d, depth=2):
                for col in range(_BLUR_NCOLS):
                    step_tap = col_taps[col][step]
                    out_hs[col].drain(buf_out_h, tap=step_tap, wait=True, group=tg)
                    in_hs[col].fill(buf_in_h, tap=step_tap, wait=True, group=tg)

        rt = Runtime(
            sequence,
            [
                buf_in,
                buf_out,
                mode,
                outer_cnt,
                inner_cnt,
                outer_stride,
                inner_stride,
                col_stride,
                valid_tail,
                in_prods,
                out_conss,
            ],
        )
        return Program(dev, rt, workers=workers).resolve_program()

    specialized = _gaussian5_blur_kernel.specialize(
        buf_in=np.ndarray[(max_elems,), np.dtype[bfloat16]],
        buf_out=np.ndarray[(max_elems,), np.dtype[bfloat16]],
    )
    return _compile_dispatch_kernel(specialized)


class PortraitBlurEffect(_CpuPortraitBlurEffect):
    """Runs streaming Video Depth Anything + AIE2 4-column blur pyramid and bokeh composite on NPU1."""

    effect_name = "portrait_blur"

    def __init__(
        self,
        *,
        use_npu_composite: bool = True,
        device: ComputeDevice | str = ComputeDevice.NPU,
        **kwargs: object,
    ) -> None:
        self.use_npu_composite = bool(use_npu_composite)
        self._npu_runtime: Any = None

        self._npu_handle: Any = None
        self._npu_kernel_obj: Any = None
        self._npu_max_pixels: int = 0
        self._npu_padded_pixels: int = 0
        self._npu_in_a: Any = None
        self._npu_in_b: Any = None
        self._npu_out: Any = None
        self._npu_insts_cache: dict[int, np.ndarray] = {}
        self._npu_insts_bo_cache: dict[tuple[int, ...], tuple[Any, int, Any]] = {}

        self._blur_handle: Any = None
        self._blur_kernel_obj: Any = None
        self._blur_max_elems: int = 0
        self._blur_buf_elems: int = 0
        self._blur_buf_a: Any = None
        self._blur_buf_b: Any = None
        self._blur_views_a: dict[int, Any] = {}
        self._blur_views_b: dict[int, Any] = {}
        self._blur_insts_cache: dict[tuple[int, ...], np.ndarray] = {}
        self._blur_insts_bo_cache: dict[tuple[int, ...], tuple[Any, int, Any]] = {}

        super().__init__(device=device, **kwargs)  # type: ignore[arg-type]

    def _release_npu_composite(self) -> None:
        self._npu_handle = None
        self._npu_kernel_obj = None
        self._npu_max_pixels = 0
        self._npu_padded_pixels = 0
        self._npu_in_a = None
        self._npu_in_b = None
        self._npu_out = None
        self._npu_insts_cache.clear()
        self._npu_insts_bo_cache.clear()

    def _release_npu_blur(self) -> None:
        self._blur_handle = None
        self._blur_kernel_obj = None
        self._blur_max_elems = 0
        self._blur_buf_elems = 0
        self._blur_buf_a = None
        self._blur_buf_b = None
        self._blur_views_a.clear()
        self._blur_views_b.clear()
        self._blur_insts_cache.clear()
        self._blur_insts_bo_cache.clear()

    def _ensure_blur_resources(self, width: int, height: int) -> None:
        import aie.iron as iron
        from aie.utils import DefaultNPURuntime

        h1, w1 = (height + 1) // 2, (width + 1) // 2
        h1_pad = _align_up(h1, _BLUR_NCOLS)
        n_strips1 = (w1 + _BLUR_TILE_W - 1) // _BLUR_TILE_W
        w1_pad = _align_up(n_strips1, _BLUR_NCOLS) * _BLUR_TILE_W
        blur_elems = h1_pad * w1_pad * 4
        max_elems = max(_MAX_BLUR_ELEMS, blur_elems)

        if self._blur_kernel_obj is None or max_elems > self._blur_max_elems:
            with NPU_DISPATCH_LOCK:
                kernel_obj = _build_npu_blur_kernel(max_elems)
                handle = DefaultNPURuntime.load(kernel_obj)
            self._npu_runtime = DefaultNPURuntime
            self._blur_kernel_obj = kernel_obj
            self._blur_handle = handle
            self._blur_max_elems = max_elems
            self._blur_insts_cache.clear()
            self._blur_insts_bo_cache.clear()

        if self._blur_buf_a is None or blur_elems != self._blur_buf_elems:
            self._blur_views_a.clear()
            self._blur_views_b.clear()
            self._blur_buf_a = iron.zeros(blur_elems, dtype=bfloat16, device="npu")
            self._blur_buf_b = iron.zeros(blur_elems, dtype=bfloat16, device="npu")
            self._blur_buf_elems = blur_elems

    def _ensure_composite_resources(self, width: int, height: int) -> None:
        if not self.use_npu_composite:
            self._release_npu_composite()
            return
        import aie.iron as iron
        from aie.utils import DefaultNPURuntime

        num_pixels = width * height
        padded_pixels = _align_up(num_pixels, _BOKEH_PIXELS_ALIGN)
        max_pixels = max(_MAX_BOKEH_PIXELS, padded_pixels)

        if self._npu_kernel_obj is None or max_pixels > self._npu_max_pixels:
            with NPU_DISPATCH_LOCK:
                kernel_obj = _build_npu_composite_kernel(max_pixels)
                handle = DefaultNPURuntime.load(kernel_obj)
            self._npu_runtime = DefaultNPURuntime
            self._npu_kernel_obj = kernel_obj
            self._npu_handle = handle
            self._npu_max_pixels = max_pixels
            self._npu_insts_cache.clear()
            self._npu_insts_bo_cache.clear()

        if self._npu_in_a is None or padded_pixels != self._npu_padded_pixels:
            self._npu_in_a = iron.zeros(padded_pixels * 4, dtype=np.uint8, device="npu")
            self._npu_in_b = iron.zeros(padded_pixels * 9, dtype=np.uint8, device="npu")
            self._npu_out = iron.zeros(padded_pixels * 3, dtype=np.uint8, device="npu")
            self._npu_padded_pixels = padded_pixels

    def _on_plans_resized(self, width: int, height: int) -> None:
        self._ensure_blur_resources(width, height)
        self._ensure_composite_resources(width, height)

    def _get_blur_insts(self, *params: int) -> np.ndarray:
        insts = self._blur_insts_cache.get(params)
        if insts is None:
            mode, outer_cnt, inner_cnt, outer_stride, inner_stride, col_stride, valid_tail = params
            insts = self._blur_kernel_obj.instructions(
                mode=mode,
                outer_cnt=outer_cnt,
                inner_cnt=inner_cnt,
                outer_stride=outer_stride,
                inner_stride=inner_stride,
                col_stride=col_stride,
                valid_tail=valid_tail,
            )
            self._blur_insts_cache[params] = insts
        return insts

    def _get_bokeh_insts(self, chunks: int) -> np.ndarray:
        insts = self._npu_insts_cache.get(chunks)
        if insts is None:
            insts = self._npu_kernel_obj.instructions(k_chunks=chunks)
            self._npu_insts_cache[chunks] = insts
        return insts

    def _get_blur_views(self, level_elems: int) -> tuple[Any, Any]:
        if level_elems == self._blur_buf_elems:
            return self._blur_buf_a, self._blur_buf_b
        va = self._blur_views_a.get(level_elems)
        vb = self._blur_views_b.get(level_elems)
        if va is None or vb is None:
            va = self._blur_buf_a.subview(0, (level_elems,))
            vb = self._blur_buf_b.subview(0, (level_elems,))
            self._blur_views_a[level_elems] = va
            self._blur_views_b[level_elems] = vb
        return va, vb

    def _gaussian5_npu(self, arr: np.ndarray, passes: int = 1) -> np.ndarray:
        """Separable 5-tap [1,4,6,4,1]/16 Gaussian blur on (H, W, 4) float32 executed fully on NPU1."""
        if arr.ndim != 3 or arr.shape[2] != 4:
            raise ValueError(f"Expected (H, W, 4) array, got {arr.shape}")
        h, w, _ = arr.shape
        if passes <= 0:
            return np.ascontiguousarray(arr, dtype=np.float32).copy()
        if self._blur_kernel_obj is None or self._blur_buf_a is None or self._blur_buf_b is None:
            raise RuntimeError("NPU Gaussian blur kernel is not initialized")

        n_strips = (w + _BLUR_TILE_W - 1) // _BLUR_TILE_W
        n_strips_pad = _align_up(n_strips, _BLUR_NCOLS)
        w_pad = n_strips_pad * _BLUR_TILE_W
        h_pad = _align_up(h, _BLUR_NCOLS)
        level_elems = h_pad * w_pad * 4
        if level_elems > self._blur_buf_elems:
            raise ValueError(
                f"Blur level ({h}, {w}) padded to {level_elems} elements exceeds allocated buffer "
                f"capacity {self._blur_buf_elems}"
            )

        buf_a_view, buf_b_view = self._get_blur_views(level_elems)
        arr_c = np.ascontiguousarray(arr, dtype=np.float32)
        with buf_a_view.overwrite() as dst_bf16:
            dst_2d = dst_bf16.view(np.uint16).reshape(h_pad, w_pad * 4)
            host_ops.f32_to_bf16_pad(arr_c.reshape(h, w * 4), dst_2d, h, w * 4, zero_tail=True)

        m0_params = (
            0,
            h_pad // _BLUR_NCOLS,
            n_strips,
            w_pad * 4,
            _BLUR_TILE_ELEMS,
            (h_pad // _BLUR_NCOLS) * w_pad * 4,
            w - (n_strips - 1) * _BLUR_TILE_W,
        )
        m1_params = (
            1,
            n_strips_pad // _BLUR_NCOLS,
            h,
            _BLUR_TILE_ELEMS,
            w_pad * 4,
            (n_strips_pad // _BLUR_NCOLS) * _BLUR_TILE_ELEMS,
            _BLUR_TILE_W,
        )
        rem = int(passes)
        if rem >= 2:
            m2_params = (2, *m0_params[1:])
            m3_params = (3, *m1_params[1:])
            insts_h2 = self._get_blur_insts(*m2_params)
            insts_v2 = self._get_blur_insts(*m3_params)
            while rem >= 2:
                with NPU_DISPATCH_LOCK:
                    self._blur_handle = _dispatch_npu_kernel(
                        self._npu_runtime,
                        self._blur_kernel_obj,
                        self._blur_handle,
                        self._blur_insts_bo_cache,
                        m2_params,
                        [buf_a_view, buf_b_view],
                        insts_h2,
                    )
                with NPU_DISPATCH_LOCK:
                    self._blur_handle = _dispatch_npu_kernel(
                        self._npu_runtime,
                        self._blur_kernel_obj,
                        self._blur_handle,
                        self._blur_insts_bo_cache,
                        m3_params,
                        [buf_b_view, buf_a_view],
                        insts_v2,
                    )
                rem -= 2
        if rem == 1:
            insts_h = self._get_blur_insts(*m0_params)
            insts_v = self._get_blur_insts(*m1_params)
            with NPU_DISPATCH_LOCK:
                self._blur_handle = _dispatch_npu_kernel(
                    self._npu_runtime,
                    self._blur_kernel_obj,
                    self._blur_handle,
                    self._blur_insts_bo_cache,
                    m0_params,
                    [buf_a_view, buf_b_view],
                    insts_h,
                )
            with NPU_DISPATCH_LOCK:
                self._blur_handle = _dispatch_npu_kernel(
                    self._npu_runtime,
                    self._blur_kernel_obj,
                    self._blur_handle,
                    self._blur_insts_bo_cache,
                    m1_params,
                    [buf_b_view, buf_a_view],
                    insts_v,
                )

        src_2d = buf_a_view.numpy().view(np.uint16).reshape(h_pad, w_pad * 4)
        out_f32 = np.empty((h, w, 4), dtype=np.float32)
        host_ops.bf16_to_f32_unpad(src_2d, out_f32.reshape(h, w * 4), h, w * 4)
        return out_f32

    def apply_depth_bokeh(
        self,
        frame: np.ndarray,
        depth_raw: np.ndarray,
        *,
        pixel_format: str | None = None,
    ) -> np.ndarray:
        """Render variable portrait bokeh on NPU with live render priority over async depth."""
        with render_priority():
            return super().apply_depth_bokeh(frame, depth_raw, pixel_format=pixel_format)

    def _build_blur_pyramid(
        self,
        frame: np.ndarray,
        coc_unscaled: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self._plan_pyr_to_full is not None
        p1, p2, p3 = self._plan_pyr_to_full
        passes_1, passes_2, passes_3 = self.blur_passes

        # Keep a tiny 1e-3 weight on foreground pixels so normalized convolution suppresses
        # foreground halo bleed while gracefully falling back inside solid foreground regions.
        bg_mask = np.clip((coc_unscaled - np.float32(0.02)) * np.float32(self.halo_suppression), 0.0, 1.0)
        w0 = bg_mask * np.float32(1.0 - 1e-3) + np.float32(1e-3)

        s1 = self._gaussian5_npu(host_ops.downsample2x_bg_hw4(frame, w0), passes=passes_1)
        l1 = host_ops.unpremul_resize_hw4_to_rgb_u8(s1, p1.out_h, p1.out_w)

        s2 = self._gaussian5_npu(host_ops.downsample2x_hw4(s1), passes=passes_2)
        l2 = host_ops.unpremul_resize_hw4_to_rgb_u8(s2, p2.out_h, p2.out_w)

        s3 = self._gaussian5_npu(host_ops.downsample2x_hw4(host_ops.downsample2x_hw4(s2)), passes=passes_3)
        l3 = host_ops.unpremul_resize_hw4_to_rgb_u8(s3, p3.out_h, p3.out_w)

        return l1, l2, l3

    def _composite_bokeh(
        self,
        l0: np.ndarray,
        l1: np.ndarray,
        l2: np.ndarray,
        l3: np.ndarray,
        coc_u8: np.ndarray,
    ) -> np.ndarray:
        if not self.use_npu_composite:
            return super()._composite_bokeh(l0, l1, l2, l3, coc_u8)
        if self._npu_kernel_obj is None or self._npu_in_a is None or self._npu_in_b is None or self._npu_out is None:
            raise RuntimeError("NPU bokeh composite kernel is not initialized")

        h, w, _ = l0.shape
        num_pixels = h * w
        padded_pixels = _align_up(num_pixels, _BOKEH_PIXELS_ALIGN)
        chunks = padded_pixels // _BOKEH_PIXELS_ALIGN

        if padded_pixels > self._npu_padded_pixels:
            self._ensure_composite_resources(w, h)

        if padded_pixels == self._npu_padded_pixels:
            in_a_view = self._npu_in_a
            in_b_view = self._npu_in_b
            out_view = self._npu_out
        else:
            in_a_view = self._npu_in_a.subview(0, (padded_pixels * 4,))
            in_b_view = self._npu_in_b.subview(0, (padded_pixels * 9,))
            out_view = self._npu_out.subview(0, (padded_pixels * 3,))

        with in_a_view.overwrite() as dst_a:
            with in_b_view.overwrite() as dst_b:
                host_ops.pack_bokeh_npu_inputs_soa32(l0, coc_u8, l1, l2, l3, dst_a, dst_b, padded_pixels)

        insts = self._get_bokeh_insts(chunks)
        with NPU_DISPATCH_LOCK:
            self._npu_handle = _dispatch_npu_kernel(
                self._npu_runtime,
                self._npu_kernel_obj,
                self._npu_handle,
                self._npu_insts_bo_cache,
                (chunks,),
                [in_a_view, in_b_view, out_view],
                insts,
            )
        return out_view.numpy()[: num_pixels * 3].reshape(h, w, 3).copy()

    def teardown(self) -> None:
        self._release_npu_composite()
        self._release_npu_blur()
        self._npu_runtime = None
        super().teardown()
