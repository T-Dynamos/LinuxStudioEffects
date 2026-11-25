"""AMD Ryzen AI NPU depth-aware portrait bokeh blur backed by Video Depth Anything."""

from functools import lru_cache
from typing import Any

import numpy as np

from ...models.video_depth_anything.npu_ops import NPU_DISPATCH_LOCK
from ..base import ComputeDevice
from .cpu import PortraitBlurEffect as _CpuPortraitBlurEffect

_BOKEH_NCOLS = 4
_BOKEH_TILE_PIXEL_CANDIDATES = (1280, 640, 320, 160, 64, 32)

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
        const uint8_t *pa = in_a + s * 128;
        const uint8_t *pb = in_b + s * 288;
        ::aie::vector<int16_t, 32> r0, g0, b0, r1, g1, b1, r2, g2, b2, r3, g3, b3, coc;
        for (int k = 0; k < 32; ++k) {
            r0[k] = pa[k * 4 + 0];
            g0[k] = pa[k * 4 + 1];
            b0[k] = pa[k * 4 + 2];
            coc[k] = pa[k * 4 + 3];
            r1[k] = pb[k * 9 + 0];
            g1[k] = pb[k * 9 + 1];
            b1[k] = pb[k * 9 + 2];
            r2[k] = pb[k * 9 + 3];
            g2[k] = pb[k * 9 + 4];
            b2[k] = pb[k * 9 + 5];
            r3[k] = pb[k * 9 + 6];
            g3[k] = pb[k * 9 + 7];
            b3[k] = pb[k * 9 + 8];
        }
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

        uint8_t *q = out + s * 96;
        for (int k = 0; k < 32; ++k) {
            q[k * 3 + 0] = orv[k];
            q[k * 3 + 1] = ogv[k];
            q[k * 3 + 2] = obv[k];
        }
    }
    for (int i = steps * 32; i < num_pixels; ++i) {
        int16_t c = in_a[i * 4 + 3];
        int16_t w0 = 0, w1 = 0, w2 = 0, w3 = 0;
        if (c < 85) {
            w1 = c * 3;
            w0 = 256 - w1;
        } else if (c < 170) {
            w2 = (c - 85) * 3;
            w1 = 256 - w2;
        } else {
            int16_t d = c - 170;
            w3 = d * 3 + (d >> 6);
            w2 = 256 - w3;
        }
        for (int ch = 0; ch < 3; ++ch) {
            int32_t acc = 128 + in_a[i * 4 + ch] * w0 + in_b[i * 9 + ch] * w1
                          + in_b[i * 9 + 3 + ch] * w2 + in_b[i * 9 + 6 + ch] * w3;
            out[i * 3 + ch] = (uint8_t)(acc >> 8);
        }
    }
}
"""


def select_npu_bokeh_tiling(num_pixels: int) -> tuple[int, int] | None:
    """Return (4, tile_pixels) for 4-column AIE2 bokeh blend, or None if unaligned."""
    if num_pixels <= 0:
        return None
    for tile_pixels in _BOKEH_TILE_PIXEL_CANDIDATES:
        if num_pixels % (_BOKEH_NCOLS * tile_pixels) == 0:
            return _BOKEH_NCOLS, tile_pixels
    return None


@lru_cache(maxsize=4)
def _build_npu_composite_kernel(num_pixels: int, num_workers: int, tile_pixels: int) -> Any:
    """Compile and cache a 4-column IRON @jit AIE2 4-level portrait bokeh composite kernel."""
    import aie.iron as iron
    from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
    from aie.iron.controlflow import range_

    iron.ensure_current_device()
    chunks = num_pixels // (num_workers * tile_pixels)

    @iron.jit
    def _bokeh_blend_kernel(
        in_a: In,
        in_b: In,
        out: Out,
        *,
        k_pixels: CompileTime[int] = num_pixels,
        k_workers: CompileTime[int] = num_workers,
        k_tile_pixels: CompileTime[int] = tile_pixels,
        k_chunks: CompileTime[int] = chunks,
    ):
        full_a_ty = np.ndarray[(k_pixels * 4,), np.dtype[np.uint8]]
        full_b_ty = np.ndarray[(k_pixels * 9,), np.dtype[np.uint8]]
        full_o_ty = np.ndarray[(k_pixels * 3,), np.dtype[np.uint8]]

        tile_a_ty = np.ndarray[(k_tile_pixels * 4,), np.dtype[np.uint8]]
        tile_b_ty = np.ndarray[(k_tile_pixels * 9,), np.dtype[np.uint8]]
        tile_o_ty = np.ndarray[(k_tile_pixels * 3,), np.dtype[np.uint8]]

        ext_fn = ExternalFunction(
            "lse_bokeh_blend4",
            source_string=_BOKEH_BLEND4_C_SRC,
            arg_types=[tile_a_ty, tile_b_ty, tile_o_ty, np.int32],
        )

        in_as = [ObjectFifo(tile_a_ty, name=f"bokeh_a{i}") for i in range(k_workers)]
        in_bs = [ObjectFifo(tile_b_ty, name=f"bokeh_b{i}") for i in range(k_workers)]
        outs = [ObjectFifo(tile_o_ty, name=f"bokeh_o{i}") for i in range(k_workers)]

        def _make_worker(idx: int) -> Worker:
            def core_fn(fa, fb, fo, fn):
                for _ in range_(k_chunks):
                    a = fa.acquire(1)
                    b = fb.acquire(1)
                    o = fo.acquire(1)
                    fn(a, b, o, k_tile_pixels)
                    fa.release(1)
                    fb.release(1)
                    fo.release(1)

            return Worker(core_fn, [in_as[idx].cons(), in_bs[idx].cons(), outs[idx].prod(), ext_fn])

        workers = [_make_worker(i) for i in range(k_workers)]

        def sequence(ha, hb, ho, *fifos):
            pa = (k_pixels * 4) // k_workers
            pb = (k_pixels * 9) // k_workers
            po = (k_pixels * 3) // k_workers
            a_prods = fifos[:k_workers]
            b_prods = fifos[k_workers : 2 * k_workers]
            o_conss = fifos[2 * k_workers :]
            for i in range(k_workers):
                a_prods[i].fill(ha, tap=ha[i * pa : (i + 1) * pa])
                b_prods[i].fill(hb, tap=hb[i * pb : (i + 1) * pb])
            for i in range(k_workers):
                o_conss[i].drain(ho, tap=ho[i * po : (i + 1) * po], wait=True)

        rt = Runtime(
            sequence,
            [
                full_a_ty,
                full_b_ty,
                full_o_ty,
                *[f.prod() for f in in_as],
                *[f.prod() for f in in_bs],
                *[f.cons() for f in outs],
            ],
        )
        return Program(iron.get_current_device(), rt, workers=workers).resolve_program()

    return _bokeh_blend_kernel


class PortraitBlurEffect(_CpuPortraitBlurEffect):
    """Runs streaming Video Depth Anything + AIE2 4-column portrait bokeh composite on NPU1."""

    effect_name = "portrait_blur"

    def __init__(
        self,
        *,
        use_npu_composite: bool = False,
        device: ComputeDevice | str = ComputeDevice.NPU,
        **kwargs: object,
    ) -> None:
        self.use_npu_composite = bool(use_npu_composite)
        self._npu_runtime: Any = None
        self._npu_handle: Any = None
        self._npu_kernel: Any = None
        self._npu_kernel_obj: Any = None
        self._npu_in_a: Any = None
        self._npu_in_b: Any = None
        self._npu_out: Any = None
        super().__init__(device=device, **kwargs)  # type: ignore[arg-type]

    def _release_npu_composite(self) -> None:
        self._npu_handle = None
        self._npu_runtime = None
        self._npu_kernel = None
        self._npu_kernel_obj = None
        self._npu_in_a = None
        self._npu_in_b = None
        self._npu_out = None

    def _on_plans_resized(self, width: int, height: int) -> None:
        self._release_npu_composite()
        if not self.use_npu_composite:
            return
        num_pixels = width * height
        tiling = select_npu_bokeh_tiling(num_pixels)
        if tiling is None:
            return
        workers, tile_pixels = tiling
        import aie.iron as iron
        from aie.utils import DefaultNPURuntime

        with NPU_DISPATCH_LOCK:
            kernel = _build_npu_composite_kernel(num_pixels, workers, tile_pixels)
            in_a = iron.zeros(num_pixels * 4, dtype=np.uint8, device="npu")
            in_b = iron.zeros(num_pixels * 9, dtype=np.uint8, device="npu")
            out = iron.zeros(num_pixels * 3, dtype=np.uint8, device="npu")
            kernel(
                in_a,
                in_b,
                out,
                k_pixels=num_pixels,
                k_workers=workers,
                k_tile_pixels=tile_pixels,
                k_chunks=num_pixels // (workers * tile_pixels),
            )
            npu_kernel = next(iter(kernel._kernel_cache.values()))
            handle = DefaultNPURuntime.load(npu_kernel)

        self._npu_runtime = DefaultNPURuntime
        self._npu_kernel = kernel
        self._npu_kernel_obj = npu_kernel
        self._npu_handle = handle
        self._npu_in_a = in_a
        self._npu_in_b = in_b
        self._npu_out = out

    def _composite_bokeh(
        self,
        l0: np.ndarray,
        l1: np.ndarray,
        l2: np.ndarray,
        l3: np.ndarray,
        coc_u8: np.ndarray,
    ) -> np.ndarray:
        if self._npu_kernel_obj is None or self._npu_in_a is None:
            return super()._composite_bokeh(l0, l1, l2, l3, coc_u8)
        from aie.utils.hostruntime.hostruntime import HostRuntimeError

        h, w, _ = l0.shape
        num_pixels = h * w
        args = [self._npu_in_a, self._npu_in_b, self._npu_out]
        with NPU_DISPATCH_LOCK:
            with self._npu_in_a.overwrite() as dst_a:
                va = dst_a.reshape(num_pixels, 4)
                va[:, :3] = l0.reshape(num_pixels, 3)
                va[:, 3] = coc_u8.reshape(num_pixels)
            with self._npu_in_b.overwrite() as dst_b:
                vb = dst_b.reshape(num_pixels, 9)
                vb[:, 0:3] = l1.reshape(num_pixels, 3)
                vb[:, 3:6] = l2.reshape(num_pixels, 3)
                vb[:, 6:9] = l3.reshape(num_pixels, 3)
            if self._npu_handle is None or not getattr(self._npu_handle, "_is_valid", False):
                self._npu_handle = self._npu_runtime.load(self._npu_kernel_obj)
            try:
                self._npu_runtime.run(self._npu_handle, args, only_if_loaded=True)
            except HostRuntimeError:
                self._npu_handle = self._npu_runtime.load(self._npu_kernel_obj)
                self._npu_runtime.run(self._npu_handle, args, only_if_loaded=True)
            return self._npu_out.numpy().reshape(h, w, 3).copy()

    def teardown(self) -> None:
        self._release_npu_composite()
        super().teardown()
