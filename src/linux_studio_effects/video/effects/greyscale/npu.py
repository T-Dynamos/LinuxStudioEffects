"""NPU-accelerated RGB -> Greyscale (BT.601) video effect using MLIR-AIE / IRON."""

from functools import lru_cache
from typing import Any

import numpy as np

from ..base import ComputeDevice, FrameContext, FrameEffect

_RGB2GRAY_C_SRC = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" void lse_rgb2gray_rgb(const uint8_t *__restrict in, uint8_t *__restrict out, int32_t num_pixels) {
    ::aie::vector<uint8_t, 32> r, g, b;
    ::aie::accum<acc32, 32> rnd;
    rnd.from_vector(::aie::broadcast<int32_t, 32>(128));
    const int steps = num_pixels / 32;
    _Pragma("clang loop unroll(disable)")
    for (int s = 0; s < steps; ++s) {
        const uint8_t *p = in + s * 96;
        for (int k = 0; k < 32; ++k) {
            r[k] = p[k * 3 + 0];
            g[k] = p[k * 3 + 1];
            b[k] = p[k * 3 + 2];
        }
        ::aie::accum<acc32, 32> acc =
            ::aie::mac(::aie::mac(::aie::mac(rnd, r, (int16_t)77), g, (int16_t)150), b, (int16_t)29);
        ::aie::vector<uint8_t, 32> y = acc.template to_vector<uint8_t>(8);
        uint8_t *q = out + s * 96;
        for (int k = 0; k < 32; ++k) {
            uint8_t v = y[k];
            q[k * 3 + 0] = v;
            q[k * 3 + 1] = v;
            q[k * 3 + 2] = v;
        }
    }
    for (int i = steps * 32; i < num_pixels; ++i) {
        int32_t rv = in[i * 3 + 0];
        int32_t gv = in[i * 3 + 1];
        int32_t bv = in[i * 3 + 2];
        uint8_t yv = (uint8_t)((77 * rv + 150 * gv + 29 * bv + 128) >> 8);
        out[i * 3 + 0] = yv;
        out[i * 3 + 1] = yv;
        out[i * 3 + 2] = yv;
    }
}
"""

_TILE_PIXEL_CANDIDATES = (2560, 1280, 640, 320, 160, 64, 32, 16, 4, 1)


def select_npu_tiling(num_pixels: int) -> tuple[int, int]:
    """Choose (num_workers, tile_pixels) that evenly partitions num_pixels across AIE columns."""
    if num_pixels <= 0:
        raise ValueError(f"num_pixels must be positive, got {num_pixels}")
    for workers in (4, 2, 1):
        for tile_pixels in _TILE_PIXEL_CANDIDATES:
            if num_pixels % (workers * tile_pixels) == 0:
                return workers, tile_pixels
    return 1, 1


@lru_cache(maxsize=16)
def _build_greyscale_kernel(total_bytes: int, num_workers: int, tile_pixels: int) -> Any:
    """Compile and cache an IRON @jit RGB->Greyscale kernel for a given frame geometry."""
    import aie.iron as iron
    from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
    from aie.iron.controlflow import range_

    iron.ensure_current_device()
    tile_bytes = tile_pixels * 3
    chunks = total_bytes // (num_workers * tile_bytes)

    @iron.jit
    def _rgb2gray_kernel(
        inp: In,
        out: Out,
        *,
        k_total_bytes: CompileTime[int] = total_bytes,
        k_workers: CompileTime[int] = num_workers,
        k_tile_pixels: CompileTime[int] = tile_pixels,
        k_tile_bytes: CompileTime[int] = tile_bytes,
        k_chunks: CompileTime[int] = chunks,
    ):
        full_ty = np.ndarray[(k_total_bytes,), np.dtype[np.uint8]]
        tile_ty = np.ndarray[(k_tile_bytes,), np.dtype[np.uint8]]

        ext_fn = ExternalFunction(
            "lse_rgb2gray_rgb",
            source_string=_RGB2GRAY_C_SRC,
            arg_types=[tile_ty, tile_ty, np.int32],
        )

        ins = [ObjectFifo(tile_ty, name=f"in{i}") for i in range(k_workers)]
        outs = [ObjectFifo(tile_ty, name=f"out{i}") for i in range(k_workers)]

        def _make_worker(idx: int) -> Worker:
            def core_fn(in_f, out_f, fn):
                for _ in range_(k_chunks):
                    a = in_f.acquire(1)
                    b = out_f.acquire(1)
                    fn(a, b, k_tile_pixels)
                    in_f.release(1)
                    out_f.release(1)

            return Worker(core_fn, [ins[idx].cons(), outs[idx].prod(), ext_fn])

        workers = [_make_worker(i) for i in range(k_workers)]

        def sequence(a_host, b_host, *fifos):
            in_prods = fifos[:k_workers]
            out_conss = fifos[k_workers:]
            part = k_total_bytes // k_workers
            for i in range(k_workers):
                s = i * part
                in_prods[i].fill(a_host, tap=a_host[s : s + part])
            for i in range(k_workers):
                s = i * part
                out_conss[i].drain(b_host, tap=b_host[s : s + part], wait=True)

        rt = Runtime(sequence, [full_ty, full_ty, *[f.prod() for f in ins], *[f.cons() for f in outs]])
        return Program(iron.get_current_device(), rt, workers=workers).resolve_program()

    return _rgb2gray_kernel


class GreyscaleEffect(FrameEffect):
    """Convert RGB video frames to BT.601 greyscale RGB on the AMD Ryzen AI NPU."""

    supported_devices = (ComputeDevice.NPU,)
    effect_name = "greyscale"

    def __init__(self, *, name: str | None = None, device: ComputeDevice | str = ComputeDevice.NPU, enabled: bool = True) -> None:
        super().__init__(name=name, device=device, enabled=enabled)
        self._shape: tuple[int, int] | None = None
        self._total_bytes: int = 0
        self._num_workers: int = 1
        self._tile_pixels: int = 1
        self._kernel: Any = None
        self._runtime: Any = None
        self._handle: Any = None
        self._in_tensor: Any = None
        self._out_tensor: Any = None

    def setup(self, width: int, height: int, fps: float) -> None:
        import aie.iron as iron
        from aie.utils import DefaultNPURuntime

        super().setup(width, height, fps)
        if self._shape == (height, width) and self._handle is not None and self._in_tensor is not None:
            return

        num_pixels = width * height
        total_bytes = num_pixels * 3
        workers, tile_pixels = select_npu_tiling(num_pixels)
        kernel = _build_greyscale_kernel(total_bytes, workers, tile_pixels)

        in_tensor = iron.zeros(total_bytes, dtype=np.uint8, device="npu")
        out_tensor = iron.zeros(total_bytes, dtype=np.uint8, device="npu")
        # Warm up JIT compilation upfront during setup rather than on frame 1.
        kernel(
            in_tensor,
            out_tensor,
            k_total_bytes=total_bytes,
            k_workers=workers,
            k_tile_pixels=tile_pixels,
            k_tile_bytes=tile_pixels * 3,
            k_chunks=total_bytes // (workers * tile_pixels * 3),
        )
        npu_kernel = next(iter(kernel._kernel_cache.values()))
        handle = DefaultNPURuntime.load(npu_kernel)

        self._shape = (height, width)
        self._total_bytes = total_bytes
        self._num_workers = workers
        self._tile_pixels = tile_pixels
        self._kernel = kernel
        self._runtime = DefaultNPURuntime
        self._handle = handle
        self._in_tensor = in_tensor
        self._out_tensor = out_tensor

    def process(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) RGB frame, got {frame.shape}")
        h, w, _ = frame.shape
        if self._shape != (h, w) or self._handle is None or self._in_tensor is None or self._out_tensor is None:
            self.setup(w, h, ctx.fps if ctx else 30.0)

        flat = np.ascontiguousarray(frame, dtype=np.uint8).reshape(-1)
        with self._in_tensor.overwrite() as dst:
            np.copyto(dst, flat)

        self._runtime.run(self._handle, [self._in_tensor, self._out_tensor])
        return self._out_tensor.numpy().reshape(h, w, 3).copy()

    def teardown(self) -> None:
        self._handle = None
        self._runtime = None
        self._kernel = None
        self._in_tensor = None
        self._out_tensor = None
        self._shape = None
        self._total_bytes = 0
        super().teardown()
