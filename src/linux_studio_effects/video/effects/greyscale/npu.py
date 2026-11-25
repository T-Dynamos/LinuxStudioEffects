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

_BGR2GRAY_C_SRC = _RGB2GRAY_C_SRC.replace(
    "lse_rgb2gray_rgb", "lse_rgb2gray_bgr"
).replace("p[k * 3 + 0];", "p[k * 3 + TEMP];", 1).replace(
    "p[k * 3 + 2];", "p[k * 3 + 0];", 1
).replace("p[k * 3 + TEMP];", "p[k * 3 + 2];", 1)
# The fallback loop uses the same channel order as the vector loop.
_BGR2GRAY_C_SRC = _BGR2GRAY_C_SRC.replace(
    "in[i * 3 + 0];", "in[i * 3 + TEMP];", 1
).replace("in[i * 3 + 2];", "in[i * 3 + 0];", 1).replace("in[i * 3 + TEMP];", "in[i * 3 + 2];", 1)

_YUYV2GRAY_C_SRC = r"""
#include <stdint.h>
extern "C" void lse_yuyv2gray(uint8_t *__restrict in, uint8_t *__restrict out, int32_t num_pairs) {
    for (int i = 0; i < num_pairs; ++i) {
        out[i * 4 + 0] = in[i * 4 + 0]; out[i * 4 + 1] = 128;
        out[i * 4 + 2] = in[i * 4 + 2]; out[i * 4 + 3] = 128;
    }
}
"""

_UYVY2GRAY_C_SRC = r"""
#include <stdint.h>
extern "C" void lse_uyvy2gray(uint8_t *__restrict in, uint8_t *__restrict out, int32_t num_pairs) {
    for (int i = 0; i < num_pairs; ++i) {
        out[i * 4 + 0] = 128; out[i * 4 + 1] = in[i * 4 + 1];
        out[i * 4 + 2] = 128; out[i * 4 + 3] = in[i * 4 + 3];
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


@lru_cache(maxsize=32)
def _build_greyscale_kernel(total_bytes: int, num_workers: int, tile_pixels: int, pixel_format: str = "RGB3") -> Any:
    """Compile and cache an IRON greyscale kernel for native RGB or packed YUV422."""
    import aie.iron as iron
    from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
    from aie.iron.controlflow import range_

    iron.ensure_current_device()
    fmt = pixel_format.upper()
    yuv422 = fmt in {"YUYV", "UYVY"}
    unit_bytes = 4 if yuv422 else 3
    tile_bytes = tile_pixels * unit_bytes
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
            "lse_yuyv2gray"
            if fmt == "YUYV"
            else "lse_uyvy2gray"
            if fmt == "UYVY"
            else "lse_rgb2gray_bgr"
            if fmt == "BGR3"
            else "lse_rgb2gray_rgb",
            source_string=(
                _YUYV2GRAY_C_SRC
                if fmt == "YUYV"
                else _UYVY2GRAY_C_SRC
                if fmt == "UYVY"
                else _BGR2GRAY_C_SRC
                if fmt == "BGR3"
                else _RGB2GRAY_C_SRC
            ),
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
    """Greyscale RGB or neutral-chroma YUV422 frames on the AMD Ryzen AI NPU."""

    effect_name = "greyscale"

    def __init__(self, *, name: str | None = None, device: ComputeDevice | str = ComputeDevice.NPU, enabled: bool = True) -> None:
        super().__init__(name=name, device=device, enabled=enabled)
        self._shape: tuple[int, int, str] | None = None
        self._total_bytes: int = 0
        self._num_workers: int = 1
        self._tile_pixels: int = 1
        self._kernel: Any = None
        self._runtime: Any = None
        self._handle: Any = None
        self._in_tensor: Any = None
        self._out_tensor: Any = None

    def setup(self, width: int, height: int, fps: float, *, pixel_format: str | None = None) -> None:
        super().setup(width, height, fps)
        if pixel_format is None:
            return
        import aie.iron as iron
        from aie.utils import DefaultNPURuntime

        fmt = pixel_format.upper()
        if self._shape == (height, width, fmt) and self._handle is not None and self._in_tensor is not None:
            return

        num_units = width * height // 2 if fmt in {"YUYV", "UYVY"} else width * height
        unit_bytes = 4 if fmt in {"YUYV", "UYVY"} else 3
        total_bytes = num_units * unit_bytes
        workers, tile_pixels = select_npu_tiling(num_units)
        kernel = _build_greyscale_kernel(total_bytes, workers, tile_pixels, fmt)

        in_tensor = iron.zeros(total_bytes, dtype=np.uint8, device="npu")
        out_tensor = iron.zeros(total_bytes, dtype=np.uint8, device="npu")
        # Warm up JIT compilation upfront during setup rather than on frame 1.
        kernel(
            in_tensor,
            out_tensor,
            k_total_bytes=total_bytes,
            k_workers=workers,
            k_tile_pixels=tile_pixels,
            k_tile_bytes=tile_pixels * unit_bytes,
            k_chunks=total_bytes // (workers * tile_pixels * unit_bytes),
        )
        npu_kernel = next(iter(kernel._kernel_cache.values()))
        handle = DefaultNPURuntime.load(npu_kernel)

        self._shape = (height, width, fmt)
        self._total_bytes = total_bytes
        self._num_workers = workers
        self._tile_pixels = tile_pixels
        self._kernel = kernel
        self._runtime = DefaultNPURuntime
        self._handle = handle
        self._in_tensor = in_tensor
        self._out_tensor = out_tensor

    def process(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        pixel_format = ctx.pixel_format.upper() if ctx else ("YUYV" if frame.ndim == 3 and frame.shape[-1] == 4 else "RGB3")
        if pixel_format in {"YUYV", "UYVY"}:
            if frame.ndim != 3 or frame.shape[2] != 4:
                raise ValueError(f"Expected packed {pixel_format} pairs, got {frame.shape}")
            h, pair_width, _ = frame.shape
            w = ctx.width if ctx else pair_width * 2
            if pair_width * 2 != w:
                raise ValueError(f"Packed {pixel_format} width does not match frame context ({pair_width * 2} != {w})")
        else:
            if pixel_format not in {"RGB3", "BGR3"} or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"Unsupported greyscale frame {pixel_format}: {frame.shape}")
            h, w, _ = frame.shape
        if self._shape != (h, w, pixel_format) or self._handle is None or self._in_tensor is None or self._out_tensor is None:
            self.setup(w, h, ctx.fps if ctx else 30.0, pixel_format=pixel_format)

        flat = np.ascontiguousarray(frame, dtype=np.uint8).reshape(-1)
        with self._in_tensor.overwrite() as dst:
            np.copyto(dst, flat)

        self._runtime.run(self._handle, [self._in_tensor, self._out_tensor])
        return self._out_tensor.numpy().reshape(frame.shape).copy()

    def teardown(self) -> None:
        self._handle = None
        self._runtime = None
        self._kernel = None
        self._in_tensor = None
        self._out_tensor = None
        self._shape = None
        self._total_bytes = 0
        super().teardown()
