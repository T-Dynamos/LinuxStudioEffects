"""NPU horizontal mirror for native packed video frames."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from ...pixels import NATIVE_PIXEL_FORMATS, is_yuv422
from ..base import ComputeDevice, FrameContext, FrameEffect

_RGB_MIRROR_C_SRC = r"""
#include <stdint.h>
extern "C" void lse_mirror_rgb(const uint8_t *__restrict in, uint8_t *__restrict out, int32_t width) {
    for (int x = 0; x < width; ++x) {
        const uint8_t *src = in + (width - 1 - x) * 3;
        uint8_t *dst = out + x * 3;
        dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
    }
}
"""

_YUYV_MIRROR_C_SRC = r"""
#include <stdint.h>
extern "C" void lse_mirror_yuyv(const uint8_t *__restrict in, uint8_t *__restrict out, int32_t pairs) {
    for (int x = 0; x < pairs; ++x) {
        const uint8_t *src = in + (pairs - 1 - x) * 4;
        uint8_t *dst = out + x * 4;
        dst[0] = src[2]; dst[1] = src[1]; dst[2] = src[0]; dst[3] = src[3];
    }
}
"""

_UYVY_MIRROR_C_SRC = r"""
#include <stdint.h>
extern "C" void lse_mirror_uyvy(const uint8_t *__restrict in, uint8_t *__restrict out, int32_t pairs) {
    for (int x = 0; x < pairs; ++x) {
        const uint8_t *src = in + (pairs - 1 - x) * 4;
        uint8_t *dst = out + x * 4;
        dst[0] = src[0]; dst[1] = src[3]; dst[2] = src[2]; dst[3] = src[1];
    }
}
"""


def select_mirror_workers(height: int) -> int:
    """Choose up to four workers while keeping every worker's row range whole."""
    if height <= 0:
        raise ValueError(f"height must be positive, got {height}")
    return next((workers for workers in (4, 2) if height % workers == 0), 1)


@lru_cache(maxsize=32)
def _build_mirror_kernel(width: int, height: int, pixel_format: str) -> Any:
    """Compile an IRON kernel whose input and output are identical packed formats."""
    import aie.iron as iron
    from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
    from aie.iron.controlflow import range_

    fmt = pixel_format.upper()
    if fmt not in NATIVE_PIXEL_FORMATS:
        raise ValueError(f"Unsupported native mirror format {pixel_format!r}")
    if is_yuv422(fmt) and width % 2:
        raise ValueError(f"{fmt} requires an even frame width, got {width}")

    workers = select_mirror_workers(height)
    bytes_per_pixel = 2 if is_yuv422(fmt) else 3
    row_bytes = width * bytes_per_pixel
    total_bytes = height * row_bytes
    chunks = height // workers
    num_units = width // 2 if is_yuv422(fmt) else width
    tile_bytes = row_bytes
    c_source, function_name = {
        "YUYV": (_YUYV_MIRROR_C_SRC, "lse_mirror_yuyv"),
        "UYVY": (_UYVY_MIRROR_C_SRC, "lse_mirror_uyvy"),
        "RGB3": (_RGB_MIRROR_C_SRC, "lse_mirror_rgb"),
        "BGR3": (_RGB_MIRROR_C_SRC, "lse_mirror_rgb"),
    }[fmt]

    @iron.jit
    def mirror_kernel(
        inp: In,
        out: Out,
        *,
        k_total_bytes: CompileTime[int] = total_bytes,
        k_workers: CompileTime[int] = workers,
        k_row_bytes: CompileTime[int] = row_bytes,
        k_tile_bytes: CompileTime[int] = tile_bytes,
        k_chunks: CompileTime[int] = chunks,
        k_num_units: CompileTime[int] = num_units,
    ):
        full_ty = np.ndarray[(k_total_bytes,), np.dtype[np.uint8]]
        row_ty = np.ndarray[(k_tile_bytes,), np.dtype[np.uint8]]
        ext_fn = ExternalFunction(function_name, source_string=c_source, arg_types=[row_ty, row_ty, np.int32])
        ins = [ObjectFifo(row_ty, name=f"mirror_in{i}") for i in range(k_workers)]
        outs = [ObjectFifo(row_ty, name=f"mirror_out{i}") for i in range(k_workers)]

        def make_worker(idx: int) -> Worker:
            def core_fn(in_f, out_f, fn):
                for _ in range_(k_chunks):
                    source_row = in_f.acquire(1)
                    output_row = out_f.acquire(1)
                    fn(source_row, output_row, k_num_units)
                    in_f.release(1)
                    out_f.release(1)

            return Worker(core_fn, [ins[idx].cons(), outs[idx].prod(), ext_fn])

        workers_list = [make_worker(i) for i in range(k_workers)]

        def sequence(input_host, output_host, *fifos):
            input_producers = fifos[:k_workers]
            output_consumers = fifos[k_workers:]
            rows_per_worker = k_chunks
            for i in range(k_workers):
                start = i * rows_per_worker * k_row_bytes
                end = start + rows_per_worker * k_row_bytes
                input_producers[i].fill(input_host, tap=input_host[start:end])
            for i in range(k_workers):
                start = i * rows_per_worker * k_row_bytes
                end = start + rows_per_worker * k_row_bytes
                output_consumers[i].drain(output_host, tap=output_host[start:end], wait=True)

        runtime = Runtime(
            sequence,
            [full_ty, full_ty, *[fifo.prod() for fifo in ins], *[fifo.cons() for fifo in outs]],
        )
        return Program(iron.get_current_device(), runtime, workers=workers_list).resolve_program()

    return mirror_kernel


class MirrorEffect(FrameEffect):
    """Horizontally mirror RGB/BGR or packed YUYV/UYVY frames on the AMD NPU."""

    effect_name = "mirror"

    def __init__(
        self,
        *,
        name: str | None = None,
        device: ComputeDevice | str = ComputeDevice.NPU,
        enabled: bool = True,
    ) -> None:
        super().__init__(name=name, device=device, enabled=enabled)
        self._shape: tuple[int, int, str] | None = None
        self._kernel: Any = None
        self._runtime: Any = None
        self._handle: Any = None
        self._input_tensor: Any = None
        self._output_tensor: Any = None

    def setup(self, width: int, height: int, fps: float, *, pixel_format: str | None = None) -> None:
        super().setup(width, height, fps)
        if pixel_format is None:
            return
        import aie.iron as iron
        from aie.utils import DefaultNPURuntime

        fmt = pixel_format.upper()
        if fmt not in NATIVE_PIXEL_FORMATS:
            raise ValueError(f"Unsupported native mirror format {pixel_format!r}")
        if is_yuv422(fmt) and width % 2:
            raise ValueError(f"{fmt} requires an even frame width, got {width}")
        if self._shape == (height, width, fmt) and self._handle is not None:
            return

        total_bytes = height * width * (2 if is_yuv422(fmt) else 3)
        row_bytes = width * (2 if is_yuv422(fmt) else 3)
        workers = select_mirror_workers(height)
        kernel = _build_mirror_kernel(width, height, fmt)
        input_tensor = iron.zeros(total_bytes, dtype=np.uint8, device="npu")
        output_tensor = iron.zeros(total_bytes, dtype=np.uint8, device="npu")
        kernel(
            input_tensor,
            output_tensor,
            k_total_bytes=total_bytes,
            k_workers=workers,
            k_row_bytes=row_bytes,
            k_tile_bytes=row_bytes,
            k_chunks=height // workers,
            k_num_units=width // 2 if is_yuv422(fmt) else width,
        )
        compiled_kernel = next(iter(kernel._kernel_cache.values()))
        handle = DefaultNPURuntime.load(compiled_kernel)

        self._shape = (height, width, fmt)
        self._kernel = kernel
        self._runtime = DefaultNPURuntime
        self._handle = handle
        self._input_tensor = input_tensor
        self._output_tensor = output_tensor

    def process(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        pixel_format = ctx.pixel_format.upper() if ctx else ("YUYV" if frame.ndim == 3 and frame.shape[-1] == 4 else "RGB3")
        if pixel_format not in NATIVE_PIXEL_FORMATS:
            raise ValueError(f"Unsupported native mirror format {pixel_format!r}")
        h = frame.shape[0]
        if is_yuv422(pixel_format):
            if frame.ndim != 3 or frame.shape[2] != 4:
                raise ValueError(f"Expected packed {pixel_format} pairs, got {frame.shape}")
            pair_width = frame.shape[1]
            width = ctx.width if ctx else pair_width * 2
            if pair_width * 2 != width:
                raise ValueError(f"Packed {pixel_format} width does not match frame context")
        else:
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"Expected {pixel_format} frame shaped (H, W, 3), got {frame.shape}")
            width = frame.shape[1]
            if ctx and ctx.width != width:
                raise ValueError(f"{pixel_format} frame width does not match frame context")

        if self._shape != (h, width, pixel_format) or self._handle is None:
            self.setup(width, h, ctx.fps if ctx else 30.0, pixel_format=pixel_format)
        flat = np.ascontiguousarray(frame, dtype=np.uint8).reshape(-1)
        with self._input_tensor.overwrite() as target:
            np.copyto(target, flat)
        self._runtime.run(self._handle, [self._input_tensor, self._output_tensor])
        return self._output_tensor.numpy().reshape(frame.shape).copy()

    def teardown(self) -> None:
        self._handle = None
        self._runtime = None
        self._kernel = None
        self._input_tensor = None
        self._output_tensor = None
        self._shape = None
        super().teardown()
