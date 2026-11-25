"""Depth-aware portrait bokeh blur driven by streaming Video Depth Anything across CPU and NPU."""

import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from ...effects.base import ComputeDevice, FrameContext, FrameEffect
from .config import VideoDepthConfig, get_video_depth_config
from .model import InferenceResult, VideoDepthAnythingModel
from .npu_ops import NPU_DISPATCH_LOCK
from .weights import VideoDepthWeights, create_synthetic_weights, load_from_hf

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


@dataclass(frozen=True)
class _BilinearPlan:
    in_h: int
    in_w: int
    out_h: int
    out_w: int
    identity: bool
    y0: np.ndarray
    y1: np.ndarray
    x0: np.ndarray
    x1: np.ndarray
    wy0_2d: np.ndarray
    wy1_2d: np.ndarray
    wx0_2d: np.ndarray
    wx1_2d: np.ndarray
    wy0_3d: np.ndarray
    wy1_3d: np.ndarray
    wx0_3d: np.ndarray
    wx1_3d: np.ndarray


def _make_bilinear_1d(in_len: int, out_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if in_len == 1 or out_len == 1:
        idx0 = np.zeros((out_len,), dtype=np.intp)
        return idx0, idx0, np.zeros((out_len,), dtype=np.float32)
    coords = np.clip(
        (np.arange(out_len, dtype=np.float32) + np.float32(0.5)) * np.float32(in_len / out_len) - np.float32(0.5),
        0.0,
        float(in_len - 1),
    )
    i0 = np.floor(coords).astype(np.intp)
    i1 = np.minimum(i0 + 1, in_len - 1)
    w = (coords - i0).astype(np.float32)
    return i0, i1, w


def _build_bilinear_plan(in_h: int, in_w: int, out_h: int, out_w: int) -> _BilinearPlan:
    y0, y1, wy = _make_bilinear_1d(in_h, out_h)
    x0, x1, wx = _make_bilinear_1d(in_w, out_w)
    wy1_2d = wy[:, None]
    wy0_2d = np.float32(1.0) - wy1_2d
    wx1_2d = wx[None, :]
    wx0_2d = np.float32(1.0) - wx1_2d
    return _BilinearPlan(
        in_h=in_h,
        in_w=in_w,
        out_h=out_h,
        out_w=out_w,
        identity=(in_h == out_h and in_w == out_w),
        y0=y0,
        y1=y1,
        x0=x0,
        x1=x1,
        wy0_2d=wy0_2d,
        wy1_2d=wy1_2d,
        wx0_2d=wx0_2d,
        wx1_2d=wx1_2d,
        wy0_3d=wy0_2d[:, :, None],
        wy1_3d=wy1_2d[:, :, None],
        wx0_3d=wx0_2d[:, :, None],
        wx1_3d=wx1_2d[:, :, None],
    )


def _resize_2d(arr: np.ndarray, plan: _BilinearPlan) -> np.ndarray:
    if plan.identity:
        return arr
    tmp = arr[:, plan.x0] * plan.wx0_2d + arr[:, plan.x1] * plan.wx1_2d
    return tmp[plan.y0, :] * plan.wy0_2d + tmp[plan.y1, :] * plan.wy1_2d


def _resize_hwc_u8(arr: np.ndarray, plan: _BilinearPlan) -> np.ndarray:
    if plan.identity:
        return np.clip(np.round(arr), 0, 255).astype(np.uint8)
    tmp = arr[:, plan.x0, :] * plan.wx0_3d + arr[:, plan.x1, :] * plan.wx1_3d
    out = tmp[plan.y0, :, :] * plan.wy0_3d + tmp[plan.y1, :, :] * plan.wy1_3d
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _box_filter_2d(arr: np.ndarray, radius: int) -> np.ndarray:
    """O(H*W) separable box filter with edge clamping using cumulative sums."""
    if radius <= 0:
        return arr
    h, w = arr.shape
    win = float(2 * radius + 1)

    px = np.pad(arr, ((0, 0), (radius, radius)), mode="edge")
    cs_x = np.empty((h, w + 2 * radius + 1), dtype=np.float32)
    cs_x[:, 0] = 0.0
    np.cumsum(px, axis=1, dtype=np.float32, out=cs_x[:, 1:])
    blur_x = (cs_x[:, 2 * radius + 1 :] - cs_x[:, :w]) / np.float32(win)

    py = np.pad(blur_x, ((radius, radius), (0, 0)), mode="edge")
    cs_y = np.empty((h + 2 * radius + 1, w), dtype=np.float32)
    cs_y[0, :] = 0.0
    np.cumsum(py, axis=0, dtype=np.float32, out=cs_y[1:, :])
    return (cs_y[2 * radius + 1 :, :] - cs_y[:h, :]) / np.float32(win)


def _downsample2x_hw4(arr: np.ndarray) -> np.ndarray:
    h, w, _ = arr.shape
    pad_h = h & 1
    pad_w = w & 1
    if pad_h or pad_w:
        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return np.float32(0.25) * (arr[0::2, 0::2] + arr[0::2, 1::2] + arr[1::2, 0::2] + arr[1::2, 1::2])


def _gaussian5_hw4(arr: np.ndarray, passes: int = 1) -> np.ndarray:
    cur = arr
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


def _composite_cpu_u8(
    l0: np.ndarray,
    l1: np.ndarray,
    l2: np.ndarray,
    l3: np.ndarray,
    coc_u8: np.ndarray,
) -> np.ndarray:
    """Fixed-point 4-level bokeh blend matching the AIE2 NPU kernel bit-for-bit."""
    c = coc_u8.astype(np.int16)
    d2 = c - 170
    t2 = d2 * 3 + (d2 >> 6)
    w0 = np.where(c < 85, 256 - c * 3, 0)[:, :, None]
    w1 = np.where(c < 85, c * 3, np.where(c < 170, 256 - (c - 85) * 3, 0))[:, :, None]
    w2 = np.where((c >= 85) & (c < 170), (c - 85) * 3, np.where(c >= 170, 256 - t2, 0))[:, :, None]
    w3 = np.where(c >= 170, t2, 0)[:, :, None]
    acc = (
        128
        + l0.astype(np.int32) * w0
        + l1.astype(np.int32) * w1
        + l2.astype(np.int32) * w2
        + l3.astype(np.int32) * w3
    )
    return np.ascontiguousarray((acc >> 8).astype(np.uint8))


class PortraitBlurEffectBase(FrameEffect):
    """Streaming depth-aware portrait bokeh effect backed by VideoDepthAnythingModel."""

    def __init__(
        self,
        *,
        variant: str = "small",
        input_size: int = 252,
        use_real_weights: bool = True,
        local_files_only: bool = False,
        weights: VideoDepthWeights | None = None,
        blur_strength: float = 0.8,
        focus_mode: str = "auto",
        focus_depth: float = 0.7,
        dof_range: float = 0.12,
        falloff: float = 0.25,
        blur_foreground: bool = False,
        temporal_alpha: float = 0.35,
        edge_refine: bool = True,
        use_npu_composite: bool = True,
        async_depth: bool = False,
        depth_interval_s: float = 0.2,
        name: str | None = None,
        device: ComputeDevice | str = ComputeDevice.AUTO,
        enabled: bool = True,
    ) -> None:
        if not 0.0 <= blur_strength <= 1.0:
            raise ValueError(f"blur_strength must be in [0, 1], got {blur_strength}")
        mode = focus_mode.lower().strip()
        if mode not in ("auto", "manual"):
            raise ValueError(f"focus_mode must be 'auto' or 'manual', got {focus_mode!r}")
        if not 0.0 <= focus_depth <= 1.0:
            raise ValueError(f"focus_depth must be in [0, 1], got {focus_depth}")
        if dof_range < 0.0:
            raise ValueError(f"dof_range must be >= 0, got {dof_range}")
        if falloff < 0.0:
            raise ValueError(f"falloff must be >= 0, got {falloff}")
        if not 0.0 <= temporal_alpha <= 1.0:
            raise ValueError(f"temporal_alpha must be in [0, 1], got {temporal_alpha}")

        super().__init__(name=name, device=device, enabled=enabled)
        self.depth_config: VideoDepthConfig = get_video_depth_config(variant, input_size=input_size)
        self.use_real_weights = use_real_weights
        self.local_files_only = local_files_only
        self._explicit_weights = weights

        self.blur_strength = float(blur_strength)
        self.focus_mode = mode
        self.focus_depth = float(focus_depth)
        self.dof_range = float(dof_range)
        self.falloff = float(falloff)
        self.blur_foreground = bool(blur_foreground)
        self.temporal_alpha = float(temporal_alpha)
        self.edge_refine = bool(edge_refine)
        self.use_npu_composite = bool(use_npu_composite)
        self.async_depth = bool(async_depth)
        self.depth_interval_s = float(depth_interval_s)

        self._model: VideoDepthAnythingModel | None = None
        self.last_result: InferenceResult | None = None
        self.last_coc: np.ndarray | None = None

        # Background 4-5 Hz depth state for decoupled ~30fps live bokeh streaming
        self._depth_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_depth: np.ndarray | None = None
        self._depth_updates: int = 0
        self._stop_event = threading.Event()
        self._frame_ready_event = threading.Event()
        self._depth_thread: threading.Thread | None = None

        self._frame_shape: tuple[int, int] | None = None
        self._plan_full_to_low: _BilinearPlan | None = None
        self._plan_low_to_full: _BilinearPlan | None = None
        self._plan_pyr_to_full: tuple[_BilinearPlan, _BilinearPlan, _BilinearPlan] | None = None

        self._ema_lo: float | None = None
        self._ema_hi: float | None = None
        self._ema_focus: float | None = None
        self._prev_luma_low: np.ndarray | None = None
        self._prev_coc_low: np.ndarray | None = None

        # NPU composite kernel state (at most 1 extra hardware context slot: 4 MatMul + 1 Bokeh <= 6)
        self._npu_runtime: Any = None
        self._npu_handle: Any = None
        self._npu_kernel: Any = None
        self._npu_in_a: Any = None
        self._npu_in_b: Any = None
        self._npu_out: Any = None

    def _release_npu_composite(self) -> None:
        self._npu_handle = None
        self._npu_runtime = None
        self._npu_kernel = None
        self._npu_in_a = None
        self._npu_in_b = None
        self._npu_out = None

    def _reset_temporal_bokeh_state(self) -> None:
        self._ema_lo = None
        self._ema_hi = None
        self._ema_focus = None
        self._prev_luma_low = None
        self._prev_coc_low = None
        self.last_coc = None

    def _setup_npu_composite(self, width: int, height: int) -> None:
        self._release_npu_composite()
        if self.active_device != ComputeDevice.NPU or not self.use_npu_composite:
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
        self._npu_handle = handle
        self._npu_in_a = in_a
        self._npu_in_b = in_b
        self._npu_out = out

    def setup(self, width: int, height: int, fps: float) -> None:
        super().setup(width, height, fps)
        self._reset_temporal_bokeh_state()

        if self._model is None:
            if self._explicit_weights is not None:
                weights = self._explicit_weights
            elif self.use_real_weights:
                weights = load_from_hf(self.depth_config, local_files_only=self.local_files_only)
            else:
                weights = create_synthetic_weights(self.depth_config)

            self._model = VideoDepthAnythingModel(
                self.depth_config,
                weights,
                backend=self.active_device.value,
            )
        else:
            self._model.reset_temporal_state()

        if self._frame_shape != (height, width):
            s = self.depth_config.input_size
            self._plan_full_to_low = _build_bilinear_plan(height, width, s, s)
            self._plan_low_to_full = _build_bilinear_plan(s, s, height, width)

            h1, w1 = (height + 1) // 2, (width + 1) // 2
            h2, w2 = (h1 + 1) // 2, (w1 + 1) // 2
            h3, w3 = (h2 + 1) // 2, (w2 + 1) // 2
            self._plan_pyr_to_full = (
                _build_bilinear_plan(h1, w1, height, width),
                _build_bilinear_plan(h2, w2, height, width),
                _build_bilinear_plan(h3, w3, height, width),
            )
            self._setup_npu_composite(width, height)
            self._frame_shape = (height, width)

    def _compute_coc_full(self, depth_raw: np.ndarray, frame: np.ndarray) -> np.ndarray:
        s = self.depth_config.input_size
        lo = float(np.percentile(depth_raw, 5.0))
        hi = float(np.percentile(depth_raw, 95.0))

        if self._ema_lo is None or self._ema_hi is None:
            self._ema_lo = lo
            self._ema_hi = hi
        else:
            norm_alpha = 1.0 if self.temporal_alpha >= 1.0 else min(0.15, self.temporal_alpha)
            self._ema_lo = norm_alpha * lo + (1.0 - norm_alpha) * self._ema_lo
            self._ema_hi = norm_alpha * hi + (1.0 - norm_alpha) * self._ema_hi

        denom = max(self._ema_hi - self._ema_lo, 1e-4)
        depth_norm = np.clip((depth_raw - self._ema_lo) / denom, 0.0, 1.0).astype(np.float32)

        if self.focus_mode == "auto":
            y0 = int(round(0.10 * s))
            y1 = max(y0 + 1, int(round(0.75 * s)))
            x0 = int(round(0.25 * s))
            x1 = max(x0 + 1, int(round(0.75 * s)))
            target_focus = float(np.percentile(depth_norm[y0:y1, x0:x1], 80.0))
            if self._ema_focus is None:
                self._ema_focus = target_focus
            else:
                focus_alpha = 1.0 if self.temporal_alpha >= 1.0 else min(0.25, self.temporal_alpha)
                self._ema_focus = focus_alpha * target_focus + (1.0 - focus_alpha) * self._ema_focus
            focus = self._ema_focus
        else:
            focus = self.focus_depth

        if self.blur_foreground:
            dist = np.abs(depth_norm - np.float32(focus))
        else:
            dist = np.float32(focus) - depth_norm

        t = np.clip((dist - np.float32(self.dof_range)) / np.float32(max(self.falloff, 1e-4)), 0.0, 1.0)
        coc_raw = (t * t * (np.float32(3.0) - np.float32(2.0) * t)).astype(np.float32, copy=False)

        luma_full = (
            frame[:, :, 0] * np.float32(0.299 / 255.0)
            + frame[:, :, 1] * np.float32(0.587 / 255.0)
            + frame[:, :, 2] * np.float32(0.114 / 255.0)
        )
        assert self._plan_full_to_low is not None and self._plan_low_to_full is not None
        luma_low = _resize_2d(luma_full, self._plan_full_to_low)

        if self._prev_luma_low is not None and self._prev_coc_low is not None:
            diff_luma = np.abs(luma_low - self._prev_luma_low)
            motion = np.clip((diff_luma - np.float32(0.02)) * np.float32(10.0), 0.0, 1.0)
            alpha_map = np.float32(self.temporal_alpha) + np.float32(1.0 - self.temporal_alpha) * motion
            coc_low = alpha_map * coc_raw + (np.float32(1.0) - alpha_map) * self._prev_coc_low
        else:
            coc_low = coc_raw

        self._prev_luma_low = luma_low
        self._prev_coc_low = coc_low

        if self.edge_refine:
            radius = max(2, s // 32)
            eps = np.float32(2e-3)
            mean_i = _box_filter_2d(luma_low, radius)
            mean_p = _box_filter_2d(coc_low, radius)
            corr_i = _box_filter_2d(luma_low * luma_low, radius)
            corr_ip = _box_filter_2d(luma_low * coc_low, radius)
            var_i = np.maximum(corr_i - mean_i * mean_i, np.float32(0.0))
            cov_ip = corr_ip - mean_i * mean_p
            a_low = cov_ip / (var_i + eps)
            b_low = mean_p - a_low * mean_i
            mean_a = _resize_2d(_box_filter_2d(a_low, radius), self._plan_low_to_full)
            mean_b = _resize_2d(_box_filter_2d(b_low, radius), self._plan_low_to_full)
            coc_full = np.clip(mean_a * luma_full + mean_b, 0.0, 1.0)
        else:
            coc_full = np.clip(_resize_2d(coc_low, self._plan_low_to_full), 0.0, 1.0)

        return coc_full.astype(np.float32, copy=False)

    def _build_blur_pyramid(self, frame: np.ndarray, coc_unscaled: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self._plan_pyr_to_full is not None
        p1, p2, p3 = self._plan_pyr_to_full

        # Keep a tiny 1e-3 weight on foreground pixels so normalized convolution suppresses
        # foreground halo bleed while gracefully falling back inside solid foreground regions.
        bg_mask = np.clip((coc_unscaled - np.float32(0.02)) * np.float32(10.0), 0.0, 1.0)
        w0 = bg_mask * np.float32(1.0 - 1e-3) + np.float32(1e-3)

        h, w, _ = frame.shape
        cur = np.empty((h, w, 4), dtype=np.float32)
        cur[:, :, :3] = frame.astype(np.float32) * w0[:, :, None]
        cur[:, :, 3] = w0

        s1 = _gaussian5_hw4(_downsample2x_hw4(cur), passes=1)
        l1_small = s1[:, :, :3] / np.maximum(s1[:, :, 3:4], np.float32(1e-5))
        l1 = _resize_hwc_u8(l1_small, p1)

        s2 = _gaussian5_hw4(_downsample2x_hw4(s1), passes=2)
        l2_small = s2[:, :, :3] / np.maximum(s2[:, :, 3:4], np.float32(1e-5))
        l2 = _resize_hwc_u8(l2_small, p2)

        s3 = _gaussian5_hw4(_downsample2x_hw4(s2), passes=3)
        l3_small = s3[:, :, :3] / np.maximum(s3[:, :, 3:4], np.float32(1e-5))
        l3 = _resize_hwc_u8(l3_small, p3)

        return l1, l2, l3

    def _composite_npu(
        self,
        l0: np.ndarray,
        l1: np.ndarray,
        l2: np.ndarray,
        l3: np.ndarray,
        coc_u8: np.ndarray,
    ) -> np.ndarray:
        h, w, _ = l0.shape
        num_pixels = h * w
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
            self._npu_runtime.run(self._npu_handle, [self._npu_in_a, self._npu_in_b, self._npu_out])
            return self._npu_out.numpy().reshape(h, w, 3).copy()

    def _start_depth_worker(self) -> None:
        if self._depth_thread is not None:
            return
        self._stop_event.clear()
        self._frame_ready_event.clear()

        def _worker() -> None:
            while not self._stop_event.is_set():
                if not self._frame_ready_event.wait(timeout=0.05):
                    continue
                self._frame_ready_event.clear()
                with self._depth_lock:
                    snap = self._latest_frame
                if snap is None or self._model is None:
                    continue
                t0 = time.perf_counter()
                res = self._model.infer(snap)
                with self._depth_lock:
                    self._latest_depth = res.depth
                    self.last_result = res
                    self._depth_updates += 1
                elapsed = time.perf_counter() - t0
                rem = self.depth_interval_s - elapsed
                if rem > 0.0:
                    self._stop_event.wait(rem)

        self._depth_thread = threading.Thread(target=_worker, daemon=True, name="vda-depth-worker")
        self._depth_thread.start()

    def _stop_depth_worker(self) -> None:
        self._stop_event.set()
        self._frame_ready_event.set()
        if self._depth_thread is not None:
            self._depth_thread.join(timeout=5.0)
            self._depth_thread = None
        with self._depth_lock:
            self._latest_frame = None
            self._latest_depth = None
            self._depth_updates = 0

    @property
    def depth_updates(self) -> int:
        with self._depth_lock:
            return self._depth_updates

    def apply_depth_bokeh(self, frame: np.ndarray, depth_raw: np.ndarray) -> np.ndarray:
        """Render variable portrait bokeh onto (H, W, 3) uint8 frame from a (S, S) depth map."""
        coc_unscaled = self._compute_coc_full(depth_raw, frame)
        coc_scaled = np.clip(coc_unscaled * np.float32(self.blur_strength), 0.0, 1.0)
        self.last_coc = coc_scaled

        l0 = np.ascontiguousarray(frame, dtype=np.uint8)
        if self.blur_strength <= 0.0:
            return l0

        l1, l2, l3 = self._build_blur_pyramid(l0, coc_unscaled)
        coc_u8 = np.clip(np.round(coc_scaled * np.float32(255.0)), 0, 255).astype(np.uint8)

        if self._npu_handle is not None and self._npu_in_a is not None:
            return self._composite_npu(l0, l1, l2, l3, coc_u8)
        return _composite_cpu_u8(l0, l1, l2, l3, coc_u8)

    def process(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) RGB frame, got {frame.shape}")
        h, w, _ = frame.shape
        if self._model is None or self._frame_shape != (h, w):
            self.setup(w, h, ctx.fps if ctx else 30.0)
        assert self._model is not None

        if not self.async_depth:
            res = self._model.infer(frame)
            self.last_result = res
            self._depth_updates += 1
            return self.apply_depth_bokeh(frame, res.depth)

        with self._depth_lock:
            depth = self._latest_depth
            self._latest_frame = frame.copy()
        if depth is None:
            res = self._model.infer(frame)
            with self._depth_lock:
                self._latest_depth = res.depth
                self.last_result = res
                self._depth_updates += 1
                depth = res.depth
            self._start_depth_worker()
        else:
            self._frame_ready_event.set()
        return self.apply_depth_bokeh(frame, depth)

    def teardown(self) -> None:
        self._stop_depth_worker()
        self._release_npu_composite()
        self._reset_temporal_bokeh_state()
        self._frame_shape = None
        self._plan_full_to_low = None
        self._plan_low_to_full = None
        self._plan_pyr_to_full = None
        if self._model is not None:
            self._model.close()
            self._model = None
        super().teardown()
