"""CPU depth-aware portrait bokeh blur driven by streaming Video Depth Anything."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from ...models.video_depth_anything import host_ops
from ...models.video_depth_anything.config import VideoDepthConfig, get_video_depth_config
from ...models.video_depth_anything.model import InferenceResult, VideoDepthAnythingModel
from ...models.video_depth_anything.weights import VideoDepthWeights, create_synthetic_weights, load_from_hf
from ...pixels import (
    NATIVE_PIXEL_FORMATS,
    channels_to_frame,
    frame_to_channels,
    is_yuv422,
    model_luma_input,
    split_yuv422,
)
from ..base import ComputeDevice, FrameContext, FrameEffect


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
    fast = host_ops.bilinear_resize_hwc_u8_fast(arr, plan.out_h, plan.out_w)
    if fast is not None:
        return fast
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
    return host_ops.bokeh_composite_u8(l0, l1, l2, l3, coc_u8)


_QUALITY_PRESETS: dict[str, dict[str, object]] = {
    "fast": {
        "variant": "small",
        "input_size": 196,
        "blur_passes": (1, 2, 4),
        "edge_refine_eps": 2e-3,
    },
    "balanced": {
        "variant": "base",
        "input_size": 252,
        "blur_passes": (2, 4, 6),
        "edge_refine_eps": 1.2e-3,
    },
    "high": {
        "variant": "base",
        "input_size": 336,
        "blur_passes": (3, 6, 8),
        "edge_refine_eps": 8e-4,
    },
}


class PortraitBlurEffect(FrameEffect):
    """Streaming depth-aware portrait bokeh effect backed by VideoDepthAnythingModel."""

    effect_name = "portrait_blur"

    def __init__(
        self,
        *,
        quality: str = "balanced",
        variant: str | None = None,
        input_size: int | None = None,
        use_real_weights: bool = True,
        local_files_only: bool = False,
        weights: VideoDepthWeights | None = None,
        blur_strength: float = 1.0,
        blur_passes: tuple[int, int, int] | None = None,
        focus_mode: str = "auto",
        focus_depth: float = 0.7,
        focus_percentile: float = 65.0,
        focus_region: tuple[float, float, float, float] = (0.10, 0.75, 0.25, 0.75),
        depth_percentiles: tuple[float, float] = (5.0, 95.0),
        dof_range: float = 0.30,
        falloff: float = 0.20,
        blur_foreground: bool = False,
        temporal_alpha: float = 0.35,
        edge_refine: bool = True,
        edge_refine_radius: int | None = None,
        edge_refine_eps: float | None = None,
        mask_gamma: float = 1.0,
        halo_suppression: float = 10.0,
        async_depth: bool = False,
        depth_interval_s: float = 0.2,
        name: str | None = None,
        device: ComputeDevice | str = ComputeDevice.CPU,
        enabled: bool = True,
    ) -> None:
        q_key = quality.lower().strip()
        if q_key not in _QUALITY_PRESETS:
            raise ValueError(f"quality must be one of {tuple(_QUALITY_PRESETS.keys())}, got {quality!r}")
        preset = _QUALITY_PRESETS[q_key]

        resolved_variant = str(preset["variant"] if variant is None else variant)
        resolved_input_size = int(preset["input_size"] if input_size is None else input_size)  # type: ignore[arg-type]
        resolved_passes = preset["blur_passes"] if blur_passes is None else blur_passes
        resolved_eps = float(preset["edge_refine_eps"] if edge_refine_eps is None else edge_refine_eps)  # type: ignore[arg-type]

        super().__init__(name=name, device=device, enabled=enabled)
        self.quality = q_key
        self.depth_config: VideoDepthConfig = get_video_depth_config(resolved_variant, input_size=resolved_input_size)
        self.use_real_weights = use_real_weights
        self.local_files_only = local_files_only
        self._explicit_weights = weights

        self.set_params(
            blur_strength=blur_strength,
            blur_passes=resolved_passes,  # type: ignore[arg-type]
            focus_mode=focus_mode,
            focus_depth=focus_depth,
            focus_percentile=focus_percentile,
            focus_region=focus_region,
            depth_percentiles=depth_percentiles,
            dof_range=dof_range,
            falloff=falloff,
            blur_foreground=blur_foreground,
            temporal_alpha=temporal_alpha,
            edge_refine=edge_refine,
            edge_refine_radius=edge_refine_radius,
            edge_refine_eps=resolved_eps,
            mask_gamma=mask_gamma,
            halo_suppression=halo_suppression,
            depth_interval_s=depth_interval_s,
        )
        self.async_depth = bool(async_depth)

        self._model: VideoDepthAnythingModel | None = None
        self.last_result: InferenceResult | None = None
        self.last_coc: np.ndarray | None = None

        # Background depth state for decoupled ~30fps live bokeh streaming
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

    def set_params(self, **params: object) -> PortraitBlurEffect:
        """Validate and update live-tunable bokeh, focus, and refinement parameters."""
        if "blur_strength" in params:
            val = float(params["blur_strength"])  # type: ignore[arg-type]
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"blur_strength must be in [0, 1], got {val}")
            self.blur_strength = val
        if "blur_passes" in params:
            passes = tuple(int(p) for p in params["blur_passes"])  # type: ignore[union-attr]
            if len(passes) != 3 or any(p < 1 for p in passes):
                raise ValueError(f"blur_passes must be 3 positive ints, got {params['blur_passes']!r}")
            self.blur_passes: tuple[int, int, int] = (passes[0], passes[1], passes[2])
        if "focus_mode" in params:
            mode = str(params["focus_mode"]).lower().strip()
            if mode not in ("auto", "manual"):
                raise ValueError(f"focus_mode must be 'auto' or 'manual', got {params['focus_mode']!r}")
            self.focus_mode = mode
        if "focus_depth" in params:
            val = float(params["focus_depth"])  # type: ignore[arg-type]
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"focus_depth must be in [0, 1], got {val}")
            self.focus_depth = val
        if "focus_percentile" in params:
            val = float(params["focus_percentile"])  # type: ignore[arg-type]
            if not 0.0 <= val <= 100.0:
                raise ValueError(f"focus_percentile must be in [0, 100], got {val}")
            self.focus_percentile = val
        if "focus_region" in params:
            region = tuple(float(v) for v in params["focus_region"])  # type: ignore[union-attr]
            if len(region) != 4 or not (0.0 <= region[0] < region[1] <= 1.0 and 0.0 <= region[2] < region[3] <= 1.0):
                raise ValueError(f"focus_region must be (y0, y1, x0, x1) within [0, 1], got {params['focus_region']!r}")
            self.focus_region: tuple[float, float, float, float] = (region[0], region[1], region[2], region[3])
        if "depth_percentiles" in params:
            pcts = tuple(float(v) for v in params["depth_percentiles"])  # type: ignore[union-attr]
            if len(pcts) != 2 or not (0.0 <= pcts[0] < pcts[1] <= 100.0):
                raise ValueError(f"depth_percentiles must satisfy 0 <= lo < hi <= 100, got {params['depth_percentiles']!r}")
            self.depth_percentiles: tuple[float, float] = (pcts[0], pcts[1])
        if "dof_range" in params:
            val = float(params["dof_range"])  # type: ignore[arg-type]
            if val < 0.0:
                raise ValueError(f"dof_range must be >= 0, got {val}")
            self.dof_range = val
        if "falloff" in params:
            val = float(params["falloff"])  # type: ignore[arg-type]
            if val < 0.0:
                raise ValueError(f"falloff must be >= 0, got {val}")
            self.falloff = val
        if "blur_foreground" in params:
            self.blur_foreground = bool(params["blur_foreground"])
        if "temporal_alpha" in params:
            val = float(params["temporal_alpha"])  # type: ignore[arg-type]
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"temporal_alpha must be in [0, 1], got {val}")
            self.temporal_alpha = val
        if "edge_refine" in params:
            self.edge_refine = bool(params["edge_refine"])
        if "edge_refine_radius" in params:
            rad = params["edge_refine_radius"]
            if rad is not None:
                rad_i = int(rad)  # type: ignore[arg-type]
                if rad_i < 1:
                    raise ValueError(f"edge_refine_radius must be >= 1 or None, got {rad}")
                self.edge_refine_radius: int | None = rad_i
            else:
                self.edge_refine_radius = None
        if "edge_refine_eps" in params:
            val = float(params["edge_refine_eps"])  # type: ignore[arg-type]
            if val <= 0.0:
                raise ValueError(f"edge_refine_eps must be > 0, got {val}")
            self.edge_refine_eps = val
        if "mask_gamma" in params:
            val = float(params["mask_gamma"])  # type: ignore[arg-type]
            if val <= 0.0:
                raise ValueError(f"mask_gamma must be > 0, got {val}")
            self.mask_gamma = val
        if "halo_suppression" in params:
            val = float(params["halo_suppression"])  # type: ignore[arg-type]
            if val <= 0.0:
                raise ValueError(f"halo_suppression must be > 0, got {val}")
            self.halo_suppression = val
        if "depth_interval_s" in params:
            val = float(params["depth_interval_s"])  # type: ignore[arg-type]
            if val < 0.0:
                raise ValueError(f"depth_interval_s must be >= 0, got {val}")
            self.depth_interval_s = val
        unknown = set(params) - {
            "blur_strength",
            "blur_passes",
            "focus_mode",
            "focus_depth",
            "focus_percentile",
            "focus_region",
            "depth_percentiles",
            "dof_range",
            "falloff",
            "blur_foreground",
            "temporal_alpha",
            "edge_refine",
            "edge_refine_radius",
            "edge_refine_eps",
            "mask_gamma",
            "halo_suppression",
            "depth_interval_s",
        }
        if unknown:
            raise ValueError(f"Unknown or non-live parameter(s): {sorted(unknown)}")
        return self

    def _reset_temporal_bokeh_state(self) -> None:
        self._ema_lo = None
        self._ema_hi = None
        self._ema_focus = None
        self._prev_luma_low = None
        self._prev_coc_low = None
        self.last_coc = None

    def _on_plans_resized(self, width: int, height: int) -> None:
        """Hook for subclasses to allocate resolution-dependent composite resources."""

    def _ensure_plans(self, width: int, height: int) -> None:
        if self._frame_shape == (height, width) and self._plan_full_to_low is not None:
            return
        s = self.depth_config.input_size
        self._plan_full_to_low = _build_bilinear_plan(height, width, s, s)
        self._plan_low_to_full = _build_bilinear_plan(s, s, height, width)

        h1, w1 = (height + 1) // 2, (width + 1) // 2
        h2, w2 = (h1 + 1) // 2, (w1 + 1) // 2
        h3_8, w3_8 = (h2 + 1) // 2, (w2 + 1) // 2
        h3, w3 = (h3_8 + 1) // 2, (w3_8 + 1) // 2
        self._plan_pyr_to_full = (
            _build_bilinear_plan(h1, w1, height, width),
            _build_bilinear_plan(h2, w2, height, width),
            _build_bilinear_plan(h3, w3, height, width),
        )
        self._on_plans_resized(width, height)
        self._frame_shape = (height, width)

    def setup(self, width: int, height: int, fps: float) -> None:
        super().setup(width, height, fps)
        self._stop_depth_worker()
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

        self._ensure_plans(width, height)

    def _compute_coc_full(
        self,
        depth_raw: np.ndarray,
        frame: np.ndarray,
        *,
        pixel_format: str = "RGB3",
    ) -> np.ndarray:
        s = self.depth_config.input_size
        lo = float(np.percentile(depth_raw, self.depth_percentiles[0]))
        hi = float(np.percentile(depth_raw, self.depth_percentiles[1]))

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
            ry0, ry1, rx0, rx1 = self.focus_region
            y0 = int(round(ry0 * s))
            y1 = max(y0 + 1, int(round(ry1 * s)))
            x0 = int(round(rx0 * s))
            x1 = max(x0 + 1, int(round(rx1 * s)))
            target_focus = float(np.percentile(depth_norm[y0:y1, x0:x1], self.focus_percentile))
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

        fmt = pixel_format.upper()
        if is_yuv422(fmt):
            y_plane, _, _ = split_yuv422(frame, fmt)
            luma_full = y_plane.astype(np.float32) * np.float32(1.0 / 255.0)
        elif fmt == "BGR3":
            luma_full = (
                frame[:, :, 0] * np.float32(0.114 / 255.0)
                + frame[:, :, 1] * np.float32(0.587 / 255.0)
                + frame[:, :, 2] * np.float32(0.299 / 255.0)
            )
        else:
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
            radius = self.edge_refine_radius if self.edge_refine_radius is not None else max(2, s // 32)
            eps = np.float32(self.edge_refine_eps)
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

        if self.mask_gamma != 1.0:
            coc_full = np.power(coc_full, np.float32(self.mask_gamma))
        return coc_full.astype(np.float32, copy=False)

    def _build_blur_pyramid(self, frame: np.ndarray, coc_unscaled: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self._plan_pyr_to_full is not None
        p1, p2, p3 = self._plan_pyr_to_full
        passes_1, passes_2, passes_3 = self.blur_passes

        # Keep a tiny 1e-3 weight on foreground pixels so normalized convolution suppresses
        # foreground halo bleed while gracefully falling back inside solid foreground regions.
        bg_mask = np.clip((coc_unscaled - np.float32(0.02)) * np.float32(self.halo_suppression), 0.0, 1.0)
        w0 = bg_mask * np.float32(1.0 - 1e-3) + np.float32(1e-3)

        s1 = _gaussian5_hw4(host_ops.downsample2x_bg_hw4(frame, w0), passes=passes_1)
        l1_small = s1[:, :, :3] / np.maximum(s1[:, :, 3:4], np.float32(1e-5))
        l1 = _resize_hwc_u8(l1_small, p1)

        s2 = _gaussian5_hw4(_downsample2x_hw4(s1), passes=passes_2)
        l2_small = s2[:, :, :3] / np.maximum(s2[:, :, 3:4], np.float32(1e-5))
        l2 = _resize_hwc_u8(l2_small, p2)

        s3 = _gaussian5_hw4(_downsample2x_hw4(_downsample2x_hw4(s2)), passes=passes_3)
        l3_small = s3[:, :, :3] / np.maximum(s3[:, :, 3:4], np.float32(1e-5))
        l3 = _resize_hwc_u8(l3_small, p3)

        return l1, l2, l3

    def _composite_bokeh(
        self,
        l0: np.ndarray,
        l1: np.ndarray,
        l2: np.ndarray,
        l3: np.ndarray,
        coc_u8: np.ndarray,
    ) -> np.ndarray:
        return _composite_cpu_u8(l0, l1, l2, l3, coc_u8)

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

    def apply_depth_bokeh(
        self,
        frame: np.ndarray,
        depth_raw: np.ndarray,
        *,
        pixel_format: str | None = None,
    ) -> np.ndarray:
        """Render variable portrait bokeh onto a native packed frame from a (S, S) depth map."""
        fmt = (pixel_format or ("YUYV" if frame.ndim == 3 and frame.shape[-1] == 4 else "RGB3")).upper()
        if fmt not in NATIVE_PIXEL_FORMATS:
            raise ValueError(f"Unsupported pixel format {fmt!r}; expected one of {NATIVE_PIXEL_FORMATS}")
        h = frame.shape[0]
        w = frame.shape[1] * 2 if is_yuv422(fmt) else frame.shape[1]
        self._ensure_plans(w, h)

        coc_unscaled = self._compute_coc_full(depth_raw, frame, pixel_format=fmt)
        coc_scaled = np.clip(coc_unscaled * np.float32(self.blur_strength), 0.0, 1.0)
        self.last_coc = coc_scaled

        if self.blur_strength <= 0.0:
            return np.ascontiguousarray(frame, dtype=np.uint8)

        l0 = np.ascontiguousarray(frame_to_channels(frame, fmt), dtype=np.uint8)
        l1, l2, l3 = self._build_blur_pyramid(l0, coc_unscaled)
        coc_u8 = np.clip(np.round(coc_scaled * np.float32(255.0)), 0, 255).astype(np.uint8)
        composited = self._composite_bokeh(l0, l1, l2, l3, coc_u8)
        return channels_to_frame(composited, fmt)

    def process(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        pixel_format = (
            ctx.pixel_format.upper()
            if ctx is not None
            else ("YUYV" if frame.ndim == 3 and frame.shape[-1] == 4 else "RGB3")
        )
        if pixel_format not in NATIVE_PIXEL_FORMATS:
            raise ValueError(f"Unsupported pixel format {pixel_format!r}; expected one of {NATIVE_PIXEL_FORMATS}")

        h = frame.shape[0]
        if is_yuv422(pixel_format):
            if frame.ndim != 3 or frame.shape[2] != 4:
                raise ValueError(f"Expected packed {pixel_format} pairs, got {frame.shape}")
            pair_width = frame.shape[1]
            w = ctx.width if ctx is not None else pair_width * 2
            if pair_width * 2 != w:
                raise ValueError(f"Packed {pixel_format} width does not match frame context ({pair_width * 2} != {w})")
        else:
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"Expected {pixel_format} frame shaped (H, W, 3), got {frame.shape}")
            w = frame.shape[1]
            if ctx is not None and ctx.width != w:
                raise ValueError(f"{pixel_format} frame width does not match frame context ({w} != {ctx.width})")
        if ctx is not None and ctx.height != h:
            raise ValueError(f"{pixel_format} frame height does not match frame context ({h} != {ctx.height})")

        if self._model is None or self._frame_shape != (h, w):
            self.setup(w, h, ctx.fps if ctx else 30.0)
        assert self._model is not None

        model_input = model_luma_input(frame, pixel_format)
        if not self.async_depth:
            res = self._model.infer(model_input)
            self.last_result = res
            self._depth_updates += 1
            return self.apply_depth_bokeh(frame, res.depth, pixel_format=pixel_format)

        latest_copy = frame.copy() if pixel_format == "RGB3" else model_input
        with self._depth_lock:
            depth = self._latest_depth
            self._latest_frame = latest_copy
        if depth is None:
            res = self._model.infer(latest_copy)
            with self._depth_lock:
                self._latest_depth = res.depth
                self.last_result = res
                self._depth_updates += 1
                depth = res.depth
            self._start_depth_worker()
        else:
            self._frame_ready_event.set()
        return self.apply_depth_bokeh(frame, depth, pixel_format=pixel_format)

    def teardown(self) -> None:
        self._stop_depth_worker()
        self._reset_temporal_bokeh_state()
        self._frame_shape = None
        self._plan_full_to_low = None
        self._plan_low_to_full = None
        self._plan_pyr_to_full = None
        if self._model is not None:
            self._model.close()
            self._model = None
        super().teardown()
