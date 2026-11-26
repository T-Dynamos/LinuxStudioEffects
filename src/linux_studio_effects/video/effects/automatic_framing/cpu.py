"""CPU Automatic Framing via MODNet portrait tracking and smooth pan/crop/zoom."""

from __future__ import annotations

import numpy as np

from ...models.modnet import (
    AsyncModnetWorker,
    ModnetConfig,
    ModnetModel,
    ModnetWeights,
    SegmentationResult,
    create_synthetic_weights,
    get_modnet_config,
    load_from_hf,
)
from ...models.modnet.host_ops import resize_hwc_u8
from ...models.video_depth_anything.host_ops import (
    crop_bilinear_resize_rgb_u8_fast,
    crop_bilinear_resize_yuyv_u8_fast,
)
from ...pixels import NATIVE_PIXEL_FORMATS, channels_to_frame, frame_to_channels, is_yuv422
from ..base import ComputeDevice, FrameContext, FrameEffect

_QUALITY_PRESETS: dict[str, int] = {
    "fast": 160,
    "balanced": 256,
    "high": 384,
}


def estimate_framing_target(
    alpha: np.ndarray,
    *,
    min_zoom: float = 1.0,
    max_zoom: float = 2.0,
    zoom_bias: float = 1.0,
    headroom: float = 0.08,
    side_margin: float = 0.10,
    min_subject_ratio: float = 0.015,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
    """Compute target (cx, cy, zoom) and subject normalized bbox (x0, y0, x1, y1) from alpha matte."""
    ah, aw = alpha.shape
    fg = alpha >= np.float32(0.45)
    fg_ratio = float(np.mean(fg))
    if fg_ratio < min_subject_ratio:
        return None

    row_counts = np.sum(fg, axis=1)
    min_row_px = max(2, int(0.02 * aw))
    valid_rows = np.flatnonzero(row_counts >= min_row_px)
    if valid_rows.size == 0:
        valid_rows = np.flatnonzero(row_counts > 0)
    if valid_rows.size == 0:
        return None

    y_top = int(valid_rows[0])
    y_bot = int(valid_rows[-1])
    span_h = max(1, y_bot - y_top + 1)

    fg_slice = fg[y_top : y_bot + 1, :]
    col_counts = np.sum(fg_slice, axis=0)
    min_col_px = max(2, int(0.02 * span_h))
    valid_cols = np.flatnonzero(col_counts >= min_col_px)
    if valid_cols.size == 0:
        valid_cols = np.flatnonzero(col_counts > 0)
    if valid_cols.size == 0:
        return None

    x0 = int(valid_cols[0])
    x1 = int(valid_cols[-1])

    # Bias horizontal tracking toward the upper portrait region (head & upper shoulders)
    # so out-of-center hand gestures weigh less than the user's head/face.
    head_end = min(y_bot + 1, y_top + max(4, int(0.48 * span_h)))
    head_alpha = np.where(fg[y_top:head_end, :], alpha[y_top:head_end, :], np.float32(0.0))
    head_mass = float(np.sum(head_alpha))
    xs_norm = (np.arange(aw, dtype=np.float32) + np.float32(0.5)) / np.float32(aw)
    if head_mass > 1e-4:
        cx_head = float(np.sum(np.sum(head_alpha, axis=0) * xs_norm) / head_mass)
    else:
        cx_head = 0.5 * (x0 + x1 + 1) / aw

    body_alpha = np.where(fg[y_top : y_bot + 1, :], alpha[y_top : y_bot + 1, :], np.float32(0.0))
    body_mass = float(np.sum(body_alpha))
    cx_body = float(np.sum(np.sum(body_alpha, axis=0) * xs_norm) / body_mass) if body_mass > 1e-4 else cx_head
    cx = 0.65 * cx_head + 0.35 * cx_body

    fg_y0 = y_top / float(ah)
    fg_y1 = (y_bot + 1) / float(ah)
    fg_x0 = x0 / float(aw)
    fg_x1 = (x1 + 1) / float(aw)

    cy_top = max(0.0, fg_y0 - headroom)
    need_h = max(0.35, min(1.0, fg_y1 + 0.02) - cy_top)
    need_w = max(0.35, min(1.0, (fg_x1 - fg_x0) + 2.0 * side_margin))

    ideal_zoom = 1.0 / max(need_h, need_w, 1.0 / max(max_zoom, 1.0))
    target_zoom = float(np.clip(ideal_zoom * zoom_bias, min_zoom, max_zoom))

    half = 0.5 / target_zoom
    crop_y0 = float(np.clip(cy_top, 0.0, max(0.0, 1.0 - 2.0 * half)))
    target_cy = crop_y0 + half
    target_cx = float(np.clip(cx, half, 1.0 - half))
    return (target_cx, target_cy, target_zoom), (fg_x0, fg_y0, fg_x1, fg_y1)


class AutomaticFramingEffect(FrameEffect):
    """Real-time automatic camera framing with MODNet face/person localization and smooth pan/crop/zoom."""

    effect_name = "automatic_framing"

    def __init__(
        self,
        *,
        quality: str = "balanced",
        variant: str = "webcam",
        max_size: int | None = None,
        use_real_weights: bool = True,
        local_files_only: bool = False,
        weights: ModnetWeights | None = None,
        min_zoom: float = 1.0,
        max_zoom: float = 2.0,
        zoom_bias: float = 1.0,
        headroom: float = 0.08,
        side_margin: float = 0.10,
        pan_smoothing: float = 0.14,
        zoom_smoothing: float = 0.10,
        deadzone: float = 0.025,
        min_subject_ratio: float = 0.015,
        lost_hold_frames: int = 30,
        async_tracking: bool = True,
        track_interval_s: float = 0.20,
        name: str | None = None,
        device: ComputeDevice | str = ComputeDevice.CPU,
        enabled: bool = True,
    ) -> None:
        q_key = quality.lower().strip()
        if q_key not in _QUALITY_PRESETS:
            raise ValueError(f"quality must be one of {tuple(_QUALITY_PRESETS)}, got {quality!r}")
        resolved_max = int(_QUALITY_PRESETS[q_key] if max_size is None else max_size)

        super().__init__(name=name, device=device, enabled=enabled)
        self.quality = q_key
        self.modnet_config: ModnetConfig = get_modnet_config(variant, max_size=resolved_max)
        self.use_real_weights = use_real_weights
        self.local_files_only = local_files_only
        self._explicit_weights = weights

        self._worker = AsyncModnetWorker(track_interval_s, thread_name="modnet-framing-worker")
        self._model: ModnetModel | None = None
        self._frame_shape: tuple[int, int] | None = None

        self.set_params(
            min_zoom=min_zoom,
            max_zoom=max_zoom,
            zoom_bias=zoom_bias,
            headroom=headroom,
            side_margin=side_margin,
            pan_smoothing=pan_smoothing,
            zoom_smoothing=zoom_smoothing,
            deadzone=deadzone,
            min_subject_ratio=min_subject_ratio,
            lost_hold_frames=lost_hold_frames,
            async_tracking=async_tracking,
            track_interval_s=track_interval_s,
        )

        self._cur_cx: float = 0.5
        self._cur_cy: float = 0.5
        self._cur_zoom: float = 1.0
        self._target_cx: float = 0.5
        self._target_cy: float = 0.5
        self._target_zoom: float = 1.0
        self._lost_frames: int = 0
        self._initialized_pose: bool = False

        self.last_result: SegmentationResult | None = None
        self.last_box: tuple[float, float, float, float] | None = None
        self.last_crop: tuple[int, int, int, int] | None = None
        self._cached_alpha_ref: np.ndarray | None = None
        self._cached_alpha_key: tuple[object, ...] | None = None
        self._cached_est: tuple[tuple[float, float, float], tuple[float, float, float, float]] | None = None

    def set_params(self, **params: object) -> AutomaticFramingEffect:
        """Validate and update live-tunable framing, zoom, and smoothing parameters."""
        if "min_zoom" in params:
            val = float(params["min_zoom"])  # type: ignore[arg-type]
            if val < 1.0:
                raise ValueError(f"min_zoom must be >= 1.0, got {val}")
            self.min_zoom = val
        if "max_zoom" in params:
            val = float(params["max_zoom"])  # type: ignore[arg-type]
            if val < 1.0:
                raise ValueError(f"max_zoom must be >= 1.0, got {val}")
            self.max_zoom = val
        if getattr(self, "max_zoom", 2.0) < getattr(self, "min_zoom", 1.0):
            raise ValueError(f"max_zoom ({self.max_zoom}) must be >= min_zoom ({self.min_zoom})")
        if "zoom_bias" in params:
            val = float(params["zoom_bias"])  # type: ignore[arg-type]
            if val <= 0.0:
                raise ValueError(f"zoom_bias must be > 0, got {val}")
            self.zoom_bias = val
        if "headroom" in params:
            val = float(params["headroom"])  # type: ignore[arg-type]
            if not 0.0 <= val <= 0.45:
                raise ValueError(f"headroom must be in [0, 0.45], got {val}")
            self.headroom = val
        if "side_margin" in params:
            val = float(params["side_margin"])  # type: ignore[arg-type]
            if not 0.0 <= val <= 0.45:
                raise ValueError(f"side_margin must be in [0, 0.45], got {val}")
            self.side_margin = val
        if "pan_smoothing" in params:
            val = float(params["pan_smoothing"])  # type: ignore[arg-type]
            if not 0.0 < val <= 1.0:
                raise ValueError(f"pan_smoothing must be in (0, 1], got {val}")
            self.pan_smoothing = val
        if "zoom_smoothing" in params:
            val = float(params["zoom_smoothing"])  # type: ignore[arg-type]
            if not 0.0 < val <= 1.0:
                raise ValueError(f"zoom_smoothing must be in (0, 1], got {val}")
            self.zoom_smoothing = val
        if "deadzone" in params:
            val = float(params["deadzone"])  # type: ignore[arg-type]
            if not 0.0 <= val <= 0.5:
                raise ValueError(f"deadzone must be in [0, 0.5], got {val}")
            self.deadzone = val
        if "min_subject_ratio" in params:
            val = float(params["min_subject_ratio"])  # type: ignore[arg-type]
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"min_subject_ratio must be in [0, 1], got {val}")
            self.min_subject_ratio = val
        if "lost_hold_frames" in params:
            val_i = int(params["lost_hold_frames"])  # type: ignore[arg-type]
            if val_i < 0:
                raise ValueError(f"lost_hold_frames must be >= 0, got {val_i}")
            self.lost_hold_frames = val_i
        if "async_tracking" in params:
            new_async = bool(params["async_tracking"])
            if getattr(self, "async_tracking", False) and not new_async:
                self._worker.stop()
            self.async_tracking = new_async
        if "track_interval_s" in params:
            val = float(params["track_interval_s"])  # type: ignore[arg-type]
            if val < 0.0:
                raise ValueError(f"track_interval_s must be >= 0, got {val}")
            self.track_interval_s = val
            self._worker.interval_s = val

        unknown = set(params) - {
            "min_zoom",
            "max_zoom",
            "zoom_bias",
            "headroom",
            "side_margin",
            "pan_smoothing",
            "zoom_smoothing",
            "deadzone",
            "min_subject_ratio",
            "lost_hold_frames",
            "async_tracking",
            "track_interval_s",
        }
        if unknown:
            raise ValueError(f"Unknown or non-live parameter(s): {sorted(unknown)}")
        return self

    def reset_tracking(self) -> None:
        """Reset pan/zoom smoothing state back to the full uncropped frame."""
        self._cur_cx = 0.5
        self._cur_cy = 0.5
        self._cur_zoom = float(self.min_zoom)
        self._target_cx = 0.5
        self._target_cy = 0.5
        self._target_zoom = float(self.min_zoom)
        self._lost_frames = 0
        self._initialized_pose = False
        self.last_box = None
        self.last_crop = None
        self._cached_alpha_ref = None
        self._cached_alpha_key = None
        self._cached_est = None

    @property
    def tracker_updates(self) -> int:
        return self._worker.updates

    @property
    def tracker_lag_frames(self) -> int:
        return self._worker.lag_frames

    @property
    def tracker_lag_ms(self) -> float:
        return self._worker.lag_ms

    def setup(self, width: int, height: int, fps: float) -> None:
        super().setup(width, height, fps)
        self._worker.stop()
        self.reset_tracking()
        if self._model is None:
            if self._explicit_weights is not None:
                weights = self._explicit_weights
            elif self.use_real_weights:
                weights = load_from_hf(self.modnet_config, local_files_only=self.local_files_only)
            else:
                weights = create_synthetic_weights(self.modnet_config)
            self._model = ModnetModel(self.modnet_config, weights, backend=self.active_device.value)
        self._frame_shape = (height, width)

    def update_framing_from_alpha(self, alpha: np.ndarray) -> tuple[float, float, float]:
        """Update smoothed normalized framing (cx, cy, zoom) from a predicted alpha matte."""
        key = (
            id(alpha),
            alpha.shape,
            self.min_zoom,
            self.max_zoom,
            self.zoom_bias,
            self.headroom,
            self.side_margin,
            self.min_subject_ratio,
        )
        if self._cached_alpha_ref is alpha and self._cached_alpha_key == key:
            est = self._cached_est
        else:
            est = estimate_framing_target(
                alpha,
                min_zoom=self.min_zoom,
                max_zoom=self.max_zoom,
                zoom_bias=self.zoom_bias,
                headroom=self.headroom,
                side_margin=self.side_margin,
                min_subject_ratio=self.min_subject_ratio,
            )
            self._cached_alpha_ref = alpha
            self._cached_alpha_key = key
            self._cached_est = est
        if est is None:
            self._lost_frames += 1
            if self._lost_frames > self.lost_hold_frames:
                self._target_cx, self._target_cy, self._target_zoom = 0.5, 0.5, float(self.min_zoom)
        else:
            (tx, ty, tz), bbox = est
            self.last_box = bbox
            self._lost_frames = 0
            if not self._initialized_pose:
                self._cur_cx = self._target_cx = tx
                self._cur_cy = self._target_cy = ty
                self._cur_zoom = self._target_zoom = tz
                self._initialized_pose = True
                return self._cur_cx, self._cur_cy, self._cur_zoom
            if (
                abs(tx - self._target_cx) > self.deadzone
                or abs(ty - self._target_cy) > self.deadzone
                or abs(tz - self._target_zoom) > self.deadzone * 1.5
            ):
                self._target_cx, self._target_cy, self._target_zoom = tx, ty, tz

        p_a = self.pan_smoothing
        z_a = self.zoom_smoothing
        self._cur_cx += p_a * (self._target_cx - self._cur_cx)
        self._cur_cy += p_a * (self._target_cy - self._cur_cy)
        self._cur_zoom += z_a * (self._target_zoom - self._cur_zoom)
        self._cur_zoom = float(np.clip(self._cur_zoom, self.min_zoom, self.max_zoom))
        half = 0.5 / self._cur_zoom
        self._cur_cx = float(np.clip(self._cur_cx, half, 1.0 - half))
        self._cur_cy = float(np.clip(self._cur_cy, half, 1.0 - half))
        return self._cur_cx, self._cur_cy, self._cur_zoom

    def apply_framing(
        self,
        frame: np.ndarray,
        alpha: np.ndarray,
        *,
        pixel_format: str | None = None,
    ) -> np.ndarray:
        """Apply smoothed pan/crop/zoom framing onto a native packed frame."""
        fmt = (pixel_format or ("YUYV" if frame.ndim == 3 and frame.shape[-1] == 4 else "RGB3")).upper()
        if fmt not in NATIVE_PIXEL_FORMATS:
            raise ValueError(f"Unsupported pixel format {fmt!r}; expected one of {NATIVE_PIXEL_FORMATS}")

        cx, cy, zoom = self.update_framing_from_alpha(alpha)
        yuv = is_yuv422(fmt)
        h = frame.shape[0]
        w = frame.shape[1] * 2 if yuv else frame.shape[1]

        crop_w = max(4, min(w, int(round(w / zoom))))
        crop_h = max(4, min(h, int(round(h / zoom))))
        x0 = int(round(cx * w - crop_w * 0.5))
        y0 = int(round(cy * h - crop_h * 0.5))
        x0 = max(0, min(w - crop_w, x0))
        y0 = max(0, min(h - crop_h, y0))
        self.last_crop = (x0, y0, crop_w, crop_h)

        if crop_w >= w and crop_h >= h:
            return np.ascontiguousarray(frame, dtype=np.uint8)

        if yuv:
            fast_yuyv = crop_bilinear_resize_yuyv_u8_fast(frame, x0, y0, crop_w, crop_h, h, w, fmt)
            if fast_yuyv is not None:
                return fast_yuyv
        else:
            fast_rgb = crop_bilinear_resize_rgb_u8_fast(frame, x0, y0, crop_w, crop_h, h, w)
            if fast_rgb is not None:
                return fast_rgb

        channels = frame_to_channels(frame, fmt)
        cropped = channels[y0 : y0 + crop_h, x0 : x0 + crop_w]
        resized = resize_hwc_u8(cropped, h, w)
        return channels_to_frame(resized, fmt)

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
            pair_w = frame.shape[1]
            w = ctx.width if ctx is not None else pair_w * 2
            if pair_w * 2 != w:
                raise ValueError(f"Packed {pixel_format} width mismatch ({pair_w * 2} != {w})")
        else:
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"Expected {pixel_format} frame shaped (H, W, 3), got {frame.shape}")
            w = frame.shape[1]
            if ctx is not None and ctx.width != w:
                raise ValueError(f"{pixel_format} frame width mismatch ({w} != {ctx.width})")
        if ctx is not None and ctx.height != h:
            raise ValueError(f"{pixel_format} frame height mismatch ({h} != {ctx.height})")

        if self._model is None or self._frame_shape != (h, w):
            self.setup(w, h, ctx.fps if ctx else 30.0)
        assert self._model is not None

        res = self._worker.step(
            self._model,
            frame,
            pixel_format,
            async_enabled=self.async_tracking,
        )
        self.last_result = res
        return self.apply_framing(frame, res.alpha, pixel_format=pixel_format)

    def teardown(self) -> None:
        self._worker.stop()
        self._frame_shape = None
        self.reset_tracking()
        if self._model is not None:
            self._model.close()
            self._model = None
        super().teardown()
