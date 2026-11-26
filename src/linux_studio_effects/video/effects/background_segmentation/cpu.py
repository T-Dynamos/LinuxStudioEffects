"""CPU background segmentation (MODNet) supporting background blur, image/color replacement, and matte view."""

from __future__ import annotations

from pathlib import Path

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
from ...models.modnet.host_ops import (
    alpha_composite_bg_u8,
    bilinear_resize_clamp01,
    downsample2x_bg_hw4_alpha,
    extract_luma_low,
    resize_hwc_u8,
    unpremul_upsample_bg_composite,
)
from ...models.video_depth_anything import host_ops as vda_host_ops
from ...pixels import NATIVE_PIXEL_FORMATS, channels_to_frame, frame_to_channels, is_yuv422
from ...virtual_cam import encode_rgb_frame, encode_rgb_to_yuyv, resize_rgb
from ..base import ComputeDevice, FrameContext, FrameEffect

_QUALITY_PRESETS: dict[str, dict[str, object]] = {
    "fast": {"max_size": 256, "blur_passes": (1, 2, 4)},
    "balanced": {"max_size": 384, "blur_passes": (2, 4, 6)},
    "high": {"max_size": 512, "blur_passes": (3, 6, 8)},
}


def _downsample2x_hw4(arr: np.ndarray) -> np.ndarray:
    return vda_host_ops.downsample2x_hw4(arr)


def _gaussian5_hw4(arr: np.ndarray, passes: int = 1) -> np.ndarray:
    return vda_host_ops.gaussian5_hw4(arr, passes=passes)


def _load_image_rgb(source: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(source, np.ndarray):
        if source.ndim != 3 or source.shape[2] != 3:
            raise ValueError(f"Expected background_image with shape (H, W, 3), got {source.shape}")
        return np.ascontiguousarray(source, dtype=np.uint8)
    import av

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Background image not found: {path}")
    with av.open(str(path)) as cont:
        return next(cont.decode(video=0)).to_ndarray(format="rgb24")


def _encode_rgb_for_native_channels(rgb: np.ndarray, pixel_format: str) -> np.ndarray:
    fmt = pixel_format.upper()
    if fmt == "UYVY":
        yuyv = encode_rgb_to_yuyv(rgb)
        uyvy = np.ascontiguousarray(yuyv[:, :, [1, 0, 3, 2]])
        return frame_to_channels(uyvy, "UYVY")
    packed = encode_rgb_frame(rgb, fmt)
    return np.ascontiguousarray(frame_to_channels(packed, fmt), dtype=np.uint8)


class BackgroundSegmentationEffect(FrameEffect):
    """Real-time MODNet portrait segmentation for background blur, image, or color replacement."""

    effect_name = "background_segmentation"

    def __init__(
        self,
        *,
        mode: str = "blur",
        background_image: str | Path | np.ndarray | None = None,
        background_color: tuple[int, int, int] = (0, 177, 64),
        quality: str = "balanced",
        variant: str = "webcam",
        max_size: int | None = None,
        use_real_weights: bool = True,
        local_files_only: bool = False,
        weights: ModnetWeights | None = None,
        blur_strength: float = 1.0,
        blur_passes: tuple[int, int, int] | None = None,
        alpha_gamma: float = 1.0,
        matte_threshold: tuple[float, float] = (0.0, 1.0),
        temporal_alpha: float = 0.65,
        async_matte: bool = True,
        matte_interval_s: float = 0.15,
        name: str | None = None,
        device: ComputeDevice | str = ComputeDevice.CPU,
        enabled: bool = True,
    ) -> None:
        q_key = quality.lower().strip()
        if q_key not in _QUALITY_PRESETS:
            raise ValueError(f"quality must be one of {tuple(_QUALITY_PRESETS)}, got {quality!r}")
        preset = _QUALITY_PRESETS[q_key]
        resolved_max_size = int(preset["max_size"] if max_size is None else max_size)  # type: ignore[arg-type]
        resolved_passes = preset["blur_passes"] if blur_passes is None else blur_passes

        super().__init__(name=name, device=device, enabled=enabled)
        self.quality = q_key
        self.modnet_config: ModnetConfig = get_modnet_config(variant, max_size=resolved_max_size)
        self.use_real_weights = use_real_weights
        self.local_files_only = local_files_only
        self._explicit_weights = weights

        self._worker = AsyncModnetWorker(matte_interval_s, thread_name="modnet-matte-worker")

        self._bg_rgb_raw: np.ndarray | None = None
        self._bg_channels_cache: dict[tuple[int, int, str], np.ndarray] = {}
        self._color_channels_cache: dict[tuple[int, int, str, tuple[int, int, int]], np.ndarray] = {}
        self._prev_alpha_low: np.ndarray | None = None
        self._prev_luma_low: np.ndarray | None = None
        self._cached_alpha_key: tuple[object, ...] | None = None
        self._cached_alpha_full: np.ndarray | None = None
        self._frame_shape: tuple[int, int] | None = None

        self.set_params(
            mode=mode,
            background_image=background_image,
            background_color=background_color,
            blur_strength=blur_strength,
            blur_passes=resolved_passes,  # type: ignore[arg-type]
            alpha_gamma=alpha_gamma,
            matte_threshold=matte_threshold,
            temporal_alpha=temporal_alpha,
            async_matte=async_matte,
            matte_interval_s=matte_interval_s,
        )

        self._model: ModnetModel | None = None
        self.last_result: SegmentationResult | None = None
        self.last_alpha: np.ndarray | None = None

    def set_params(self, **params: object) -> BackgroundSegmentationEffect:
        """Validate and update live segmentation mode, background source, and refinement parameters."""
        if "mode" in params:
            m = str(params["mode"]).lower().strip()
            if m not in ("blur", "image", "color", "matte"):
                raise ValueError(f"mode must be 'blur', 'image', 'color', or 'matte', got {params['mode']!r}")
            self.mode = m
        if "background_image" in params:
            src = params["background_image"]
            self.background_image = src
            self._bg_rgb_raw = _load_image_rgb(src) if src is not None else None  # type: ignore[arg-type]
            self._bg_channels_cache.clear()
        if "background_color" in params:
            cols = tuple(int(c) for c in params["background_color"])  # type: ignore[union-attr]
            if len(cols) != 3 or any(not 0 <= c <= 255 for c in cols):
                raise ValueError(f"background_color must be (R, G, B) ints in [0, 255], got {params['background_color']!r}")
            self.background_color: tuple[int, int, int] = (cols[0], cols[1], cols[2])
            self._color_channels_cache.clear()
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
        if "alpha_gamma" in params:
            val = float(params["alpha_gamma"])  # type: ignore[arg-type]
            if val <= 0.0:
                raise ValueError(f"alpha_gamma must be > 0, got {val}")
            self.alpha_gamma = val
        if "matte_threshold" in params:
            rng = tuple(float(v) for v in params["matte_threshold"])  # type: ignore[union-attr]
            if len(rng) != 2 or not (0.0 <= rng[0] < rng[1] <= 1.0):
                raise ValueError(f"matte_threshold must satisfy 0 <= lo < hi <= 1, got {params['matte_threshold']!r}")
            self.matte_threshold: tuple[float, float] = (rng[0], rng[1])
        if "temporal_alpha" in params:
            val = float(params["temporal_alpha"])  # type: ignore[arg-type]
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"temporal_alpha must be in [0, 1], got {val}")
            self.temporal_alpha = val
        if "async_matte" in params:
            new_async = bool(params["async_matte"])
            if getattr(self, "async_matte", False) and not new_async:
                self._worker.stop()
            self.async_matte = new_async
        if "matte_interval_s" in params:
            val = float(params["matte_interval_s"])  # type: ignore[arg-type]
            if val < 0.0:
                raise ValueError(f"matte_interval_s must be >= 0, got {val}")
            self.matte_interval_s = val
            self._worker.interval_s = val

        unknown = set(params) - {
            "mode",
            "background_image",
            "background_color",
            "blur_strength",
            "blur_passes",
            "alpha_gamma",
            "matte_threshold",
            "temporal_alpha",
            "async_matte",
            "matte_interval_s",
        }
        if unknown:
            raise ValueError(f"Unknown or non-live parameter(s): {sorted(unknown)}")
        return self

    def setup(self, width: int, height: int, fps: float) -> None:
        super().setup(width, height, fps)
        self._worker.stop()
        self._prev_alpha_low = None
        self._prev_luma_low = None
        self._cached_alpha_key = None
        self._cached_alpha_full = None
        self.last_alpha = None
        if self._model is None:
            if self._explicit_weights is not None:
                weights = self._explicit_weights
            elif self.use_real_weights:
                weights = load_from_hf(self.modnet_config, local_files_only=self.local_files_only)
            else:
                weights = create_synthetic_weights(self.modnet_config)
            self._model = ModnetModel(self.modnet_config, weights, backend=self.active_device.value)
        self._frame_shape = (height, width)

    @property
    def matte_updates(self) -> int:
        return self._worker.updates

    @property
    def matte_lag_frames(self) -> int:
        return self._worker.lag_frames

    @property
    def matte_lag_ms(self) -> float:
        return self._worker.lag_ms

    def _refine_alpha(self, alpha_raw: np.ndarray, frame_or_l0: np.ndarray, pixel_format: str) -> np.ndarray:
        ah, aw = alpha_raw.shape
        fmt = pixel_format.upper()
        is_packed_yuv = is_yuv422(fmt) and frame_or_l0.ndim == 3 and frame_or_l0.shape[2] == 4
        h = int(frame_or_l0.shape[0])
        w = int(frame_or_l0.shape[1] * 2) if is_packed_yuv else int(frame_or_l0.shape[1])
        cache_key = (id(alpha_raw), ah, aw, h, w, self.temporal_alpha, self.matte_threshold, self.alpha_gamma)
        if self._cached_alpha_full is not None and self._cached_alpha_key == cache_key:
            return self._cached_alpha_full

        # Compute low-res luma directly from native frame (zero full-res unpack) for motion-aware EMA.
        if is_packed_yuv or not is_yuv422(fmt):
            y_low = extract_luma_low(frame_or_l0, fmt, ah, aw)
        else:
            y_low = None
        if y_low is None:
            l0_ch = frame_or_l0 if frame_or_l0.shape[2] == 3 else frame_to_channels(frame_or_l0, fmt)
            luma_src = l0_ch[:, :, 0:1].repeat(3, axis=2) if is_yuv422(fmt) else l0_ch
            _, y_low = vda_host_ops.extract_luma_and_downsample(
                luma_src,
                "RGB3" if is_yuv422(fmt) else fmt,
                ah,
                aw,
            )

        if self._prev_luma_low is not None and self._prev_alpha_low is not None and self._prev_alpha_low.shape == alpha_raw.shape:
            diff = np.abs(y_low - self._prev_luma_low)
            motion = np.clip((diff - np.float32(0.02)) * np.float32(10.0), 0.0, 1.0)
            w_t = np.float32(self.temporal_alpha) + np.float32(1.0 - self.temporal_alpha) * motion
            alpha_low = w_t * alpha_raw + (np.float32(1.0) - w_t) * self._prev_alpha_low
        else:
            alpha_low = alpha_raw
        self._prev_luma_low = y_low
        self._prev_alpha_low = alpha_low

        lo, hi = self.matte_threshold
        if lo > 0.0 or hi < 1.0:
            alpha_low = np.clip((alpha_low - np.float32(lo)) / np.float32(max(hi - lo, 1e-4)), 0.0, 1.0)
        if self.alpha_gamma != 1.0:
            alpha_low = np.power(alpha_low, np.float32(self.alpha_gamma))

        out = bilinear_resize_clamp01(alpha_low, h, w)
        self._cached_alpha_key = cache_key
        self._cached_alpha_full = out
        return out

    def _build_blur_pyramid_hw4(self, frame: np.ndarray, pixel_format: str, alpha_full: np.ndarray) -> np.ndarray:
        p1, p2, p3 = self.blur_passes
        s1_in = downsample2x_bg_hw4_alpha(frame, pixel_format, alpha_full)
        if s1_in is None:
            l0 = np.ascontiguousarray(frame_to_channels(frame, pixel_format), dtype=np.uint8)
            bg_weight = np.clip((np.float32(1.0) - alpha_full) * np.float32(1.0 - 1e-3) + np.float32(1e-3), 1e-3, 1.0)
            s1_in = vda_host_ops.downsample2x_bg_hw4(l0, bg_weight)

        s1 = _gaussian5_hw4(s1_in, passes=p1)
        if self.blur_strength < 0.35:
            return s1
        s2 = _gaussian5_hw4(_downsample2x_hw4(s1), passes=p2)
        if self.blur_strength < 0.75:
            return s2
        return _gaussian5_hw4(_downsample2x_hw4(_downsample2x_hw4(s2)), passes=p3)

    def _build_blurred_bg(self, l0: np.ndarray, alpha_full: np.ndarray) -> np.ndarray:
        h, w, _ = l0.shape
        s_top = self._build_blur_pyramid_hw4(l0, "RGB3", alpha_full)
        return vda_host_ops.unpremul_resize_hw4_to_rgb_u8(s_top, h, w)

    def _get_image_bg_channels(self, h: int, w: int, pixel_format: str) -> np.ndarray:
        if self._bg_rgb_raw is None:
            raise ValueError("background_image must be set when mode='image'")
        key = (h, w, pixel_format)
        cached = self._bg_channels_cache.get(key)
        if cached is not None:
            return cached
        resized_rgb = resize_rgb(self._bg_rgb_raw, w, h, mode="cover")
        bg_ch = _encode_rgb_for_native_channels(resized_rgb, pixel_format)
        self._bg_channels_cache[key] = bg_ch
        return bg_ch

    def _get_color_bg_channels(self, h: int, w: int, pixel_format: str) -> np.ndarray:
        key = (h, w, pixel_format, self.background_color)
        cached = self._color_channels_cache.get(key)
        if cached is not None:
            return cached
        rgb = np.empty((h, w, 3), dtype=np.uint8)
        rgb[:, :, 0] = self.background_color[0]
        rgb[:, :, 1] = self.background_color[1]
        rgb[:, :, 2] = self.background_color[2]
        bg_ch = _encode_rgb_for_native_channels(rgb, pixel_format)
        self._color_channels_cache[key] = bg_ch
        return bg_ch

    def apply_alpha_background(
        self,
        frame: np.ndarray,
        alpha_raw: np.ndarray,
        *,
        pixel_format: str | None = None,
    ) -> np.ndarray:
        """Composite segmented foreground onto blurred, image, color, or matte output in native format."""
        fmt = (pixel_format or ("YUYV" if frame.ndim == 3 and frame.shape[-1] == 4 else "RGB3")).upper()
        if fmt not in NATIVE_PIXEL_FORMATS:
            raise ValueError(f"Unsupported pixel format {fmt!r}; expected one of {NATIVE_PIXEL_FORMATS}")

        alpha_full = self._refine_alpha(alpha_raw, frame, fmt)
        self.last_alpha = alpha_full

        if self.mode == "matte":
            a_u8 = np.clip(np.round(alpha_full * np.float32(255.0)), 0, 255).astype(np.uint8)
            if is_yuv422(fmt):
                out_yuv = np.empty_like(frame)
                if fmt == "UYVY":
                    out_yuv[:, :, 0] = 128
                    out_yuv[:, :, 1] = a_u8[:, 0::2]
                    out_yuv[:, :, 2] = 128
                    out_yuv[:, :, 3] = a_u8[:, 1::2]
                else:
                    out_yuv[:, :, 0] = a_u8[:, 0::2]
                    out_yuv[:, :, 1] = 128
                    out_yuv[:, :, 2] = a_u8[:, 1::2]
                    out_yuv[:, :, 3] = 128
                return out_yuv
            return np.repeat(a_u8[:, :, None], 3, axis=2)

        if self.mode == "blur":
            if self.blur_strength <= 0.0:
                return np.ascontiguousarray(frame, dtype=np.uint8)
            s_top = self._build_blur_pyramid_hw4(frame, fmt, alpha_full)
            fused = unpremul_upsample_bg_composite(frame, fmt, s_top, alpha_full, self.blur_strength)
            if fused is not None:
                return fused
            h, w = alpha_full.shape
            l0 = np.ascontiguousarray(frame_to_channels(frame, fmt), dtype=np.uint8)
            bg_ch = vda_host_ops.unpremul_resize_hw4_to_rgb_u8(s_top, h, w)
            blend_alpha = alpha_full
            if self.blur_strength < 1.0:
                blend_alpha = np.clip(np.float32(1.0) - np.float32(self.blur_strength) * (np.float32(1.0) - alpha_full), 0.0, 1.0)
            return channels_to_frame(alpha_composite_bg_u8(l0, bg_ch, blend_alpha), fmt)

        l0 = np.ascontiguousarray(frame_to_channels(frame, fmt), dtype=np.uint8)
        h, w, _ = l0.shape
        if self.mode == "image":
            bg_ch = self._get_image_bg_channels(h, w, fmt)
            out_ch = alpha_composite_bg_u8(l0, bg_ch, alpha_full)
        else:
            bg_ch = self._get_color_bg_channels(h, w, fmt)
            out_ch = alpha_composite_bg_u8(l0, bg_ch, alpha_full)

        return channels_to_frame(out_ch, fmt)

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
            async_enabled=self.async_matte,
        )
        self.last_result = res
        return self.apply_alpha_background(frame, res.alpha, pixel_format=pixel_format)

    def teardown(self) -> None:
        self._worker.stop()
        self._prev_alpha_low = None
        self._prev_luma_low = None
        self._cached_alpha_key = None
        self._cached_alpha_full = None
        self.last_alpha = None
        self._bg_channels_cache.clear()
        self._color_channels_cache.clear()
        self._frame_shape = None
        if self._model is not None:
            self._model.close()
            self._model = None
        super().teardown()
