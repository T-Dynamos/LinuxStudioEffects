"""CPU depth-guided Aesthetic Light effect for warm-white side / split portrait lighting."""

from __future__ import annotations

import numpy as np

from ...models.video_depth_anything import host_ops as vda_host_ops
from ...models.video_depth_anything.weights import VideoDepthWeights
from ...pixels import NATIVE_PIXEL_FORMATS, is_yuv422, pack_yuv422, split_yuv422
from ..base import ComputeDevice
from ..portrait_light.cpu import (
    PortraitLightEffect,
    _box_filter_2d,
    _relight_luma_u8,
    _relight_rgb_u8,
)

LIGHT_SOURCE_PRESETS: dict[str, tuple[float, float, float]] = {
    "left": (-0.12, 0.34, 0.68),
    "right": (1.12, 0.34, 0.68),
    "top_left": (0.04, 0.08, 0.72),
    "top_right": (0.96, 0.08, 0.72),
    "bottom_left": (0.04, 0.88, 0.72),
    "bottom_right": (0.96, 0.88, 0.72),
}

LIGHT_COLOR_PRESETS: dict[str, tuple[float, float, float]] = {
    "warm_white": (1.0, 0.90, 0.75),
    "golden": (1.0, 0.76, 0.48),
    "candle": (1.0, 0.82, 0.56),
    "soft_white": (1.0, 0.96, 0.90),
    "cool_fill": (0.80, 0.90, 1.0),
}


class AestheticLightEffect(PortraitLightEffect):
    """Cinematic directional split/side key light (e.g. warm white glow on one side of the face)."""

    effect_name = "aesthetic_light"

    def __init__(
        self,
        *,
        quality: str = "balanced",
        variant: str | None = None,
        input_size: int | None = None,
        use_real_weights: bool = True,
        local_files_only: bool = False,
        weights: VideoDepthWeights | None = None,
        source: str = "left",
        color_preset: str = "warm_white",
        light_pos: tuple[float, float, float] | None = None,
        light_color: tuple[float, float, float] | None = None,
        shadow_color: tuple[float, float, float] = (0.86, 0.92, 1.02),
        intensity: float = 2.40,
        ambient: float = 0.44,
        bg_dim: float = 0.56,
        bg_spill: float = 0.22,
        light_radius: float = 0.82,
        split_sharpness: float = 1.85,
        specular_strength: float = 0.28,
        normal_strength: float = 4.2,
        normal_smooth_radius: int = 5,
        depth_focus: float = 0.42,
        depth_softness: float = 0.22,
        depth_percentiles: tuple[float, float] = (5.0, 95.0),
        temporal_alpha: float = 0.35,
        visualize_light: bool = False,
        async_depth: bool = True,
        depth_interval_s: float = 0.05,
        name: str | None = None,
        device: ComputeDevice | str = ComputeDevice.CPU,
        enabled: bool = True,
    ) -> None:
        src_key = source.lower().strip()
        if light_pos is None:
            if src_key not in LIGHT_SOURCE_PRESETS:
                raise ValueError(f"source must be one of {tuple(LIGHT_SOURCE_PRESETS)}, got {source!r}")
            resolved_pos = LIGHT_SOURCE_PRESETS[src_key]
        else:
            src_key = "custom"
            resolved_pos = light_pos

        col_key = color_preset.lower().strip()
        if light_color is None:
            if col_key not in LIGHT_COLOR_PRESETS:
                raise ValueError(f"color_preset must be one of {tuple(LIGHT_COLOR_PRESETS)}, got {color_preset!r}")
            resolved_color = LIGHT_COLOR_PRESETS[col_key]
        else:
            col_key = "custom"
            resolved_color = light_color

        self.source = src_key
        self.color_preset = col_key
        self.shadow_color: tuple[float, float, float] = (0.86, 0.92, 1.02)
        self.split_sharpness: float = float(split_sharpness)
        self.specular_strength: float = float(specular_strength)
        self.bg_spill: float = float(bg_spill)
        self._last_key_map: np.ndarray | None = None
        self._last_base_map: np.ndarray | None = None

        super().__init__(
            quality=quality,
            variant=variant,
            input_size=input_size,
            use_real_weights=use_real_weights,
            local_files_only=local_files_only,
            weights=weights,
            light_pos=resolved_pos,
            intensity=intensity,
            ambient=ambient,
            bg_dim=bg_dim,
            light_radius=light_radius,
            light_color=resolved_color,
            normal_strength=normal_strength,
            normal_smooth_radius=normal_smooth_radius,
            depth_focus=depth_focus,
            depth_softness=depth_softness,
            depth_percentiles=depth_percentiles,
            temporal_alpha=temporal_alpha,
            visualize_light=visualize_light,
            async_depth=async_depth,
            depth_interval_s=depth_interval_s,
            name=name,
            device=device,
            enabled=enabled,
        )
        self.set_params(
            shadow_color=shadow_color,
            split_sharpness=split_sharpness,
            specular_strength=specular_strength,
            bg_spill=bg_spill,
        )

    def set_params(self, **params: object) -> AestheticLightEffect:
        """Validate and update live-tunable source side, warm tint, contrast, and light coordinates."""
        remaining = dict(params)
        if "source" in remaining:
            src_key = str(remaining.pop("source")).lower().strip()
            if src_key != "custom":
                if src_key not in LIGHT_SOURCE_PRESETS:
                    raise ValueError(f"source must be one of {tuple(LIGHT_SOURCE_PRESETS)} or 'custom', got {src_key!r}")
                if "light_pos" not in remaining:
                    remaining["light_pos"] = LIGHT_SOURCE_PRESETS[src_key]
            self.source = src_key
        elif "light_pos" in remaining:
            self.source = "custom"

        if "color_preset" in remaining:
            col_key = str(remaining.pop("color_preset")).lower().strip()
            if col_key != "custom":
                if col_key not in LIGHT_COLOR_PRESETS:
                    raise ValueError(f"color_preset must be one of {tuple(LIGHT_COLOR_PRESETS)} or 'custom', got {col_key!r}")
                if "light_color" not in remaining:
                    remaining["light_color"] = LIGHT_COLOR_PRESETS[col_key]
            self.color_preset = col_key
        elif "light_color" in remaining:
            self.color_preset = "custom"

        if "shadow_color" in remaining:
            sc = tuple(float(v) for v in remaining.pop("shadow_color"))  # type: ignore[union-attr]
            if len(sc) != 3 or any(v < 0.0 for v in sc):
                raise ValueError(f"shadow_color must be 3 non-negative floats (R, G, B), got {params['shadow_color']!r}")
            self.shadow_color = (sc[0], sc[1], sc[2])
        if "split_sharpness" in remaining:
            val = float(remaining.pop("split_sharpness"))  # type: ignore[arg-type]
            if val <= 0.0:
                raise ValueError(f"split_sharpness must be > 0, got {val}")
            self.split_sharpness = val
        if "specular_strength" in remaining:
            val = float(remaining.pop("specular_strength"))  # type: ignore[arg-type]
            if val < 0.0:
                raise ValueError(f"specular_strength must be >= 0, got {val}")
            self.specular_strength = val
        if "bg_spill" in remaining:
            val = float(remaining.pop("bg_spill"))  # type: ignore[arg-type]
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"bg_spill must be in [0, 1], got {val}")
            self.bg_spill = val

        if remaining:
            super().set_params(**remaining)
        return self

    def setup(self, width: int, height: int, fps: float) -> None:
        super().setup(width, height, fps)
        self._last_key_map = None
        self._last_base_map = None

    def _compute_split_fields(self, depth_raw: np.ndarray, out_h: int, out_w: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (base_hw, key_hw) encoding shadow fill and directional warm-white key intensity."""
        lo = float(np.percentile(depth_raw, self.depth_percentiles[0]))
        hi = float(np.percentile(depth_raw, self.depth_percentiles[1]))
        if self._ema_lo is None or self._ema_hi is None:
            self._ema_lo, self._ema_hi = lo, hi
        else:
            alpha = 1.0 if self.temporal_alpha >= 1.0 else min(0.15, self.temporal_alpha)
            self._ema_lo = alpha * lo + (1.0 - alpha) * self._ema_lo
            self._ema_hi = alpha * hi + (1.0 - alpha) * self._ema_hi

        denom = max(self._ema_hi - self._ema_lo, 1e-4)
        d_norm = np.clip((depth_raw - self._ema_lo) / denom, 0.0, 1.0).astype(np.float32)
        if self._prev_depth_norm is not None and self._prev_depth_norm.shape == d_norm.shape:
            t_a = np.float32(self.temporal_alpha)
            d_norm = t_a * d_norm + (np.float32(1.0) - t_a) * self._prev_depth_norm
        self._prev_depth_norm = d_norm

        d_smooth = _box_filter_2d(d_norm, self.normal_smooth_radius)

        gy, gx = np.gradient(d_smooth)
        max_grad = np.float32(3.0 / max(1, d_smooth.shape[0]))
        gx = np.clip(gx, -max_grad, max_grad)
        gy = np.clip(gy, -max_grad, max_grad)
        scale = np.float32(self.normal_strength * max(1, d_smooth.shape[0]) * 0.06)
        nx = -gx * scale
        ny = -gy * scale
        nz = np.ones_like(nx, dtype=np.float32)
        inv_n = np.float32(1.0) / np.sqrt(nx * nx + ny * ny + nz * nz + np.float32(1e-6))
        nx *= inv_n
        ny *= inv_n
        nz *= inv_n

        assert self._coord_xy is not None
        xs, ys = self._coord_xy
        lx, ly, lz = self.light_pos
        dx = np.float32(lx) - xs
        dy = np.float32(ly) - ys
        dz = np.float32(lz) - (np.float32(1.0) - d_smooth) * np.float32(0.8)

        dist2 = dx * dx + dy * dy + dz * dz
        inv_l = np.float32(1.0) / np.sqrt(dist2 + np.float32(1e-6))
        ldx = dx * inv_l
        ldy = dy * inv_l
        ldz = dz * inv_l

        ndotl = np.maximum(nx * ldx + ny * ldy + nz * ldz, np.float32(0.0))
        # High-contrast split wrap: strong key illumination on the lit cheek, smooth rolloff into shadow.
        wrap = np.power(np.clip(np.float32(0.75) * ndotl + np.float32(0.25), 0.0, 1.0), np.float32(self.split_sharpness))

        # Subtle Blinn-Phong cheekbone/bridge specular highlight towards camera view vector (0, 0, 1).
        hz = ldz + np.float32(1.0)
        inv_h = np.float32(1.0) / np.sqrt(ldx * ldx + ldy * ldy + hz * hz + np.float32(1e-6))
        ndoth = np.maximum((nx * ldx + ny * ldy + nz * hz) * inv_h, np.float32(0.0))
        spec = np.float32(self.specular_strength) * np.power(ndoth, np.float32(8.0))

        rad2 = np.float32(self.light_radius * self.light_radius)
        falloff = np.float32(1.0) / np.power(np.float32(1.0) + (dx * dx + dy * dy) / rad2, np.float32(1.25))

        fg_mask = np.clip(
            (d_smooth - np.float32(self.depth_focus - self.depth_softness)) / np.float32(self.depth_softness * 2.0),
            0.0,
            1.0,
        )
        fg_mask = fg_mask * fg_mask * (np.float32(3.0) - np.float32(2.0) * fg_mask)

        base_low = np.float32(self.ambient) * (np.float32(1.0) - np.float32(self.bg_dim) * (np.float32(1.0) - fg_mask))
        key_scale = np.float32(max(0.0, self.intensity - self.ambient))
        spill = np.float32(self.bg_spill) * (np.float32(1.0) - fg_mask) * (falloff * falloff)
        key_low = key_scale * (fg_mask * (wrap + spec) * falloff + spill)

        base_low = np.clip(base_low, 0.05, 4.0).astype(np.float32)
        key_low = np.clip(key_low, 0.0, 4.0).astype(np.float32)

        if d_smooth.shape == (out_h, out_w):
            return base_low, key_low
        stacked = np.stack([base_low, key_low], axis=0)[None, :, :, :]
        up = vda_host_ops.bilinear_resize_nchw(stacked, out_h, out_w, align_corners=True)[0]
        return up[0], up[1]

    def compute_light_map(self, depth_raw: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        """Compute total directional light intensity map `(out_h, out_w)` from `(S, S)` depth."""
        base_hw, key_hw = self._compute_split_fields(depth_raw, out_h, out_w)
        self._last_base_map = base_hw
        self._last_key_map = key_hw
        return np.clip(base_hw + key_hw, 0.05, 4.5).astype(np.float32)

    def apply_portrait_light(
        self,
        frame: np.ndarray,
        depth_raw: np.ndarray,
        *,
        pixel_format: str | None = None,
    ) -> np.ndarray:
        """Apply directional warm-white aesthetic key lighting and cool shadow fill to a frame."""
        fmt = (pixel_format or ("YUYV" if frame.ndim == 3 and frame.shape[-1] == 4 else "RGB3")).upper()
        if fmt not in NATIVE_PIXEL_FORMATS:
            raise ValueError(f"Unsupported pixel format {fmt!r}; expected one of {NATIVE_PIXEL_FORMATS}")

        h = frame.shape[0]
        w = frame.shape[1] * 2 if is_yuv422(fmt) else frame.shape[1]
        if self._coord_xy is None or self._frame_shape != (h, w):
            self.setup(w, h, 30.0)

        light_hw = self.compute_light_map(depth_raw, h, w)
        self.last_light_map = light_hw
        assert self._last_base_map is not None and self._last_key_map is not None
        base_hw = self._last_base_map
        key_hw = self._last_key_map

        if self.visualize_light:
            vis = np.clip(np.round((light_hw / np.float32(max(self.intensity, 1.0))) * np.float32(255.0)), 0, 255).astype(np.uint8)
            if is_yuv422(fmt):
                neutral = np.full((h, w // 2), 128, dtype=np.uint8)
                return pack_yuv422(vis, neutral, neutral, fmt)
            return np.repeat(vis[:, :, None], 3, axis=2)

        sc = self.shadow_color
        lc = self.light_color

        if is_yuv422(fmt):
            y, u, v = split_yuv422(frame, fmt)
            sc_y = np.float32(0.299 * sc[0] + 0.587 * sc[1] + 0.114 * sc[2])
            lc_y = np.float32(0.299 * lc[0] + 0.587 * lc[1] + 0.114 * lc[2])
            y_gain = base_hw * sc_y + key_hw * lc_y
            y_lit = _relight_luma_u8(y, y_gain)

            # Apply proportional warm-white / cool-shadow chroma shift on 4:2:2 U/V pairs.
            key_pair = np.float32(0.5) * (key_hw[:, 0::2] + key_hw[:, 1::2])
            base_pair = np.float32(0.5) * (base_hw[:, 0::2] + base_hw[:, 1::2])
            y_pair = (y_lit[:, 0::2].astype(np.float32) + y_lit[:, 1::2].astype(np.float32)) * np.float32(0.5 / 255.0)
            y_norm = np.float32(0.5) * (y_gain[:, 0::2] + y_gain[:, 1::2])
            r_pair = base_pair * np.float32(sc[0]) + key_pair * np.float32(lc[0])
            b_pair = base_pair * np.float32(sc[2]) + key_pair * np.float32(lc[2])
            du = (b_pair - y_norm) * np.float32(64.0) * y_pair
            dv = (r_pair - y_norm) * np.float32(64.0) * y_pair
            u_out = np.clip(np.round(u.astype(np.float32) + du), 0.0, 255.0).astype(np.uint8)
            v_out = np.clip(np.round(v.astype(np.float32) + dv), 0.0, 255.0).astype(np.uint8)
            return pack_yuv422(y_lit, u_out, v_out, fmt)

        r_gain = base_hw * np.float32(sc[0]) + key_hw * np.float32(lc[0])
        g_gain = base_hw * np.float32(sc[1]) + key_hw * np.float32(lc[1])
        b_gain = base_hw * np.float32(sc[2]) + key_hw * np.float32(lc[2])
        if fmt == "BGR3":
            gain_3 = np.stack([b_gain, g_gain, r_gain], axis=2)
        else:
            gain_3 = np.stack([r_gain, g_gain, b_gain], axis=2)
        return _relight_rgb_u8(frame, gain_3)

    def teardown(self) -> None:
        self._last_key_map = None
        self._last_base_map = None
        super().teardown()
