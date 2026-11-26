"""CPU depth-guided 3D Portrait Light effect backed by streaming Video Depth Anything."""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from ...models.video_depth_anything import host_ops as vda_host_ops
from ...models.video_depth_anything.config import VideoDepthConfig, get_video_depth_config
from ...models.video_depth_anything.model import InferenceResult, VideoDepthAnythingModel
from ...models.video_depth_anything.weights import VideoDepthWeights, create_synthetic_weights, load_from_hf
from ...models.video_depth_anything.worker import AsyncDepthWorker
from ...pixels import (
    NATIVE_PIXEL_FORMATS,
    is_yuv422,
    pack_yuv422,
    split_yuv422,
)
from ..base import ComputeDevice, FrameContext, FrameEffect

_SRC_PATH = Path(__file__).with_name("_light_ops.c")
_REQUIRED_CPU_FLAGS = frozenset({"avx512f", "avx512bw", "avx512vl", "fma"})
_C_FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
_C_U8_PTR = ctypes.POINTER(ctypes.c_uint8)


def _load_light_lib() -> ctypes.CDLL | None:
    if not _SRC_PATH.is_file():
        return None
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    if not any(
        line.startswith("flags") and _REQUIRED_CPU_FLAGS.issubset(set(line.split(":", 1)[1].split()))
        for line in cpuinfo.splitlines()
    ):
        return None
    src_hash = hashlib.sha256(_SRC_PATH.read_bytes()).hexdigest()[:16]
    cache_dir = Path.home() / ".cache" / "linux_studio_effects"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        cache_dir = Path(tempfile.gettempdir())
    so_path = cache_dir / f"_light_ops_{src_hash}.so"
    if not so_path.is_file():
        gcc = shutil.which("gcc")
        if gcc is None:
            return None
        tmp_so = cache_dir / f"_light_ops_{src_hash}.{os.getpid()}.tmp.so"
        proc = subprocess.run(
            [gcc, "-O3", "-mavx512f", "-mavx512bw", "-mavx512vl", "-mfma", "-shared", "-fPIC", "-o", str(tmp_so), str(_SRC_PATH), "-lm"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not tmp_so.is_file():
            tmp_so.unlink(missing_ok=True)
            return None
        os.replace(tmp_so, so_path)
    try:
        lib = ctypes.CDLL(str(so_path))
    except OSError:
        return None
    lib.relight_rgb_u8_f32.argtypes = [_C_U8_PTR, _C_FLOAT_PTR, _C_U8_PTR, ctypes.c_int64]
    lib.relight_rgb_u8_f32.restype = None
    lib.relight_luma_u8_f32.argtypes = [_C_U8_PTR, _C_FLOAT_PTR, _C_U8_PTR, ctypes.c_int64]
    lib.relight_luma_u8_f32.restype = None
    return lib


_LIGHT_LIB: ctypes.CDLL | None = _load_light_lib()


def _box_filter_2d(arr: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return arr
    h, w = arr.shape
    win = np.float32(2 * radius + 1)
    px = np.pad(arr, ((0, 0), (radius, radius)), mode="edge")
    cs_x = np.empty((h, w + 2 * radius + 1), dtype=np.float32)
    cs_x[:, 0] = 0.0
    np.cumsum(px, axis=1, dtype=np.float32, out=cs_x[:, 1:])
    bx = (cs_x[:, 2 * radius + 1 :] - cs_x[:, :w]) / win

    py = np.pad(bx, ((radius, radius), (0, 0)), mode="edge")
    cs_y = np.empty((h + 2 * radius + 1, w), dtype=np.float32)
    cs_y[0, :] = 0.0
    np.cumsum(py, axis=0, dtype=np.float32, out=cs_y[1:, :])
    return (cs_y[2 * radius + 1 :, :] - cs_y[:h, :]) / win


def _relight_rgb_u8(src_rgb: np.ndarray, gain_hw3: np.ndarray) -> np.ndarray:
    src_c = np.ascontiguousarray(src_rgb, dtype=np.uint8)
    gain_c = np.ascontiguousarray(gain_hw3, dtype=np.float32)
    if _LIGHT_LIB is not None:
        out = np.empty_like(src_c)
        _LIGHT_LIB.relight_rgb_u8_f32(
            src_c.ctypes.data_as(_C_U8_PTR),
            gain_c.ctypes.data_as(_C_FLOAT_PTR),
            out.ctypes.data_as(_C_U8_PTR),
            src_c.shape[0] * src_c.shape[1],
        )
        return out
    v = src_c.astype(np.float32) * np.float32(1.0 / 255.0)
    g = np.maximum(gain_c, np.float32(0.0))
    out_f = ((g * v) / (np.float32(1.0) + (g - np.float32(1.0)) * v)) * np.float32(255.0)
    return np.clip(np.round(out_f), 0.0, 255.0).astype(np.uint8)


def _relight_luma_u8(src_y: np.ndarray, gain_hw: np.ndarray) -> np.ndarray:
    src_c = np.ascontiguousarray(src_y, dtype=np.uint8)
    gain_c = np.ascontiguousarray(gain_hw, dtype=np.float32)
    if _LIGHT_LIB is not None:
        out = np.empty_like(src_c)
        _LIGHT_LIB.relight_luma_u8_f32(
            src_c.ctypes.data_as(_C_U8_PTR),
            gain_c.ctypes.data_as(_C_FLOAT_PTR),
            out.ctypes.data_as(_C_U8_PTR),
            src_c.size,
        )
        return out
    v = src_c.astype(np.float32) * np.float32(1.0 / 255.0)
    g = np.maximum(gain_c, np.float32(0.0))
    out_f = ((g * v) / (np.float32(1.0) + (g - np.float32(1.0)) * v)) * np.float32(255.0)
    return np.clip(np.round(out_f), 0.0, 255.0).astype(np.uint8)


_QUALITY_PRESETS: dict[str, dict[str, object]] = {
    "fast": {"variant": "small", "input_size": 196},
    "balanced": {"variant": "base", "input_size": 252},
    "high": {"variant": "base", "input_size": 336},
}


class PortraitLightEffect(FrameEffect):
    """Depth-guided 3D virtual key-light simulation for face illumination & exposure balancing."""

    effect_name = "portrait_light"

    def __init__(
        self,
        *,
        quality: str = "balanced",
        variant: str | None = None,
        input_size: int | None = None,
        use_real_weights: bool = True,
        local_files_only: bool = False,
        weights: VideoDepthWeights | None = None,
        light_pos: tuple[float, float, float] = (0.35, 0.25, 1.15),
        intensity: float = 1.65,
        ambient: float = 0.92,
        bg_dim: float = 0.20,
        light_radius: float = 0.75,
        light_color: tuple[float, float, float] = (1.0, 0.96, 0.90),
        normal_strength: float = 3.2,
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
        q_key = quality.lower().strip()
        if q_key not in _QUALITY_PRESETS:
            raise ValueError(f"quality must be one of {tuple(_QUALITY_PRESETS)}, got {quality!r}")
        preset = _QUALITY_PRESETS[q_key]
        resolved_variant = str(preset["variant"] if variant is None else variant)
        resolved_input_size = int(preset["input_size"] if input_size is None else input_size)  # type: ignore[arg-type]

        super().__init__(name=name, device=device, enabled=enabled)
        self.quality = q_key
        self.depth_config: VideoDepthConfig = get_video_depth_config(resolved_variant, input_size=resolved_input_size)
        self.use_real_weights = use_real_weights
        self.local_files_only = local_files_only
        self._explicit_weights = weights

        self._depth_worker = AsyncDepthWorker(depth_interval_s, thread_name="vda-light-worker")

        self.set_params(
            light_pos=light_pos,
            intensity=intensity,
            ambient=ambient,
            bg_dim=bg_dim,
            light_radius=light_radius,
            light_color=light_color,
            normal_strength=normal_strength,
            normal_smooth_radius=normal_smooth_radius,
            depth_focus=depth_focus,
            depth_softness=depth_softness,
            depth_percentiles=depth_percentiles,
            temporal_alpha=temporal_alpha,
            visualize_light=visualize_light,
            async_depth=async_depth,
            depth_interval_s=depth_interval_s,
        )

        self._model: VideoDepthAnythingModel | None = None
        self._frame_shape: tuple[int, int] | None = None
        self._ema_lo: float | None = None
        self._ema_hi: float | None = None
        self._prev_depth_norm: np.ndarray | None = None
        self._coord_xy: tuple[np.ndarray, np.ndarray] | None = None

        self.last_result: InferenceResult | None = None
        self.last_light_map: np.ndarray | None = None

    def set_params(self, **params: object) -> PortraitLightEffect:
        """Validate and update live-tunable 3D light position, color, and falloff controls."""
        if "light_pos" in params:
            lp = tuple(float(v) for v in params["light_pos"])  # type: ignore[union-attr]
            if len(lp) != 3 or lp[2] <= 0.05:
                raise ValueError(f"light_pos must be (x, y, z) with z > 0.05, got {params['light_pos']!r}")
            self.light_pos: tuple[float, float, float] = (lp[0], lp[1], lp[2])
        if "intensity" in params:
            val = float(params["intensity"])  # type: ignore[arg-type]
            if val < 0.0:
                raise ValueError(f"intensity must be >= 0, got {val}")
            self.intensity = val
        if "ambient" in params:
            val = float(params["ambient"])  # type: ignore[arg-type]
            if val < 0.0:
                raise ValueError(f"ambient must be >= 0, got {val}")
            self.ambient = val
        if "bg_dim" in params:
            val = float(params["bg_dim"])  # type: ignore[arg-type]
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"bg_dim must be in [0, 1], got {val}")
            self.bg_dim = val
        if "light_radius" in params:
            val = float(params["light_radius"])  # type: ignore[arg-type]
            if val <= 0.05:
                raise ValueError(f"light_radius must be > 0.05, got {val}")
            self.light_radius = val
        if "light_color" in params:
            lc = tuple(float(v) for v in params["light_color"])  # type: ignore[union-attr]
            if len(lc) != 3 or any(v < 0.0 for v in lc):
                raise ValueError(f"light_color must be 3 non-negative floats (R, G, B), got {params['light_color']!r}")
            self.light_color: tuple[float, float, float] = (lc[0], lc[1], lc[2])
        if "normal_strength" in params:
            val = float(params["normal_strength"])  # type: ignore[arg-type]
            if val < 0.0:
                raise ValueError(f"normal_strength must be >= 0, got {val}")
            self.normal_strength = val
        if "normal_smooth_radius" in params:
            val_i = int(params["normal_smooth_radius"])  # type: ignore[arg-type]
            if val_i < 0:
                raise ValueError(f"normal_smooth_radius must be >= 0, got {val_i}")
            self.normal_smooth_radius = val_i
        if "depth_focus" in params:
            val = float(params["depth_focus"])  # type: ignore[arg-type]
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"depth_focus must be in [0, 1], got {val}")
            self.depth_focus = val
        if "depth_softness" in params:
            val = float(params["depth_softness"])  # type: ignore[arg-type]
            if val <= 0.0:
                raise ValueError(f"depth_softness must be > 0, got {val}")
            self.depth_softness = val
        if "depth_percentiles" in params:
            pcts = tuple(float(v) for v in params["depth_percentiles"])  # type: ignore[union-attr]
            if len(pcts) != 2 or not (0.0 <= pcts[0] < pcts[1] <= 100.0):
                raise ValueError(f"depth_percentiles must satisfy 0 <= lo < hi <= 100, got {params['depth_percentiles']!r}")
            self.depth_percentiles: tuple[float, float] = (pcts[0], pcts[1])
        if "temporal_alpha" in params:
            val = float(params["temporal_alpha"])  # type: ignore[arg-type]
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"temporal_alpha must be in [0, 1], got {val}")
            self.temporal_alpha = val
        if "visualize_light" in params:
            self.visualize_light = bool(params["visualize_light"])
        if "async_depth" in params:
            new_async = bool(params["async_depth"])
            if getattr(self, "async_depth", False) and not new_async:
                self._depth_worker.stop()
            self.async_depth = new_async
        if "depth_interval_s" in params:
            val = float(params["depth_interval_s"])  # type: ignore[arg-type]
            if val < 0.0:
                raise ValueError(f"depth_interval_s must be >= 0, got {val}")
            self.depth_interval_s = val
            self._depth_worker.interval_s = val

        unknown = set(params) - {
            "light_pos",
            "intensity",
            "ambient",
            "bg_dim",
            "light_radius",
            "light_color",
            "normal_strength",
            "normal_smooth_radius",
            "depth_focus",
            "depth_softness",
            "depth_percentiles",
            "temporal_alpha",
            "visualize_light",
            "async_depth",
            "depth_interval_s",
        }
        if unknown:
            raise ValueError(f"Unknown or non-live parameter(s): {sorted(unknown)}")
        return self

    def setup(self, width: int, height: int, fps: float) -> None:
        super().setup(width, height, fps)
        self._depth_worker.stop()
        self._ema_lo = None
        self._ema_hi = None
        self._prev_depth_norm = None
        self.last_light_map = None

        s = self.depth_config.input_size
        xs = np.linspace(0.0, 1.0, s, dtype=np.float32)[None, :]
        ys = np.linspace(0.0, 1.0, s, dtype=np.float32)[:, None]
        self._coord_xy = (xs, ys)

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
        self._frame_shape = (height, width)

    @property
    def depth_updates(self) -> int:
        return self._depth_worker.updates

    @property
    def depth_lag_frames(self) -> int:
        return self._depth_worker.lag_frames

    @property
    def depth_lag_ms(self) -> float:
        return self._depth_worker.lag_ms

    def compute_light_map(self, depth_raw: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        """Compute smooth 3D key-light shading map (out_h, out_w) from (S, S) depth."""
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

        # Smooth 3D surface normals (gx, gy), clamped to suppress foreground/background edge spikes
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
        # Map disparity [0..1] to subtle camera depth offset so near face gets stronger key reach
        dz = np.float32(lz) - (np.float32(1.0) - d_smooth) * np.float32(0.8)

        dist2 = dx * dx + dy * dy + dz * dz
        inv_l = np.float32(1.0) / np.sqrt(dist2 + np.float32(1e-6))
        ndotl = np.maximum(nx * (dx * inv_l) + ny * (dy * inv_l) + nz * (dz * inv_l), np.float32(0.0))
        # Half-Lambert wrap for soft portrait skin transitions without harsh shadow terminators
        wrap = np.power(np.float32(0.55) * ndotl + np.float32(0.45), np.float32(1.4))

        rad2 = np.float32(self.light_radius * self.light_radius)
        falloff = np.float32(1.0) / (np.float32(1.0) + (dx * dx + dy * dy) / rad2)

        # Smooth foreground isolation curve around depth_focus
        fg_mask = np.clip((d_smooth - np.float32(self.depth_focus - self.depth_softness)) / np.float32(self.depth_softness * 2.0), 0.0, 1.0)
        fg_mask = fg_mask * fg_mask * (np.float32(3.0) - np.float32(2.0) * fg_mask)

        base_level = np.float32(self.ambient) * (np.float32(1.0) - np.float32(self.bg_dim) * (np.float32(1.0) - fg_mask))
        key_contrib = np.float32(max(0.0, self.intensity - self.ambient)) * fg_mask * wrap * falloff
        light_low = np.clip(base_level + key_contrib, 0.05, 4.0).astype(np.float32)

        if d_smooth.shape == (out_h, out_w):
            return light_low
        return vda_host_ops.bilinear_resize_nchw(light_low[None, None, :, :], out_h, out_w, align_corners=True)[0, 0]

    def apply_portrait_light(
        self,
        frame: np.ndarray,
        depth_raw: np.ndarray,
        *,
        pixel_format: str | None = None,
    ) -> np.ndarray:
        """Apply smooth 3D key-light exposure and tint onto a native packed frame."""
        fmt = (pixel_format or ("YUYV" if frame.ndim == 3 and frame.shape[-1] == 4 else "RGB3")).upper()
        if fmt not in NATIVE_PIXEL_FORMATS:
            raise ValueError(f"Unsupported pixel format {fmt!r}; expected one of {NATIVE_PIXEL_FORMATS}")

        h = frame.shape[0]
        w = frame.shape[1] * 2 if is_yuv422(fmt) else frame.shape[1]
        if self._coord_xy is None or self._frame_shape != (h, w):
            self.setup(w, h, 30.0)

        light_hw = self.compute_light_map(depth_raw, h, w)
        self.last_light_map = light_hw

        if self.visualize_light:
            vis = np.clip(np.round((light_hw / np.float32(max(self.intensity, 1.0))) * np.float32(255.0)), 0, 255).astype(np.uint8)
            if is_yuv422(fmt):
                neutral = np.full((h, w // 2), 128, dtype=np.uint8)
                return pack_yuv422(vis, neutral, neutral, fmt)
            return np.repeat(vis[:, :, None], 3, axis=2)

        if is_yuv422(fmt):
            y, u, v = split_yuv422(frame, fmt)
            y_lit = _relight_luma_u8(y, light_hw)
            return pack_yuv422(y_lit, u, v, fmt)

        lc = self.light_color
        # Blend tint only above ambient baseline so dark room ambient isn't shifted color-wise
        base = np.float32(self.ambient)
        delta = np.maximum(light_hw - base, np.float32(0.0))
        common = np.minimum(light_hw, base)
        if fmt == "BGR3":
            gain_3 = np.stack([common + delta * np.float32(lc[2]), common + delta * np.float32(lc[1]), common + delta * np.float32(lc[0])], axis=2)
        else:
            gain_3 = np.stack([common + delta * np.float32(lc[0]), common + delta * np.float32(lc[1]), common + delta * np.float32(lc[2])], axis=2)
        return _relight_rgb_u8(frame, gain_3)

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

        res = self._depth_worker.step(self._model, frame, pixel_format, async_enabled=self.async_depth)
        self.last_result = res
        return self.apply_portrait_light(frame, res.depth, pixel_format=pixel_format)

    def teardown(self) -> None:
        self._depth_worker.stop()
        self._ema_lo = None
        self._ema_hi = None
        self._prev_depth_norm = None
        self._coord_xy = None
        self.last_light_map = None
        self._frame_shape = None
        if self._model is not None:
            self._model.close()
            self._model = None
        super().teardown()
