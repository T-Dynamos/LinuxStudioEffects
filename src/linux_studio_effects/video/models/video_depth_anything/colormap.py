"""Fast Turbo-style colormap visualization for monocular video depth maps."""

from __future__ import annotations

import numpy as np

# 9-stop Turbo-inspired colormap anchors (near=warm/red-yellow, far=cool/blue-purple)
_TURBO_STOPS = np.array(
    [
        [48, 18, 59],     # far (0)
        [68, 84, 196],
        [40, 156, 232],
        [34, 210, 175],
        [124, 240, 90],
        [224, 224, 52],
        [253, 152, 42],
        [232, 70, 26],
        [144, 12, 0],     # near (255)
    ],
    dtype=np.float32,
)


def _build_turbo_lut() -> np.ndarray:
    t = np.linspace(0.0, len(_TURBO_STOPS) - 1, 256, dtype=np.float32)
    idx0 = np.floor(t).astype(np.int32)
    idx1 = np.minimum(idx0 + 1, len(_TURBO_STOPS) - 1)
    frac = (t - idx0)[:, None]
    rgb = _TURBO_STOPS[idx0] * (1.0 - frac) + _TURBO_STOPS[idx1] * frac
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


TURBO_LUT = _build_turbo_lut()


def depth_to_colormap_rgb(
    depth: np.ndarray,
    *,
    out_width: int | None = None,
    out_height: int | None = None,
    lut: np.ndarray = TURBO_LUT,
) -> np.ndarray:
    """Normalize a 2D depth map to [0, 255] and map to (H, W, 3) uint8 RGB."""
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim != 2:
        raise ValueError(f"Expected 2D depth map, got shape {d.shape}")

    lo = float(np.percentile(d, 2.0))
    hi = float(np.percentile(d, 98.0))
    scale = 255.0 / max(hi - lo, 1e-6)
    indices = np.clip(np.round((d - lo) * scale), 0, 255).astype(np.uint8)

    h, w = indices.shape
    if out_width is not None and out_height is not None and (out_width != w or out_height != h):
        y_idx = (np.arange(out_height, dtype=np.int64) * h) // out_height
        x_idx = (np.arange(out_width, dtype=np.int64) * w) // out_width
        indices = indices[y_idx[:, None], x_idx]

    return np.ascontiguousarray(lut[indices], dtype=np.uint8)
