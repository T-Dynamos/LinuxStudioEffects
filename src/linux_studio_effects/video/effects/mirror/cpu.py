"""CPU horizontal mirror video effect."""

from __future__ import annotations

import numpy as np

from ..base import ComputeDevice, FrameContext, FrameEffect
from ...pixels import is_yuv422


class MirrorEffect(FrameEffect):
    """Horizontally mirror the frame (natural webcam self-view)."""

    effect_name = "mirror"

    def process(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        pixel_format = ctx.pixel_format if ctx else ("YUYV" if frame.ndim == 3 and frame.shape[2] == 4 else "RGB3")
        if is_yuv422(pixel_format):
            mirrored = frame[:, ::-1, :].copy()
            y0_index, y1_index = (0, 2) if pixel_format.upper() == "YUYV" else (1, 3)
            y0 = mirrored[:, :, y0_index].copy()
            mirrored[:, :, y0_index] = mirrored[:, :, y1_index]
            mirrored[:, :, y1_index] = y0
            return mirrored
        return np.ascontiguousarray(frame[:, ::-1, :])
