"""CPU horizontal mirror video effect."""

from __future__ import annotations

import numpy as np

from ..base import ComputeDevice, FrameContext, FrameEffect


class MirrorEffect(FrameEffect):
    """Horizontally mirror the frame (natural webcam self-view)."""

    effect_name = "mirror"

    def process(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        return np.ascontiguousarray(frame[:, ::-1, :])
