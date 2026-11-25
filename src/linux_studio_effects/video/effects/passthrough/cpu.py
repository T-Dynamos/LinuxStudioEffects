"""CPU passthrough (identity) video effect."""

from __future__ import annotations

import numpy as np

from ..base import ComputeDevice, FrameContext, FrameEffect


class PassthroughEffect(FrameEffect):
    """Identity effect returning the input RGB frame unchanged."""

    supported_devices = (ComputeDevice.CPU,)
    effect_name = "passthrough"

    def process(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        return frame
