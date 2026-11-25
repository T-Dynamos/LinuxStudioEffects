"""AMD Ryzen AI NPU Depth-Aware Portrait Bokeh video effect backed by Video Depth Anything."""

from __future__ import annotations

from ..base import ComputeDevice
from ...models.video_depth_anything.portrait_blur import PortraitBlurEffectBase


class PortraitBlurEffect(PortraitBlurEffectBase):
    """Runs streaming Video Depth Anything + AIE2 4-column portrait bokeh composite on NPU1."""

    effect_name = "portrait_blur"

    def __init__(self, *, device: ComputeDevice | str = ComputeDevice.NPU, **kwargs: object) -> None:
        super().__init__(device=device, **kwargs)  # type: ignore[arg-type]
