"""CPU Depth-Aware Portrait Bokeh video effect backed by Video Depth Anything."""

from __future__ import annotations

from ..base import ComputeDevice
from ...models.video_depth_anything.portrait_blur import PortraitBlurEffectBase


class PortraitBlurEffect(PortraitBlurEffectBase):
    """Runs streaming Video Depth Anything + guided-filtered portrait bokeh on CPU."""

    supported_devices = (ComputeDevice.CPU,)
    effect_name = "portrait_blur"

    def __init__(self, *, device: ComputeDevice | str = ComputeDevice.CPU, **kwargs: object) -> None:
        super().__init__(device=device, **kwargs)  # type: ignore[arg-type]
