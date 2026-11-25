"""PyTorch GPU Video Depth Anything streaming monocular depth -> Turbo colormap video effect."""

from __future__ import annotations

from ..base import ComputeDevice
from ...models.video_depth_anything.effect_base import VideoDepthEffectBase


class GpuVideoDepthEffect(VideoDepthEffectBase):
    """Runs streaming Video Depth Anything on GPU using PyTorch."""

    supported_devices = (ComputeDevice.GPU,)

    def __init__(self, *, device: ComputeDevice | str = ComputeDevice.GPU, **kwargs: object) -> None:
        super().__init__(device=device, **kwargs)  # type: ignore[arg-type]
