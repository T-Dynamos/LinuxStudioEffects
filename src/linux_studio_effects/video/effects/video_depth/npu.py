"""AMD Ryzen AI NPU Video Depth Anything streaming monocular depth -> Turbo colormap video effect."""

from __future__ import annotations

from ..base import ComputeDevice
from ...models.video_depth_anything.effect_base import VideoDepthEffectBase


class VideoDepthEffect(VideoDepthEffectBase):
    """Runs streaming Video Depth Anything ViT backbone on NPU1 (AIE2) and Temporal DPT head on host."""

    supported_devices = (ComputeDevice.NPU,)
    effect_name = "video_depth"

    def __init__(self, *, device: ComputeDevice | str = ComputeDevice.NPU, **kwargs: object) -> None:
        super().__init__(device=device, **kwargs)  # type: ignore[arg-type]
