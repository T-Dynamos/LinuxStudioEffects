"""CPU Video Depth Anything streaming monocular depth -> Turbo colormap video effect."""

from __future__ import annotations

from ..base import ComputeDevice
from ...models.video_depth_anything.effect_base import VideoDepthEffectBase


class VideoDepthEffect(VideoDepthEffectBase):
    """Runs streaming Video Depth Anything (Small/Base/Large) on CPU via NumPy/OpenBLAS."""

    effect_name = "video_depth"

    def __init__(self, *, device: ComputeDevice | str = ComputeDevice.CPU, **kwargs: object) -> None:
        super().__init__(device=device, **kwargs)  # type: ignore[arg-type]
