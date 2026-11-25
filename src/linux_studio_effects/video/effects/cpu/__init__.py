"""CPU-backed video frame effects (one effect per module)."""

from .fast_blur import FastBlurEffect
from .mirror import MirrorEffect
from .passthrough import PassthroughEffect
from .portrait_blur import CpuPortraitBlurEffect
from .spotlight import SpotlightEffect
from .video_depth import CpuVideoDepthEffect
from .zoom_crop import ZoomCropEffect

__all__ = [
    "CpuPortraitBlurEffect",
    "CpuVideoDepthEffect",
    "FastBlurEffect",
    "MirrorEffect",
    "PassthroughEffect",
    "SpotlightEffect",
    "ZoomCropEffect",
]
