"""CPU-backed video frame effects (one effect per module)."""

from .fast_blur import FastBlurEffect
from .mirror import MirrorEffect
from .passthrough import PassthroughEffect
from .spotlight import SpotlightEffect
from .video_depth import CpuVideoDepthEffect
from .zoom_crop import ZoomCropEffect

__all__ = [
    "CpuVideoDepthEffect",
    "FastBlurEffect",
    "MirrorEffect",
    "PassthroughEffect",
    "SpotlightEffect",
    "ZoomCropEffect",
]
