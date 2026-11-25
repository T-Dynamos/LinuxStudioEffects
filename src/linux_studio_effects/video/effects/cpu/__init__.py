"""CPU-backed video frame effects (one effect per module)."""

from .depth_anything import CpuDepthAnythingEffect
from .fast_blur import FastBlurEffect
from .mirror import MirrorEffect
from .passthrough import PassthroughEffect
from .spotlight import SpotlightEffect
from .zoom_crop import ZoomCropEffect

__all__ = [
    "CpuDepthAnythingEffect",
    "FastBlurEffect",
    "MirrorEffect",
    "PassthroughEffect",
    "SpotlightEffect",
    "ZoomCropEffect",
]
