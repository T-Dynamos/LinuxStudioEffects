"""NPU-accelerated video frame effects (one effect per module)."""

from .greyscale import GreyscaleEffect
from .portrait_blur import NpuPortraitBlurEffect
from .video_depth import NpuVideoDepthEffect

__all__ = ["GreyscaleEffect", "NpuPortraitBlurEffect", "NpuVideoDepthEffect"]
