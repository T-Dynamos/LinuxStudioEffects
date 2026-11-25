"""NPU-accelerated video frame effects (one effect per module)."""

from .greyscale import GreyscaleEffect
from .video_depth import NpuVideoDepthEffect

__all__ = ["GreyscaleEffect", "NpuVideoDepthEffect"]
