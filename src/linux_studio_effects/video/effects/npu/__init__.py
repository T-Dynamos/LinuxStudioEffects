"""NPU-accelerated video frame effects (one effect per module)."""

from .depth_anything import NpuDepthAnythingEffect
from .greyscale import GreyscaleEffect

__all__ = ["GreyscaleEffect", "NpuDepthAnythingEffect"]
