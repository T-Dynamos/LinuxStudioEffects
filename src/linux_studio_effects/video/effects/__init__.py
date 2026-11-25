"""Video frame effects organized by execution backend (cpu, npu, gpu)."""

from __future__ import annotations

from collections.abc import Callable

from .base import (
    CallableEffect,
    ComputeDevice,
    DeviceUnavailableError,
    EffectChain,
    FrameContext,
    FrameEffect,
    as_frame_effect,
    is_device_available,
    resolve_compute_device,
)
from .cpu import (
    CpuVideoDepthEffect,
    FastBlurEffect,
    MirrorEffect,
    PassthroughEffect,
    SpotlightEffect,
    ZoomCropEffect,
)
from .gpu import GpuVideoDepthEffect
from .npu import GreyscaleEffect, NpuVideoDepthEffect

BUILTIN_EFFECTS: dict[str, Callable[[], FrameEffect]] = {
    "none": PassthroughEffect,
    "mirror": MirrorEffect,
    "zoom": ZoomCropEffect,
    "spotlight": SpotlightEffect,
    "blur": FastBlurEffect,
    "greyscale": GreyscaleEffect,
    "depth_cpu": CpuVideoDepthEffect,
    "depth_npu": NpuVideoDepthEffect,
    "video_depth_cpu": CpuVideoDepthEffect,
    "video_depth_npu": NpuVideoDepthEffect,
}

__all__ = [
    "BUILTIN_EFFECTS",
    "CallableEffect",
    "ComputeDevice",
    "CpuVideoDepthEffect",
    "DeviceUnavailableError",
    "EffectChain",
    "FastBlurEffect",
    "FrameContext",
    "FrameEffect",
    "GpuVideoDepthEffect",
    "GreyscaleEffect",
    "MirrorEffect",
    "NpuVideoDepthEffect",
    "PassthroughEffect",
    "SpotlightEffect",
    "ZoomCropEffect",
    "as_frame_effect",
    "is_device_available",
    "resolve_compute_device",
]
