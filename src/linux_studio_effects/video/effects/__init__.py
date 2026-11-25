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
    CpuDepthAnythingEffect,
    FastBlurEffect,
    MirrorEffect,
    PassthroughEffect,
    SpotlightEffect,
    ZoomCropEffect,
)
from .gpu import GpuDepthAnythingEffect
from .npu import GreyscaleEffect, NpuDepthAnythingEffect

BUILTIN_EFFECTS: dict[str, Callable[[], FrameEffect]] = {
    "none": PassthroughEffect,
    "mirror": MirrorEffect,
    "zoom": ZoomCropEffect,
    "spotlight": SpotlightEffect,
    "blur": FastBlurEffect,
    "greyscale": GreyscaleEffect,
    "depth_cpu": CpuDepthAnythingEffect,
    "depth_npu": NpuDepthAnythingEffect,
}

__all__ = [
    "BUILTIN_EFFECTS",
    "CallableEffect",
    "ComputeDevice",
    "CpuDepthAnythingEffect",
    "DeviceUnavailableError",
    "EffectChain",
    "FastBlurEffect",
    "FrameContext",
    "FrameEffect",
    "GpuDepthAnythingEffect",
    "GreyscaleEffect",
    "MirrorEffect",
    "NpuDepthAnythingEffect",
    "PassthroughEffect",
    "SpotlightEffect",
    "ZoomCropEffect",
    "as_frame_effect",
    "is_device_available",
    "resolve_compute_device",
]
