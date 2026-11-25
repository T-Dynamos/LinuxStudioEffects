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
    FastBlurEffect,
    MirrorEffect,
    PassthroughEffect,
    SpotlightEffect,
    ZoomCropEffect,
)

BUILTIN_EFFECTS: dict[str, Callable[[], FrameEffect]] = {
    "none": PassthroughEffect,
    "mirror": MirrorEffect,
    "zoom": ZoomCropEffect,
    "spotlight": SpotlightEffect,
    "blur": FastBlurEffect,
}

__all__ = [
    "BUILTIN_EFFECTS",
    "CallableEffect",
    "ComputeDevice",
    "DeviceUnavailableError",
    "EffectChain",
    "FastBlurEffect",
    "FrameContext",
    "FrameEffect",
    "MirrorEffect",
    "PassthroughEffect",
    "SpotlightEffect",
    "ZoomCropEffect",
    "as_frame_effect",
    "is_device_available",
    "resolve_compute_device",
]
