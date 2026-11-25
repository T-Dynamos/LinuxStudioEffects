"""Video frame effects organized by effect, with optional backend modules."""

from __future__ import annotations

from collections.abc import Callable
from ..backend import (
    available_effect_backends,
    create_effect,
    is_device_available,
    load_effect_backend,
    resolve_compute_device,
)
from .base import (
    CallableEffect,
    ComputeDevice,
    DeviceUnavailableError,
    EffectChain,
    FrameGraph,
    FrameContext,
    FrameEffect,
    as_frame_effect,
)
from .fast_blur.cpu import FastBlurEffect
from .greyscale.npu import GreyscaleEffect
from .mirror.cpu import MirrorEffect
from .passthrough.cpu import PassthroughEffect
from .portrait_blur.cpu import PortraitBlurEffect
from .spotlight.cpu import SpotlightEffect
from .video_depth.cpu import VideoDepthEffect
from .zoom_crop.cpu import ZoomCropEffect

BUILTIN_EFFECTS: dict[str, Callable[[], FrameEffect]] = {
    "none": PassthroughEffect,
    "mirror": MirrorEffect,
    "zoom": ZoomCropEffect,
    "spotlight": SpotlightEffect,
    "blur": FastBlurEffect,
    "greyscale": lambda: create_effect("greyscale"),
    "depth_cpu": lambda: create_effect("video_depth", device="cpu"),
    "depth_npu": lambda: create_effect("video_depth", device="npu"),
    "video_depth_cpu": lambda: create_effect("video_depth", device="cpu"),
    "video_depth_npu": lambda: create_effect("video_depth", device="npu"),
    "portrait_blur_cpu": lambda: create_effect("portrait_blur", device="cpu"),
    "portrait_blur_npu": lambda: create_effect("portrait_blur", device="npu"),
}

__all__ = [
    "BUILTIN_EFFECTS",
    "create_effect",
    "available_effect_backends",
    "load_effect_backend",
    "CallableEffect",
    "ComputeDevice",
    "PortraitBlurEffect",
    "DeviceUnavailableError",
    "EffectChain",
    "FrameGraph",
    "FastBlurEffect",
    "FrameContext",
    "FrameEffect",
    "GreyscaleEffect",
    "MirrorEffect",
    "PassthroughEffect",
    "SpotlightEffect",
    "ZoomCropEffect",
    "VideoDepthEffect",
    "as_frame_effect",
    "is_device_available",
    "resolve_compute_device",
]
