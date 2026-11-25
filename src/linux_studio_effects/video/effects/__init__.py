"""Video frame effects organized by execution backend (cpu, npu, gpu)."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from pathlib import Path

from .base import (
    CallableEffect,
    ComputeDevice,
    DeviceUnavailableError,
    EffectChain,
    FrameGraph,
    FrameContext,
    FrameEffect,
    as_frame_effect,
    is_device_available,
    resolve_compute_device,
)
from .fast_blur.cpu import FastBlurEffect
from .greyscale.npu import GreyscaleEffect
from .mirror.cpu import MirrorEffect
from .passthrough.cpu import PassthroughEffect
from .portrait_blur.cpu import PortraitBlurEffect
from .portrait_blur.npu import PortraitBlurEffect as _NpuPortraitBlurEffect
from .spotlight.cpu import SpotlightEffect
from .video_depth.cpu import VideoDepthEffect
from .video_depth.gpu import VideoDepthEffect as _GpuVideoDepthEffect
from .video_depth.npu import VideoDepthEffect as _NpuVideoDepthEffect
from .zoom_crop.cpu import ZoomCropEffect

BUILTIN_EFFECTS: dict[str, Callable[[], FrameEffect]] = {
    "none": PassthroughEffect,
    "mirror": MirrorEffect,
    "zoom": ZoomCropEffect,
    "spotlight": SpotlightEffect,
    "blur": FastBlurEffect,
    "greyscale": GreyscaleEffect,
    "depth_cpu": VideoDepthEffect,
    "depth_npu": _NpuVideoDepthEffect,
    "video_depth_cpu": VideoDepthEffect,
    "video_depth_npu": _NpuVideoDepthEffect,
    "portrait_blur_cpu": PortraitBlurEffect,
    "portrait_blur_npu": _NpuPortraitBlurEffect,
}


def available_effect_backends() -> dict[str, tuple[str, ...]]:
    """List implemented backends by effect; a missing backend file is skipped."""
    root = Path(__file__).parent
    return {
        effect.name: tuple(backend for backend in ("cpu", "gpu", "npu") if (effect / f"{backend}.py").is_file())
        for effect in sorted(root.iterdir())
        if effect.is_dir() and effect.name not in {"cpu", "gpu", "npu"} and (effect / "__init__.py").is_file()
    }


def load_effect_backend(effect: str, backend: str):
    """Import a backend implementation on demand, raising ImportError if absent."""
    if backend not in {"cpu", "gpu", "npu"}:
        raise ValueError(f"Unknown backend {backend!r}")
    module_path = Path(__file__).parent / effect / f"{backend}.py"
    if not module_path.is_file():
        raise ImportError(f"Effect {effect!r} has no {backend!r} implementation")
    return import_module(f"{__name__}.{effect}.{backend}")

__all__ = [
    "BUILTIN_EFFECTS",
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
