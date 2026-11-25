"""Video effect interfaces and lazy exports for concrete effect backends."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

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

_LAZY_EFFECTS = {
    "FastBlurEffect": ("fast_blur.cpu", "FastBlurEffect"),
    "GreyscaleEffect": ("greyscale.npu", "GreyscaleEffect"),
    "MirrorEffect": ("mirror.cpu", "MirrorEffect"),
    "PassthroughEffect": ("passthrough.cpu", "PassthroughEffect"),
    "PortraitBlurEffect": ("portrait_blur.cpu", "PortraitBlurEffect"),
    "SpotlightEffect": ("spotlight.cpu", "SpotlightEffect"),
    "ZoomCropEffect": ("zoom_crop.cpu", "ZoomCropEffect"),
}


def _factory(effect_name: str, device: str = "auto") -> Callable[[], FrameEffect]:
    """Defer backend resolution and effect imports until a preset is selected."""
    return lambda: create_effect(effect_name, device=device)


BUILTIN_EFFECTS: dict[str, Callable[[], FrameEffect]] = {
    "none": _factory("passthrough"),
    "mirror": _factory("mirror"),
    "zoom": _factory("zoom_crop"),
    "spotlight": _factory("spotlight"),
    "blur": _factory("fast_blur"),
    "greyscale": _factory("greyscale"),
    "portrait_blur": _factory("portrait_blur"),
    "portrait_blur_cpu": _factory("portrait_blur", "cpu"),
    "portrait_blur_npu": _factory("portrait_blur", "npu"),
}


def __getattr__(name: str) -> Any:
    """Import a concrete effect only when its public class is requested."""
    target = _LAZY_EFFECTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, class_name = target
    value = getattr(import_module(f".{module_name}", __name__), class_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EFFECTS})


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
    "as_frame_effect",
    "is_device_available",
    "resolve_compute_device",
]
