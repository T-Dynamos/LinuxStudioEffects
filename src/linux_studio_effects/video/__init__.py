"""Video and camera studio effects capture pipeline."""

from .benchmark import CameraBenchmarkResult, benchmark_camera, format_benchmark_result
from .backend import create_effect
from .backend import (
    available_effect_backends,
    is_device_available,
    load_effect_backend,
    resolve_compute_device,
)
from .camera import CameraConfig, CapturedFrame, V4L2Camera
from .capabilities import (
    CameraCapabilitiesSummary,
    PixelFormatSummary,
    RecommendedMode,
    format_all_camera_summaries,
    format_camera_summary,
    summarize_all_cameras,
    summarize_camera_capabilities,
    summarize_device_capabilities,
)
from .decoder import FrameDecoder, decode_yuyv_to_rgb, has_pyav
from .deps import CameraDepsReport, CheckItem, check_camera_deps, format_deps_report
from .effects.base import (
    CallableEffect,
    ComputeDevice,
    DeviceUnavailableError,
    EffectChain,
    FrameGraph,
    FrameContext,
    FrameEffect,
    as_frame_effect,
)
from .pipeline import PipelineConfig, PipelineStats, VideoPipeline
from .v4l2 import (
    ActiveStreamFormat,
    VideoControlInfo,
    VideoDeviceInfo,
    VideoMode,
    list_loopback_devices,
    list_video_devices,
    pick_default_camera,
    pick_default_loopback,
    query_video_device,
)
from .virtual_cam import (
    VirtualCameraConfig,
    VirtualCameraOutput,
    encode_rgb_frame,
    encode_rgb_to_yuyv,
    resize_rgb,
)

_LAZY_EFFECT_EXPORTS = {
    "BUILTIN_EFFECTS",
    "FastBlurEffect",
    "GreyscaleEffect",
    "MirrorEffect",
    "PassthroughEffect",
    "PortraitBlurEffect",
    "SpotlightEffect",
    "VideoDepthEffect",
    "ZoomCropEffect",
}


def __getattr__(name: str):
    """Load concrete effect implementations only when their API is requested."""
    if name not in _LAZY_EFFECT_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import effects

    value = getattr(effects, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EFFECT_EXPORTS})

__all__ = [
    "ActiveStreamFormat",
    "BUILTIN_EFFECTS",
    "create_effect",
    "CallableEffect",
    "CameraBenchmarkResult",
    "CameraCapabilitiesSummary",
    "CameraConfig",
    "CameraDepsReport",
    "CapturedFrame",
    "CheckItem",
    "ComputeDevice",
    "DeviceUnavailableError",
    "EffectChain",
    "FrameGraph",
    "FastBlurEffect",
    "FrameContext",
    "FrameDecoder",
    "FrameEffect",
    "GreyscaleEffect",
    "MirrorEffect",
    "PortraitBlurEffect",
    "VideoDepthEffect",
    "PassthroughEffect",
    "PipelineConfig",
    "PipelineStats",
    "PixelFormatSummary",
    "RecommendedMode",
    "SpotlightEffect",
    "V4L2Camera",
    "VideoControlInfo",
    "VideoDeviceInfo",
    "VideoMode",
    "VideoPipeline",
    "VirtualCameraConfig",
    "VirtualCameraOutput",
    "ZoomCropEffect",
    "as_frame_effect",
    "available_effect_backends",
    "benchmark_camera",
    "check_camera_deps",
    "decode_yuyv_to_rgb",
    "encode_rgb_frame",
    "encode_rgb_to_yuyv",
    "format_all_camera_summaries",
    "format_benchmark_result",
    "format_camera_summary",
    "format_deps_report",
    "has_pyav",
    "is_device_available",
    "load_effect_backend",
    "list_loopback_devices",
    "list_video_devices",
    "pick_default_camera",
    "pick_default_loopback",
    "query_video_device",
    "resize_rgb",
    "resolve_compute_device",
    "summarize_all_cameras",
    "summarize_camera_capabilities",
    "summarize_device_capabilities",
]
