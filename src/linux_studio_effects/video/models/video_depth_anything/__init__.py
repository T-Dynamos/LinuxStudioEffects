"""Streaming Video Depth Anything model package (Small, Base, Large) for CPU, NPU, and GPU."""

from .colormap import TURBO_LUT, depth_to_colormap_rgb
from .config import VideoDepthConfig, get_video_depth_config
from .feasibility import FeasibilityReport, analyze_feasibility, format_feasibility_table
from .model import InferenceResult, VideoDepthAnythingModel
from .npu_ops import NpuExecStats, NpuMatMulExecutor
from .weights import (
    VideoDepthWeights,
    create_synthetic_weights,
    load_from_hf,
    load_pth_state_dict,
)

__all__ = [
    "FeasibilityReport",
    "InferenceResult",
    "NpuExecStats",
    "NpuMatMulExecutor",
    "TURBO_LUT",
    "VideoDepthAnythingModel",
    "VideoDepthConfig",
    "VideoDepthWeights",
    "analyze_feasibility",
    "create_synthetic_weights",
    "depth_to_colormap_rgb",
    "format_feasibility_table",
    "get_video_depth_config",
    "load_from_hf",
    "load_pth_state_dict",
]
