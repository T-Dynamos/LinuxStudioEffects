"""Depth Anything V2 model package (Small, Base, Large) for CPU, NPU, and GPU."""

from .colormap import TURBO_LUT, depth_to_colormap_rgb
from .config import DepthAnythingConfig, get_depth_config
from .feasibility import FeasibilityReport, analyze_feasibility, format_feasibility_table
from .model import DepthAnythingV2Model, InferenceResult
from .npu_ops import NpuExecStats, NpuMatMulExecutor
from .weights import DepthAnythingWeights, create_synthetic_weights, load_from_hf

__all__ = [
    "DepthAnythingConfig",
    "DepthAnythingV2Model",
    "DepthAnythingWeights",
    "FeasibilityReport",
    "InferenceResult",
    "NpuExecStats",
    "NpuMatMulExecutor",
    "TURBO_LUT",
    "analyze_feasibility",
    "create_synthetic_weights",
    "depth_to_colormap_rgb",
    "format_feasibility_table",
    "get_depth_config",
    "load_from_hf",
]
