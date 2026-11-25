"""NPU runtime and hardware detection helpers."""

from .runtime import NpuStatus, probe_npu

__all__ = ["NpuStatus", "probe_npu"]
