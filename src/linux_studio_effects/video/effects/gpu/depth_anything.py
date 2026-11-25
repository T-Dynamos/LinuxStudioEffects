"""PyTorch GPU (ROCm/CUDA/XPU) Depth Anything V2 monocular depth -> Turbo colormap video effect."""

from __future__ import annotations

from ..base import ComputeDevice
from ...models.depth_anything_v2.effect_base import DepthAnythingEffectBase


class GpuDepthAnythingEffect(DepthAnythingEffectBase):
    """Runs Depth Anything V2 on GPU using PyTorch."""

    supported_devices = (ComputeDevice.GPU,)

    def __init__(self, *, device: ComputeDevice | str = ComputeDevice.GPU, **kwargs: object) -> None:
        super().__init__(device=device, **kwargs)  # type: ignore[arg-type]
