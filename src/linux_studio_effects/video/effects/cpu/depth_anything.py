"""CPU Depth Anything V2 monocular depth -> Turbo colormap video effect."""

from __future__ import annotations

from ..base import ComputeDevice
from ...models.depth_anything_v2.effect_base import DepthAnythingEffectBase


class CpuDepthAnythingEffect(DepthAnythingEffectBase):
    """Runs Depth Anything V2 (Small/Base/Large) on CPU via NumPy/OpenBLAS."""

    supported_devices = (ComputeDevice.CPU,)

    def __init__(self, *, device: ComputeDevice | str = ComputeDevice.CPU, **kwargs: object) -> None:
        super().__init__(device=device, **kwargs)  # type: ignore[arg-type]
