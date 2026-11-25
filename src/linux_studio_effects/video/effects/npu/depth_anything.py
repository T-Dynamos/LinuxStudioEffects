"""AMD Ryzen AI NPU Depth Anything V2 monocular depth -> Turbo colormap video effect."""

from __future__ import annotations

from ..base import ComputeDevice
from ...models.depth_anything_v2.effect_base import DepthAnythingEffectBase


class NpuDepthAnythingEffect(DepthAnythingEffectBase):
    """Runs Depth Anything V2 ViT projections on NPU1 (AIE2) and DPT head on host."""

    supported_devices = (ComputeDevice.NPU,)

    def __init__(self, *, device: ComputeDevice | str = ComputeDevice.NPU, **kwargs: object) -> None:
        super().__init__(device=device, **kwargs)  # type: ignore[arg-type]
