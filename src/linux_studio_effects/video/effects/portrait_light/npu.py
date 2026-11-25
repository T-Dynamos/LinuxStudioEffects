"""AMD Ryzen AI NPU backend for depth-guided 3D Portrait Light."""

from ..base import ComputeDevice
from .cpu import PortraitLightEffect as _CpuPortraitLightEffect


class PortraitLightEffect(_CpuPortraitLightEffect):
    """Runs streaming Video Depth Anything on NPU1 with smooth 3D key-light relighting."""

    effect_name = "portrait_light"

    def __init__(self, *, device: ComputeDevice | str = ComputeDevice.NPU, **kwargs: object) -> None:
        super().__init__(device=device, **kwargs)  # type: ignore[arg-type]
