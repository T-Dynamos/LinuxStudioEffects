"""AMD Ryzen AI NPU backend for depth-guided Aesthetic Light."""

from ..base import ComputeDevice
from .cpu import AestheticLightEffect as _CpuAestheticLightEffect


class AestheticLightEffect(_CpuAestheticLightEffect):
    """Runs streaming Video Depth Anything on NPU1 with warm-white / split aesthetic lighting."""

    effect_name = "aesthetic_light"

    def __init__(self, *, device: ComputeDevice | str = ComputeDevice.NPU, **kwargs: object) -> None:
        super().__init__(device=device, **kwargs)  # type: ignore[arg-type]
