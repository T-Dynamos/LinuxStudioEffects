"""AMD Ryzen AI NPU backend for Automatic Framing."""

from ..base import ComputeDevice
from .cpu import AutomaticFramingEffect as _CpuAutomaticFramingEffect


class AutomaticFramingEffect(_CpuAutomaticFramingEffect):
    """Runs MODNet subject tracking on NPU1 with smooth pan/crop/zoom framing."""

    effect_name = "automatic_framing"

    def __init__(self, *, device: ComputeDevice | str = ComputeDevice.NPU, **kwargs: object) -> None:
        super().__init__(device=device, **kwargs)  # type: ignore[arg-type]
