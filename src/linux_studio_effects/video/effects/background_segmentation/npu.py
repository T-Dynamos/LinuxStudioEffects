"""AMD Ryzen AI NPU backend for MODNet portrait background segmentation."""

from ..base import ComputeDevice
from .cpu import BackgroundSegmentationEffect as _CpuBackgroundSegmentationEffect


class BackgroundSegmentationEffect(_CpuBackgroundSegmentationEffect):
    """Runs MODNet with high-FLOP dense convolutions offloaded to NPU1."""

    effect_name = "background_segmentation"

    def __init__(self, *, device: ComputeDevice | str = ComputeDevice.NPU, **kwargs: object) -> None:
        super().__init__(device=device, **kwargs)  # type: ignore[arg-type]
