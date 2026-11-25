"""Shared streaming FrameEffect adapter for Video Depth Anything across CPU, NPU, and GPU."""

from __future__ import annotations

import numpy as np

from ...effects.base import ComputeDevice, FrameContext, FrameEffect
from .colormap import depth_to_colormap_rgb
from .config import VideoDepthConfig, get_video_depth_config
from .model import InferenceResult, VideoDepthAnythingModel
from .weights import VideoDepthWeights, create_synthetic_weights, load_from_hf


class VideoDepthEffectBase(FrameEffect):
    """Runs streaming Video Depth Anything and renders temporally consistent depth maps."""

    def __init__(
        self,
        *,
        variant: str = "small",
        input_size: int = 252,
        use_real_weights: bool = True,
        local_files_only: bool = False,
        weights: VideoDepthWeights | None = None,
        name: str | None = None,
        device: ComputeDevice | str = ComputeDevice.AUTO,
        enabled: bool = True,
    ) -> None:
        super().__init__(name=name, device=device, enabled=enabled)
        self.depth_config: VideoDepthConfig = get_video_depth_config(variant, input_size=input_size)
        self.use_real_weights = use_real_weights
        self.local_files_only = local_files_only
        self._explicit_weights = weights
        self._model: VideoDepthAnythingModel | None = None
        self.last_result: InferenceResult | None = None

    def setup(self, width: int, height: int, fps: float) -> None:
        super().setup(width, height, fps)
        if self._model is not None:
            self._model.reset_temporal_state()
            return
        if self._explicit_weights is not None:
            weights = self._explicit_weights
        elif self.use_real_weights:
            weights = load_from_hf(self.depth_config, local_files_only=self.local_files_only)
        else:
            weights = create_synthetic_weights(self.depth_config)

        self._model = VideoDepthAnythingModel(
            self.depth_config,
            weights,
            backend=self.active_device.value,
        )

    def process(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        h, w, _ = frame.shape
        if self._model is None:
            self.setup(w, h, ctx.fps if ctx else 30.0)
        assert self._model is not None
        res = self._model.infer(frame)
        self.last_result = res
        return depth_to_colormap_rgb(res.depth, out_width=w, out_height=h)

    def teardown(self) -> None:
        if self._model is not None:
            self._model.close()
            self._model = None
        super().teardown()
