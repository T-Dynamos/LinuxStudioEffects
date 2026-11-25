"""CPU downsampled separable box blur video effect."""

from __future__ import annotations

import numpy as np

from ..base import ComputeDevice, FrameContext, FrameEffect
from ...pixels import channels_to_frame, frame_to_channels, is_yuv422


class FastBlurEffect(FrameEffect):
    """Fast CPU downsampled separable box blur (useful baseline before NPU blur)."""

    effect_name = "fast_blur"

    def __init__(self, scale: int = 4, passes: int = 10, **kwargs: object) -> None:
        if scale < 1 or passes < 1:
            raise ValueError("scale and passes must be >= 1")
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.scale = scale
        self.passes = passes
        self._y_up: np.ndarray | None = None
        self._x_up: np.ndarray | None = None

    def setup(self, width: int, height: int, fps: float) -> None:
        super().setup(width, height, fps)
        sh = max(1, height // self.scale)
        sw = max(1, width // self.scale)
        self._y_up = (np.arange(height, dtype=np.int64) * sh) // height
        self._x_up = (np.arange(width, dtype=np.int64) * sw) // width

    def process(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        pixel_format = ctx.pixel_format if ctx else ("YUYV" if frame.ndim == 3 and frame.shape[2] == 4 else "RGB3")
        image = frame_to_channels(frame, pixel_format)
        h, w = image.shape[:2]
        effect_width = ctx.width if ctx and is_yuv422(pixel_format) else w
        if self._y_up is None or self._x_up is None or len(self._y_up) != h or len(self._x_up) != effect_width:
            self.setup(effect_width, h, ctx.fps if ctx else 30.0)
        assert self._y_up is not None and self._x_up is not None

        small = image[:: self.scale, :: self.scale].astype(np.uint16)
        for _ in range(self.passes):
            # 3-tap horizontal + vertical averaging with edge clamping.
            padded_x = np.pad(small, ((0, 0), (1, 1), (0, 0)), mode="edge")
            small = (padded_x[:, :-2] + padded_x[:, 1:-1] + padded_x[:, 2:]) // 3
            padded_y = np.pad(small, ((1, 1), (0, 0), (0, 0)), mode="edge")
            small = (padded_y[:-2, :] + padded_y[1:-1, :] + padded_y[2:, :]) // 3
        small_u8 = small.astype(np.uint8)
        result = np.ascontiguousarray(small_u8[self._y_up[:, None], self._x_up])
        return channels_to_frame(result, pixel_format)

    def teardown(self) -> None:
        self._y_up = None
        self._x_up = None
        super().teardown()
