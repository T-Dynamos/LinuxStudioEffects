"""CPU center-crop zoom video effect."""

from __future__ import annotations

import numpy as np

from ..base import ComputeDevice, FrameContext, FrameEffect
from ...pixels import channels_to_frame, frame_to_channels, is_yuv422


class ZoomCropEffect(FrameEffect):
    """Center-crop zoom on CPU (simulates framing crop before resize)."""

    effect_name = "zoom_crop"

    def __init__(self, zoom: float = 1.25, **kwargs: object) -> None:
        if zoom < 1.0:
            raise ValueError(f"zoom must be >= 1.0, got {zoom}")
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.zoom = zoom
        self._y_idx: np.ndarray | None = None
        self._x_idx: np.ndarray | None = None

    def setup(self, width: int, height: int, fps: float) -> None:
        super().setup(width, height, fps)
        crop_w = max(2, int(round(width / self.zoom)))
        crop_h = max(2, int(round(height / self.zoom)))
        x0 = (width - crop_w) // 2
        y0 = (height - crop_h) // 2
        self._y_idx = y0 + (np.arange(height, dtype=np.int64) * crop_h) // height
        self._x_idx = x0 + (np.arange(width, dtype=np.int64) * crop_w) // width

    def process(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        if self.zoom <= 1.0:
            return frame
        pixel_format = ctx.pixel_format if ctx else ("YUYV" if frame.ndim == 3 and frame.shape[2] == 4 else "RGB3")
        image = frame_to_channels(frame, pixel_format)
        h, w = image.shape[:2]
        effect_width = ctx.width if ctx and is_yuv422(pixel_format) else w
        if self._y_idx is None or self._x_idx is None or len(self._y_idx) != h or len(self._x_idx) != w:
            self.setup(effect_width, h, ctx.fps if ctx else 30.0)
        assert self._y_idx is not None and self._x_idx is not None
        if len(self._x_idx) != w:
            self.setup(effect_width, h, ctx.fps if ctx else 30.0)
        result = np.ascontiguousarray(image[self._y_idx[:, None], self._x_idx])
        return channels_to_frame(result, pixel_format)

    def teardown(self) -> None:
        self._y_idx = None
        self._x_idx = None
        super().teardown()
