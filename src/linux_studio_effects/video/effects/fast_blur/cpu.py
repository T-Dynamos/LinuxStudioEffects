"""CPU downsampled separable box blur video effect."""

from __future__ import annotations

import numpy as np

from ..base import ComputeDevice, FrameContext, FrameEffect


class FastBlurEffect(FrameEffect):
    """Fast CPU downsampled separable box blur (useful baseline before NPU blur)."""

    supported_devices = (ComputeDevice.CPU,)
    effect_name = "fast_blur"

    def __init__(self, scale: int = 4, passes: int = 2, **kwargs: object) -> None:
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
        h, w = frame.shape[:2]
        if self._y_up is None or self._x_up is None or len(self._y_up) != h or len(self._x_up) != w:
            self.setup(w, h, ctx.fps if ctx else 30.0)
        assert self._y_up is not None and self._x_up is not None

        small = frame[:: self.scale, :: self.scale].astype(np.uint16)
        for _ in range(self.passes):
            # 3-tap horizontal + vertical averaging with edge clamping.
            padded_x = np.pad(small, ((0, 0), (1, 1), (0, 0)), mode="edge")
            small = (padded_x[:, :-2] + padded_x[:, 1:-1] + padded_x[:, 2:]) // 3
            padded_y = np.pad(small, ((1, 1), (0, 0), (0, 0)), mode="edge")
            small = (padded_y[:-2, :] + padded_y[1:-1, :] + padded_y[2:, :]) // 3
        small_u8 = small.astype(np.uint8)
        return np.ascontiguousarray(small_u8[self._y_up[:, None], self._x_up])

    def teardown(self) -> None:
        self._y_up = None
        self._x_up = None
        super().teardown()
