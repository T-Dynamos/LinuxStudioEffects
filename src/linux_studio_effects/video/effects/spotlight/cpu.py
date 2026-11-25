"""CPU radial spotlight vignette video effect."""

from __future__ import annotations

import numpy as np

from ..base import ComputeDevice, FrameContext, FrameEffect


class SpotlightEffect(FrameEffect):
    """Precomputed radial vignette spotlight to brighten center subject relative to edges."""

    effect_name = "spotlight"

    def __init__(self, edge_dim: float = 0.45, **kwargs: object) -> None:
        if not 0.0 <= edge_dim <= 1.0:
            raise ValueError(f"edge_dim must be in [0, 1], got {edge_dim}")
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.edge_dim = edge_dim
        self._mask_q8: np.ndarray | None = None
        self._buf: np.ndarray | None = None

    def setup(self, width: int, height: int, fps: float) -> None:
        super().setup(width, height, fps)
        yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
        dist2 = np.clip(xx * xx + yy * yy, 0.0, 1.0)
        gain = 1.0 - self.edge_dim * (dist2 * dist2)
        self._mask_q8 = np.clip(np.round(gain * 256.0), 0, 256).astype(np.uint16)[:, :, None]
        self._buf = np.empty((height, width, 3), dtype=np.uint16)

    def process(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        h, w = frame.shape[:2]
        if self._mask_q8 is None or self._buf is None or self._buf.shape[:2] != (h, w):
            self.setup(w, h, ctx.fps if ctx else 30.0)
        assert self._mask_q8 is not None and self._buf is not None
        np.multiply(frame, self._mask_q8, out=self._buf, dtype=np.uint16)
        self._buf >>= 8
        return self._buf.astype(np.uint8)

    def teardown(self) -> None:
        self._mask_q8 = None
        self._buf = None
        super().teardown()
