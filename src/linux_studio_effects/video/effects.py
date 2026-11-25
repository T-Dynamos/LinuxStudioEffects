"""Device-agnostic frame-in -> frame-out effect interface and CPU reference effects."""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum

import numpy as np


class ComputeDevice(str, Enum):
    """Execution backend for a frame effect."""

    AUTO = "auto"
    CPU = "cpu"
    NPU = "npu"
    GPU = "gpu"


class DeviceUnavailableError(RuntimeError):
    """Raised when an effect cannot run on the requested compute device."""


@dataclass(frozen=True)
class FrameContext:
    """Per-frame metadata passed to frame processors during streaming."""

    frame_index: int
    timestamp_s: float
    width: int
    height: int
    fps: float


def _is_npu_available() -> bool:
    try:
        from ..npu import probe_npu

        return probe_npu().available
    except Exception:
        return False


def _is_gpu_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            return True
        xpu = getattr(torch, "xpu", None)
        return bool(xpu is not None and xpu.is_available())
    except Exception:
        return False


def resolve_compute_device(
    requested: ComputeDevice | str,
    supported: tuple[ComputeDevice, ...],
) -> ComputeDevice:
    """Resolve 'auto' or validate an explicit ComputeDevice against effect capabilities."""
    req = ComputeDevice(requested.lower()) if isinstance(requested, str) else requested
    if not supported:
        raise DeviceUnavailableError("Effect declares no supported compute devices")

    if req == ComputeDevice.AUTO:
        for candidate in (ComputeDevice.NPU, ComputeDevice.GPU, ComputeDevice.CPU):
            if candidate in supported and is_device_available(candidate):
                return candidate
        raise DeviceUnavailableError(f"None of supported devices {[d.value for d in supported]} are available")

    if req not in supported:
        raise DeviceUnavailableError(
            f"Requested device '{req.value}' is not supported by this effect "
            f"(supported: {[d.value for d in supported]})"
        )
    if not is_device_available(req):
        raise DeviceUnavailableError(f"Requested device '{req.value}' is not available on this host")
    return req


def is_device_available(device: ComputeDevice | str) -> bool:
    """Return True if the requested compute backend is usable on this host."""
    dev = ComputeDevice(device.lower()) if isinstance(device, str) else device
    if dev == ComputeDevice.CPU:
        return True
    if dev == ComputeDevice.NPU:
        return _is_npu_available()
    if dev == ComputeDevice.GPU:
        return _is_gpu_available()
    if dev == ComputeDevice.AUTO:
        return True
    return False


class FrameEffect(ABC):
    """Base interface for a frame-in -> frame-out video processor on CPU, NPU, or GPU."""

    supported_devices: tuple[ComputeDevice, ...] = (ComputeDevice.CPU,)

    def __init__(
        self,
        *,
        name: str | None = None,
        device: ComputeDevice | str = ComputeDevice.AUTO,
        enabled: bool = True,
    ) -> None:
        self.name = name or type(self).__name__
        self.requested_device = ComputeDevice(device.lower()) if isinstance(device, str) else device
        self.active_device = resolve_compute_device(self.requested_device, self.supported_devices)
        self.enabled = enabled
        self._is_setup = False

    def setup(self, width: int, height: int, fps: float) -> None:
        """Allocate backend buffers/models for the negotiated input stream dimensions."""
        self._is_setup = True

    @abstractmethod
    def process(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        """Transform one (H, W, 3) uint8 RGB frame into a uint8 RGB frame."""

    def teardown(self) -> None:
        """Release backend buffers/models when streaming stops."""
        self._is_setup = False

    def __call__(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        if not self.enabled:
            return frame
        return self.process(frame, ctx)


class CallableEffect(FrameEffect):
    """Wrap any plain ``fn(frame) -> frame`` callable into a FrameEffect."""

    supported_devices = (ComputeDevice.CPU,)

    def __init__(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        *,
        name: str | None = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(name=name or getattr(fn, "__name__", "callable_effect"), device=ComputeDevice.CPU, enabled=enabled)
        self._fn = fn

    def process(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        out = self._fn(frame)
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(out)


class EffectChain(FrameEffect):
    """Compose multiple FrameEffects sequentially with shared lifecycle management."""

    supported_devices = (ComputeDevice.NPU, ComputeDevice.GPU, ComputeDevice.CPU)

    def __init__(
        self,
        effects: Iterable[FrameEffect | Callable[[np.ndarray], np.ndarray]] = (),
        *,
        name: str = "EffectChain",
    ) -> None:
        self.name = name
        self.requested_device = ComputeDevice.AUTO
        self.active_device = ComputeDevice.CPU
        self.enabled = True
        self._is_setup = False
        self._effects: list[FrameEffect] = [as_frame_effect(e) for e in effects]

    @property
    def effects(self) -> tuple[FrameEffect, ...]:
        return tuple(self._effects)

    def add(self, effect: FrameEffect | Callable[[np.ndarray], np.ndarray]) -> EffectChain:
        self._effects.append(as_frame_effect(effect))
        return self

    def set_enabled(self, name: str, enabled: bool) -> bool:
        """Enable or disable the first matching effect by name; returns True if found."""
        for eff in self._effects:
            if eff.name == name:
                eff.enabled = enabled
                return True
        return False

    def toggle(self, name: str) -> bool | None:
        """Toggle an effect's enabled state by name; returns new state or None if missing."""
        for eff in self._effects:
            if eff.name == name:
                eff.enabled = not eff.enabled
                return eff.enabled
        return None

    def setup(self, width: int, height: int, fps: float) -> None:
        cur_w, cur_h = width, height
        for eff in self._effects:
            eff.setup(cur_w, cur_h, fps)
        self._is_setup = True

    def process(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        cur = frame
        for eff in self._effects:
            if eff.enabled:
                cur = eff.process(cur, ctx)
        return cur

    def teardown(self) -> None:
        for eff in reversed(self._effects):
            eff.teardown()
        self._is_setup = False


def as_frame_effect(
    processor: FrameEffect | Callable[[np.ndarray], np.ndarray],
) -> FrameEffect:
    """Normalize a FrameEffect or plain callable into a FrameEffect instance."""
    if isinstance(processor, FrameEffect):
        return processor
    if callable(processor):
        return CallableEffect(processor)
    raise TypeError(f"Expected FrameEffect or callable, got {type(processor).__name__}")


# --- Built-in CPU reference effects ---


class PassthroughEffect(FrameEffect):
    """Identity effect returning the input RGB frame unchanged."""

    supported_devices = (ComputeDevice.CPU,)

    def process(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        return frame


class MirrorEffect(FrameEffect):
    """Horizontally mirror the frame (natural webcam self-view)."""

    supported_devices = (ComputeDevice.CPU,)

    def process(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        return np.ascontiguousarray(frame[:, ::-1, :])


class ZoomCropEffect(FrameEffect):
    """Center-crop zoom on CPU (simulates framing crop before resize)."""

    supported_devices = (ComputeDevice.CPU,)

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
        h, w = frame.shape[:2]
        if self._y_idx is None or self._x_idx is None or len(self._y_idx) != h or len(self._x_idx) != w:
            self.setup(w, h, ctx.fps if ctx else 30.0)
        assert self._y_idx is not None and self._x_idx is not None
        return np.ascontiguousarray(frame[self._y_idx[:, None], self._x_idx])

    def teardown(self) -> None:
        self._y_idx = None
        self._x_idx = None
        super().teardown()


class SpotlightEffect(FrameEffect):
    """Precomputed radial vignette spotlight to brighten center subject relative to edges."""

    supported_devices = (ComputeDevice.CPU,)

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


class FastBlurEffect(FrameEffect):
    """Fast CPU downsampled separable box blur (useful baseline before NPU blur)."""

    supported_devices = (ComputeDevice.CPU,)

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


BUILTIN_EFFECTS: dict[str, Callable[[], FrameEffect]] = {
    "none": PassthroughEffect,
    "mirror": MirrorEffect,
    "zoom": ZoomCropEffect,
    "spotlight": SpotlightEffect,
    "blur": FastBlurEffect,
}
