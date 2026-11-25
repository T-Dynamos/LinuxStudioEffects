"""Device-agnostic frame-in -> frame-out effect interface and backend resolution."""

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
        from ...npu import probe_npu

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


class FrameEffect(ABC):
    """Base interface for a frame-in -> frame-out video processor on CPU, NPU, or GPU."""

    supported_devices: tuple[ComputeDevice, ...] = (ComputeDevice.CPU,)
    effect_name: str | None = None

    def __init__(
        self,
        *,
        name: str | None = None,
        device: ComputeDevice | str = ComputeDevice.AUTO,
        enabled: bool = True,
    ) -> None:
        self.name = name or self.effect_name or type(self).__name__
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
        super().__init__(
            name=name or getattr(fn, "__name__", "callable_effect"),
            device=ComputeDevice.CPU,
            enabled=enabled,
        )
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
        for eff in self._effects:
            if eff.name == name:
                eff.enabled = enabled
                return True
        return False

    def toggle(self, name: str) -> bool | None:
        for eff in self._effects:
            if eff.name == name:
                eff.enabled = not eff.enabled
                return eff.enabled
        return None

    def setup(self, width: int, height: int, fps: float) -> None:
        for eff in self._effects:
            eff.setup(width, height, fps)
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


class FrameGraph(FrameEffect):
    """A named, acyclic graph of frame effects with selectable upstream inputs.

    Each node receives either the original source frame (``source="input"``)
    or the output of an earlier node. The selected output node is returned.
    Nodes execute in insertion order, which makes forward references invalid.
    """

    def __init__(self, *, output: str | None = None, name: str = "FrameGraph") -> None:
        self.name = name
        self.requested_device = ComputeDevice.AUTO
        self.active_device = ComputeDevice.CPU
        self.enabled = True
        self._is_setup = False
        self._nodes: list[tuple[str, str, FrameEffect]] = []
        self._output = output

    @property
    def nodes(self) -> tuple[tuple[str, str, FrameEffect], ...]:
        return tuple(self._nodes)

    def add(self, name: str, effect: FrameEffect | Callable[[np.ndarray], np.ndarray], *, source: str = "input") -> FrameGraph:
        if not name or name == "input" or any(n == name for n, _, _ in self._nodes):
            raise ValueError(f"Invalid or duplicate graph node name: {name!r}")
        available = {"input", *(n for n, _, _ in self._nodes)}
        if source not in available:
            raise ValueError(f"Unknown source {source!r}; nodes can only use earlier outputs")
        self._nodes.append((name, source, as_frame_effect(effect)))
        if self._output is None:
            self._output = name
        return self

    def set_output(self, name: str) -> FrameGraph:
        if name != "input" and name not in {n for n, _, _ in self._nodes}:
            raise ValueError(f"Unknown graph output {name!r}")
        self._output = name
        return self

    def setup(self, width: int, height: int, fps: float) -> None:
        for _, _, effect in self._nodes:
            effect.setup(width, height, fps)
        self._is_setup = True

    def process(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        outputs: dict[str, np.ndarray] = {"input": frame}
        for name, source, effect in self._nodes:
            outputs[name] = effect(outputs[source], ctx) if effect.enabled else outputs[source]
        if self._output is None:
            return frame
        return outputs[self._output]

    def teardown(self) -> None:
        for _, _, effect in reversed(self._nodes):
            effect.teardown()
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
