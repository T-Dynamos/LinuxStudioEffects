"""Backend discovery and automatic effect implementation selection."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path


_BACKENDS = ("npu", "gpu", "cpu")
_EFFECTS_DIR = Path(__file__).parent / "effects"


def available_effect_backends() -> dict[str, tuple[str, ...]]:
    """Return backend files present for each effect folder."""
    return {
        folder.name: tuple(b for b in reversed(_BACKENDS) if (folder / f"{b}.py").is_file())
        for folder in sorted(_EFFECTS_DIR.iterdir())
        if folder.is_dir() and folder.name not in _BACKENDS and (folder / "__init__.py").is_file()
    }


def is_device_available(device: str) -> bool:
    """Check whether the requested compute device is usable on this host."""
    normalized = device.lower()
    if normalized == "cpu":
        return True
    if normalized == "npu":
        try:
            from ..npu import probe_npu

            return probe_npu().available
        except Exception:
            return False
    if normalized == "gpu":
        try:
            import torch

            if torch.cuda.is_available():
                return True
            xpu = getattr(torch, "xpu", None)
            return bool(xpu is not None and xpu.is_available())
        except Exception:
            return False
    return False


def resolve_compute_device(
    effect: str | None,
    requested: str = "auto",
    *,
    implementation_module: str | None = None,
) -> str:
    """Choose an implemented and available backend, preferring NPU, then GPU, then CPU.

    When called by a class imported from ``effects/<name>/<backend>.py``, the
    concrete module pins selection to that implementation. ``create_effect``
    uses the same resolver before importing a backend module.
    """
    from .effects.base import DeviceUnavailableError

    wanted = requested.lower()
    if wanted not in {"auto", "cpu", "gpu", "npu"}:
        raise ValueError(f"Unknown compute backend {requested!r}")

    files = available_effect_backends().get(effect or "", ())
    implementation = None
    if implementation_module:
        for backend in ("cpu", "gpu", "npu"):
            if implementation_module.endswith(f".effects.{effect}.{backend}"):
                implementation = backend
                break

    if not files:
        # User-defined FrameEffects outside the built-in catalog are CPU effects.
        files = ("cpu",)
    if implementation is not None:
        if implementation not in files:
            raise DeviceUnavailableError(f"No {implementation!r} implementation exists for effect {effect!r}")
        if wanted != "auto" and wanted != implementation:
            raise DeviceUnavailableError(
                f"This class is the {implementation!r} implementation of {effect!r}, not {wanted!r}"
            )
        candidates = (implementation,)
    elif wanted == "auto":
        candidates = tuple(b for b in _BACKENDS if b in files)
    else:
        candidates = (wanted,)

    if wanted != "auto" and wanted not in files:
        raise DeviceUnavailableError(f"Effect {effect!r} has no {wanted!r} implementation")
    for backend in candidates:
        if is_device_available(backend):
            return backend
    if wanted != "auto":
        raise DeviceUnavailableError(f"Requested compute backend {wanted!r} is not available on this host")
    raise DeviceUnavailableError(f"No available backend for effect {effect!r}; implemented: {files}")


def load_effect_backend(effect: str, backend: str):
    """Import one implementation module, rejecting missing backend files."""
    if backend not in {"cpu", "gpu", "npu"}:
        raise ValueError(f"Unknown backend {backend!r}")
    module_path = _EFFECTS_DIR / effect / f"{backend}.py"
    if not module_path.is_file():
        raise ImportError(f"Effect {effect!r} has no {backend!r} implementation")
    return import_module(f"linux_studio_effects.video.effects.{effect}.{backend}")


def create_effect(effect: str, *, device: str = "auto", **kwargs: object):
    """Create the selected backend class for an effect folder."""
    backend = resolve_compute_device(effect, device)
    module = load_effect_backend(effect, backend)
    class_name = "".join(part.title() for part in effect.split("_")) + "Effect"
    try:
        effect_class = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(f"{module.__name__} must define {class_name}") from exc
    return effect_class(device=backend, **kwargs)
