"""Configuration presets for MODNet portrait background matting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModnetConfig:
    """Hyperparameters and pretrained checkpoint metadata for MODNet (MobileNetV2)."""

    variant: str
    max_size: int
    hr_channels: int = 32
    enc_channels: tuple[int, ...] = (16, 24, 32, 96, 1280)
    norm_eps: float = 1e-5
    hf_repo_id: str = "DavG25/modnet-pretrained-models"
    checkpoint_filename: str = "models/modnet_webcam_portrait_matting.ckpt"
    license_name: str = "Apache-2.0"

    def __post_init__(self) -> None:
        if self.max_size <= 0 or self.max_size % 32 != 0:
            raise ValueError(f"max_size must be a positive multiple of 32, got {self.max_size}")
        if self.hr_channels <= 0 or self.hr_channels % 2 != 0:
            raise ValueError(f"hr_channels must be a positive even integer, got {self.hr_channels}")

    def resolve_input_hw(self, src_h: int, src_w: int) -> tuple[int, int]:
        """Compute aspect-preserving (H, W) rounded to multiples of 32 with max side ~ max_size."""
        if src_h <= 0 or src_w <= 0:
            raise ValueError(f"Expected positive image shape, got ({src_h}, {src_w})")
        scale = float(self.max_size) / float(max(src_h, src_w))
        h = max(32, int(round((src_h * scale) / 32.0)) * 32)
        w = max(32, int(round((src_w * scale) / 32.0)) * 32)
        return h, w


_VARIANT_PRESETS: dict[str, dict[str, object]] = {
    "webcam": {
        "hf_repo_id": "DavG25/modnet-pretrained-models",
        "checkpoint_filename": "models/modnet_webcam_portrait_matting.ckpt",
    },
    "photographic": {
        "hf_repo_id": "DavG25/modnet-pretrained-models",
        "checkpoint_filename": "models/modnet_photographic_portrait_matting.ckpt",
    },
}


def get_modnet_config(variant: str = "webcam", *, max_size: int = 384) -> ModnetConfig:
    """Return a validated ModnetConfig for 'webcam' or 'photographic' checkpoints."""
    key = variant.lower().strip()
    if key not in _VARIANT_PRESETS:
        raise ValueError(f"Unknown MODNet variant {variant!r}; expected one of {tuple(_VARIANT_PRESETS)}")
    preset = _VARIANT_PRESETS[key]
    return ModnetConfig(
        variant=key,
        max_size=int(max_size),
        hf_repo_id=str(preset["hf_repo_id"]),
        checkpoint_filename=str(preset["checkpoint_filename"]),
    )
