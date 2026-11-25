"""Video Depth Anything architecture configurations for Small, Base, and Large variants."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoDepthConfig:
    """Parameterized architecture config covering ViT-S/14, ViT-B/14, and ViT-L/14 + Temporal DPT."""

    variant: str
    backbone_name: str
    embed_dim: int
    depth: int
    num_heads: int
    mlp_ratio: int
    patch_size: int
    default_image_size: int
    input_size: int
    tap_indices: tuple[int, int, int, int]
    fusion_hidden_size: int
    neck_hidden_sizes: tuple[int, int, int, int]
    reassemble_factors: tuple[float, float, float, float]
    head_hidden_size: int
    num_frames: int
    temporal_heads: int
    temporal_norm_groups: int
    hf_repo_id: str
    checkpoint_filename: str
    license_name: str
    layer_norm_eps: float = 1e-6
    temporal_norm_eps: float = 1e-5
    group_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.input_size <= 0 or self.input_size % self.patch_size != 0:
            raise ValueError(
                f"input_size must be a positive multiple of patch_size ({self.patch_size}), got {self.input_size}"
            )
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @property
    def mlp_dim(self) -> int:
        return self.embed_dim * self.mlp_ratio

    @property
    def patch_grid(self) -> int:
        return self.input_size // self.patch_size

    @property
    def num_patches(self) -> int:
        g = self.patch_grid
        return g * g

    @property
    def num_tokens(self) -> int:
        return self.num_patches + 1  # +1 CLS token

    @property
    def motion_channels(self) -> tuple[int, int, int, int]:
        return (
            self.neck_hidden_sizes[2],
            self.neck_hidden_sizes[3],
            self.fusion_hidden_size,
            self.fusion_hidden_size,
        )


_VARIANT_PRESETS: dict[str, dict[str, object]] = {
    "small": {
        "variant": "small",
        "backbone_name": "DINOv2 ViT-S/14",
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4,
        "patch_size": 14,
        "default_image_size": 518,
        "tap_indices": (2, 5, 8, 11),
        "fusion_hidden_size": 64,
        "neck_hidden_sizes": (48, 96, 192, 384),
        "reassemble_factors": (4.0, 2.0, 1.0, 0.5),
        "head_hidden_size": 32,
        "num_frames": 32,
        "temporal_heads": 8,
        "temporal_norm_groups": 32,
        "hf_repo_id": "depth-anything/Video-Depth-Anything-Small",
        "checkpoint_filename": "video_depth_anything_vits.pth",
        "license_name": "Apache-2.0",
    },
    "base": {
        "variant": "base",
        "backbone_name": "DINOv2 ViT-B/14",
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
        "patch_size": 14,
        "default_image_size": 518,
        "tap_indices": (2, 5, 8, 11),
        "fusion_hidden_size": 128,
        "neck_hidden_sizes": (96, 192, 384, 768),
        "reassemble_factors": (4.0, 2.0, 1.0, 0.5),
        "head_hidden_size": 32,
        "num_frames": 32,
        "temporal_heads": 8,
        "temporal_norm_groups": 32,
        "hf_repo_id": "depth-anything/Video-Depth-Anything-Base",
        "checkpoint_filename": "video_depth_anything_vitb.pth",
        "license_name": "CC-BY-NC-4.0",
    },
    "large": {
        "variant": "large",
        "backbone_name": "DINOv2 ViT-L/14",
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4,
        "patch_size": 14,
        "default_image_size": 518,
        "tap_indices": (4, 11, 17, 23),
        "fusion_hidden_size": 256,
        "neck_hidden_sizes": (256, 512, 1024, 1024),
        "reassemble_factors": (4.0, 2.0, 1.0, 0.5),
        "head_hidden_size": 32,
        "num_frames": 32,
        "temporal_heads": 8,
        "temporal_norm_groups": 32,
        "hf_repo_id": "depth-anything/Video-Depth-Anything-Large",
        "checkpoint_filename": "video_depth_anything_vitl.pth",
        "license_name": "CC-BY-NC-4.0",
    },
}

_ALIASES = {
    "vits": "small",
    "vit-s": "small",
    "vitb": "base",
    "vit-b": "base",
    "vitl": "large",
    "vit-l": "large",
}


def get_video_depth_config(variant: str = "small", *, input_size: int = 252) -> VideoDepthConfig:
    """Build a VideoDepthConfig for 'small', 'base', or 'large' (or vits/vitb/vitl)."""
    key = variant.lower().strip()
    key = _ALIASES.get(key, key)
    if key not in _VARIANT_PRESETS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {tuple(_VARIANT_PRESETS.keys())}")
    params = dict(_VARIANT_PRESETS[key])
    params["input_size"] = input_size
    return VideoDepthConfig(**params)  # type: ignore[arg-type]
