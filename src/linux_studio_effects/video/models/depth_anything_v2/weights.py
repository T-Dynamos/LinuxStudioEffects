"""Hugging Face safetensors loader and synthetic weight builder for Depth Anything V2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DepthAnythingConfig


@dataclass(frozen=True)
class EncoderBlockWeights:
    norm1_weight: np.ndarray
    norm1_bias: np.ndarray
    q_weight: np.ndarray
    q_bias: np.ndarray
    k_weight: np.ndarray
    k_bias: np.ndarray
    v_weight: np.ndarray
    v_bias: np.ndarray
    proj_weight: np.ndarray
    proj_bias: np.ndarray
    ls1: np.ndarray
    norm2_weight: np.ndarray
    norm2_bias: np.ndarray
    fc1_weight: np.ndarray
    fc1_bias: np.ndarray
    fc2_weight: np.ndarray
    fc2_bias: np.ndarray
    ls2: np.ndarray


@dataclass(frozen=True)
class ReassembleWeights:
    proj_weight: np.ndarray
    proj_bias: np.ndarray
    resize_weight: np.ndarray | None
    resize_bias: np.ndarray | None


@dataclass(frozen=True)
class ResidualConvWeights:
    conv1_weight: np.ndarray
    conv1_bias: np.ndarray
    conv2_weight: np.ndarray
    conv2_bias: np.ndarray


@dataclass(frozen=True)
class FusionLayerWeights:
    projection_weight: np.ndarray
    projection_bias: np.ndarray
    residual1: ResidualConvWeights
    residual2: ResidualConvWeights


@dataclass(frozen=True)
class DepthAnythingWeights:
    """All backbone, DPT neck, and depth head parameters in float32."""

    patch_weight: np.ndarray  # (D, 3, 14, 14)
    patch_bias: np.ndarray  # (D,)
    cls_token: np.ndarray  # (1, D)
    pos_embed: np.ndarray  # (1370, D) or ((default_size//14)^2 + 1, D)
    blocks: tuple[EncoderBlockWeights, ...]
    backbone_norm_weight: np.ndarray
    backbone_norm_bias: np.ndarray
    reassemble: tuple[ReassembleWeights, ...]
    neck_convs: tuple[np.ndarray, ...]  # (F, C_i, 3, 3) no bias
    fusion_layers: tuple[FusionLayerWeights, ...]
    head_conv1_weight: np.ndarray
    head_conv1_bias: np.ndarray
    head_conv2_weight: np.ndarray
    head_conv2_bias: np.ndarray
    head_conv3_weight: np.ndarray
    head_conv3_bias: np.ndarray


def load_from_hf(
    cfg: DepthAnythingConfig,
    *,
    local_files_only: bool = False,
) -> DepthAnythingWeights:
    """Download (or read from HF cache) and parse model.safetensors into DepthAnythingWeights."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download(
        cfg.hf_repo_id,
        "model.safetensors",
        local_files_only=local_files_only,
    )
    tensors: dict[str, np.ndarray] = {}
    with safe_open(path, framework="numpy") as f:
        for k in f.keys():
            tensors[k] = np.ascontiguousarray(f.get_tensor(k), dtype=np.float32)

    blocks: list[EncoderBlockWeights] = []
    for i in range(cfg.depth):
        p = f"backbone.encoder.layer.{i}"
        blocks.append(
            EncoderBlockWeights(
                norm1_weight=tensors[f"{p}.norm1.weight"],
                norm1_bias=tensors[f"{p}.norm1.bias"],
                q_weight=tensors[f"{p}.attention.attention.query.weight"],
                q_bias=tensors[f"{p}.attention.attention.query.bias"],
                k_weight=tensors[f"{p}.attention.attention.key.weight"],
                k_bias=tensors[f"{p}.attention.attention.key.bias"],
                v_weight=tensors[f"{p}.attention.attention.value.weight"],
                v_bias=tensors[f"{p}.attention.attention.value.bias"],
                proj_weight=tensors[f"{p}.attention.output.dense.weight"],
                proj_bias=tensors[f"{p}.attention.output.dense.bias"],
                ls1=tensors[f"{p}.layer_scale1.lambda1"],
                norm2_weight=tensors[f"{p}.norm2.weight"],
                norm2_bias=tensors[f"{p}.norm2.bias"],
                fc1_weight=tensors[f"{p}.mlp.fc1.weight"],
                fc1_bias=tensors[f"{p}.mlp.fc1.bias"],
                fc2_weight=tensors[f"{p}.mlp.fc2.weight"],
                fc2_bias=tensors[f"{p}.mlp.fc2.bias"],
                ls2=tensors[f"{p}.layer_scale2.lambda1"],
            )
        )

    reassemble: list[ReassembleWeights] = []
    neck_convs: list[np.ndarray] = []
    fusion_layers: list[FusionLayerWeights] = []
    for i in range(4):
        rp = f"neck.reassemble_stage.layers.{i}"
        reassemble.append(
            ReassembleWeights(
                proj_weight=tensors[f"{rp}.projection.weight"],
                proj_bias=tensors[f"{rp}.projection.bias"],
                resize_weight=tensors.get(f"{rp}.resize.weight"),
                resize_bias=tensors.get(f"{rp}.resize.bias"),
            )
        )
        neck_convs.append(tensors[f"neck.convs.{i}.weight"])

        fp = f"neck.fusion_stage.layers.{i}"
        fusion_layers.append(
            FusionLayerWeights(
                projection_weight=tensors[f"{fp}.projection.weight"],
                projection_bias=tensors[f"{fp}.projection.bias"],
                residual1=ResidualConvWeights(
                    conv1_weight=tensors[f"{fp}.residual_layer1.convolution1.weight"],
                    conv1_bias=tensors[f"{fp}.residual_layer1.convolution1.bias"],
                    conv2_weight=tensors[f"{fp}.residual_layer1.convolution2.weight"],
                    conv2_bias=tensors[f"{fp}.residual_layer1.convolution2.bias"],
                ),
                residual2=ResidualConvWeights(
                    conv1_weight=tensors[f"{fp}.residual_layer2.convolution1.weight"],
                    conv1_bias=tensors[f"{fp}.residual_layer2.convolution1.bias"],
                    conv2_weight=tensors[f"{fp}.residual_layer2.convolution2.weight"],
                    conv2_bias=tensors[f"{fp}.residual_layer2.convolution2.bias"],
                ),
            )
        )

    return DepthAnythingWeights(
        patch_weight=tensors["backbone.embeddings.patch_embeddings.projection.weight"],
        patch_bias=tensors["backbone.embeddings.patch_embeddings.projection.bias"],
        cls_token=tensors["backbone.embeddings.cls_token"].reshape(1, cfg.embed_dim),
        pos_embed=tensors["backbone.embeddings.position_embeddings"].reshape(-1, cfg.embed_dim),
        blocks=tuple(blocks),
        backbone_norm_weight=tensors["backbone.layernorm.weight"],
        backbone_norm_bias=tensors["backbone.layernorm.bias"],
        reassemble=tuple(reassemble),
        neck_convs=tuple(neck_convs),
        fusion_layers=tuple(fusion_layers),
        head_conv1_weight=tensors["head.conv1.weight"],
        head_conv1_bias=tensors["head.conv1.bias"],
        head_conv2_weight=tensors["head.conv2.weight"],
        head_conv2_bias=tensors["head.conv2.bias"],
        head_conv3_weight=tensors["head.conv3.weight"],
        head_conv3_bias=tensors["head.conv3.bias"],
    )


def create_synthetic_weights(cfg: DepthAnythingConfig, *, seed: int = 0) -> DepthAnythingWeights:
    """Construct properly-shaped synthetic weights for offline testing & benchmarking."""
    rng = np.random.default_rng(seed)

    def _w(shape: tuple[int, ...], scale: float = 0.02) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def _ones(n: int) -> np.ndarray:
        return np.ones((n,), dtype=np.float32)

    def _zeros(n: int) -> np.ndarray:
        return np.zeros((n,), dtype=np.float32)

    d = cfg.embed_dim
    mlp = cfg.mlp_dim
    p = cfg.patch_size
    f = cfg.fusion_hidden_size
    default_tokens = (cfg.default_image_size // p) ** 2 + 1

    blocks = tuple(
        EncoderBlockWeights(
            norm1_weight=_ones(d),
            norm1_bias=_zeros(d),
            q_weight=_w((d, d)),
            q_bias=_zeros(d),
            k_weight=_w((d, d)),
            k_bias=_zeros(d),
            v_weight=_w((d, d)),
            v_bias=_zeros(d),
            proj_weight=_w((d, d)),
            proj_bias=_zeros(d),
            ls1=np.full((d,), 0.1, dtype=np.float32),
            norm2_weight=_ones(d),
            norm2_bias=_zeros(d),
            fc1_weight=_w((mlp, d)),
            fc1_bias=_zeros(mlp),
            fc2_weight=_w((d, mlp)),
            fc2_bias=_zeros(d),
            ls2=np.full((d,), 0.1, dtype=np.float32),
        )
        for _ in range(cfg.depth)
    )

    reassemble: list[ReassembleWeights] = []
    neck_convs: list[np.ndarray] = []
    for c_mid, factor in zip(cfg.neck_hidden_sizes, cfg.reassemble_factors):
        if factor == 4.0:
            rw, rb = _w((c_mid, c_mid, 4, 4)), _zeros(c_mid)
        elif factor == 2.0:
            rw, rb = _w((c_mid, c_mid, 2, 2)), _zeros(c_mid)
        elif factor == 1.0:
            rw, rb = None, None
        else:
            rw, rb = _w((c_mid, c_mid, 3, 3)), _zeros(c_mid)
        reassemble.append(ReassembleWeights(_w((c_mid, d, 1, 1)), _zeros(c_mid), rw, rb))
        neck_convs.append(_w((f, c_mid, 3, 3)))

    fusion_layers = tuple(
        FusionLayerWeights(
            projection_weight=_w((f, f, 1, 1)),
            projection_bias=_zeros(f),
            residual1=ResidualConvWeights(_w((f, f, 3, 3)), _zeros(f), _w((f, f, 3, 3)), _zeros(f)),
            residual2=ResidualConvWeights(_w((f, f, 3, 3)), _zeros(f), _w((f, f, 3, 3)), _zeros(f)),
        )
        for _ in range(4)
    )

    c1 = f // 2
    c2 = cfg.head_hidden_size
    return DepthAnythingWeights(
        patch_weight=_w((d, 3, p, p)),
        patch_bias=_zeros(d),
        cls_token=_w((1, d)),
        pos_embed=_w((default_tokens, d)),
        blocks=blocks,
        backbone_norm_weight=_ones(d),
        backbone_norm_bias=_zeros(d),
        reassemble=tuple(reassemble),
        neck_convs=tuple(neck_convs),
        fusion_layers=fusion_layers,
        head_conv1_weight=_w((c1, f, 3, 3)),
        head_conv1_bias=_zeros(c1),
        head_conv2_weight=_w((c2, c1, 3, 3)),
        head_conv2_bias=_zeros(c2),
        head_conv3_weight=_w((1, c2, 1, 1)),
        head_conv3_bias=_zeros(1),
    )
