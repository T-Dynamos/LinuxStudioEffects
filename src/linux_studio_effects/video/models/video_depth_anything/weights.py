"""Pure-Python PyTorch .pth loader and synthetic weight builder for Video Depth Anything."""

from __future__ import annotations

import math
import pickle
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np
from ml_dtypes import bfloat16

from .config import VideoDepthConfig


@dataclass(frozen=True)
class EncoderBlockWeights:
    norm1_weight: np.ndarray
    norm1_bias: np.ndarray
    qkv_weight: np.ndarray  # (3*D, D)
    qkv_bias: np.ndarray  # (3*D,)
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
class RefineNetWeights:
    out_conv_weight: np.ndarray
    out_conv_bias: np.ndarray
    res1: ResidualConvWeights
    res2: ResidualConvWeights


@dataclass(frozen=True)
class TemporalAttentionWeights:
    to_q_weight: np.ndarray  # (C, C)
    to_k_weight: np.ndarray  # (C, C)
    to_v_weight: np.ndarray  # (C, C)
    to_out_weight: np.ndarray  # (C, C)
    to_out_bias: np.ndarray  # (C,)
    pe: np.ndarray  # (1, T_max, C)


@dataclass(frozen=True)
class MotionModuleWeights:
    norm_weight: np.ndarray  # GroupNorm weight (C,)
    norm_bias: np.ndarray  # GroupNorm bias (C,)
    proj_in_weight: np.ndarray  # (C, C)
    proj_in_bias: np.ndarray  # (C,)
    attn0_norm_weight: np.ndarray  # LayerNorm (C,)
    attn0_norm_bias: np.ndarray
    attn0: TemporalAttentionWeights
    attn1_norm_weight: np.ndarray  # LayerNorm (C,)
    attn1_norm_bias: np.ndarray
    attn1: TemporalAttentionWeights
    ff_norm_weight: np.ndarray  # LayerNorm (C,)
    ff_norm_bias: np.ndarray
    ff_proj_weight: np.ndarray  # GEGLU (8*C, C)
    ff_proj_bias: np.ndarray  # (8*C,)
    ff_out_weight: np.ndarray  # (C, 4*C)
    ff_out_bias: np.ndarray  # (C,)
    proj_out_weight: np.ndarray  # (C, C)
    proj_out_bias: np.ndarray  # (C,)


@dataclass(frozen=True)
class VideoDepthWeights:
    """All DINOv2 backbone, DPT neck/head, and spatiotemporal MotionModule weights in float32."""

    patch_weight: np.ndarray  # (D, 3, 14, 14)
    patch_bias: np.ndarray  # (D,)
    cls_token: np.ndarray  # (1, D)
    pos_embed: np.ndarray  # (1370, D)
    blocks: tuple[EncoderBlockWeights, ...]
    backbone_norm_weight: np.ndarray
    backbone_norm_bias: np.ndarray
    reassemble: tuple[ReassembleWeights, ...]
    layer_rn_weights: tuple[np.ndarray, ...]  # (F, C_i, 3, 3) no bias
    refinenets: tuple[RefineNetWeights, ...]  # 1, 2, 3, 4 (index 0 = refinenet1)
    motion_modules: tuple[MotionModuleWeights, ...]  # 0 (layer3), 1 (layer4), 2 (path4), 3 (path3)
    output_conv1_weight: np.ndarray
    output_conv1_bias: np.ndarray
    output_conv2_0_weight: np.ndarray
    output_conv2_0_bias: np.ndarray
    output_conv2_2_weight: np.ndarray
    output_conv2_2_bias: np.ndarray


_TORCH_DTYPE_MAP: dict[str, np.dtype[Any]] = {
    "FloatStorage": np.dtype("<f4"),
    "HalfStorage": np.dtype("<f2"),
    "BFloat16Storage": np.dtype(bfloat16),
    "DoubleStorage": np.dtype("<f8"),
    "LongStorage": np.dtype("<i8"),
    "IntStorage": np.dtype("<i4"),
    "BoolStorage": np.dtype("?"),
}


@dataclass(frozen=True)
class _StorageRef:
    storage_type: str
    key: str
    numel: int


@dataclass(frozen=True)
class _TensorRef:
    storage: _StorageRef
    offset: int
    size: tuple[int, ...]
    stride: tuple[int, ...]


def load_pth_state_dict(path: str) -> dict[str, np.ndarray]:
    """Read a zip-format PyTorch .pth state_dict cleanly using stdlib zipfile + pickle."""

    def _rebuild_tensor_v2(
        storage: _StorageRef,
        storage_offset: int,
        size: tuple[int, ...],
        stride: tuple[int, ...],
        _requires_grad: bool,
        _backward_hooks: Any,
    ) -> _TensorRef:
        return _TensorRef(storage=storage, offset=int(storage_offset), size=tuple(size), stride=tuple(stride))

    class _RestrictedUnpickler(pickle.Unpickler):
        def persistent_load(self, pid: Any) -> _StorageRef:
            if not isinstance(pid, tuple) or len(pid) < 5 or pid[0] != "storage":
                raise pickle.UnpicklingError(f"Unsupported persistent id: {pid!r}")
            _, stype, key, _loc, numel = pid
            if isinstance(stype, tuple) and len(stype) == 2 and stype[0] == "torch":
                sname = str(stype[1])
            else:
                sname = str(stype)
            return _StorageRef(storage_type=sname, key=str(key), numel=int(numel))

        def find_class(self, module: str, name: str) -> Any:
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                return _rebuild_tensor_v2
            if module == "torch":
                return ("torch", name)
            if module == "collections" and name == "OrderedDict":
                return OrderedDict
            raise pickle.UnpicklingError(f"Blocked unpickle global: {module}.{name}")

    tensors: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        pkl_candidates = [n for n in names if n.endswith("data.pkl")]
        if not pkl_candidates:
            raise ValueError(f"Checkpoint {path!r} is not a zip-archive PyTorch checkpoint")
        pkl_name = pkl_candidates[0]
        prefix = pkl_name[: -len("data.pkl")]

        with zf.open(pkl_name) as f:
            raw_sd = _RestrictedUnpickler(f).load()

        raw_cache: dict[str, np.ndarray] = {}
        for name, ref in raw_sd.items():
            if not isinstance(ref, _TensorRef):
                continue
            dt = _TORCH_DTYPE_MAP.get(ref.storage.storage_type)
            if dt is None:
                raise ValueError(f"Unsupported PyTorch storage type: {ref.storage.storage_type}")
            buf = raw_cache.get(ref.storage.key)
            if buf is None:
                with zf.open(f"{prefix}data/{ref.storage.key}") as sf:
                    buf = np.frombuffer(sf.read(), dtype=dt)
                raw_cache[ref.storage.key] = buf

            itemsize = dt.itemsize
            sub = buf[ref.offset :]
            if ref.size:
                byte_strides = tuple(st * itemsize for st in ref.stride)
                arr = np.lib.stride_tricks.as_strided(sub, shape=ref.size, strides=byte_strides)
            else:
                arr = sub[:1].reshape(())
            tensors[name] = np.ascontiguousarray(arr, dtype=np.float32)

    return tensors


def _parse_temporal_attn(tensors: dict[str, np.ndarray], prefix: str) -> TemporalAttentionWeights:
    return TemporalAttentionWeights(
        to_q_weight=tensors[f"{prefix}.to_q.weight"],
        to_k_weight=tensors[f"{prefix}.to_k.weight"],
        to_v_weight=tensors[f"{prefix}.to_v.weight"],
        to_out_weight=tensors[f"{prefix}.to_out.0.weight"],
        to_out_bias=tensors[f"{prefix}.to_out.0.bias"],
        pe=tensors[f"{prefix}.pos_encoder.pe"],
    )


def load_from_hf(
    cfg: VideoDepthConfig,
    *,
    local_files_only: bool = False,
) -> VideoDepthWeights:
    """Download (or read from HF cache) and parse Video Depth Anything .pth weights."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        cfg.hf_repo_id,
        cfg.checkpoint_filename,
        local_files_only=local_files_only,
    )
    t = load_pth_state_dict(path)

    blocks = tuple(
        EncoderBlockWeights(
            norm1_weight=t[f"pretrained.blocks.{i}.norm1.weight"],
            norm1_bias=t[f"pretrained.blocks.{i}.norm1.bias"],
            qkv_weight=t[f"pretrained.blocks.{i}.attn.qkv.weight"],
            qkv_bias=t[f"pretrained.blocks.{i}.attn.qkv.bias"],
            proj_weight=t[f"pretrained.blocks.{i}.attn.proj.weight"],
            proj_bias=t[f"pretrained.blocks.{i}.attn.proj.bias"],
            ls1=t[f"pretrained.blocks.{i}.ls1.gamma"],
            norm2_weight=t[f"pretrained.blocks.{i}.norm2.weight"],
            norm2_bias=t[f"pretrained.blocks.{i}.norm2.bias"],
            fc1_weight=t[f"pretrained.blocks.{i}.mlp.fc1.weight"],
            fc1_bias=t[f"pretrained.blocks.{i}.mlp.fc1.bias"],
            fc2_weight=t[f"pretrained.blocks.{i}.mlp.fc2.weight"],
            fc2_bias=t[f"pretrained.blocks.{i}.mlp.fc2.bias"],
            ls2=t[f"pretrained.blocks.{i}.ls2.gamma"],
        )
        for i in range(cfg.depth)
    )

    reassemble = tuple(
        ReassembleWeights(
            proj_weight=t[f"head.projects.{i}.weight"],
            proj_bias=t[f"head.projects.{i}.bias"],
            resize_weight=t.get(f"head.resize_layers.{i}.weight"),
            resize_bias=t.get(f"head.resize_layers.{i}.bias"),
        )
        for i in range(4)
    )

    layer_rn_weights = tuple(t[f"head.scratch.layer{i}_rn.weight"] for i in range(1, 5))

    refinenets = tuple(
        RefineNetWeights(
            out_conv_weight=t[f"head.scratch.refinenet{i}.out_conv.weight"],
            out_conv_bias=t[f"head.scratch.refinenet{i}.out_conv.bias"],
            res1=ResidualConvWeights(
                conv1_weight=t[f"head.scratch.refinenet{i}.resConfUnit1.conv1.weight"],
                conv1_bias=t[f"head.scratch.refinenet{i}.resConfUnit1.conv1.bias"],
                conv2_weight=t[f"head.scratch.refinenet{i}.resConfUnit1.conv2.weight"],
                conv2_bias=t[f"head.scratch.refinenet{i}.resConfUnit1.conv2.bias"],
            ),
            res2=ResidualConvWeights(
                conv1_weight=t[f"head.scratch.refinenet{i}.resConfUnit2.conv1.weight"],
                conv1_bias=t[f"head.scratch.refinenet{i}.resConfUnit2.conv1.bias"],
                conv2_weight=t[f"head.scratch.refinenet{i}.resConfUnit2.conv2.weight"],
                conv2_bias=t[f"head.scratch.refinenet{i}.resConfUnit2.conv2.bias"],
            ),
        )
        for i in range(1, 5)
    )

    motion_modules = []
    for i in range(4):
        mp = f"head.motion_modules.{i}.temporal_transformer"
        tb = f"{mp}.transformer_blocks.0"
        motion_modules.append(
            MotionModuleWeights(
                norm_weight=t[f"{mp}.norm.weight"],
                norm_bias=t[f"{mp}.norm.bias"],
                proj_in_weight=t[f"{mp}.proj_in.weight"],
                proj_in_bias=t[f"{mp}.proj_in.bias"],
                attn0_norm_weight=t[f"{tb}.norms.0.weight"],
                attn0_norm_bias=t[f"{tb}.norms.0.bias"],
                attn0=_parse_temporal_attn(t, f"{tb}.attention_blocks.0"),
                attn1_norm_weight=t[f"{tb}.norms.1.weight"],
                attn1_norm_bias=t[f"{tb}.norms.1.bias"],
                attn1=_parse_temporal_attn(t, f"{tb}.attention_blocks.1"),
                ff_norm_weight=t[f"{tb}.ff_norm.weight"],
                ff_norm_bias=t[f"{tb}.ff_norm.bias"],
                ff_proj_weight=t[f"{tb}.ff.net.0.proj.weight"],
                ff_proj_bias=t[f"{tb}.ff.net.0.proj.bias"],
                ff_out_weight=t[f"{tb}.ff.net.2.weight"],
                ff_out_bias=t[f"{tb}.ff.net.2.bias"],
                proj_out_weight=t[f"{mp}.proj_out.weight"],
                proj_out_bias=t[f"{mp}.proj_out.bias"],
            )
        )

    return VideoDepthWeights(
        patch_weight=t["pretrained.patch_embed.proj.weight"],
        patch_bias=t["pretrained.patch_embed.proj.bias"],
        cls_token=t["pretrained.cls_token"].reshape(1, cfg.embed_dim),
        pos_embed=t["pretrained.pos_embed"].reshape(-1, cfg.embed_dim),
        blocks=blocks,
        backbone_norm_weight=t["pretrained.norm.weight"],
        backbone_norm_bias=t["pretrained.norm.bias"],
        reassemble=reassemble,
        layer_rn_weights=layer_rn_weights,
        refinenets=refinenets,
        motion_modules=tuple(motion_modules),
        output_conv1_weight=t["head.scratch.output_conv1.weight"],
        output_conv1_bias=t["head.scratch.output_conv1.bias"],
        output_conv2_0_weight=t["head.scratch.output_conv2.0.weight"],
        output_conv2_0_bias=t["head.scratch.output_conv2.0.bias"],
        output_conv2_2_weight=t["head.scratch.output_conv2.2.weight"],
        output_conv2_2_bias=t["head.scratch.output_conv2.2.bias"],
    )


def _sinusoidal_pe(max_len: int, d_model: int) -> np.ndarray:
    pos = np.arange(max_len, dtype=np.float32)[:, None]
    div = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * (-math.log(10000.0) / d_model))
    pe = np.zeros((1, max_len, d_model), dtype=np.float32)
    pe[0, :, 0::2] = np.sin(pos * div)
    pe[0, :, 1::2] = np.cos(pos * div)
    return pe


def create_synthetic_weights(cfg: VideoDepthConfig, *, seed: int = 0) -> VideoDepthWeights:
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
            qkv_weight=_w((3 * d, d)),
            qkv_bias=_zeros(3 * d),
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
    layer_rn_weights: list[np.ndarray] = []
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
        layer_rn_weights.append(_w((f, c_mid, 3, 3)))

    refinenets = tuple(
        RefineNetWeights(
            out_conv_weight=_w((f, f, 1, 1)),
            out_conv_bias=_zeros(f),
            res1=ResidualConvWeights(_w((f, f, 3, 3)), _zeros(f), _w((f, f, 3, 3)), _zeros(f)),
            res2=ResidualConvWeights(_w((f, f, 3, 3)), _zeros(f), _w((f, f, 3, 3)), _zeros(f)),
        )
        for _ in range(4)
    )

    def _mk_attn(ch: int) -> TemporalAttentionWeights:
        return TemporalAttentionWeights(
            to_q_weight=_w((ch, ch)),
            to_k_weight=_w((ch, ch)),
            to_v_weight=_w((ch, ch)),
            to_out_weight=_w((ch, ch)),
            to_out_bias=_zeros(ch),
            pe=_sinusoidal_pe(cfg.num_frames, ch),
        )

    motion_modules = tuple(
        MotionModuleWeights(
            norm_weight=_ones(ch),
            norm_bias=_zeros(ch),
            proj_in_weight=_w((ch, ch)),
            proj_in_bias=_zeros(ch),
            attn0_norm_weight=_ones(ch),
            attn0_norm_bias=_zeros(ch),
            attn0=_mk_attn(ch),
            attn1_norm_weight=_ones(ch),
            attn1_norm_bias=_zeros(ch),
            attn1=_mk_attn(ch),
            ff_norm_weight=_ones(ch),
            ff_norm_bias=_zeros(ch),
            ff_proj_weight=_w((8 * ch, ch)),
            ff_proj_bias=_zeros(8 * ch),
            ff_out_weight=_w((ch, 4 * ch)),
            ff_out_bias=_zeros(ch),
            proj_out_weight=_w((ch, ch), scale=0.005),
            proj_out_bias=_zeros(ch),
        )
        for ch in cfg.motion_channels
    )

    c1 = f // 2
    c2 = cfg.head_hidden_size
    return VideoDepthWeights(
        patch_weight=_w((d, 3, p, p)),
        patch_bias=_zeros(d),
        cls_token=_w((1, d)),
        pos_embed=_w((default_tokens, d)),
        blocks=blocks,
        backbone_norm_weight=_ones(d),
        backbone_norm_bias=_zeros(d),
        reassemble=tuple(reassemble),
        layer_rn_weights=tuple(layer_rn_weights),
        refinenets=refinenets,
        motion_modules=motion_modules,
        output_conv1_weight=_w((c1, f, 3, 3)),
        output_conv1_bias=_zeros(c1),
        output_conv2_0_weight=_w((c2, c1, 3, 3)),
        output_conv2_0_bias=_zeros(c2),
        output_conv2_2_weight=_w((1, c2, 1, 1)),
        output_conv2_2_bias=_zeros(1),
    )
