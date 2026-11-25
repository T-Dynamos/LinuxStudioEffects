"""Zero-PyTorch weight loader, BatchNorm folding, and synthetic weights for MODNet."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import pickle
import struct
from typing import Any
import zipfile

from ml_dtypes import bfloat16
import numpy as np

from .config import ModnetConfig

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
class ConvLayerWeights:
    """Convolution weights pre-folded with BatchNorm2d (and BN half of IBNorm)."""

    weight_2d: np.ndarray  # (C_out, C_in * kh * kw) or (C, 9) for DWConv
    bias: np.ndarray  # (C_out,)
    kh: int
    kw: int
    stride: int = 1
    padding: int = 0
    has_inorm: bool = False
    relu_mode: str | None = None  # None, 'relu', or 'relu6'


@dataclass(frozen=True)
class InvertedResidualWeights:
    pw: ConvLayerWeights | None
    dw: ConvLayerWeights
    pw_linear: ConvLayerWeights
    use_res_connect: bool


@dataclass(frozen=True)
class ModnetWeights:
    conv0: ConvLayerWeights
    blocks: tuple[InvertedResidualWeights, ...]
    conv_last: ConvLayerWeights
    se_fc1_t: np.ndarray  # (1280, 320)
    se_fc2_t: np.ndarray  # (320, 1280)
    conv_lr16x: ConvLayerWeights
    conv_lr8x: ConvLayerWeights
    tohr_enc2x: ConvLayerWeights
    conv_enc2x: ConvLayerWeights
    tohr_enc4x: ConvLayerWeights
    conv_enc4x: ConvLayerWeights
    conv_hr4x: tuple[ConvLayerWeights, ...]
    conv_hr2x: tuple[ConvLayerWeights, ...]
    conv_lr4x: ConvLayerWeights
    conv_f2x: ConvLayerWeights
    conv_f0: ConvLayerWeights
    conv_f1: ConvLayerWeights


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


def load_checkpoint_state_dict(path: str) -> dict[str, np.ndarray]:
    """Load a zip-archive or legacy tar/pickle PyTorch checkpoint without importing torch."""

    def _rebuild_tensor_v2(
        storage: _StorageRef,
        storage_offset: int,
        size: tuple[int, ...],
        stride: tuple[int, ...],
        _requires_grad: bool,
        _backward_hooks: Any,
    ) -> _TensorRef:
        return _TensorRef(storage=storage, offset=int(storage_offset), size=tuple(size), stride=tuple(stride))

    refs_by_key: dict[str, _StorageRef] = {}

    class _RestrictedUnpickler(pickle.Unpickler):
        def persistent_load(self, pid: Any) -> _StorageRef:
            if not isinstance(pid, tuple) or len(pid) < 5 or pid[0] != "storage":
                raise pickle.UnpicklingError(f"Unsupported persistent id: {pid!r}")
            _, stype, key, _loc, numel = pid[:5]
            sname = str(stype[1]) if isinstance(stype, tuple) and len(stype) >= 2 else str(stype)
            ref = _StorageRef(storage_type=sname, key=str(key), numel=int(numel))
            refs_by_key[ref.key] = ref
            return ref

        def find_class(self, module: str, name: str) -> Any:
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                return _rebuild_tensor_v2
            if module == "torch":
                return ("torch", name)
            if module == "collections" and name == "OrderedDict":
                return OrderedDict
            raise pickle.UnpicklingError(f"Blocked unpickle global: {module}.{name}")

    def _materialize(raw_sd: Any, read_buf: Any) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        raw_cache: dict[str, np.ndarray] = {}
        for name, ref in raw_sd.items():
            if not isinstance(ref, _TensorRef):
                continue
            dt = _TORCH_DTYPE_MAP.get(ref.storage.storage_type)
            if dt is None:
                raise ValueError(f"Unsupported PyTorch storage type: {ref.storage.storage_type}")
            buf = raw_cache.get(ref.storage.key)
            if buf is None:
                buf = read_buf(ref.storage.key, dt)
                raw_cache[ref.storage.key] = buf
            sub = buf[ref.offset :]
            if ref.size:
                byte_strides = tuple(st * dt.itemsize for st in ref.stride)
                arr = np.lib.stride_tricks.as_strided(sub, shape=ref.size, strides=byte_strides)
            else:
                arr = sub[:1].reshape(())
            out[name] = np.ascontiguousarray(arr, dtype=np.float32)
        return out

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as zf:
            pkl_candidates = [n for n in zf.namelist() if n.endswith("data.pkl")]
            if not pkl_candidates:
                raise ValueError(f"Zip checkpoint {path!r} is missing data.pkl")
            pkl_name = pkl_candidates[0]
            prefix = pkl_name[: -len("data.pkl")]
            with zf.open(pkl_name) as f:
                raw_sd = _RestrictedUnpickler(f).load()

            def _read_zip(key: str, dt: np.dtype[Any]) -> np.ndarray:
                with zf.open(f"{prefix}data/{key}") as sf:
                    return np.frombuffer(sf.read(), dtype=dt)

            return _materialize(raw_sd, _read_zip)

    with open(path, "rb") as f:
        _magic = _RestrictedUnpickler(f).load()
        _proto = _RestrictedUnpickler(f).load()
        _sys_info = _RestrictedUnpickler(f).load()
        raw_sd = _RestrictedUnpickler(f).load()
        storage_keys = _RestrictedUnpickler(f).load()

        raw_bufs: dict[str, np.ndarray] = {}
        for k in storage_keys:
            (numel,) = struct.unpack("<Q", f.read(8))
            ref = refs_by_key.get(str(k))
            if ref is None:
                raise ValueError(f"Unknown storage key {k!r} in legacy checkpoint")
            dt = _TORCH_DTYPE_MAP.get(ref.storage_type)
            if dt is None:
                raise ValueError(f"Unsupported PyTorch storage type: {ref.storage_type}")
            raw_bufs[str(k)] = np.frombuffer(f.read(numel * dt.itemsize), dtype=dt)

    return _materialize(raw_sd, lambda key, _dt: raw_bufs[key])


def _fold_conv_bn(
    weight: np.ndarray,
    bias: np.ndarray | None,
    bn_weight: np.ndarray,
    bn_bias: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    scale = (bn_weight / np.sqrt(running_var + np.float32(eps))).astype(np.float32)
    b_in = bias if bias is not None else np.zeros((weight.shape[0],), dtype=np.float32)
    w_folded = np.ascontiguousarray(weight * scale.reshape(-1, 1, 1, 1), dtype=np.float32)
    b_folded = np.ascontiguousarray((b_in - running_mean) * scale + bn_bias, dtype=np.float32)
    return w_folded, b_folded


def _fold_conv_ibn(
    weight: np.ndarray,
    bias: np.ndarray | None,
    bn_weight: np.ndarray,
    bn_bias: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    c_out = weight.shape[0]
    c_bn = c_out // 2
    b_in = bias if bias is not None else np.zeros((c_out,), dtype=np.float32)
    w_out = np.array(weight, dtype=np.float32, copy=True)
    b_out = np.array(b_in, dtype=np.float32, copy=True)
    w_bn, b_bn = _fold_conv_bn(weight[:c_bn], b_in[:c_bn], bn_weight, bn_bias, running_mean, running_var, eps=eps)
    w_out[:c_bn] = w_bn
    b_out[:c_bn] = b_bn
    return np.ascontiguousarray(w_out, dtype=np.float32), np.ascontiguousarray(b_out, dtype=np.float32)


def _parse_conv_bn(
    sd: dict[str, np.ndarray],
    conv_prefix: str,
    bn_prefix: str,
    *,
    stride: int,
    padding: int,
    relu_mode: str | None,
    eps: float,
) -> ConvLayerWeights:
    w4 = sd[f"{conv_prefix}.weight"]
    b = sd.get(f"{conv_prefix}.bias")
    w_f, b_f = _fold_conv_bn(
        w4,
        b,
        sd[f"{bn_prefix}.weight"],
        sd[f"{bn_prefix}.bias"],
        sd[f"{bn_prefix}.running_mean"],
        sd[f"{bn_prefix}.running_var"],
        eps=eps,
    )
    c_out, _, kh, kw = w_f.shape
    return ConvLayerWeights(
        weight_2d=np.ascontiguousarray(w_f.reshape(c_out, -1)),
        bias=b_f,
        kh=kh,
        kw=kw,
        stride=stride,
        padding=padding,
        has_inorm=False,
        relu_mode=relu_mode,
    )


def _parse_ibn_conv(
    sd: dict[str, np.ndarray],
    prefix: str,
    *,
    stride: int = 1,
    padding: int = 0,
    with_ibn: bool = True,
    with_relu: bool = True,
    eps: float = 1e-5,
) -> ConvLayerWeights:
    w4 = sd[f"{prefix}.layers.0.weight"]
    b = sd.get(f"{prefix}.layers.0.bias")
    c_out, _, kh, kw = w4.shape
    if with_ibn:
        w4, b_f = _fold_conv_ibn(
            w4,
            b,
            sd[f"{prefix}.layers.1.bnorm.weight"],
            sd[f"{prefix}.layers.1.bnorm.bias"],
            sd[f"{prefix}.layers.1.bnorm.running_mean"],
            sd[f"{prefix}.layers.1.bnorm.running_var"],
            eps=eps,
        )
    else:
        b_f = np.ascontiguousarray(b if b is not None else np.zeros((c_out,), dtype=np.float32), dtype=np.float32)
    return ConvLayerWeights(
        weight_2d=np.ascontiguousarray(w4.reshape(c_out, -1), dtype=np.float32),
        bias=b_f,
        kh=kh,
        kw=kw,
        stride=stride,
        padding=padding,
        has_inorm=with_ibn,
        relu_mode="relu" if with_relu else None,
    )


def load_from_hf(cfg: ModnetConfig, *, local_files_only: bool = False) -> ModnetWeights:
    """Download/load a pretrained MODNet checkpoint from Hugging Face with folded BatchNorms."""
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(
        repo_id=cfg.hf_repo_id,
        filename=cfg.checkpoint_filename,
        local_files_only=local_files_only,
    )
    raw = load_checkpoint_state_dict(ckpt_path)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in raw.items()}
    eps = cfg.norm_eps

    bb = "lr_branch.backbone.model" if "lr_branch.backbone.model.features.0.0.weight" in sd else "backbone.model"
    conv0 = _parse_conv_bn(sd, f"{bb}.features.0.0", f"{bb}.features.0.1", stride=2, padding=1, relu_mode="relu6", eps=eps)

    mbv2_spec = [
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]
    blocks: list[InvertedResidualWeights] = []
    feat_idx = 1
    in_ch = 32
    for t, c, n, s in mbv2_spec:
        for i in range(n):
            stride = s if i == 0 else 1
            res = stride == 1 and in_ch == c
            p = f"{bb}.features.{feat_idx}.conv"
            if t == 1:
                dw = _parse_conv_bn(sd, f"{p}.0", f"{p}.1", stride=stride, padding=1, relu_mode="relu6", eps=eps)
                pw_lin = _parse_conv_bn(sd, f"{p}.3", f"{p}.4", stride=1, padding=0, relu_mode=None, eps=eps)
                blocks.append(InvertedResidualWeights(pw=None, dw=dw, pw_linear=pw_lin, use_res_connect=res))
            else:
                pw = _parse_conv_bn(sd, f"{p}.0", f"{p}.1", stride=1, padding=0, relu_mode="relu6", eps=eps)
                dw = _parse_conv_bn(sd, f"{p}.3", f"{p}.4", stride=stride, padding=1, relu_mode="relu6", eps=eps)
                pw_lin = _parse_conv_bn(sd, f"{p}.6", f"{p}.7", stride=1, padding=0, relu_mode=None, eps=eps)
                blocks.append(InvertedResidualWeights(pw=pw, dw=dw, pw_linear=pw_lin, use_res_connect=res))
            in_ch = c
            feat_idx += 1

    conv_last = _parse_conv_bn(sd, f"{bb}.features.18.0", f"{bb}.features.18.1", stride=1, padding=0, relu_mode="relu6", eps=eps)

    return ModnetWeights(
        conv0=conv0,
        blocks=tuple(blocks),
        conv_last=conv_last,
        se_fc1_t=np.ascontiguousarray(sd["lr_branch.se_block.fc.0.weight"].T, dtype=np.float32),
        se_fc2_t=np.ascontiguousarray(sd["lr_branch.se_block.fc.2.weight"].T, dtype=np.float32),
        conv_lr16x=_parse_ibn_conv(sd, "lr_branch.conv_lr16x", stride=1, padding=2, eps=eps),
        conv_lr8x=_parse_ibn_conv(sd, "lr_branch.conv_lr8x", stride=1, padding=2, eps=eps),
        tohr_enc2x=_parse_ibn_conv(sd, "hr_branch.tohr_enc2x", stride=1, padding=0, eps=eps),
        conv_enc2x=_parse_ibn_conv(sd, "hr_branch.conv_enc2x", stride=2, padding=1, eps=eps),
        tohr_enc4x=_parse_ibn_conv(sd, "hr_branch.tohr_enc4x", stride=1, padding=0, eps=eps),
        conv_enc4x=_parse_ibn_conv(sd, "hr_branch.conv_enc4x", stride=1, padding=1, eps=eps),
        conv_hr4x=tuple(_parse_ibn_conv(sd, f"hr_branch.conv_hr4x.{i}", stride=1, padding=1, eps=eps) for i in range(3)),
        conv_hr2x=tuple(_parse_ibn_conv(sd, f"hr_branch.conv_hr2x.{i}", stride=1, padding=1, eps=eps) for i in range(4)),
        conv_lr4x=_parse_ibn_conv(sd, "f_branch.conv_lr4x", stride=1, padding=2, eps=eps),
        conv_f2x=_parse_ibn_conv(sd, "f_branch.conv_f2x", stride=1, padding=1, eps=eps),
        conv_f0=_parse_ibn_conv(sd, "f_branch.conv_f.0", stride=1, padding=1, eps=eps),
        conv_f1=_parse_ibn_conv(sd, "f_branch.conv_f.1", stride=1, padding=0, with_ibn=False, with_relu=False, eps=eps),
    )


def create_synthetic_weights(cfg: ModnetConfig, *, seed: int = 0) -> ModnetWeights:
    """Create lightweight random MODNet weights for unit tests without downloading checkpoints."""
    rng = np.random.default_rng(seed)

    def _conv(c_out: int, c_in: int, k: int, *, stride: int = 1, pad: int = 0, ibn: bool = False, relu: str | None = None) -> ConvLayerWeights:
        scale = np.float32(0.1 / max(1.0, np.sqrt(c_in * k * k)))
        w = (rng.standard_normal((c_out, c_in * k * k)) * scale).astype(np.float32)
        b = (rng.standard_normal((c_out,)) * np.float32(0.05)).astype(np.float32)
        return ConvLayerWeights(weight_2d=w, bias=b, kh=k, kw=k, stride=stride, padding=pad, has_inorm=ibn, relu_mode=relu)

    conv0 = _conv(32, 3, 3, stride=2, pad=1, relu="relu6")
    mbv2_spec = [
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]
    blocks: list[InvertedResidualWeights] = []
    in_ch = 32
    for t, c, n, s in mbv2_spec:
        for i in range(n):
            stride = s if i == 0 else 1
            res = stride == 1 and in_ch == c
            if t == 1:
                blocks.append(
                    InvertedResidualWeights(
                        pw=None,
                        dw=_conv(in_ch, 1, 3, stride=stride, pad=1, relu="relu6"),
                        pw_linear=_conv(c, in_ch, 1, stride=1, pad=0, relu=None),
                        use_res_connect=res,
                    )
                )
            else:
                hid = in_ch * t
                blocks.append(
                    InvertedResidualWeights(
                        pw=_conv(hid, in_ch, 1, stride=1, pad=0, relu="relu6"),
                        dw=_conv(hid, 1, 3, stride=stride, pad=1, relu="relu6"),
                        pw_linear=_conv(c, hid, 1, stride=1, pad=0, relu=None),
                        use_res_connect=res,
                    )
                )
            in_ch = c

    hr = cfg.hr_channels
    enc_ch = cfg.enc_channels
    return ModnetWeights(
        conv0=conv0,
        blocks=tuple(blocks),
        conv_last=_conv(enc_ch[4], 320, 1, relu="relu6"),
        se_fc1_t=(rng.standard_normal((enc_ch[4], enc_ch[4] // 4)) * np.float32(0.05)).astype(np.float32),
        se_fc2_t=(rng.standard_normal((enc_ch[4] // 4, enc_ch[4])) * np.float32(0.05)).astype(np.float32),
        conv_lr16x=_conv(enc_ch[3], enc_ch[4], 5, pad=2, ibn=True, relu="relu"),
        conv_lr8x=_conv(enc_ch[2], enc_ch[3], 5, pad=2, ibn=True, relu="relu"),
        tohr_enc2x=_conv(hr, enc_ch[0], 1, ibn=True, relu="relu"),
        conv_enc2x=_conv(hr, hr + 3, 3, stride=2, pad=1, ibn=True, relu="relu"),
        tohr_enc4x=_conv(hr, enc_ch[1], 1, ibn=True, relu="relu"),
        conv_enc4x=_conv(2 * hr, 2 * hr, 3, pad=1, ibn=True, relu="relu"),
        conv_hr4x=(
            _conv(2 * hr, 3 * hr + 3, 3, pad=1, ibn=True, relu="relu"),
            _conv(2 * hr, 2 * hr, 3, pad=1, ibn=True, relu="relu"),
            _conv(hr, 2 * hr, 3, pad=1, ibn=True, relu="relu"),
        ),
        conv_hr2x=(
            _conv(2 * hr, 2 * hr, 3, pad=1, ibn=True, relu="relu"),
            _conv(hr, 2 * hr, 3, pad=1, ibn=True, relu="relu"),
            _conv(hr, hr, 3, pad=1, ibn=True, relu="relu"),
            _conv(hr, hr, 3, pad=1, ibn=True, relu="relu"),
        ),
        conv_lr4x=_conv(hr, enc_ch[2], 5, pad=2, ibn=True, relu="relu"),
        conv_f2x=_conv(hr, 2 * hr, 3, pad=1, ibn=True, relu="relu"),
        conv_f0=_conv(hr // 2, hr + 3, 3, pad=1, ibn=True, relu="relu"),
        conv_f1=_conv(1, hr // 2, 1, ibn=False, relu=None),
    )
