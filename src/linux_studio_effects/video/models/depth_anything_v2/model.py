"""From-scratch Depth Anything V2 (Small / Base / Large) across CPU, NPU, and PyTorch GPU."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import DepthAnythingConfig
from .npu_ops import NpuExecStats, NpuMatMulExecutor
from .weights import DepthAnythingWeights

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


@dataclass(frozen=True)
class InferenceResult:
    """Predicted depth map and backend timing breakdown."""

    depth: np.ndarray  # (H_in, W_in) float32
    total_ms: float
    backbone_ms: float
    neck_head_ms: float
    npu_stats: NpuExecStats | None = None


def _bilinear_resize_nchw(x: np.ndarray, out_h: int, out_w: int, *, align_corners: bool = True) -> np.ndarray:
    """Vectorized NumPy bilinear interpolation matching torch.nn.functional.interpolate."""
    n, c, in_h, in_w = x.shape
    if in_h == out_h and in_w == out_w:
        return x

    if out_h == 1:
        y_coords = np.zeros((1,), dtype=np.float32)
    elif align_corners:
        y_coords = np.linspace(0.0, in_h - 1, out_h, dtype=np.float32)
    else:
        y_coords = np.clip((np.arange(out_h, dtype=np.float32) + 0.5) * (in_h / out_h) - 0.5, 0.0, in_h - 1)

    if out_w == 1:
        x_coords = np.zeros((1,), dtype=np.float32)
    elif align_corners:
        x_coords = np.linspace(0.0, in_w - 1, out_w, dtype=np.float32)
    else:
        x_coords = np.clip((np.arange(out_w, dtype=np.float32) + 0.5) * (in_w / out_w) - 0.5, 0.0, in_w - 1)

    y0 = np.floor(y_coords).astype(np.int64)
    x0 = np.floor(x_coords).astype(np.int64)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)

    wy = (y_coords - y0)[None, None, :, None]
    wx = (x_coords - x0)[None, None, None, :]

    v00 = x[:, :, y0[:, None], x0[None, :]]
    v01 = x[:, :, y0[:, None], x1[None, :]]
    v10 = x[:, :, y1[:, None], x0[None, :]]
    v11 = x[:, :, y1[:, None], x1[None, :]]

    top = v00 * (1.0 - wx) + v01 * wx
    bot = v10 * (1.0 - wx) + v11 * wx
    return (top * (1.0 - wy) + bot * wy).astype(np.float32)


def _bicubic_keys_weights(t: np.ndarray, a: float = -0.75) -> np.ndarray:
    """Catmull-Rom / Keys bicubic filter kernel matching PyTorch bicubic resize."""
    abs_t = np.abs(t)
    w = np.zeros_like(abs_t, dtype=np.float32)
    m1 = abs_t <= 1.0
    m2 = (abs_t > 1.0) & (abs_t < 2.0)
    t1 = abs_t[m1]
    t2 = abs_t[m2]
    w[m1] = (a + 2.0) * (t1**3) - (a + 3.0) * (t1**2) + 1.0
    w[m2] = a * (t2**3) - 5.0 * a * (t2**2) + 8.0 * a * t2 - 4.0 * a
    return w


def _bicubic_resize_hwc(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bicubic resize for (H, W, C) float32 tensors (align_corners=False)."""
    in_h, in_w, c = arr.shape
    if in_h == out_h and in_w == out_w:
        return arr

    # Horizontal pass
    gx = (np.arange(out_w, dtype=np.float32) + 0.5) * (in_w / out_w) - 0.5
    ix = np.floor(gx).astype(np.int64)
    fx = gx - ix
    tmp = np.zeros((in_h, out_w, c), dtype=np.float32)
    for k in range(-1, 3):
        idx = np.clip(ix + k, 0, in_w - 1)
        wk = _bicubic_keys_weights(fx - k)[None, :, None]
        tmp += arr[:, idx, :] * wk

    # Vertical pass
    gy = (np.arange(out_h, dtype=np.float32) + 0.5) * (in_h / out_h) - 0.5
    iy = np.floor(gy).astype(np.int64)
    fy = gy - iy
    out = np.zeros((out_h, out_w, c), dtype=np.float32)
    for k in range(-1, 3):
        idx = np.clip(iy + k, 0, in_h - 1)
        wk = _bicubic_keys_weights(fy - k)[:, None, None]
        out += tmp[idx, :, :] * wk
    return out


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)) * weight + bias


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.vectorize(math.erf, otypes=[np.float32])(x * 0.7071067811865476)) if x.size < 4096 else (
        # Fast accurate erf/tanh approximation for large matrices
        0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * (x * x * x))))
    )


def _softmax(x: np.ndarray) -> np.ndarray:
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def _conv2d_nchw(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """Im2col + BLAS Conv2d for N=1 NCHW float32 tensors."""
    _, c_in, h, w = x.shape
    c_out, _, kh, kw = weight.shape
    if kh == 1 and kw == 1 and stride == 1 and padding == 0:
        out = weight.reshape(c_out, c_in) @ x.reshape(c_in, h * w)
        out = out.reshape(1, c_out, h, w)
        if bias is not None:
            out = out + bias.reshape(1, c_out, 1, 1)
        return out

    x_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant") if padding > 0 else x
    oh = (h + 2 * padding - kh) // stride + 1
    ow = (w + 2 * padding - kw) // stride + 1

    # Strided view for zero-copy im2col patches
    s_b, s_c, s_h, s_w = x_pad.strides
    patches = np.lib.stride_tricks.as_strided(
        x_pad,
        shape=(c_in, kh, kw, oh, ow),
        strides=(s_c, s_h, s_w, s_h * stride, s_w * stride),
    ).reshape(c_in * kh * kw, oh * ow)

    out = (weight.reshape(c_out, c_in * kh * kw) @ patches).reshape(1, c_out, oh, ow)
    if bias is not None:
        out = out + bias.reshape(1, c_out, 1, 1)
    return out


def _conv_transpose2d_nchw(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    stride: int,
) -> np.ndarray:
    """ConvTranspose2d for stride == kernel_size (4x4 stride 4, 2x2 stride 2) via col2im."""
    _, c_in, h, w = x.shape
    _, c_out, kh, kw = weight.shape
    assert kh == stride and kw == stride
    # (c_in, kh*kw*c_out)^T @ (c_in, h*w) -> (c_out, kh, kw, h, w)
    w_mat = weight.reshape(c_in, c_out * kh * kw).T
    cols = (w_mat @ x.reshape(c_in, h * w)).reshape(c_out, kh, kw, h, w)
    out = cols.transpose(0, 3, 1, 4, 2).reshape(1, c_out, h * kh, w * kw)
    if bias is not None:
        out = out + bias.reshape(1, c_out, 1, 1)
    return np.ascontiguousarray(out, dtype=np.float32)


class DepthAnythingV2Model:
    """From-scratch Depth Anything V2 inference across CPU, NPU1, and PyTorch GPU."""

    def __init__(
        self,
        config: DepthAnythingConfig,
        weights: DepthAnythingWeights,
        *,
        backend: str = "cpu",
    ) -> None:
        self.config = config
        self.weights = weights
        self.backend = backend.lower()
        if self.backend not in ("cpu", "npu", "gpu"):
            raise ValueError(f"Unsupported backend {backend!r}; choose 'cpu', 'npu', or 'gpu'")

        self._npu_mm: NpuMatMulExecutor | None = None
        self._pos_embed_cache: dict[tuple[int, int], np.ndarray] = {}
        self._wt_cache: dict[int, np.ndarray] = {}
        self._qkv_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._torch_ctx: dict[str, Any] | None = None

        if self.backend == "npu":
            self._npu_mm = NpuMatMulExecutor()
        elif self.backend == "gpu":
            self._init_torch_gpu()

    def _init_torch_gpu(self) -> None:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
            dev = torch.device("xpu")
        else:
            raise RuntimeError("PyTorch GPU backend requested, but neither ROCm/CUDA nor XPU is available")

        w = self.weights
        torch_tensors: dict[str, Any] = {
            "device": dev,
            "patch_w": torch.from_numpy(w.patch_weight).to(dev),
            "patch_b": torch.from_numpy(w.patch_bias).to(dev),
            "cls": torch.from_numpy(w.cls_token).unsqueeze(0).to(dev),
            "pos": torch.from_numpy(w.pos_embed).to(dev),
            "norm_w": torch.from_numpy(w.backbone_norm_weight).to(dev),
            "norm_b": torch.from_numpy(w.backbone_norm_bias).to(dev),
            "mean": torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1),
            "std": torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1),
            "blocks": [
                {
                    "n1w": torch.from_numpy(b.norm1_weight).to(dev),
                    "n1b": torch.from_numpy(b.norm1_bias).to(dev),
                    "qw": torch.from_numpy(b.q_weight).to(dev),
                    "qb": torch.from_numpy(b.q_bias).to(dev),
                    "kw": torch.from_numpy(b.k_weight).to(dev),
                    "kb": torch.from_numpy(b.k_bias).to(dev),
                    "vw": torch.from_numpy(b.v_weight).to(dev),
                    "vb": torch.from_numpy(b.v_bias).to(dev),
                    "pw": torch.from_numpy(b.proj_weight).to(dev),
                    "pb": torch.from_numpy(b.proj_bias).to(dev),
                    "ls1": torch.from_numpy(b.ls1).to(dev),
                    "n2w": torch.from_numpy(b.norm2_weight).to(dev),
                    "n2b": torch.from_numpy(b.norm2_bias).to(dev),
                    "f1w": torch.from_numpy(b.fc1_weight).to(dev),
                    "f1b": torch.from_numpy(b.fc1_bias).to(dev),
                    "f2w": torch.from_numpy(b.fc2_weight).to(dev),
                    "f2b": torch.from_numpy(b.fc2_bias).to(dev),
                    "ls2": torch.from_numpy(b.ls2).to(dev),
                }
                for b in w.blocks
            ],
            "reassemble": [
                {
                    "pw": torch.from_numpy(r.proj_weight).to(dev),
                    "pb": torch.from_numpy(r.proj_bias).to(dev),
                    "rw": torch.from_numpy(r.resize_weight).to(dev) if r.resize_weight is not None else None,
                    "rb": torch.from_numpy(r.resize_bias).to(dev) if r.resize_bias is not None else None,
                }
                for r in w.reassemble
            ],
            "convs": [torch.from_numpy(c).to(dev) for c in w.neck_convs],
            "fusion": [
                {
                    "pw": torch.from_numpy(fl.projection_weight).to(dev),
                    "pb": torch.from_numpy(fl.projection_bias).to(dev),
                    "r1c1w": torch.from_numpy(fl.residual1.conv1_weight).to(dev),
                    "r1c1b": torch.from_numpy(fl.residual1.conv1_bias).to(dev),
                    "r1c2w": torch.from_numpy(fl.residual1.conv2_weight).to(dev),
                    "r1c2b": torch.from_numpy(fl.residual1.conv2_bias).to(dev),
                    "r2c1w": torch.from_numpy(fl.residual2.conv1_weight).to(dev),
                    "r2c1b": torch.from_numpy(fl.residual2.conv1_bias).to(dev),
                    "r2c2w": torch.from_numpy(fl.residual2.conv2_weight).to(dev),
                    "r2c2b": torch.from_numpy(fl.residual2.conv2_bias).to(dev),
                }
                for fl in w.fusion_layers
            ],
            "h1w": torch.from_numpy(w.head_conv1_weight).to(dev),
            "h1b": torch.from_numpy(w.head_conv1_bias).to(dev),
            "h2w": torch.from_numpy(w.head_conv2_weight).to(dev),
            "h2b": torch.from_numpy(w.head_conv2_bias).to(dev),
            "h3w": torch.from_numpy(w.head_conv3_weight).to(dev),
            "h3b": torch.from_numpy(w.head_conv3_bias).to(dev),
        }
        self._torch_ctx = torch_tensors

    def _get_pos_embed(self, ph: int, pw: int) -> np.ndarray:
        key = (ph, pw)
        if key in self._pos_embed_cache:
            return self._pos_embed_cache[key]

        pos = self.weights.pos_embed  # (N_def, D)
        cls_pos = pos[:1, :]
        patch_pos = pos[1:, :]
        def_g = int(round(math.sqrt(patch_pos.shape[0])))
        if ph == def_g and pw == def_g:
            res = pos
        else:
            grid = patch_pos.reshape(def_g, def_g, self.config.embed_dim)
            resized = _bicubic_resize_hwc(grid, ph, pw).reshape(ph * pw, self.config.embed_dim)
            res = np.concatenate([cls_pos, resized], axis=0)
        self._pos_embed_cache[key] = res
        return res

    def _linear_wt(self, x: np.ndarray, wt: np.ndarray, bias: np.ndarray) -> np.ndarray:
        if self._npu_mm is not None:
            y = self._npu_mm.matmul(x, wt, cache_b=True)
        else:
            y = x @ wt
        return y + bias

    def _linear(self, x: np.ndarray, weight_out_in: np.ndarray, bias: np.ndarray) -> np.ndarray:
        wid = id(weight_out_in)
        wt = self._wt_cache.get(wid)
        if wt is None:
            wt = np.ascontiguousarray(weight_out_in.T)
            self._wt_cache[wid] = wt
        return self._linear_wt(x, wt, bias)

    def _get_qkv_wt(self, idx: int, blk: Any) -> tuple[np.ndarray, np.ndarray]:
        cached = self._qkv_cache.get(idx)
        if cached is not None:
            return cached
        wt = np.ascontiguousarray(np.concatenate([blk.q_weight, blk.k_weight, blk.v_weight], axis=0).T)
        bias = np.ascontiguousarray(np.concatenate([blk.q_bias, blk.k_bias, blk.v_bias], axis=0))
        self._qkv_cache[idx] = (wt, bias)
        return wt, bias

    def preprocess(self, rgb: np.ndarray) -> np.ndarray:
        """Resize (H, W, 3) uint8 RGB frame to (1, 3, S, S) normalized float32."""
        s = self.config.input_size
        h, w, _ = rgb.shape
        x = rgb.astype(np.float32) * (1.0 / 255.0)
        x = x.transpose(2, 0, 1)[None, :, :, :]
        if h != s or w != s:
            x = _bilinear_resize_nchw(x, s, s, align_corners=False)
        return (x - _IMAGENET_MEAN) / _IMAGENET_STD

    def _forward_backbone(self, pixel_values: np.ndarray) -> tuple[list[np.ndarray], int, int]:
        cfg = self.config
        w = self.weights
        p = cfg.patch_size
        _, _, ih, iw = pixel_values.shape
        ph, pw = ih // p, iw // p

        # PatchEmbed as im2col + MatMul (runs on NPU when backend='npu')
        # pixel_values: (1, 3, ph*p, pw*p) -> (ph*pw, 3*p*p)
        patches = (
            pixel_values.reshape(3, ph, p, pw, p)
            .transpose(1, 3, 0, 2, 4)
            .reshape(ph * pw, 3 * p * p)
        )
        patch_w_flat = np.ascontiguousarray(w.patch_weight.reshape(cfg.embed_dim, 3 * p * p).T)
        if self._npu_mm is not None:
            patch_tokens = self._npu_mm.matmul(patches, patch_w_flat, cache_b=True) + w.patch_bias
        else:
            patch_tokens = patches @ patch_w_flat + w.patch_bias

        x = np.concatenate([w.cls_token, patch_tokens], axis=0) + self._get_pos_embed(ph, pw)

        taps: list[np.ndarray] = []
        tap_set = set(cfg.tap_indices)
        scale = 1.0 / math.sqrt(cfg.head_dim)
        n_tok = x.shape[0]

        for idx, blk in enumerate(w.blocks):
            h_norm = _layer_norm(x, blk.norm1_weight, blk.norm1_bias, cfg.layer_norm_eps)

            # Fused QKV projection: 1 dispatch instead of 3 separate Q, K, V MatMuls
            qkv_wt, qkv_bias = self._get_qkv_wt(idx, blk)
            qkv = self._linear_wt(h_norm, qkv_wt, qkv_bias)
            q, k, v = np.split(qkv, 3, axis=-1)

            qh = q.reshape(n_tok, cfg.num_heads, cfg.head_dim).transpose(1, 0, 2)
            kh = k.reshape(n_tok, cfg.num_heads, cfg.head_dim).transpose(1, 2, 0)
            vh = v.reshape(n_tok, cfg.num_heads, cfg.head_dim).transpose(1, 0, 2)

            attn = _softmax((qh @ kh) * scale)
            ctx = (attn @ vh).transpose(1, 0, 2).reshape(n_tok, cfg.embed_dim)
            attn_out = self._linear(ctx, blk.proj_weight, blk.proj_bias)
            x = x + attn_out * blk.ls1

            h_norm2 = _layer_norm(x, blk.norm2_weight, blk.norm2_bias, cfg.layer_norm_eps)
            mlp_hid = _gelu(self._linear(h_norm2, blk.fc1_weight, blk.fc1_bias))
            mlp_out = self._linear(mlp_hid, blk.fc2_weight, blk.fc2_bias)
            x = x + mlp_out * blk.ls2

            if idx in tap_set:
                tapped = _layer_norm(x, w.backbone_norm_weight, w.backbone_norm_bias, cfg.layer_norm_eps)
                taps.append(tapped)

        return taps, ph, pw

    def _residual_unit(self, x: np.ndarray, rcw: Any) -> np.ndarray:
        h = np.maximum(x, 0.0)
        h = _conv2d_nchw(h, rcw.conv1_weight, rcw.conv1_bias, stride=1, padding=1)
        h = np.maximum(h, 0.0)
        h = _conv2d_nchw(h, rcw.conv2_weight, rcw.conv2_bias, stride=1, padding=1)
        return x + h

    def _forward_neck_and_head(self, taps: list[np.ndarray], ph: int, pw: int) -> np.ndarray:
        cfg = self.config
        w = self.weights

        # 1. ReassembleStage
        reassembled: list[np.ndarray] = []
        for i, hidden in enumerate(taps):
            # Drop CLS token and reshape to (1, D, ph, pw)
            feat = hidden[1:, :].reshape(ph, pw, cfg.embed_dim).transpose(2, 0, 1)[None, :, :, :]
            rw = w.reassemble[i]
            feat = _conv2d_nchw(feat, rw.proj_weight, rw.proj_bias)
            factor = cfg.reassemble_factors[i]
            if rw.resize_weight is not None:
                if factor > 1.0:
                    feat = _conv_transpose2d_nchw(feat, rw.resize_weight, rw.resize_bias, stride=int(factor))
                else:
                    feat = _conv2d_nchw(feat, rw.resize_weight, rw.resize_bias, stride=2, padding=1)
            reassembled.append(feat)

        # 2. Neck 3x3 convs (no bias)
        features = [
            _conv2d_nchw(feat, w.neck_convs[i], None, stride=1, padding=1)
            for i, feat in enumerate(reassembled)
        ]

        # 3. FeatureFusionStage (deepest to shallowest)
        rev = features[::-1]
        fused: np.ndarray | None = None
        for idx, (feat, fl) in enumerate(zip(rev, w.fusion_layers)):
            target_size = rev[idx + 1].shape[2:] if idx != len(rev) - 1 else None
            if fused is None:
                cur = feat
            else:
                res = feat
                if fused.shape != res.shape:
                    res = _bilinear_resize_nchw(res, fused.shape[2], fused.shape[3], align_corners=False)
                cur = fused + self._residual_unit(res, fl.residual1)
            cur = self._residual_unit(cur, fl.residual2)
            if target_size is None:
                cur = _bilinear_resize_nchw(cur, cur.shape[2] * 2, cur.shape[3] * 2, align_corners=True)
            else:
                cur = _bilinear_resize_nchw(cur, target_size[0], target_size[1], align_corners=True)
            fused = _conv2d_nchw(cur, fl.projection_weight, fl.projection_bias)

        assert fused is not None
        # 4. Depth head
        d = _conv2d_nchw(fused, w.head_conv1_weight, w.head_conv1_bias, stride=1, padding=1)
        d = _bilinear_resize_nchw(d, ph * cfg.patch_size, pw * cfg.patch_size, align_corners=True)
        d = np.maximum(_conv2d_nchw(d, w.head_conv2_weight, w.head_conv2_bias, stride=1, padding=1), 0.0)
        d = np.maximum(_conv2d_nchw(d, w.head_conv3_weight, w.head_conv3_bias, stride=1, padding=0), 0.0)
        return d[0, 0]

    def _infer_gpu(self, rgb: np.ndarray) -> InferenceResult:
        import torch  # type: ignore[import-not-found]
        import torch.nn.functional as F  # type: ignore[import-not-found]

        assert self._torch_ctx is not None
        tc = self._torch_ctx
        dev = tc["device"]
        cfg = self.config

        t0 = time.perf_counter()
        with torch.no_grad():
            img = torch.from_numpy(rgb).to(dev, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            img = F.interpolate(img, size=(cfg.input_size, cfg.input_size), mode="bilinear", align_corners=False)
            x = (img - tc["mean"]) / tc["std"]

            ph = pw = cfg.input_size // cfg.patch_size
            x = F.conv2d(x, tc["patch_w"], tc["patch_b"], stride=cfg.patch_size)
            x = x.flatten(2).transpose(1, 2)
            cls_tok = tc["cls"].expand(1, -1, -1)
            x = torch.cat([cls_tok, x], dim=1)

            pos = tc["pos"]
            def_g = int(round(math.sqrt(pos.shape[0] - 1)))
            if ph != def_g or pw != def_g:
                p_grid = pos[1:].reshape(1, def_g, def_g, cfg.embed_dim).permute(0, 3, 1, 2)
                p_grid = F.interpolate(p_grid, size=(ph, pw), mode="bicubic", align_corners=False)
                pos_use = torch.cat([pos[:1], p_grid.permute(0, 2, 3, 1).reshape(ph * pw, cfg.embed_dim)], dim=0)
            else:
                pos_use = pos
            x = x + pos_use.unsqueeze(0)

            taps = []
            tap_set = set(cfg.tap_indices)
            n_tok = x.shape[1]
            for idx, b in enumerate(tc["blocks"]):
                h1 = F.layer_norm(x, (cfg.embed_dim,), b["n1w"], b["n1b"], cfg.layer_norm_eps)
                q = F.linear(h1, b["qw"], b["qb"]).view(1, n_tok, cfg.num_heads, cfg.head_dim).transpose(1, 2)
                k = F.linear(h1, b["kw"], b["kb"]).view(1, n_tok, cfg.num_heads, cfg.head_dim).transpose(1, 2)
                v = F.linear(h1, b["vw"], b["vb"]).view(1, n_tok, cfg.num_heads, cfg.head_dim).transpose(1, 2)
                attn = F.scaled_dot_product_attention(q, k, v)
                ctx = attn.transpose(1, 2).contiguous().view(1, n_tok, cfg.embed_dim)
                x = x + F.linear(ctx, b["pw"], b["pb"]) * b["ls1"]

                h2 = F.layer_norm(x, (cfg.embed_dim,), b["n2w"], b["n2b"], cfg.layer_norm_eps)
                mlp = F.linear(F.gelu(F.linear(h2, b["f1w"], b["f1b"]), approximate="tanh"), b["f2w"], b["f2b"])
                x = x + mlp * b["ls2"]

                if idx in tap_set:
                    taps.append(F.layer_norm(x, (cfg.embed_dim,), tc["norm_w"], tc["norm_b"], cfg.layer_norm_eps))

            if hasattr(torch, "cuda") and dev.type == "cuda":
                torch.cuda.synchronize()
            t_bb = time.perf_counter()

            reassembled = []
            for i, hidden in enumerate(taps):
                feat = hidden[:, 1:].reshape(1, ph, pw, cfg.embed_dim).permute(0, 3, 1, 2).contiguous()
                r = tc["reassemble"][i]
                feat = F.conv2d(feat, r["pw"], r["pb"])
                factor = cfg.reassemble_factors[i]
                if r["rw"] is not None:
                    if factor > 1.0:
                        feat = F.conv_transpose2d(feat, r["rw"], r["rb"], stride=int(factor))
                    else:
                        feat = F.conv2d(feat, r["rw"], r["rb"], stride=2, padding=1)
                reassembled.append(feat)

            features = [F.conv2d(feat, tc["convs"][i], padding=1) for i, feat in enumerate(reassembled)]
            rev = features[::-1]
            fused = None
            for idx, (feat, fl) in enumerate(zip(rev, tc["fusion"])):
                target_sz = rev[idx + 1].shape[2:] if idx != len(rev) - 1 else None
                if fused is None:
                    cur = feat
                else:
                    res = feat if fused.shape == feat.shape else F.interpolate(feat, size=fused.shape[2:], mode="bilinear", align_corners=False)
                    r1 = F.conv2d(F.relu(F.conv2d(F.relu(res), fl["r1c1w"], fl["r1c1b"], padding=1)), fl["r1c2w"], fl["r1c2b"], padding=1)
                    cur = fused + res + r1
                r2 = F.conv2d(F.relu(F.conv2d(F.relu(cur), fl["r2c1w"], fl["r2c1b"], padding=1)), fl["r2c2w"], fl["r2c2b"], padding=1)
                cur = cur + r2
                if target_sz is None:
                    cur = F.interpolate(cur, scale_factor=2, mode="bilinear", align_corners=True)
                else:
                    cur = F.interpolate(cur, size=target_sz, mode="bilinear", align_corners=True)
                fused = F.conv2d(cur, fl["pw"], fl["pb"])

            d = F.conv2d(fused, tc["h1w"], tc["h1b"], padding=1)
            d = F.interpolate(d, size=(ph * cfg.patch_size, pw * cfg.patch_size), mode="bilinear", align_corners=True)
            d = F.relu(F.conv2d(d, tc["h2w"], tc["h2b"], padding=1))
            d = F.relu(F.conv2d(d, tc["h3w"], tc["h3b"]))
            if hasattr(torch, "cuda") and dev.type == "cuda":
                torch.cuda.synchronize()
            depth_np = d[0, 0].cpu().numpy()

        t_end = time.perf_counter()
        return InferenceResult(
            depth=depth_np,
            total_ms=(t_end - t0) * 1000.0,
            backbone_ms=(t_bb - t0) * 1000.0,
            neck_head_ms=(t_end - t_bb) * 1000.0,
        )

    def infer(self, rgb: np.ndarray) -> InferenceResult:
        """Run full Depth Anything V2 inference on an (H, W, 3) uint8 RGB image."""
        if self.backend == "gpu":
            return self._infer_gpu(rgb)

        if self._npu_mm is not None:
            self._npu_mm.reset_stats()

        t0 = time.perf_counter()
        pixel_values = self.preprocess(rgb)
        taps, ph, pw = self._forward_backbone(pixel_values)
        t_bb = time.perf_counter()
        depth = self._forward_neck_and_head(taps, ph, pw)
        t_end = time.perf_counter()

        stats = None
        if self._npu_mm is not None:
            s = self._npu_mm.stats
            stats = NpuExecStats(dispatches=s.dispatches, npu_wall_s=s.npu_wall_s, host_copy_s=s.host_copy_s)

        return InferenceResult(
            depth=depth,
            total_ms=(t_end - t0) * 1000.0,
            backbone_ms=(t_bb - t0) * 1000.0,
            neck_head_ms=(t_end - t_bb) * 1000.0,
            npu_stats=stats,
        )

    def close(self) -> None:
        if self._npu_mm is not None:
            self._npu_mm.close()
        self._torch_ctx = None
        self._wt_cache.clear()
        self._pos_embed_cache.clear()
