"""From-scratch streaming Video Depth Anything (Small / Base / Large) across CPU, NPU1, and PyTorch GPU."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import host_ops
from .config import VideoDepthConfig
from .npu_ops import NpuExecStats, NpuMatMulExecutor
from .weights import MotionModuleWeights, TemporalAttentionWeights, VideoDepthWeights

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

# Upstream VideoDepthAnything streaming window schedule
_INFER_LEN = 32
_OVERLAP = 10
_INTERP_LEN = 8
_GAP = (_INFER_LEN - _OVERLAP) * 2 - 1 - (_OVERLAP - _INTERP_LEN)  # 41

# Crossover thresholds measured on NPU1 (4x4 bf16->f32 with N_align=128) vs AVX-512 OpenBLAS:
# below these sizes, NPU dispatch + N=128 alignment padding outweighs compute savings.
_NPU_LINEAR_MIN_MACS = 120_000_000
_NPU_CONV_MIN_MACS = 250_000_000


@dataclass(frozen=True)
class InferenceResult:
    """Predicted video depth map and per-stage timing breakdown."""

    depth: np.ndarray  # (H_in, W_in) float32
    total_ms: float
    backbone_ms: float
    neck_head_ms: float
    frame_index: int = 0
    npu_stats: NpuExecStats | None = None


@dataclass
class _TemporalAttnCtx:
    w_q_t: np.ndarray  # (C, C)
    w_kv_t: np.ndarray  # (C, 2*C) fused K/V linear weight (no bias)
    w_out_t: np.ndarray  # (C, C)
    b_out: np.ndarray  # (C,)
    pe: np.ndarray  # (T_max, C)
    k_pe: np.ndarray  # (T_max, C) precomputed pe @ W_k^T
    v_pe: np.ndarray  # (T_max, C) precomputed pe @ W_v^T


@dataclass
class _MotionCtx:
    norm_weight: np.ndarray
    norm_bias: np.ndarray
    proj_in_wt: np.ndarray
    proj_in_b: np.ndarray
    attn0_nw: np.ndarray
    attn0_nb: np.ndarray
    attn0: _TemporalAttnCtx
    attn1_nw: np.ndarray
    attn1_nb: np.ndarray
    attn1: _TemporalAttnCtx
    ff_nw: np.ndarray
    ff_nb: np.ndarray
    ff_proj_wt: np.ndarray
    ff_proj_b: np.ndarray
    ff_out_wt: np.ndarray
    ff_out_b: np.ndarray
    proj_out_wt: np.ndarray
    proj_out_b: np.ndarray


def _bilinear_resize_nchw(x: np.ndarray, out_h: int, out_w: int, *, align_corners: bool = True) -> np.ndarray:
    """AVX-512 / two-pass separable bilinear interpolation matching torch.nn.functional.interpolate."""
    return host_ops.bilinear_resize_nchw(x, out_h, out_w, align_corners=align_corners)


def _bicubic_keys_weights(t: np.ndarray, a: float = -0.75) -> np.ndarray:
    abs_t = np.abs(t)
    w = np.zeros_like(abs_t, dtype=np.float32)
    m1 = abs_t <= 1.0
    m2 = (abs_t > 1.0) & (abs_t < 2.0)
    t1 = abs_t[m1]
    t2 = abs_t[m2]
    w[m1] = (a + 2.0) * (t1**3) - (a + 3.0) * (t1**2) + 1.0
    w[m2] = a * (t2**3) - 5.0 * a * (t2**2) + 8.0 * a * t2 - 4.0 * a
    return w


def _bicubic_resize_hwc(
    arr: np.ndarray,
    out_h: int,
    out_w: int,
    *,
    offset: float = 0.1,
) -> np.ndarray:
    """Bicubic resize for DINOv2 positional embeddings matching interpolate_offset."""
    in_h, in_w, c = arr.shape
    if in_h == out_h and in_w == out_w:
        return arr

    sx = (out_w + offset) / in_w
    sy = (out_h + offset) / in_h

    gx = (np.arange(out_w, dtype=np.float32) + 0.5) / sx - 0.5
    ix = np.floor(gx).astype(np.int64)
    fx = gx - ix
    tmp = np.zeros((in_h, out_w, c), dtype=np.float32)
    for k in range(-1, 3):
        idx = np.clip(ix + k, 0, in_w - 1)
        tmp += arr[:, idx, :] * _bicubic_keys_weights(fx - k)[None, :, None]

    gy = (np.arange(out_h, dtype=np.float32) + 0.5) / sy - 0.5
    iy = np.floor(gy).astype(np.int64)
    fy = gy - iy
    out = np.zeros((out_h, out_w, c), dtype=np.float32)
    for k in range(-1, 3):
        idx = np.clip(iy + k, 0, in_h - 1)
        out += tmp[idx, :, :] * _bicubic_keys_weights(fy - k)[:, None, None]
    return out


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    return host_ops.layer_norm(x, weight, bias, eps=eps)


def _group_norm_nchw(
    x: np.ndarray,
    num_groups: int,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    return host_ops.group_norm_nchw(x, num_groups, weight, bias, eps=eps)


def _gelu(x: np.ndarray) -> np.ndarray:
    out = np.array(x, dtype=np.float32, copy=True)
    return host_ops.gelu_inplace(out)


def _softmax(x: np.ndarray) -> np.ndarray:
    return host_ops.softmax_rows(x)


def _conv2d_nchw(
    x: np.ndarray,
    weight_2d_or_4d: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    c_out: int | None = None,
    kh: int = 1,
    kw: int = 1,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """Im2col + OpenBLAS Conv2d for N=1 NCHW float32 tensors."""
    _, c_in, h, w = x.shape
    if weight_2d_or_4d.ndim == 4:
        c_out, _, kh, kw = weight_2d_or_4d.shape
        w_mat = weight_2d_or_4d.reshape(c_out, c_in * kh * kw)
    else:
        assert c_out is not None
        w_mat = weight_2d_or_4d

    if kh == 1 and kw == 1 and stride == 1 and padding == 0:
        out = (w_mat @ x.reshape(c_in, h * w)).reshape(1, c_out, h, w)
        if bias is not None:
            out = out + bias.reshape(1, c_out, 1, 1)
        return out

    if kh == 3 and kw == 3 and stride == 1 and padding == 1:
        return host_ops.conv2d_3x3_s1_p1(x, w_mat, bias)

    x_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant") if padding > 0 else x
    oh = (h + 2 * padding - kh) // stride + 1
    ow = (w + 2 * padding - kw) // stride + 1

    _, s_c, s_h, s_w = x_pad.strides
    patches = np.ascontiguousarray(
        np.lib.stride_tricks.as_strided(
            x_pad,
            shape=(c_in, kh, kw, oh, ow),
            strides=(s_c, s_h, s_w, s_h * stride, s_w * stride),
        ).reshape(c_in * kh * kw, oh * ow)
    )
    out = (w_mat @ patches).reshape(1, c_out, oh, ow)
    if bias is not None:
        out = out + bias.reshape(1, c_out, 1, 1)
    return out


def _conv_transpose2d_nchw(
    x: np.ndarray,
    weight_2d: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    c_out: int,
    stride: int,
) -> np.ndarray:
    """ConvTranspose2d for stride == kernel_size (4x4 stride 4, 2x2 stride 2) via col2im."""
    _, c_in, h, w = x.shape
    cols = (weight_2d @ x.reshape(c_in, h * w)).reshape(c_out, stride, stride, h, w)
    out = cols.transpose(0, 3, 1, 4, 2).reshape(1, c_out, h * stride, w * stride)
    if bias is not None:
        out = out + bias.reshape(1, c_out, 1, 1)
    return np.ascontiguousarray(out, dtype=np.float32)


def _build_temporal_attn_ctx(w: TemporalAttentionWeights) -> _TemporalAttnCtx:
    pe = np.ascontiguousarray(w.pe[0], dtype=np.float32)
    wt_k = np.ascontiguousarray(w.to_k_weight.T)
    wt_v = np.ascontiguousarray(w.to_v_weight.T)
    return _TemporalAttnCtx(
        w_q_t=np.ascontiguousarray(w.to_q_weight.T),
        w_kv_t=np.ascontiguousarray(np.concatenate([w.to_k_weight, w.to_v_weight], axis=0).T),
        w_out_t=np.ascontiguousarray(w.to_out_weight.T),
        b_out=w.to_out_bias,
        pe=pe,
        k_pe=np.ascontiguousarray(pe @ wt_k, dtype=np.float32),
        v_pe=np.ascontiguousarray(pe @ wt_v, dtype=np.float32),
    )


def _build_motion_ctx(m: MotionModuleWeights) -> _MotionCtx:
    return _MotionCtx(
        norm_weight=m.norm_weight,
        norm_bias=m.norm_bias,
        proj_in_wt=np.ascontiguousarray(m.proj_in_weight.T),
        proj_in_b=m.proj_in_bias,
        attn0_nw=m.attn0_norm_weight,
        attn0_nb=m.attn0_norm_bias,
        attn0=_build_temporal_attn_ctx(m.attn0),
        attn1_nw=m.attn1_norm_weight,
        attn1_nb=m.attn1_norm_bias,
        attn1=_build_temporal_attn_ctx(m.attn1),
        ff_nw=m.ff_norm_weight,
        ff_nb=m.ff_norm_bias,
        ff_proj_wt=np.ascontiguousarray(m.ff_proj_weight.T),
        ff_proj_b=m.ff_proj_bias,
        ff_out_wt=np.ascontiguousarray(m.ff_out_weight.T),
        ff_out_b=m.ff_out_bias,
        proj_out_wt=np.ascontiguousarray(m.proj_out_weight.T),
        proj_out_b=m.proj_out_bias,
    )


class VideoDepthAnythingModel:
    """Streaming Video Depth Anything inference across CPU, NPU1, and optional PyTorch GPU."""

    def __init__(
        self,
        config: VideoDepthConfig,
        weights: VideoDepthWeights,
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
        self._conv_wt_cache: dict[int, np.ndarray] = {}
        self._conv_kykxc_wt_cache: dict[int, np.ndarray] = {}
        self._deconv_wt_cache: dict[int, np.ndarray] = {}
        self._motion_ctxs: tuple[_MotionCtx, ...] = tuple(_build_motion_ctx(m) for m in weights.motion_modules)
        self._torch_ctx: dict[str, Any] | None = None

        # Streaming temporal KV-factorized history state (matches infer_video_depth_one)
        self._frame_id: int = -1
        self._frame_id_list: list[int] = []
        self._frame_cache_list: list[tuple[tuple[np.ndarray, np.ndarray], ...]] = []
        # Preallocated rolling (d, 32, ch) K/V buffers per temporal attention slot
        self._kv_work: list[tuple[np.ndarray, np.ndarray]] = []
        self._slot1_id: int | None = None

        if self.backend == "npu":
            self._npu_mm = NpuMatMulExecutor()
        elif self.backend == "gpu":
            self._init_torch_gpu()

    def reset_temporal_state(self) -> None:
        """Clear streaming frame history while preserving compiled NPU kernels and weights."""
        self._frame_id = -1
        self._frame_id_list.clear()
        self._frame_cache_list.clear()
        self._kv_work.clear()
        self._slot1_id = None

    def _init_torch_gpu(self) -> None:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            dev = torch.device("xpu")
        else:
            from ...effects.base import DeviceUnavailableError

            raise DeviceUnavailableError("PyTorch has neither CUDA/ROCm nor XPU GPU device active on this host.")
        self._torch_ctx = {"device": dev}

    def _get_pos_embed(self, ph: int, pw: int) -> np.ndarray:
        key = (ph, pw)
        cached = self._pos_embed_cache.get(key)
        if cached is not None:
            return cached

        pos = self.weights.pos_embed
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

    def _wt(self, weight_out_in: np.ndarray) -> np.ndarray:
        wid = id(weight_out_in)
        wt = self._wt_cache.get(wid)
        if wt is None:
            wt = np.ascontiguousarray(weight_out_in.T)
            self._wt_cache[wid] = wt
        return wt

    def _conv_wt(self, w4: np.ndarray) -> np.ndarray:
        wid = id(w4)
        w2 = self._conv_wt_cache.get(wid)
        if w2 is None:
            c_out = w4.shape[0]
            w2 = np.ascontiguousarray(w4.reshape(c_out, -1))
            self._conv_wt_cache[wid] = w2
        return w2

    def _conv_kykxc_wt(self, w4: np.ndarray) -> np.ndarray:
        wid = id(w4)
        w2 = self._conv_kykxc_wt_cache.get(wid)
        if w2 is None:
            c_out, c_in, kh, kw = w4.shape
            w2 = np.ascontiguousarray(w4.transpose(2, 3, 1, 0).reshape(kh * kw * c_in, c_out))
            self._conv_kykxc_wt_cache[wid] = w2
        return w2

    def _deconv_wt(self, w4: np.ndarray) -> np.ndarray:
        wid = id(w4)
        w2 = self._deconv_wt_cache.get(wid)
        if w2 is None:
            c_in, c_out, kh, kw = w4.shape
            w2 = np.ascontiguousarray(w4.reshape(c_in, c_out * kh * kw).T)
            self._deconv_wt_cache[wid] = w2
        return w2

    def _proj(
        self,
        x: np.ndarray,
        wt: np.ndarray,
        bias: np.ndarray | None = None,
        *,
        activation: str | None = None,
    ) -> np.ndarray:
        """Linear projection taking pre-transposed weight (K, N), routing heavy layers to NPU."""
        m, k = x.shape
        _, n = wt.shape
        if self._npu_mm is not None and n >= 128 and m * k * n >= _NPU_LINEAR_MIN_MACS:
            return self._npu_mm.matmul(x, wt, bias=bias, activation=activation, cache_b=True)
        y = x @ wt
        if bias is not None:
            y = y + bias
        if activation == "gelu":
            return host_ops.gelu_inplace(y)
        return y

    def _conv2d(
        self,
        x: np.ndarray,
        w4: np.ndarray,
        bias: np.ndarray | None = None,
        *,
        stride: int = 1,
        padding: int = 0,
        apply_relu: bool = False,
    ) -> np.ndarray:
        """Conv2d wrapper routing high-FLOP 3x3 s=1 p=1 and 1x1 convolutions (C_out >= 128) to NPU."""
        _, c_in, h, w = x.shape
        c_out, _, kh, kw = w4.shape
        w_mat = self._conv_wt(w4)

        if self._npu_mm is not None and stride == 1 and c_out >= 128:
            macs = c_out * c_in * kh * kw * h * w
            if kh == 3 and kw == 3 and padding == 1 and macs >= _NPU_CONV_MIN_MACS:
                return self._npu_mm.conv2d_3x3_s1_p1_nchw(
                    x,
                    self._conv_kykxc_wt(w4),
                    self._wt(w_mat),
                    bias=bias,
                    apply_relu=apply_relu,
                )
            if kh == 1 and kw == 1 and padding == 0 and macs >= _NPU_CONV_MIN_MACS:
                return self._npu_mm.matmul(
                    x.reshape(c_in, h * w),
                    self._wt(w_mat),
                    bias=bias,
                    activation="relu" if apply_relu else None,
                    cache_b=True,
                    transpose_a=True,
                    out_nchw_hw=(h, w),
                )

        out = _conv2d_nchw(x, w_mat, bias, c_out=c_out, kh=kh, kw=kw, stride=stride, padding=padding)
        if apply_relu:
            return np.maximum(out, np.float32(0.0))
        return out

    def _linear(
        self,
        x: np.ndarray,
        weight_out_in: np.ndarray,
        bias: np.ndarray,
        *,
        activation: str | None = None,
    ) -> np.ndarray:
        wt = self._wt(weight_out_in)
        if self._npu_mm is not None:
            return self._npu_mm.matmul(x, wt, bias=bias, activation=activation, cache_b=True)
        y = x @ wt + bias
        if activation == "gelu":
            return host_ops.gelu_inplace(y)
        return y

    def preprocess(self, rgb: np.ndarray) -> np.ndarray:
        """Resize (H, W, 3) uint8 RGB frame to (1, 3, S, S) normalized float32."""
        s = self.config.input_size
        h, w, _ = rgb.shape
        x = (rgb.astype(np.float32) * (1.0 / 255.0)).transpose(2, 0, 1)[None, :, :, :]
        if h != s or w != s:
            x = _bilinear_resize_nchw(x, s, s, align_corners=False)
        return (x - _IMAGENET_MEAN) / _IMAGENET_STD

    def _forward_backbone(self, pixel_values: np.ndarray) -> tuple[list[np.ndarray], int, int]:
        cfg = self.config
        w = self.weights
        p = cfg.patch_size
        _, _, ih, iw = pixel_values.shape
        ph, pw = ih // p, iw // p

        patches = _conv2d_nchw(
            pixel_values,
            self._conv_wt(w.patch_weight),
            w.patch_bias,
            c_out=cfg.embed_dim,
            kh=p,
            kw=p,
            stride=p,
            padding=0,
        )
        patch_tokens = patches[0].reshape(cfg.embed_dim, ph * pw).T
        x = np.concatenate([w.cls_token, patch_tokens], axis=0)
        x = x + self._get_pos_embed(ph, pw)

        taps: list[np.ndarray] = []
        tap_set = set(cfg.tap_indices)
        scale = np.float32(1.0 / math.sqrt(cfg.head_dim))
        n_tok = x.shape[0]

        for idx, blk in enumerate(w.blocks):
            h_norm = _layer_norm(x, blk.norm1_weight, blk.norm1_bias, cfg.layer_norm_eps)

            # Fused QKV projection: 1 dispatch per block (weights already packed in .pth)
            qkv = self._linear(h_norm, blk.qkv_weight, blk.qkv_bias)
            q, k, v = np.split(qkv, 3, axis=-1)

            qh = q.reshape(n_tok, cfg.num_heads, cfg.head_dim).transpose(1, 0, 2)
            kh = k.reshape(n_tok, cfg.num_heads, cfg.head_dim).transpose(1, 2, 0)
            vh = v.reshape(n_tok, cfg.num_heads, cfg.head_dim).transpose(1, 0, 2)

            attn = _softmax((qh @ kh) * scale)
            ctx = (attn @ vh).transpose(1, 0, 2).reshape(n_tok, cfg.embed_dim)
            attn_out = self._linear(ctx, blk.proj_weight, blk.proj_bias)
            x = x + attn_out * blk.ls1

            h_norm2 = _layer_norm(x, blk.norm2_weight, blk.norm2_bias, cfg.layer_norm_eps)
            mlp_hid = self._linear(h_norm2, blk.fc1_weight, blk.fc1_bias, activation="gelu")
            mlp_out = self._linear(mlp_hid, blk.fc2_weight, blk.fc2_bias)
            x = x + mlp_out * blk.ls2

            if idx in tap_set:
                tapped = _layer_norm(x, w.backbone_norm_weight, w.backbone_norm_bias, cfg.layer_norm_eps)
                taps.append(tapped[1:, :])  # drop CLS token (use_clstoken=False)

        return taps, ph, pw

    def _run_temporal_attn(
        self,
        slot: int,
        x_dc: np.ndarray,
        actx: _TemporalAttnCtx,
        has_history: bool,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Algebraically exact KV-factorized streaming temporal self-attention with rolling buffers."""
        d, ch = x_dc.shape
        n_heads = self.config.temporal_heads
        dh = ch // n_heads
        scale = float(1.0 / math.sqrt(dh))

        kv_cur = self._proj(x_dc, actx.w_kv_t)
        k_cur_raw = np.ascontiguousarray(kv_cur[:, :ch])
        v_cur_raw = np.ascontiguousarray(kv_cur[:, ch:])

        while len(self._kv_work) <= slot:
            self._kv_work.append(
                (
                    np.empty((d, _INFER_LEN, ch), dtype=np.float32),
                    np.empty((d, _INFER_LEN, ch), dtype=np.float32),
                )
            )
        k_buf, v_buf = self._kv_work[slot]
        if k_buf.shape[0] != d or k_buf.shape[2] != ch:
            k_buf = np.empty((d, _INFER_LEN, ch), dtype=np.float32)
            v_buf = np.empty((d, _INFER_LEN, ch), dtype=np.float32)
            self._kv_work[slot] = (k_buf, v_buf)

        if not has_history:
            attn_out = host_ops.temporal_attn_roll(
                k_cur_raw,
                k_cur_raw,
                v_cur_raw,
                k_buf,
                v_buf,
                actx.k_pe,
                actx.v_pe,
                n_heads,
                scale,
                False,
            )
        else:
            q = np.ascontiguousarray(self._proj(x_dc + actx.pe[_INFER_LEN - 1], actx.w_q_t))
            attn_out = host_ops.temporal_attn_roll(
                q,
                k_cur_raw,
                v_cur_raw,
                k_buf,
                v_buf,
                actx.k_pe,
                actx.v_pe,
                n_heads,
                scale,
                True,
            )

        out = self._proj(attn_out, actx.w_out_t, actx.b_out)
        return out, (k_cur_raw, v_cur_raw)

    def _run_motion_module(
        self,
        feat_nchw: np.ndarray,
        mctx: _MotionCtx,
        slot_offset: int,
        has_history: bool,
    ) -> tuple[np.ndarray, tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]]:
        _, ch, h, w = feat_nchw.shape
        d = h * w
        eps = self.config.temporal_norm_eps

        residual = feat_nchw
        x = _group_norm_nchw(
            feat_nchw,
            self.config.temporal_norm_groups,
            mctx.norm_weight,
            mctx.norm_bias,
            eps=self.config.group_norm_eps,
        )
        tokens = np.ascontiguousarray(x[0].transpose(1, 2, 0).reshape(d, ch))
        hs = self._proj(tokens, mctx.proj_in_wt, mctx.proj_in_b)

        out0, kv0 = self._run_temporal_attn(slot_offset, _layer_norm(hs, mctx.attn0_nw, mctx.attn0_nb, eps), mctx.attn0, has_history)
        hs = hs + out0

        out1, kv1 = self._run_temporal_attn(slot_offset + 1, _layer_norm(hs, mctx.attn1_nw, mctx.attn1_nb, eps), mctx.attn1, has_history)
        hs = hs + out1

        ff_in = _layer_norm(hs, mctx.ff_nw, mctx.ff_nb, eps)
        ff_proj = self._proj(ff_in, mctx.ff_proj_wt, mctx.ff_proj_b)
        ff_val, ff_gate = np.split(ff_proj, 2, axis=-1)
        ff_in_out = np.ascontiguousarray(ff_val * _gelu(ff_gate))
        ff_out = self._proj(ff_in_out, mctx.ff_out_wt, mctx.ff_out_b)
        hs = hs + ff_out

        out_tokens = self._proj(hs, mctx.proj_out_wt, mctx.proj_out_b)
        out_nchw = out_tokens.reshape(1, h, w, ch).transpose(0, 3, 1, 2)
        return np.ascontiguousarray(out_nchw + residual, dtype=np.float32), (kv0, kv1)

    def _residual_unit(self, x: np.ndarray, rcw: Any) -> np.ndarray:
        h = np.maximum(x, np.float32(0.0))
        h = self._conv2d(h, rcw.conv1_weight, rcw.conv1_bias, stride=1, padding=1, apply_relu=True)
        h = self._conv2d(h, rcw.conv2_weight, rcw.conv2_bias, stride=1, padding=1)
        return x + h

    def _forward_temporal_dpt(
        self,
        taps: list[np.ndarray],
        ph: int,
        pw: int,
    ) -> tuple[np.ndarray, tuple[tuple[np.ndarray, np.ndarray], ...]]:
        cfg = self.config
        w = self.weights

        has_history = bool(self._frame_cache_list)
        if has_history:
            slot1 = self._frame_cache_list[1]
            if id(slot1) != self._slot1_id and len(self._kv_work) >= 8:
                for slot in range(8):
                    k_buf, v_buf = self._kv_work[slot]
                    k_buf[:, 1, :] = slot1[slot][0]
                    v_buf[:, 1, :] = slot1[slot][1]
                self._slot1_id = id(slot1)

        # 1. Projects + Resize
        reassembled: list[np.ndarray] = []
        for i, hidden in enumerate(taps):
            feat = np.ascontiguousarray(hidden.reshape(ph, pw, cfg.embed_dim).transpose(2, 0, 1)[None, :, :, :])
            rw = w.reassemble[i]
            c_mid = rw.proj_weight.shape[0]
            feat = _conv2d_nchw(feat, self._conv_wt(rw.proj_weight), rw.proj_bias, c_out=c_mid)
            factor = cfg.reassemble_factors[i]
            if rw.resize_weight is not None:
                if factor > 1.0:
                    feat = _conv_transpose2d_nchw(
                        feat,
                        self._deconv_wt(rw.resize_weight),
                        rw.resize_bias,
                        c_out=c_mid,
                        stride=int(factor),
                    )
                else:
                    feat = _conv2d_nchw(
                        feat,
                        self._conv_wt(rw.resize_weight),
                        rw.resize_bias,
                        c_out=c_mid,
                        kh=3,
                        kw=3,
                        stride=2,
                        padding=1,
                    )
            reassembled.append(feat)

        layer_1, layer_2, layer_3, layer_4 = reassembled

        # 2. Temporal modules 0 and 1 on layer_3 and layer_4
        layer_3, kv_m0 = self._run_motion_module(layer_3, self._motion_ctxs[0], 0, has_history)
        layer_4, kv_m1 = self._run_motion_module(layer_4, self._motion_ctxs[1], 2, has_history)

        # 3. Scratch 3x3 projections (no bias)
        layer_1_rn = self._conv2d(layer_1, w.layer_rn_weights[0], None, padding=1)
        layer_2_rn = self._conv2d(layer_2, w.layer_rn_weights[1], None, padding=1)
        layer_3_rn = self._conv2d(layer_3, w.layer_rn_weights[2], None, padding=1)
        layer_4_rn = self._conv2d(layer_4, w.layer_rn_weights[3], None, padding=1)

        # 4. RefineNet4 + MotionModule 2
        rf4 = w.refinenets[3]
        path_4 = self._residual_unit(layer_4_rn, rf4.res2)
        path_4 = _bilinear_resize_nchw(path_4, layer_3_rn.shape[2], layer_3_rn.shape[3], align_corners=True)
        path_4 = self._conv2d(path_4, rf4.out_conv_weight, rf4.out_conv_bias)
        path_4, kv_m2 = self._run_motion_module(path_4, self._motion_ctxs[2], 4, has_history)

        # 5. RefineNet3 + MotionModule 3
        rf3 = w.refinenets[2]
        path_3 = self._residual_unit(path_4 + self._residual_unit(layer_3_rn, rf3.res1), rf3.res2)
        path_3 = _bilinear_resize_nchw(path_3, layer_2_rn.shape[2], layer_2_rn.shape[3], align_corners=True)
        path_3 = self._conv2d(path_3, rf3.out_conv_weight, rf3.out_conv_bias)
        path_3, kv_m3 = self._run_motion_module(path_3, self._motion_ctxs[3], 6, has_history)

        # 6. RefineNet2 & RefineNet1
        rf2 = w.refinenets[1]
        path_2 = self._residual_unit(path_3 + self._residual_unit(layer_2_rn, rf2.res1), rf2.res2)
        path_2 = _bilinear_resize_nchw(path_2, layer_1_rn.shape[2], layer_1_rn.shape[3], align_corners=True)
        path_2 = self._conv2d(path_2, rf2.out_conv_weight, rf2.out_conv_bias)

        rf1 = w.refinenets[0]
        path_1 = self._residual_unit(path_2 + self._residual_unit(layer_1_rn, rf1.res1), rf1.res2)
        path_1 = _bilinear_resize_nchw(path_1, path_1.shape[2] * 2, path_1.shape[3] * 2, align_corners=True)
        path_1 = self._conv2d(path_1, rf1.out_conv_weight, rf1.out_conv_bias)

        # 7. Output head
        out = self._conv2d(path_1, w.output_conv1_weight, w.output_conv1_bias, padding=1)
        out = _bilinear_resize_nchw(out, ph * cfg.patch_size, pw * cfg.patch_size, align_corners=True)
        out = self._conv2d(out, w.output_conv2_0_weight, w.output_conv2_0_bias, padding=1, apply_relu=True)
        out = np.maximum(
            _conv2d_nchw(out, self._conv_wt(w.output_conv2_2_weight), w.output_conv2_2_bias, c_out=1),
            0.0,
        )

        new_cache = kv_m0 + kv_m1 + kv_m2 + kv_m3
        return out[0, 0], new_cache

    def infer(self, rgb: np.ndarray) -> InferenceResult:
        """Run one streaming frame of Video Depth Anything and update temporal KV history."""
        if self.backend == "gpu":
            raise NotImplementedError("PyTorch GPU path requires PyTorch CUDA/XPU (not active on this host)")

        if self._npu_mm is not None:
            self._npu_mm.reset_stats()

        t0 = time.perf_counter()
        pixel_values = self.preprocess(rgb)
        taps, ph, pw = self._forward_backbone(pixel_values)
        t_bb = time.perf_counter()
        depth, new_cache = self._forward_temporal_dpt(taps, ph, pw)

        # Update streaming window cache matching upstream video_depth_stream.py
        self._frame_id += 1
        if self._frame_id == 0:
            self._frame_cache_list = [new_cache] * _INFER_LEN
            self._frame_id_list = [0] * (_INFER_LEN - 1) + [0]
        else:
            self._frame_cache_list.append(new_cache)
            self._frame_id_list.append(self._frame_id)
            if self._frame_id + _INFER_LEN > _GAP + 1:
                del self._frame_id_list[1]
                del self._frame_cache_list[1]

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
            frame_index=self._frame_id,
            npu_stats=stats,
        )

    def close(self) -> None:
        if self._npu_mm is not None:
            self._npu_mm.close()
        self.reset_temporal_state()
        self._torch_ctx = None
        self._wt_cache.clear()
        self._conv_wt_cache.clear()
        self._conv_kykxc_wt_cache.clear()
        self._deconv_wt_cache.clear()
        self._pos_embed_cache.clear()
