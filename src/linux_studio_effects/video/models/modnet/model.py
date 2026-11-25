"""From-scratch MODNet inference engine (CPU AVX-512 + optional NPU1 bf16 conv offload)."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from .config import ModnetConfig
from . import host_ops
from .weights import ConvLayerWeights, InvertedResidualWeights, ModnetWeights


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -40.0, 40.0, dtype=np.float32)
    return (np.float32(1.0) / (np.float32(1.0) + np.exp(-x_clip))).astype(np.float32, copy=False)


@dataclass(frozen=True)
class SegmentationResult:
    """Predicted portrait alpha matte in [0, 1] and inference timing breakdown."""

    alpha: np.ndarray  # (H_net, W_net) float32 in [0, 1]
    total_ms: float
    backbone_ms: float
    decoder_ms: float


class ModnetModel:
    """Real-time MODNet portrait alpha matting inference engine."""

    def __init__(self, config: ModnetConfig, weights: ModnetWeights, *, backend: str = "cpu") -> None:
        self.config = config
        self.weights = weights
        self.backend = backend.lower().strip()
        if self.backend not in ("cpu", "npu", "gpu"):
            raise ValueError(f"Unsupported backend {backend!r}; expected 'cpu', 'npu', or 'gpu'")
        if self.backend == "gpu":
            from ...effects.base import DeviceUnavailableError

            raise DeviceUnavailableError("MODNet GPU backend requires PyTorch CUDA/XPU (not active on this host)")

        self._npu_mm: Any = None
        self._wt_cache: dict[int, np.ndarray] = {}
        if self.backend == "npu":
            from ..video_depth_anything.npu_ops import NpuMatMulExecutor

            self._npu_mm = NpuMatMulExecutor()

    def _wt(self, weight_2d: np.ndarray) -> np.ndarray:
        wid = id(weight_2d)
        wt = self._wt_cache.get(wid)
        if wt is None:
            wt = np.ascontiguousarray(weight_2d.T, dtype=np.float32)
            self._wt_cache[wid] = wt
        return wt

    def _conv(self, x: np.ndarray, cw: ConvLayerWeights) -> np.ndarray:
        if cw.has_inorm:
            # High-FLOP IBNorm dense convs (e.g., conv_lr16x 5x5 1280->96) benefit from NPU bf16 GEMM.
            if (
                self._npu_mm is not None
                and cw.weight_2d.shape[0] >= 64
                and (x.shape[1] * cw.kh * cw.kw * cw.weight_2d.shape[0] * ((x.shape[2] + 2 * cw.padding - cw.kh) // cw.stride + 1) * ((x.shape[3] + 2 * cw.padding - cw.kw) // cw.stride + 1)) >= 600_000_000
            ):
                oh = (x.shape[2] + 2 * cw.padding - cw.kh) // cw.stride + 1
                ow = (x.shape[3] + 2 * cw.padding - cw.kw) // cw.stride + 1
                cols = x.reshape(x.shape[1], oh * ow) if (cw.kh == 1 and cw.stride == 1 and cw.padding == 0) else host_ops.im2col_kxk_p(x, kh=cw.kh, kw=cw.kw, stride=cw.stride, padding=cw.padding)
                y = self._npu_mm.matmul(
                    cols,
                    self._wt(cw.weight_2d),
                    bias=cw.bias,
                    cache_b=True,
                    transpose_a=True,
                    out_nchw_hw=(oh, ow),
                )
            else:
                y = host_ops.conv2d_nchw(
                    x,
                    cw.weight_2d,
                    cw.bias,
                    kh=cw.kh,
                    kw=cw.kw,
                    stride=cw.stride,
                    padding=cw.padding,
                    relu_mode=None,
                )
            return host_ops.ibnorm_tail_relu(
                y,
                eps=self.config.norm_eps,
                apply_relu=(cw.relu_mode == "relu"),
            )

        # Offload only large-channel dense 1x1 convolutions on NPU (where C_out >= 128 and M >= 128).
        if (
            self._npu_mm is not None
            and cw.kh == 1
            and cw.kw == 1
            and cw.stride == 1
            and cw.padding == 0
            and cw.weight_2d.shape[0] >= 128
            and (x.shape[2] * x.shape[3]) >= 128
            and (x.shape[1] * x.shape[2] * x.shape[3] * cw.weight_2d.shape[0]) >= 120_000_000
        ):
            _, c_in, h, w = x.shape
            y = self._npu_mm.matmul(
                x.reshape(c_in, h * w),
                self._wt(cw.weight_2d),
                bias=cw.bias,
                cache_b=True,
                transpose_a=True,
                out_nchw_hw=(h, w),
            )
            if cw.relu_mode == "relu":
                return np.maximum(y, np.float32(0.0))
            if cw.relu_mode == "relu6":
                return np.clip(y, np.float32(0.0), np.float32(6.0))
            return y

        return host_ops.conv2d_nchw(
            x,
            cw.weight_2d,
            cw.bias,
            kh=cw.kh,
            kw=cw.kw,
            stride=cw.stride,
            padding=cw.padding,
            relu_mode=cw.relu_mode,
        )

    def _inv_res(self, x: np.ndarray, blk: InvertedResidualWeights) -> np.ndarray:
        h = self._conv(x, blk.pw) if blk.pw is not None else x
        h = host_ops.dwconv_3x3_p1(
            h,
            blk.dw.weight_2d,
            blk.dw.bias,
            stride=blk.dw.stride,
            relu6=(blk.dw.relu_mode == "relu6"),
        )
        h = self._conv(h, blk.pw_linear)
        return (x + h) if blk.use_res_connect else h

    def preprocess(self, rgb: np.ndarray) -> np.ndarray:
        """Resize uint8 RGB (H, W, 3) to aspect-preserving multiple-of-32 and normalize to [-1, 1]."""
        h, w, _ = rgb.shape
        th, tw = self.config.resolve_input_hw(h, w)
        x = ((rgb.astype(np.float32) * np.float32(1.0 / 127.5)) - np.float32(1.0)).transpose(2, 0, 1)[None, :, :, :]
        if h != th or w != tw:
            x = host_ops.bilinear_resize_align_false(x, th, tw)
        return np.ascontiguousarray(x, dtype=np.float32)

    def infer(self, rgb: np.ndarray) -> SegmentationResult:
        """Run MODNet inference and return the soft alpha matte in [0, 1]."""
        t0 = time.perf_counter()
        img = self.preprocess(rgb)
        _, _, th, tw = img.shape
        w = self.weights

        # MobileNetV2 Backbone
        x = self._conv(img, w.conv0)
        x = self._inv_res(x, w.blocks[0])
        enc2x = x

        x = self._inv_res(x, w.blocks[1])
        x = self._inv_res(x, w.blocks[2])
        enc4x = x

        for blk in w.blocks[3:]:
            x = self._inv_res(x, blk)
        enc32x = self._conv(x, w.conv_last)
        t_bb = time.perf_counter()

        # LRBranch (semantic branch, inference=True skips conv_lr)
        se = np.mean(enc32x, axis=(2, 3), keepdims=False)  # (1, 1280)
        se_w = _sigmoid(np.maximum(se @ w.se_fc1_t, np.float32(0.0)) @ w.se_fc2_t).reshape(1, -1, 1, 1)
        enc32x = enc32x * se_w

        lr16x = host_ops.bilinear_resize_align_false(enc32x, th // 16, tw // 16)
        lr16x = self._conv(lr16x, w.conv_lr16x)
        lr8x = host_ops.bilinear_resize_align_false(lr16x, th // 8, tw // 8)
        lr8x = self._conv(lr8x, w.conv_lr8x)

        # HRBranch (detail branch, inference=True skips conv_hr)
        img2x = host_ops.bilinear_resize_align_false(img, th // 2, tw // 2)
        img4x = host_ops.bilinear_resize_align_false(img, th // 4, tw // 4)

        enc2x_hr = self._conv(enc2x, w.tohr_enc2x)
        hr4x = self._conv(np.concatenate((img2x, enc2x_hr), axis=1), w.conv_enc2x)
        enc4x_hr = self._conv(enc4x, w.tohr_enc4x)
        hr4x = self._conv(np.concatenate((hr4x, enc4x_hr), axis=1), w.conv_enc4x)

        lr4x_hr = host_ops.bilinear_resize_align_false(lr8x, th // 4, tw // 4)
        hr4x = np.concatenate((hr4x, lr4x_hr, img4x), axis=1)
        for layer in w.conv_hr4x:
            hr4x = self._conv(hr4x, layer)

        hr2x = host_ops.bilinear_resize_align_false(hr4x, th // 2, tw // 2)
        hr2x = np.concatenate((hr2x, enc2x_hr), axis=1)
        for layer in w.conv_hr2x:
            hr2x = self._conv(hr2x, layer)

        # FusionBranch
        lr4x = host_ops.bilinear_resize_align_false(lr8x, th // 4, tw // 4)
        lr4x = self._conv(lr4x, w.conv_lr4x)
        lr2x = host_ops.bilinear_resize_align_false(lr4x, th // 2, tw // 2)

        f2x = self._conv(np.concatenate((lr2x, hr2x), axis=1), w.conv_f2x)
        f = host_ops.bilinear_resize_align_false(f2x, th, tw)
        f = self._conv(np.concatenate((f, img), axis=1), w.conv_f0)
        f = self._conv(f, w.conv_f1)
        alpha = _sigmoid(f[0, 0])

        t_end = time.perf_counter()
        return SegmentationResult(
            alpha=np.ascontiguousarray(alpha, dtype=np.float32),
            total_ms=(t_end - t0) * 1000.0,
            backbone_ms=(t_bb - t0) * 1000.0,
            decoder_ms=(t_end - t_bb) * 1000.0,
        )

    def close(self) -> None:
        if self._npu_mm is not None:
            self._npu_mm.close()
            self._npu_mm = None
        self._wt_cache.clear()
