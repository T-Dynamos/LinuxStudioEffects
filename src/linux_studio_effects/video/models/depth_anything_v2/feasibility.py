"""Analytical FLOP, memory, and hardware feasibility calculator for Depth Anything V2."""

from __future__ import annotations

from dataclasses import dataclass

from .config import DepthAnythingConfig, get_depth_config


def _align_up(val: int, mult: int) -> int:
    return ((val + mult - 1) // mult) * mult


@dataclass(frozen=True)
class FeasibilityReport:
    """Resource, compute, and NPU1/GPU/CPU feasibility metrics for one config."""

    variant: str
    input_size: int
    tokens: int
    tokens_padded_npu: int
    total_params: int
    backbone_gflops: float
    dpt_gflops: float
    total_gflops: float
    weights_fp32_mb: float
    weights_bf16_mb: float
    weights_int8_mb: float
    peak_activation_bf16_mb: float
    npu_matmul_dispatches: int
    npu1_sram_tile_fit: bool
    npu1_estimated_ms: float
    npu1_estimated_fps: float
    gpu_780m_estimated_ms: float
    gpu_780m_estimated_fps: float
    cpu_7840u_estimated_ms: float
    cpu_7840u_estimated_fps: float
    realtime_viability: str


def _count_params_and_flops(cfg: DepthAnythingConfig) -> tuple[int, float, float]:
    d = cfg.embed_dim
    mlp = cfg.mlp_dim
    n_tok = cfg.num_tokens
    g = cfg.patch_grid
    p = cfg.patch_size

    # 1. Embeddings: patch conv (3 -> d, 14x14) + cls + pos_embed(518->1370 tokens)
    patch_params = d * (3 * p * p + 1)
    default_tokens = (cfg.default_image_size // p) ** 2 + 1
    embed_params = patch_params + d + default_tokens * d + d  # + mask_token
    patch_flops = 2.0 * cfg.num_patches * d * (3 * p * p)

    # 2. Transformer blocks (QKV, OutProj, Attn QK^T & Attn*V, MLP FC1/FC2, LayerNorms, LayerScales)
    block_params = (
        4 * d * (d + 1)  # Q, K, V, OutProj + biases
        + 2 * d * mlp + mlp + d  # FC1 + FC2 + biases
        + 4 * d  # norm1/norm2 weight+bias
        + 2 * d  # layer_scale1/2
    )
    backbone_params = embed_params + cfg.depth * block_params + 2 * d  # + backbone.layernorm

    qkv_proj_flops = 4.0 * (2.0 * n_tok * d * d)
    attn_flops = 2.0 * cfg.num_heads * (2.0 * n_tok * n_tok * cfg.head_dim)
    mlp_flops = 2.0 * (2.0 * n_tok * d * mlp)
    backbone_flops = patch_flops + cfg.depth * (qkv_proj_flops + attn_flops + mlp_flops)

    # 3. DPT Neck + Head
    f = cfg.fusion_hidden_size
    dpt_params = 0
    dpt_flops = 0.0
    stage_hw: list[tuple[int, int]] = []
    for c_mid, factor in zip(cfg.neck_hidden_sizes, cfg.reassemble_factors):
        # 1x1 projection: embed_dim -> c_mid at (g, g)
        dpt_params += c_mid * (d + 1)
        dpt_flops += 2.0 * g * g * c_mid * d

        if factor == 4.0:
            sh, sw = g * 4, g * 4
            dpt_params += c_mid * c_mid * 16 + c_mid
            dpt_flops += 2.0 * g * g * c_mid * c_mid * 16
        elif factor == 2.0:
            sh, sw = g * 2, g * 2
            dpt_params += c_mid * c_mid * 4 + c_mid
            dpt_flops += 2.0 * g * g * c_mid * c_mid * 4
        elif factor == 1.0:
            sh, sw = g, g
        else:  # 0.5 downsample conv 3x3 stride 2
            sh, sw = (g + 1) // 2, (g + 1) // 2
            dpt_params += c_mid * c_mid * 9 + c_mid
            dpt_flops += 2.0 * sh * sw * c_mid * c_mid * 9
        stage_hw.append((sh, sw))

        # neck.convs: 3x3 c_mid -> f (no bias)
        dpt_params += f * c_mid * 9
        dpt_flops += 2.0 * sh * sw * f * c_mid * 9

    # fusion_stage: 4 layers, each has projection(1x1) + 2 residual conv units (2 * 3x3 convs)
    for idx, (sh, sw) in enumerate(reversed(stage_hw)):
        res1_convs = 0 if idx == 0 else 2
        convs_3x3 = res1_convs + 2
        dpt_params += (2 + 2) * (f * f * 9 + f) + (f * f + f)
        dpt_flops += 2.0 * sh * sw * (convs_3x3 * f * f * 9 + f * f)

    # Head: conv1 (f -> f//2, 3x3 at stage_0*2), conv2 (f//2 -> 32, 3x3 at input_size), conv3 (32 -> 1, 1x1)
    h0, w0 = stage_hw[0][0] * 2, stage_hw[0][1] * 2
    ih, iw = cfg.input_size, cfg.input_size
    c1 = f // 2
    c2 = cfg.head_hidden_size
    dpt_params += c1 * f * 9 + c1 + c2 * c1 * 9 + c2 + c2 + 1
    dpt_flops += 2.0 * (
        h0 * w0 * c1 * f * 9
        + ih * iw * c2 * c1 * 9
        + ih * iw * c2
    )

    return backbone_params + dpt_params, backbone_flops / 1e9, dpt_flops / 1e9


def analyze_feasibility(
    cfg: DepthAnythingConfig,
    *,
    npu_matmul_gflops: float = 20.0,
    npu_dispatch_overhead_ms: float = 12.0,
    gpu_fp16_gflops: float = 1200.0,
    cpu_fp32_gflops: float = 140.0,
) -> FeasibilityReport:
    """Estimate memory footprint, NPU1 SRAM/dispatch feasibility, and realistic FPS bounds.

    Default throughput baselines reflect single-process/4-column NPU1 bf16 measurements
    (~17-25 GFLOP/s effective + ~12 ms dispatch overhead per IRON kernel call),
    Ryzen 7 7840U CPU (OpenBLAS), and Radeon 780M iGPU (gfx1103 FP16 effective).
    """
    total_params, bb_gf, dpt_gf = _count_params_and_flops(cfg)
    tot_gf = bb_gf + dpt_gf

    w_fp32_mb = (total_params * 4) / (1024 * 1024)
    w_bf16_mb = (total_params * 2) / (1024 * 1024)
    w_int8_mb = total_params / (1024 * 1024)

    tokens_pad = _align_up(cfg.num_tokens, 128)  # 4 cols * 32-row tile
    # Largest per-layer activation is attention matrix (num_heads, tokens, tokens) or MLP (tokens, mlp_dim)
    attn_act = cfg.num_heads * cfg.num_tokens * cfg.num_tokens * 2
    mlp_act = tokens_pad * cfg.mlp_dim * 2
    peak_act_mb = max(attn_act, mlp_act) / (1024 * 1024)

    # Dispatches when each Transformer linear projection (QKV, OutProj, FC1, FC2) runs on NPU:
    # 1 patch embed + depth * (QKV-fused or 3 QKV + OutProj + FC1 + FC2)
    npu_dispatches = 1 + cfg.depth * 6

    # NPU1 tile fit: 32x64 @ 64x64 bf16 uses 4KB + 8KB + 4KB = 16KB (*2 ping-pong = 32KB <= 64KB SRAM)
    npu_sram_fit = True

    # Realistic NPU1 latency = compute time + per-dispatch HRX/IRON overhead
    npu_ms = (bb_gf / npu_matmul_gflops) * 1000.0 + npu_dispatches * npu_dispatch_overhead_ms + (dpt_gf / cpu_fp32_gflops) * 1000.0
    gpu_ms = (tot_gf / gpu_fp16_gflops) * 1000.0 + 4.0  # kernel launch overhead
    cpu_ms = (tot_gf / cpu_fp32_gflops) * 1000.0 + 5.0

    npu_fps = 1000.0 / npu_ms
    gpu_fps = 1000.0 / gpu_ms
    cpu_fps = 1000.0 / cpu_ms

    if npu_fps >= 15.0:
        viability = "Real-time on NPU1"
    elif gpu_fps >= 15.0:
        viability = "Real-time on GPU; NPU1 dispatch-bound without operator fusion"
    elif cpu_fps >= 5.0:
        viability = "Interactive on GPU/CPU; too slow on NPU1 w/o INT8+fusion"
    else:
        viability = "Offline / low-FPS on laptop APU (very heavy)"

    return FeasibilityReport(
        variant=cfg.variant,
        input_size=cfg.input_size,
        tokens=cfg.num_tokens,
        tokens_padded_npu=tokens_pad,
        total_params=total_params,
        backbone_gflops=bb_gf,
        dpt_gflops=dpt_gf,
        total_gflops=tot_gf,
        weights_fp32_mb=w_fp32_mb,
        weights_bf16_mb=w_bf16_mb,
        weights_int8_mb=w_int8_mb,
        peak_activation_bf16_mb=peak_act_mb,
        npu_matmul_dispatches=npu_dispatches,
        npu1_sram_tile_fit=npu_sram_fit,
        npu1_estimated_ms=npu_ms,
        npu1_estimated_fps=npu_fps,
        gpu_780m_estimated_ms=gpu_ms,
        gpu_780m_estimated_fps=gpu_fps,
        cpu_7840u_estimated_ms=cpu_ms,
        cpu_7840u_estimated_fps=cpu_fps,
        realtime_viability=viability,
    )


def format_feasibility_table(
    variants: tuple[str, ...] = ("small", "base", "large"),
    input_sizes: tuple[int, ...] = (252, 336, 518),
) -> str:
    """Render an ASCII feasibility comparison matrix across variants and resolutions."""
    header = (
        f"{'Variant':7s} {'Res':>7s} {'Params':>8s} {'GFLOPs':>8s} {'BF16 MB':>8s} {'INT8 MB':>8s} "
        f"{'NPU1 ms':>9s} {'CPU ms':>8s} {'GPU ms':>8s}  {'Assessment'}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for var in variants:
        for sz in input_sizes:
            rep = analyze_feasibility(get_depth_config(var, input_size=sz))
            lines.append(
                f"{rep.variant:7s} {f'{sz}x{sz}':>7s} {rep.total_params/1e6:7.1f}M "
                f"{rep.total_gflops:8.1f} {rep.weights_bf16_mb:8.1f} {rep.weights_int8_mb:8.1f} "
                f"{rep.npu1_estimated_ms:9.0f} {rep.cpu_7840u_estimated_ms:8.0f} {rep.gpu_780m_estimated_ms:8.0f}  "
                f"{rep.realtime_viability}"
            )
    return "\n".join(lines)
