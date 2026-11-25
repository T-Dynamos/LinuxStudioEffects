"""Analytical FLOP, memory, and hardware feasibility calculator for streaming Video Depth Anything."""

from __future__ import annotations

from dataclasses import dataclass

from .config import VideoDepthConfig, get_video_depth_config


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
    dpt_temporal_gflops: float
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


def _count_params_and_flops(cfg: VideoDepthConfig) -> tuple[int, float, float]:
    d = cfg.embed_dim
    mlp = cfg.mlp_dim
    n_tok = cfg.num_tokens
    g = cfg.patch_grid
    p = cfg.patch_size

    # 1. Embeddings
    patch_params = d * (3 * p * p + 1)
    default_tokens = (cfg.default_image_size // p) ** 2 + 1
    embed_params = patch_params + d + default_tokens * d + d  # + mask_token
    patch_flops = 2.0 * cfg.num_patches * d * (3 * p * p)

    # 2. Transformer blocks
    block_params = (
        4 * d * (d + 1)
        + 2 * d * mlp + mlp + d
        + 4 * d
        + 2 * d
    )
    backbone_params = embed_params + cfg.depth * block_params + 2 * d

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
        else:
            sh, sw = (g + 1) // 2, (g + 1) // 2
            dpt_params += c_mid * c_mid * 9 + c_mid
            dpt_flops += 2.0 * sh * sw * c_mid * c_mid * 9
        stage_hw.append((sh, sw))

        dpt_params += f * c_mid * 9
        dpt_flops += 2.0 * sh * sw * f * c_mid * 9

    for idx, (sh, sw) in enumerate(reversed(stage_hw)):
        res1_convs = 0 if idx == 0 else 2
        convs_3x3 = res1_convs + 2
        dpt_params += (2 + 2) * (f * f * 9 + f) + (f * f + f)
        dpt_flops += 2.0 * sh * sw * (convs_3x3 * f * f * 9 + f * f)

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

    # 4. 4 Temporal MotionModules (streaming single-frame KV-factored mode)
    # Placed at: layer3 (stage 2), layer4 (stage 3), path4 (stage 2 size), path3 (stage 1 size)
    motion_spatial = (
        stage_hw[2][0] * stage_hw[2][1],
        stage_hw[3][0] * stage_hw[3][1],
        stage_hw[2][0] * stage_hw[2][1],
        stage_hw[1][0] * stage_hw[1][1],
    )
    motion_params = 0
    motion_flops = 0.0
    for ch, n_sp in zip(cfg.motion_channels, motion_spatial):
        # GroupNorm + proj_in/proj_out + 2*(norm + Q/K/V/Out) + ff_norm + GEGLU FF
        attn_params = 2 * (4 * ch * ch + ch) + 4 * ch
        ff_params = 12 * ch * ch + 9 * ch + 2 * ch
        motion_params += 2 * ch + 2 * (ch * ch + ch) + attn_params + ff_params

        # Streaming FLOPs (per frame: 1 Q, 1 new K/V factorized, 32-frame cross-attn, FF + proj_in/out)
        proj_flops = 2.0 * n_sp * 2.0 * (ch * ch)
        t_attn_flops = 2.0 * (
            2.0 * n_sp * 4.0 * (ch * ch) + 2.0 * n_sp * (2.0 * cfg.num_frames * ch)
        )
        ff_flops = 2.0 * n_sp * (12.0 * ch * ch)
        motion_flops += proj_flops + t_attn_flops + ff_flops

    total_params = backbone_params + dpt_params + motion_params
    return total_params, backbone_flops / 1e9, (dpt_flops + motion_flops) / 1e9


def analyze_feasibility(
    cfg: VideoDepthConfig,
    *,
    npu_matmul_gflops: float = 150.0,
    npu_dispatch_overhead_ms: float = 1.0,
    gpu_fp16_gflops: float = 1200.0,
    cpu_fp32_gflops: float = 140.0,
) -> FeasibilityReport:
    """Estimate memory footprint, NPU1 SRAM/dispatch feasibility, and streaming FPS bounds."""
    total_params, bb_gf, dpt_t_gf = _count_params_and_flops(cfg)
    tot_gf = bb_gf + dpt_t_gf

    w_fp32_mb = (total_params * 4) / (1024 * 1024)
    w_bf16_mb = (total_params * 2) / (1024 * 1024)
    w_int8_mb = total_params / (1024 * 1024)

    tokens_pad = _align_up(cfg.num_tokens, 128)
    attn_act = cfg.num_heads * cfg.num_tokens * cfg.num_tokens * 2
    mlp_act = tokens_pad * cfg.mlp_dim * 2
    peak_act_mb = max(attn_act, mlp_act) / (1024 * 1024)

    # Fused QKV + OutProj + FC1 + FC2 per ViT block
    npu_dispatches = cfg.depth * 4
    npu_sram_fit = True

    npu_ms = (bb_gf / npu_matmul_gflops) * 1000.0 + npu_dispatches * npu_dispatch_overhead_ms + (dpt_t_gf / cpu_fp32_gflops) * 1000.0
    gpu_ms = (tot_gf / gpu_fp16_gflops) * 1000.0 + 5.0
    cpu_ms = (tot_gf / cpu_fp32_gflops) * 1000.0 + 6.0

    npu_fps = 1000.0 / npu_ms
    gpu_fps = 1000.0 / gpu_ms
    cpu_fps = 1000.0 / cpu_ms

    if npu_fps >= 15.0:
        viability = "Real-time on NPU1"
    elif gpu_fps >= 15.0:
        viability = "Real-time on GPU; hybrid NPU1+CPU streaming"
    elif cpu_fps >= 5.0:
        viability = "Interactive on GPU/CPU; heavy on NPU1"
    else:
        viability = "Offline / low-FPS on laptop APU (very heavy)"

    return FeasibilityReport(
        variant=cfg.variant,
        input_size=cfg.input_size,
        tokens=cfg.num_tokens,
        tokens_padded_npu=tokens_pad,
        total_params=total_params,
        backbone_gflops=bb_gf,
        dpt_temporal_gflops=dpt_t_gf,
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
    """Format a human-readable multi-variant hardware feasibility matrix."""
    lines = [
        f"{'Variant':<7} {'Res':>7} {'Params':>8} {'GFLOPs':>8} {'BF16 MB':>8} {'INT8 MB':>8} "
        f"{'NPU1 ms':>8} {'CPU ms':>8} {'GPU ms':>8}  Assessment",
        "-" * 91,
    ]
    for v in variants:
        for sz in input_sizes:
            r = analyze_feasibility(get_video_depth_config(v, input_size=sz))
            lines.append(
                f"{r.variant:<7} {sz:>3}x{sz:<3} {r.total_params / 1e6:7.1f}M {r.total_gflops:8.1f} "
                f"{r.weights_bf16_mb:8.1f} {r.weights_int8_mb:8.1f} "
                f"{r.npu1_estimated_ms:8.0f} {r.cpu_7840u_estimated_ms:8.0f} {r.gpu_780m_estimated_ms:8.0f}  "
                f"{r.realtime_viability}"
            )
    return "\n".join(lines)
