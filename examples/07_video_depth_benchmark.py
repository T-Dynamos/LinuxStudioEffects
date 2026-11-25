"""Example 07: Streaming Video Depth Anything (Small/Base/Large) feasibility & NPU vs CPU vs GPU benchmark."""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from linux_studio_effects.video import (
    NpuVideoDepthEffect,
    PipelineConfig,
    V4L2Camera,
    VideoPipeline,
    is_device_available,
    pick_default_camera,
    pick_default_loopback,
)
from linux_studio_effects.video.models.video_depth_anything import (
    VideoDepthAnythingModel,
    create_synthetic_weights,
    format_feasibility_table,
    get_video_depth_config,
    load_from_hf,
)


def _load_sample_rgb(width: int = 640, height: int = 480) -> np.ndarray:
    cam_dev = pick_default_camera()
    if cam_dev is not None:
        try:
            with V4L2Camera() as cam:
                frame = cam.read_frame(decode=True, timeout_s=2.0)
                if frame.data is not None:
                    return frame.data
        except Exception:
            pass
    y, x = np.mgrid[0:height, 0:width]
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = (x * 255 // width).astype(np.uint8)
    rgb[:, :, 1] = (y * 255 // height).astype(np.uint8)
    rgb[height // 4 : 3 * height // 4, width // 3 : 2 * width // 3] = [210, 180, 150]
    return rgb


def _bench_variant(
    variant: str,
    input_size: int,
    sample_rgb: np.ndarray,
    *,
    use_real_weights: bool,
    runs: int,
    backends: list[str],
) -> None:
    cfg = get_video_depth_config(variant, input_size=input_size)
    weights = load_from_hf(cfg) if use_real_weights else create_synthetic_weights(cfg, seed=0)

    print(f"\n=== Benchmark: Video Depth Anything {variant.upper()} ({cfg.backbone_name}, {input_size}x{input_size}) ===")
    results = {}
    for backend in backends:
        if not is_device_available(backend):
            print(f"  {backend.upper():4s}: [UNAVAILABLE on this host]")
            continue

        model = VideoDepthAnythingModel(cfg, weights, backend=backend)
        try:
            # Frame 0 primes JIT kernels + initializes the 32-frame temporal KV cache
            model.infer(sample_rgb)
            total_ms_list = []
            bb_ms_list = []
            nh_ms_list = []
            last_res = model.infer(sample_rgb)
            total_ms_list.append(last_res.total_ms)
            bb_ms_list.append(last_res.backbone_ms)
            nh_ms_list.append(last_res.neck_head_ms)
            for _ in range(max(0, runs - 1)):
                last_res = model.infer(sample_rgb)
                total_ms_list.append(last_res.total_ms)
                bb_ms_list.append(last_res.backbone_ms)
                nh_ms_list.append(last_res.neck_head_ms)

            avg_tot = sum(total_ms_list) / len(total_ms_list)
            avg_bb = sum(bb_ms_list) / len(bb_ms_list)
            avg_nh = sum(nh_ms_list) / len(nh_ms_list)
            fps = 1000.0 / avg_tot
            extra = ""
            if last_res.npu_stats is not None:
                s = last_res.npu_stats
                extra = (
                    f" | npu_dispatches={s.dispatches}, "
                    f"npu_kernel={s.npu_wall_s*1000:.1f}ms, host_copy={s.host_copy_s*1000:.1f}ms"
                )
            print(
                f"  {backend.upper():4s}: {avg_tot:7.1f} ms ({fps:5.2f} fps) "
                f"[backbone={avg_bb:.1f}ms, temporal_dpt={avg_nh:.1f}ms]{extra}"
            )
            results[backend] = last_res.depth
        finally:
            model.close()

    if "cpu" in results and "npu" in results:
        c = results["cpu"].ravel()
        n = results["npu"].ravel()
        corr = float(np.corrcoef(c, n)[0, 1])
        mae = float(np.mean(np.abs(c - n)))
        print(f"  Parity (NPU bf16 vs CPU fp32): Pearson r={corr:.5f}, MAE={mae:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Streaming Video Depth Anything feasibility + NPU/CPU/GPU benchmark.")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("small", "base", "large"),
        default=["small"],
        help="Variants to benchmark (default: small)",
    )
    parser.add_argument("--input-size", type=int, default=252, help="Model input W=H multiple of 14 (default: 252)")
    parser.add_argument("--runs", type=int, default=3, help="Timed steady-state streaming runs per backend (default: 3)")
    parser.add_argument("--real-weights", action="store_true", help="Load real Video-Depth-Anything .pth weights from HF")
    parser.add_argument("--skip-feasibility", action="store_true", help="Skip printing the full feasibility table")
    parser.add_argument("--live", action="store_true", help="Stream live NPU video depth colormap to v4l2loopback until Ctrl+C")
    args = parser.parse_args()

    if not args.skip_feasibility:
        print("=== Video Depth Anything Hardware Feasibility Matrix ===")
        print(format_feasibility_table(("small", "base", "large"), (252, 336, 518)))

    if args.live:
        loopback = pick_default_loopback()
        if loopback is None:
            print("No v4l2loopback device found for --live mode.", file=sys.stderr)
            return 1
        eff = NpuVideoDepthEffect(
            variant=args.variants[0],
            input_size=args.input_size,
            use_real_weights=args.real_weights or (args.variants[0] == "small"),
        )
        pipe = VideoPipeline(
            PipelineConfig(output_device=loopback.path, output_width=640, output_height=480),
            effect=eff,
        )
        print(f"\nStreaming live NPU Video Depth Anything ({args.variants[0]} @ {args.input_size}x{args.input_size}) -> {loopback.path}")
        print(f"Preview with: ffplay {loopback.path}\nPress Ctrl+C to stop.")
        pipe.run(
            on_stats=lambda st: print(
                f"[live-vda] out={st.output_fps:.2f} fps | proc={st.avg_process_ms:.1f} ms | CPU={st.cpu_percent:.1f}%",
                flush=True,
            )
        )
        return 0

    sample = _load_sample_rgb()
    for var in args.variants:
        _bench_variant(
            var,
            args.input_size,
            sample,
            use_real_weights=args.real_weights,
            runs=args.runs,
            backends=["cpu", "npu", "gpu"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
