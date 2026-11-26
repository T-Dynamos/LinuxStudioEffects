"""Reusable microbenchmark & per-layer profiler for MODNet CPU vs NPU inference."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import linux_studio_effects  # noqa: E402 - initializes 1-thread defaults
import numpy as np  # noqa: E402 - import after 1-thread BLAS/OMP defaults
from linux_studio_effects.npu import probe_npu  # noqa: E402
from linux_studio_effects.video.models.modnet import (  # noqa: E402
    ModnetModel,
    create_synthetic_weights,
    get_modnet_config,
    host_ops as modnet_host_ops,
    load_from_hf,
)
from linux_studio_effects.video.models.video_depth_anything import npu_ops  # noqa: E402


def _make_input_rgb(image_path: str | None, width: int = 1280, height: int = 720) -> np.ndarray:
    if image_path:
        import av

        with av.open(image_path) as cont:
            rgb = next(cont.decode(video=0)).to_ndarray(format="rgb24")
        if (rgb.shape[1], rgb.shape[0]) != (width, height):
            rgb = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(width=width, height=height).to_ndarray(format="rgb24")
        return np.ascontiguousarray(rgb, dtype=np.uint8)
    yy, xx = np.indices((height, width), dtype=np.uint16)
    return np.stack(((xx * 3) % 256, (yy * 5) % 256, ((xx + yy) * 2) % 256), axis=2).astype(np.uint8)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).ravel() - float(np.mean(a))
    bf = b.astype(np.float64).ravel() - float(np.mean(b))
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    return float(np.dot(af, bf) / denom) if denom > 0.0 else 1.0


def _profile_model_layers(
    model_cpu: ModnetModel,
    model_npu: ModnetModel | None,
    rgb: np.ndarray,
    *,
    test_npu_candidates: bool = True,
) -> None:
    """Instrument every conv/dwconv/resize in MODNet and show CPU vs NPU timing per layer."""
    records: list[dict[str, Any]] = []
    npu_mm = model_npu._npu_mm if model_npu is not None else None

    orig_conv = model_cpu._conv
    orig_inv = model_cpu._inv_res
    layer_tag: list[str] = ["?"]

    def wrapped_conv(x: np.ndarray, cw: Any) -> np.ndarray:
        _, cin, h, w = x.shape
        cout = int(cw.weight_2d.shape[0])
        oh = (h + 2 * cw.padding - cw.kh) // cw.stride + 1
        ow = (w + 2 * cw.padding - cw.kw) // cw.stride + 1
        flops = cin * cw.kh * cw.kw * cout * oh * ow * 2
        t0 = time.perf_counter()
        out = orig_conv(x, cw)
        cpu_ms = (time.perf_counter() - t0) * 1000.0

        npu_ms: float | None = None
        npu_note = ""
        if test_npu_candidates and npu_mm is not None and flops >= 20_000_000:
            if cw.kh == 1 and cw.kw == 1 and cw.stride == 1 and cw.padding == 0 and oh * ow >= 32:
                wt = np.ascontiguousarray(cw.weight_2d.T, dtype=np.float32)
                _ = npu_mm.matmul(
                    x.reshape(cin, h * w),
                    wt,
                    bias=cw.bias,
                    cache_b=True,
                    transpose_a=True,
                    out_nchw_hw=(oh, ow),
                )
                tn0 = time.perf_counter()
                y = npu_mm.matmul(
                    x.reshape(cin, h * w),
                    wt,
                    bias=cw.bias,
                    cache_b=True,
                    transpose_a=True,
                    out_nchw_hw=(oh, ow),
                )
                if cw.has_inorm:
                    _ = modnet_host_ops.ibnorm_tail_relu(
                        y,
                        eps=model_cpu.config.norm_eps,
                        apply_relu=(cw.relu_mode == "relu"),
                    )
                npu_ms = (time.perf_counter() - tn0) * 1000.0
                npu_note = "mm1x1"
            elif cw.kh == 3 and cw.kw == 3 and cw.stride == 1 and cw.padding == 1 and cout >= 16 and oh * ow >= 32:
                w_ck = np.ascontiguousarray(cw.weight_2d.T, dtype=np.float32)
                w_kykxc = np.ascontiguousarray(
                    cw.weight_2d.reshape(cout, cin, 3, 3).transpose(2, 3, 1, 0).reshape(9 * cin, cout),
                    dtype=np.float32,
                )
                use_relu = (cw.relu_mode == "relu") and (not cw.has_inorm)
                _ = npu_mm.conv2d_3x3_s1_p1_nchw(x, w_kykxc, w_ck, bias=cw.bias, apply_relu=use_relu)
                tn0 = time.perf_counter()
                y = npu_mm.conv2d_3x3_s1_p1_nchw(x, w_kykxc, w_ck, bias=cw.bias, apply_relu=use_relu)
                if cw.has_inorm:
                    _ = modnet_host_ops.ibnorm_tail_relu(
                        y,
                        eps=model_cpu.config.norm_eps,
                        apply_relu=(cw.relu_mode == "relu"),
                    )
                elif cw.relu_mode == "relu6":
                    np.clip(y, np.float32(0.0), np.float32(6.0), out=y)
                npu_ms = (time.perf_counter() - tn0) * 1000.0
                npu_note = "conv3x3"
            elif cout >= 16 and oh * ow >= 32:
                wt = np.ascontiguousarray(cw.weight_2d.T, dtype=np.float32)
                cols = modnet_host_ops.im2col_kxk_p(x, kh=cw.kh, kw=cw.kw, stride=cw.stride, padding=cw.padding)
                _ = npu_mm.matmul(
                    cols,
                    wt,
                    bias=cw.bias,
                    cache_b=True,
                    transpose_a=True,
                    out_nchw_hw=(oh, ow),
                )
                tn0 = time.perf_counter()
                cols = modnet_host_ops.im2col_kxk_p(x, kh=cw.kh, kw=cw.kw, stride=cw.stride, padding=cw.padding)
                y = npu_mm.matmul(
                    cols,
                    wt,
                    bias=cw.bias,
                    cache_b=True,
                    transpose_a=True,
                    out_nchw_hw=(oh, ow),
                )
                if cw.has_inorm:
                    _ = modnet_host_ops.ibnorm_tail_relu(
                        y,
                        eps=model_cpu.config.norm_eps,
                        apply_relu=(cw.relu_mode == "relu"),
                    )
                npu_ms = (time.perf_counter() - tn0) * 1000.0
                npu_note = f"im2+{cw.kh}x{cw.kw}"

        records.append(
            {
                "name": layer_tag[0],
                "kind": f"{cw.kh}x{cw.kw}s{cw.stride}" + ("+IN" if cw.has_inorm else ""),
                "in": (cin, h, w),
                "out": (cout, oh, ow),
                "mflops": flops / 1e6,
                "cpu_ms": cpu_ms,
                "npu_ms": npu_ms,
                "npu_note": npu_note,
            }
        )
        return out

    model_cpu._conv = wrapped_conv  # type: ignore[method-assign]
    orig_dw = modnet_host_ops.dwconv_3x3_p1

    def wrapped_dw(x: np.ndarray, w_dw: np.ndarray, bias: np.ndarray, *, stride: int = 1, relu6: bool = True) -> np.ndarray:
        _, cin, h, w = x.shape
        oh = (h + 2 - 3) // stride + 1
        ow = (w + 2 - 3) // stride + 1
        t0 = time.perf_counter()
        out = orig_dw(x, w_dw, bias, stride=stride, relu6=relu6)
        cpu_ms = (time.perf_counter() - t0) * 1000.0
        records.append(
            {
                "name": layer_tag[0],
                "kind": f"dw3x3s{stride}",
                "in": (cin, h, w),
                "out": (cin, oh, ow),
                "mflops": (cin * 9 * oh * ow * 2) / 1e6,
                "cpu_ms": cpu_ms,
                "npu_ms": None,
                "npu_note": "",
            }
        )
        return out

    modnet_host_ops.dwconv_3x3_p1 = wrapped_dw
    try:
        # Warm run once cleanly, then capture traced run
        _ = model_cpu.infer(rgb)
        records.clear()
        t_start = time.perf_counter()
        img = model_cpu.preprocess(rgb)
        t_pre = (time.perf_counter() - t_start) * 1000.0
        _, _, th, tw = img.shape
        w = model_cpu.weights

        layer_tag[0] = "conv0"
        x = model_cpu._conv(img, w.conv0)
        layer_tag[0] = "blk0"
        x = orig_inv(x, w.blocks[0])
        enc2x = x
        layer_tag[0] = "blk1"
        x = orig_inv(x, w.blocks[1])
        layer_tag[0] = "blk2"
        x = orig_inv(x, w.blocks[2])
        enc4x = x
        for bi, blk in enumerate(w.blocks[3:], start=3):
            layer_tag[0] = f"blk{bi}"
            x = orig_inv(x, blk)
        layer_tag[0] = "conv_last"
        enc32x = model_cpu._conv(x, w.conv_last)

        t_se0 = time.perf_counter()
        se = np.mean(enc32x, axis=(2, 3), keepdims=False)
        se_w = (1.0 / (1.0 + np.exp(-np.clip(np.maximum(se @ w.se_fc1_t, 0.0) @ w.se_fc2_t, -40.0, 40.0)))).reshape(1, -1, 1, 1)
        enc32x = enc32x * se_w
        t_se_ms = (time.perf_counter() - t_se0) * 1000.0

        lr16x = modnet_host_ops.bilinear_resize_align_false(enc32x, th // 16, tw // 16)
        layer_tag[0] = "conv_lr16x"
        lr16x = model_cpu._conv(lr16x, w.conv_lr16x)
        lr8x = modnet_host_ops.bilinear_resize_align_false(lr16x, th // 8, tw // 8)
        layer_tag[0] = "conv_lr8x"
        lr8x = model_cpu._conv(lr8x, w.conv_lr8x)

        img2x = modnet_host_ops.bilinear_resize_align_false(img, th // 2, tw // 2)
        img4x = modnet_host_ops.bilinear_resize_align_false(img, th // 4, tw // 4)
        layer_tag[0] = "tohr_enc2x"
        enc2x_hr = model_cpu._conv(enc2x, w.tohr_enc2x)
        layer_tag[0] = "conv_enc2x"
        hr4x = model_cpu._conv(np.concatenate((img2x, enc2x_hr), axis=1), w.conv_enc2x)
        layer_tag[0] = "tohr_enc4x"
        enc4x_hr = model_cpu._conv(enc4x, w.tohr_enc4x)
        layer_tag[0] = "conv_enc4x"
        hr4x = model_cpu._conv(np.concatenate((hr4x, enc4x_hr), axis=1), w.conv_enc4x)

        lr4x_hr = modnet_host_ops.bilinear_resize_align_false(lr8x, th // 4, tw // 4)
        hr4x = np.concatenate((hr4x, lr4x_hr, img4x), axis=1)
        for i, layer in enumerate(w.conv_hr4x):
            layer_tag[0] = f"conv_hr4x[{i}]"
            hr4x = model_cpu._conv(hr4x, layer)

        hr2x = modnet_host_ops.bilinear_resize_align_false(hr4x, th // 2, tw // 2)
        hr2x = np.concatenate((hr2x, enc2x_hr), axis=1)
        for i, layer in enumerate(w.conv_hr2x):
            layer_tag[0] = f"conv_hr2x[{i}]"
            hr2x = model_cpu._conv(hr2x, layer)

        lr4x = modnet_host_ops.bilinear_resize_align_false(lr8x, th // 4, tw // 4)
        layer_tag[0] = "conv_lr4x"
        lr4x = model_cpu._conv(lr4x, w.conv_lr4x)
        lr2x = modnet_host_ops.bilinear_resize_align_false(lr4x, th // 2, tw // 2)
        layer_tag[0] = "conv_f2x"
        f2x = model_cpu._conv(np.concatenate((lr2x, hr2x), axis=1), w.conv_f2x)
        f = modnet_host_ops.bilinear_resize_align_false(f2x, th, tw)
        layer_tag[0] = "conv_f0"
        f = model_cpu._conv(np.concatenate((f, img), axis=1), w.conv_f0)
        layer_tag[0] = "conv_f1"
        _ = model_cpu._conv(f, w.conv_f1)
    finally:
        model_cpu._conv = orig_conv  # type: ignore[method-assign]
        modnet_host_ops.dwconv_3x3_p1 = orig_dw

    print(f"\n=== Layer Profile ({model_cpu.config.max_size}px, pre={t_pre:.2f}ms, se={t_se_ms:.2f}ms) ===")
    print(f"{'Layer':14s} {'Kind':11s} {'In':15s} {'Out':15s} {'MFLOPs':>8s} {'CPU ms':>8s} {'NPU ms':>8s} {'Speedup':>8s} {'Note':>8s}")
    total_cpu_conv = sum(r["cpu_ms"] for r in records)
    records.sort(key=lambda r: r["cpu_ms"], reverse=True)
    for r in records[:25]:
        npu_s = f"{r['npu_ms']:8.2f}" if r["npu_ms"] is not None else "       -"
        spd_s = f"{(r['cpu_ms'] / r['npu_ms']):7.2f}x" if (r["npu_ms"] and r["npu_ms"] > 0) else "       -"
        print(
            f"{r['name']:14s} {r['kind']:11s} {str(r['in']):15s} {str(r['out']):15s} "
            f"{r['mflops']:8.2f} {r['cpu_ms']:8.2f} {npu_s} {spd_s} {r['npu_note']:>8s}"
        )
    print(f"Sum profiled conv+dw CPU time: {total_cpu_conv:.2f} ms across {len(records)} ops")


def _thread_names() -> list[str]:
    names: list[str] = []
    for tid in sorted(os.listdir("/proc/self/task")):
        try:
            with open(f"/proc/self/task/{tid}/comm", encoding="utf-8") as fh:
                names.append(fh.read().strip())
        except OSError:
            pass
    return names


def _bench_render_1080p(max_size: int, *, has_npu: bool, run_async: bool = False) -> None:
    """Measure 1080p YUYV BackgroundSegmentationEffect per-frame render stages and thread count."""
    from linux_studio_effects.video import FrameContext, create_effect
    from linux_studio_effects.video.pixels import channels_to_frame, frame_to_channels

    yuyv = np.random.default_rng(2).integers(0, 256, (1080, 960, 4), dtype=np.uint8)
    for dev in (("cpu", "npu") if has_npu else ("cpu",)):
        print(f"--- _bench_render_1080p({dev}) start threads={len(_thread_names())}: {_thread_names()} ---")
        eff = create_effect("background_segmentation", device=dev, max_size=max_size, use_real_weights=False, async_matte=run_async)
        eff.setup(1920, 1080, 30.0)
        print(f"  after setup: threads={len(_thread_names())} {_thread_names()}")
        res = eff._worker.step(eff._model, yuyv, "YUYV", async_enabled=False)
        print(f"  after matte step: threads={len(_thread_names())} {_thread_names()}")
        alpha_raw = res.alpha
        _ = eff.apply_alpha_background(yuyv, alpha_raw, pixel_format="YUYV")

        t0 = time.perf_counter()
        eff._cached_alpha_key = None
        a_full = eff._refine_alpha(alpha_raw, yuyv, "YUYV"); t1 = time.perf_counter()
        s1_in = modnet_host_ops.downsample2x_bg_hw4_alpha(yuyv, "YUYV", a_full); t1b = time.perf_counter()
        s_top = eff._build_blur_pyramid_hw4(yuyv, "YUYV", a_full); t2 = time.perf_counter()
        _ = modnet_host_ops.unpremul_upsample_bg_composite(yuyv, "YUYV", s_top, a_full, eff.blur_strength); t3 = time.perf_counter()

        ts = []
        t_c0, t_w0 = time.process_time(), time.perf_counter()
        for i in range(10):
            ta = time.perf_counter()
            eff.apply_alpha_background(yuyv, alpha_raw, pixel_format="YUYV")
            ts.append((time.perf_counter() - ta) * 1000.0)
        cpu_pct = 100.0 * (time.process_time() - t_c0) / max(time.perf_counter() - t_w0, 1e-6)
        thrs = _thread_names()
        print(
            f"1080p YUYV bg_seg {dev.upper()} (threads={len(thrs)} {thrs}, cpu={cpu_pct:.0f}%): med={np.median(ts):.2f}ms | "
            f"refine_uncached={(t1-t0)*1000:.2f} ds2x={(t1b-t1)*1000:.2f} g5_pyr={(t2-t1b)*1000:.2f} fused_up_comp={(t3-t2)*1000:.2f}"
        )
        if run_async:
            # Warmup async worker briefly
            for k in range(3):
                eff.process(
                    yuyv,
                    FrameContext(frame_index=k, timestamp_s=k / 30.0, width=1920, height=1080, fps=30.0, pixel_format="YUYV"),
                )
                time.sleep(0.01)
            pts = []
            ac0, aw0 = time.process_time(), time.perf_counter()
            next_t = aw0
            for k in range(45):
                fctx = FrameContext(
                    frame_index=k + 3,
                    timestamp_s=(k + 3) / 30.0,
                    width=1920,
                    height=1080,
                    fps=30.0,
                    pixel_format="YUYV",
                )
                tp = time.perf_counter()
                eff.process(yuyv, fctx)
                pts.append((time.perf_counter() - tp) * 1000.0)
                next_t += 1.0 / 30.0
                rem = next_t - time.perf_counter()
                if rem > 0:
                    time.sleep(rem)
            wall = time.perf_counter() - aw0
            acpu = 100.0 * (time.process_time() - ac0) / max(wall, 1e-6)
            print(
                f"1080p ASYNC 30fps {dev.upper()} (45f, interval={eff._worker.interval_s:.2f}s): "
                f"eff_fps={45.0 / wall:.2f} process_med={np.median(pts):.2f}ms p95={np.percentile(pts, 95):.2f}ms "
                f"cpu={acpu:.1f}% threads={len(_thread_names())} matte_lag={eff.matte_lag_ms:.1f}ms"
            )
        eff.teardown()


def _print_npu_dispatch_breakdown(cpu_model: ModnetModel, npu_model: ModnetModel, rgb: np.ndarray) -> None:
    """Measure exact per-layer CPU vs NPU execution on only the layers routed to NPU."""
    orig_npu_conv = npu_model._conv
    rows: list[tuple[str, str, int, float, float, float]] = []

    def traced_npu_conv(x: np.ndarray, cw: Any) -> np.ndarray:
        assert npu_model._npu_mm is not None
        d0 = npu_model._npu_mm.stats.dispatches
        n0 = npu_model._npu_mm.stats.npu_wall_s
        c0 = npu_model._npu_mm.stats.host_copy_s
        t0 = time.perf_counter()
        out = orig_npu_conv(x, cw)
        tot_npu_ms = (time.perf_counter() - t0) * 1000.0
        dd = npu_model._npu_mm.stats.dispatches - d0
        if dd > 0:
            tc0 = time.perf_counter()
            _ = cpu_model._conv(x, cw)
            tot_cpu_ms = (time.perf_counter() - tc0) * 1000.0
            rows.append(
                (
                    f"{cw.kh}x{cw.kw}s{cw.stride} ({x.shape[1]},{x.shape[2]},{x.shape[3]})->{cw.weight_2d.shape[0]}",
                    "inorm" if cw.has_inorm else (cw.relu_mode or "none"),
                    dd,
                    tot_cpu_ms,
                    tot_npu_ms,
                    (npu_model._npu_mm.stats.host_copy_s - c0) * 1000.0,
                )
            )
        return out

    npu_model._conv = traced_npu_conv  # type: ignore[method-assign]
    try:
        _ = npu_model.infer(rgb)
    finally:
        npu_model._conv = orig_npu_conv  # type: ignore[method-assign]

    print(f"\n=== Active NPU Offloaded Layers Breakdown ({npu_model.config.max_size}px) ===")
    print(f"{'Shape':34s} {'Act':7s} {'Disp':>4s} {'CPU ms':>8s} {'NPU tot':>8s} {'Copy ms':>8s} {'Win':>8s}")
    for shp, act, dd, c_ms, n_ms, cp_ms in rows:
        print(f"{shp:34s} {act:7s} {dd:4d} {c_ms:8.2f} {n_ms:8.2f} {cp_ms:8.2f} {c_ms - n_ms:+7.2f}ms")
    print(
        f"Offloaded sum: CPU={sum(r[3] for r in rows):.2f}ms vs NPU={sum(r[4] for r in rows):.2f}ms "
        f"(delta={sum(r[3] - r[4] for r in rows):+.2f}ms)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Microbenchmark MODNet CPU vs NPU inference and per-layer profile.")
    parser.add_argument("--variant", choices=("webcam", "photographic"), default="webcam")
    parser.add_argument("--max-size", type=int, default=256, help="MODNet max side (160=fast, 256=balanced, 384=high)")
    parser.add_argument("--runs", type=int, default=6, help="Timed runs per backend after warmup")
    parser.add_argument("--bf16-out", action="store_true", help="Benchmark NPU executor with bf16 output path")
    parser.add_argument("--real-weights", action="store_true", help="Load real HuggingFace weights instead of synthetic")
    parser.add_argument("--local-only", action="store_true", help="Offline HuggingFace cache only")
    parser.add_argument("--image", help="Optional input image path")
    parser.add_argument("--profile-layers", action="store_true", help="Print top 25 layers by CPU time + NPU probe")
    parser.add_argument("--npu-breakdown", action="store_true", help="Compare only layers offloaded to NPU vs CPU")
    parser.add_argument("--render-1080p", action="store_true", help="Benchmark 1080p background_segmentation YUYV blur render stages")
    parser.add_argument("--async-1080p", action="store_true", help="Simulate 30fps live async 1080p background_segmentation and measure CPU and threads")
    args = parser.parse_args(argv)

    npu = probe_npu()
    print(
        f"Env: OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')} OMP={os.environ.get('OMP_NUM_THREADS')} "
        f"LSE={os.environ.get('LSE_THREADS')} | AVX512_MODNET={modnet_host_ops._LIB is not None} | NPU={npu.available} ({npu.reason or 'ready'})"
    )
    cfg = get_modnet_config(args.variant, max_size=args.max_size)
    weights = load_from_hf(cfg, local_files_only=args.local_only) if args.real_weights else create_synthetic_weights(cfg)
    rgb = _make_input_rgb(args.image)

    print(f"Threads before models: {len(_thread_names())}")
    cpu_model = ModnetModel(cfg, weights, backend="cpu")
    print(f"Threads after cpu_model init: {len(_thread_names())}")
    npu_model = ModnetModel(cfg, weights, backend="npu") if npu.available else None
    print(f"Threads after npu_model init: {len(_thread_names())}")
    if npu_model is not None and args.bf16_out:
        npu_model._npu_mm.close()
        npu_model._npu_mm = npu_ops.NpuMatMulExecutor(use_bf16_out=True)

    try:
        # Warmup both backends cleanly, then interleave runs to prevent thermal drift bias
        ref_res = cpu_model.infer(rgb)
        print(f"Threads after cpu_model infer: {len(_thread_names())}")
        if npu_model is not None:
            _ = npu_model.infer(rgb)
            print(f"Threads after npu_model infer: {len(_thread_names())}")
            npu_model._npu_mm.reset_stats()

        cpu_times = []
        npu_times = []
        for _ in range(args.runs):
            cpu_times.append(cpu_model.infer(rgb))
            if npu_model is not None:
                npu_times.append(npu_model.infer(rgb))

        cpu_tot = [r.total_ms for r in cpu_times]
        cpu_bb = [r.backbone_ms for r in cpu_times]
        cpu_dec = [r.decoder_ms for r in cpu_times]
        print(
            f"CPU  (max={args.max_size}): med={np.median(cpu_tot):.2f}ms (bb={np.median(cpu_bb):.2f}ms, dec={np.median(cpu_dec):.2f}ms, p95={np.percentile(cpu_tot, 95):.2f}ms)"
        )

        if npu_model is not None:
            stats = npu_model._npu_mm.stats
            n_runs = args.runs
            t_last = npu_times[-1]
            npu_tot = [r.total_ms for r in npu_times]
            npu_bb = [r.backbone_ms for r in npu_times]
            npu_dec = [r.decoder_ms for r in npu_times]
            r_corr = _pearson(ref_res.alpha, t_last.alpha)
            mae = float(np.mean(np.abs(ref_res.alpha - t_last.alpha)))
            print(
                f"NPU  (max={args.max_size}): med={np.median(npu_tot):.2f}ms (bb={np.median(npu_bb):.2f}ms, dec={np.median(npu_dec):.2f}ms, p95={np.percentile(npu_tot, 95):.2f}ms) | "
                f"dispatches/infer={stats.dispatches / n_runs:.1f} npu_ms={stats.npu_wall_s * 1000.0 / n_runs:.2f} copy_ms={stats.host_copy_s * 1000.0 / n_runs:.2f} | "
                f"pearson_r={r_corr:.6f} mae={mae:.5f}"
            )

        if args.profile_layers:
            _profile_model_layers(cpu_model, npu_model, rgb, test_npu_candidates=True)
        if npu_model is not None and args.npu_breakdown:
            _print_npu_dispatch_breakdown(cpu_model, npu_model, rgb)
        if args.render_1080p or args.async_1080p:
            _bench_render_1080p(args.max_size, has_npu=npu.available, run_async=args.async_1080p)
    finally:
        cpu_model.close()
        if npu_model is not None:
            npu_model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
