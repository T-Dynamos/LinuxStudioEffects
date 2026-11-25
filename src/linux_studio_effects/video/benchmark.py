"""Camera dependency check and real-time V4L2 FPS benchmark CLI."""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass

from .camera import CameraConfig, V4L2Camera
from .deps import check_camera_deps, format_deps_report


@dataclass(frozen=True)
class CameraBenchmarkResult:
    """Measured throughput and timing statistics for a V4L2 camera capture run."""

    config: CameraConfig
    decode: bool
    frames: int
    elapsed_s: float
    wall_fps: float
    driver_fps: float
    avg_interval_ms: float
    min_interval_ms: float
    p95_interval_ms: float
    max_interval_ms: float
    avg_decode_ms: float
    p95_decode_ms: float
    avg_payload_kb: float
    dropped_frames: int
    frame_shape: tuple[int, ...] | None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def benchmark_camera(
    config: CameraConfig | None = None,
    *,
    duration_s: float = 3.0,
    warmup_frames: int = 5,
    decode: bool = True,
) -> CameraBenchmarkResult:
    """Stream frames for duration_s and compute actual capture & decode FPS metrics."""
    with V4L2Camera(config) as cam:
        negotiated = cam.negotiated
        for _ in range(warmup_frames):
            cam.read_frame(decode=decode)

        first = cam.read_frame(decode=decode)
        start_wall = first.capture_monotonic_s
        prev_wall = start_wall
        first_drv = first.driver_timestamp_s
        last_drv = first_drv

        intervals_ms: list[float] = []
        decode_ms: list[float] = [first.decode_time_s * 1000.0] if decode else []
        payload_sizes: list[int] = [first.raw_size]
        dropped = 0
        count = 1
        frame_shape = tuple(first.data.shape) if first.data is not None else None

        while (prev_wall - start_wall) < duration_s:
            frame = cam.read_frame(decode=decode)
            count += 1
            intervals_ms.append((frame.capture_monotonic_s - prev_wall) * 1000.0)
            prev_wall = frame.capture_monotonic_s
            last_drv = frame.driver_timestamp_s
            if decode:
                decode_ms.append(frame.decode_time_s * 1000.0)
            payload_sizes.append(frame.raw_size)
            dropped += frame.dropped_since_last

        elapsed_s = max(prev_wall - start_wall, 1e-9)
        drv_elapsed_s = last_drv - first_drv
        wall_fps = (count - 1) / elapsed_s if count > 1 else 0.0
        driver_fps = (count - 1) / drv_elapsed_s if (count > 1 and drv_elapsed_s > 0) else 0.0

        return CameraBenchmarkResult(
            config=negotiated,
            decode=decode,
            frames=count,
            elapsed_s=elapsed_s,
            wall_fps=wall_fps,
            driver_fps=driver_fps,
            avg_interval_ms=statistics.fmean(intervals_ms) if intervals_ms else 0.0,
            min_interval_ms=min(intervals_ms) if intervals_ms else 0.0,
            p95_interval_ms=_percentile(intervals_ms, 95.0),
            max_interval_ms=max(intervals_ms) if intervals_ms else 0.0,
            avg_decode_ms=statistics.fmean(decode_ms) if decode_ms else 0.0,
            p95_decode_ms=_percentile(decode_ms, 95.0),
            avg_payload_kb=(statistics.fmean(payload_sizes) / 1024.0) if payload_sizes else 0.0,
            dropped_frames=dropped,
            frame_shape=frame_shape,
        )


def format_benchmark_result(res: CameraBenchmarkResult) -> str:
    """Format a CameraBenchmarkResult into a concise report."""
    mode = "capture+decode (RGB)" if res.decode else "raw V4L2 capture"
    lines = [
        f"=== Benchmark: {res.config.device} {res.config.width}x{res.config.height} "
        f"{res.config.pixel_format} (target {res.config.fps:g} fps) [{mode}] ===",
        f"  Frames        : {res.frames} in {res.elapsed_s:.2f}s (dropped={res.dropped_frames})",
        f"  Measured FPS  : {res.wall_fps:.2f} fps (driver timestamps: {res.driver_fps:.2f} fps)",
        f"  Interval (ms) : avg={res.avg_interval_ms:.2f}, min={res.min_interval_ms:.2f}, "
        f"p95={res.p95_interval_ms:.2f}, max={res.max_interval_ms:.2f}",
        f"  Payload       : avg={res.avg_payload_kb:.1f} KiB/frame",
    ]
    if res.decode:
        lines.append(
            f"  Decode (ms)   : avg={res.avg_decode_ms:.2f}, p95={res.p95_decode_ms:.2f}, "
            f"shape={res.frame_shape}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check V4L2 camera dependencies and benchmark capture FPS.")
    parser.add_argument("--device", default="", help="V4L2 device path (default: auto-detect)")
    parser.add_argument("--width", type=int, default=1280, help="Capture width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Capture height (default: 720)")
    parser.add_argument("--format", default="MJPG", help="V4L2 FourCC pixel format (default: MJPG)")
    parser.add_argument("--fps", type=float, default=30.0, help="Requested FPS (default: 30)")
    parser.add_argument("--duration", type=float, default=3.0, help="Benchmark duration in seconds (default: 3.0)")
    parser.add_argument("--check-only", action="store_true", help="Only run dependency & device checks")
    parser.add_argument("--no-decode", action="store_true", help="Skip RGB decode (measure raw V4L2 capture only)")
    parser.add_argument("--compare-raw", action="store_true", help="Run both raw V4L2 and decoded RGB benchmarks")
    args = parser.parse_args(argv)

    report = check_camera_deps()
    print(format_deps_report(report))
    if not report.ok:
        return 1
    if args.check_only:
        return 0

    cfg = CameraConfig(
        device=args.device,
        width=args.width,
        height=args.height,
        pixel_format=args.format.upper(),
        fps=args.fps,
    )

    print()
    if args.compare_raw:
        raw_res = benchmark_camera(cfg, duration_s=args.duration, decode=False)
        print(format_benchmark_result(raw_res))
        print()
        dec_res = benchmark_camera(cfg, duration_s=args.duration, decode=True)
        print(format_benchmark_result(dec_res))
    else:
        res = benchmark_camera(cfg, duration_s=args.duration, decode=not args.no_decode)
        print(format_benchmark_result(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
