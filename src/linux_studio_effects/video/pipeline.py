"""Configurable camera input -> frame processor -> virtual camera output pipeline."""

from __future__ import annotations

import argparse
import statistics
import threading
import time
from contextlib import ExitStack
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .camera import CameraConfig, V4L2Camera
from .effects import BUILTIN_EFFECTS, EffectChain, FrameContext, FrameEffect, as_frame_effect
from .pixels import NATIVE_PIXEL_FORMATS, is_yuv422, resize_native
from .virtual_cam import VirtualCameraConfig, VirtualCameraOutput

FrameProcessor = FrameEffect | Callable[[np.ndarray], np.ndarray]
StatsCallback = Callable[["PipelineStats"], None]


@dataclass(frozen=True)
class PipelineConfig:
    """End-to-end camera input and virtual camera output stream configuration."""

    input_device: str = ""
    input_width: int = 1280
    input_height: int = 720
    input_format: str = "YUYV"
    input_fps: float = 30.0
    output_device: str = ""
    output_width: int = 1280
    output_height: int = 720
    output_format: str = "YUYV"
    output_fps: float = 30.0
    resize_mode: str = "cover"


@dataclass(frozen=True)
class PipelineStats:
    """Throughput, latency, and CPU utilization summary for a pipeline run."""

    input_config: CameraConfig
    output_config: VirtualCameraConfig
    frames_in: int
    frames_out: int
    dropped_in: int
    elapsed_s: float
    input_fps: float
    output_fps: float
    avg_process_ms: float
    cpu_percent: float
    avg_lag_ms: float = 0.0


def _colorimetry_signature(config: CameraConfig | VirtualCameraConfig, pixel_format: str) -> tuple[int, int, int]:
    """Resolve V4L2 default YCbCr/range values for the same native format.

    v4l2loopback commonly reports XFER_FUNC_DEFAULT even when it was asked to
    preserve the capture device's explicit transfer-function value. The encoded
    YCbCr matrix and quantization range are the fields that determine how these
    unchanged packed YUV bytes are interpreted by V4L2 consumers.
    """
    colorspace = config.colorspace
    ycbcr_enc = config.ycbcr_enc
    if ycbcr_enc == 0:
        ycbcr_enc = {
            2: 8,  # SMPTE 240M
            3: 2,  # Rec. 709
            10: 6,  # BT.2020
            12: 2,  # DCI-P3
        }.get(colorspace, 1)  # Default for SRGB/SMPTE170M/JPEG/etc. is 601.
    quantization = config.quantization
    if quantization == 0:
        quantization = 1 if pixel_format in {"RGB3", "BGR3"} or colorspace == 7 else 2
    return colorspace, ycbcr_enc, quantization


class VideoPipeline:
    """Streams frames from a physical V4L2 camera into a v4l2loopback virtual camera."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        frame_processor: FrameProcessor | None = None,
        effect: FrameEffect | list[FrameProcessor] | tuple[FrameProcessor, ...] | None = None,
    ) -> None:
        self._config = config or PipelineConfig()
        chosen = effect if effect is not None else frame_processor
        if chosen is None:
            self._effect: FrameEffect | None = None
        elif isinstance(chosen, (list, tuple)):
            self._effect = EffectChain(chosen)
        else:
            self._effect = as_frame_effect(chosen)
        self._stop_event = threading.Event()
        self._negotiated_in: CameraConfig | None = None
        self._negotiated_out: VirtualCameraConfig | None = None

    @property
    def effect(self) -> FrameEffect | None:
        return self._effect

    @property
    def negotiated_input(self) -> CameraConfig | None:
        return self._negotiated_in

    @property
    def negotiated_output(self) -> VirtualCameraConfig | None:
        return self._negotiated_out

    def stop(self) -> None:
        """Signal a running pipeline loop to exit cleanly after the current frame."""
        self._stop_event.set()

    def run(
        self,
        *,
        duration_s: float | None = None,
        max_frames: int | None = None,
        stats_interval_s: float = 1.0,
        on_stats: StatsCallback | None = None,
    ) -> PipelineStats:
        """Run the capture -> process -> resize -> virtual-cam output loop."""
        self._stop_event.clear()
        cfg = self._config

        input_format = cfg.input_format.upper()
        output_format = cfg.output_format.upper()
        if input_format not in (*NATIVE_PIXEL_FORMATS, "MJPG"):
            raise ValueError(
                f"Unsupported input format {input_format}; choose one of {(*NATIVE_PIXEL_FORMATS, 'MJPG')}"
            )
        if output_format not in NATIVE_PIXEL_FORMATS:
            raise ValueError(f"Unsupported native output format {output_format}; choose one of {NATIVE_PIXEL_FORMATS}")
        if input_format != "MJPG" and input_format != output_format:
            raise ValueError(
                f"Input {input_format} and output {output_format} differ; "
                "pixel/color-space conversion is disabled. Use the same format at both ends."
            )
        in_cfg = CameraConfig(
            device=cfg.input_device,
            width=cfg.input_width,
            height=cfg.input_height,
            pixel_format=input_format,
            fps=cfg.input_fps,
        )

        with ExitStack() as stack:
            cam = stack.enter_context(V4L2Camera(in_cfg))
            out_cfg = VirtualCameraConfig(
                device=cfg.output_device,
                width=cfg.output_width,
                height=cfg.output_height,
                pixel_format=output_format,
                fps=cfg.output_fps,
                colorspace=cam.negotiated.colorspace,
                ycbcr_enc=cam.negotiated.ycbcr_enc,
                quantization=1 if input_format == "MJPG" else cam.negotiated.quantization,
                xfer_func=cam.negotiated.xfer_func,
            )
            vcam = stack.enter_context(VirtualCameraOutput(out_cfg))
            in_colorimetry = _colorimetry_signature(cam.negotiated, input_format) if input_format != "MJPG" else None
            out_colorimetry = _colorimetry_signature(vcam.negotiated, output_format)
            if in_colorimetry is not None and in_colorimetry != out_colorimetry:
                raise RuntimeError(
                    f"Input/output colorspace, YCbCr encoding, or quantization differs "
                    f"({in_colorimetry} != {out_colorimetry}); "
                    "color conversion is disabled"
                )
            self._negotiated_in = cam.negotiated
            self._negotiated_out = vcam.negotiated

            if self._effect is not None:
                self._effect.setup(cam.negotiated.width, cam.negotiated.height, cam.negotiated.fps)

            wall_start = time.monotonic()
            cpu_start = time.process_time()
            last_report_wall = wall_start
            last_report_cpu = cpu_start
            last_report_in = 0
            last_report_out = 0

            frames_in = 0
            frames_out = 0
            dropped_in = 0
            proc_ms_samples: list[float] = []
            lag_ms_samples: list[float] = []

            cam_fps = cam.negotiated.fps if cam.negotiated.fps > 0 else cfg.input_fps
            emit_every_frame = cfg.output_fps <= 0.0 or (cam_fps > 0.0 and cfg.output_fps >= cam_fps - 0.5)
            cam_period = (1.0 / cam_fps) if cam_fps > 0.0 else 0.0
            last_frame_now = wall_start
            target_out_period = (1.0 / cfg.output_fps) if cfg.output_fps > 0 else 0.0
            next_out_deadline = wall_start

            def _render_and_write(frame: Any, frame_idx: int) -> None:
                nonlocal frames_out
                if frame.data is None:
                    return
                frame_now = frame.capture_monotonic_s
                t0 = time.perf_counter()
                if self._effect is not None and self._effect.enabled:
                    ctx = FrameContext(
                        frame_index=frame_idx,
                        timestamp_s=frame.driver_timestamp_s,
                        width=frame.width,
                        height=frame.height,
                        fps=cam.negotiated.fps,
                        pixel_format=output_format,
                    )
                    out_frame = self._effect.process(frame.data, ctx)
                else:
                    out_frame = frame.data
                if out_frame.shape[0] != vcam.negotiated.height or (
                    out_frame.shape[1] * 2 if is_yuv422(output_format) else out_frame.shape[1]
                ) != vcam.negotiated.width:
                    out_frame = resize_native(
                        out_frame,
                        vcam.negotiated.width,
                        vcam.negotiated.height,
                        output_format,
                        mode=cfg.resize_mode,
                    )
                vcam.write_frame(out_frame)
                end_now = time.perf_counter()
                proc_ms_samples.append((end_now - t0) * 1000.0)
                lag_ms_samples.append(max(0.0, (time.monotonic() - frame_now) * 1000.0))
                frames_out += 1

            def _maybe_report_stats(frame_now: float, cur_in: int, cur_drop: int) -> None:
                nonlocal last_report_wall, last_report_cpu, last_report_in, last_report_out
                if on_stats is None or stats_interval_s <= 0 or (frame_now - last_report_wall) < stats_interval_s:
                    return
                dt = max(frame_now - last_report_wall, 1e-9)
                cpu_now = time.process_time()
                d_cpu = max(cpu_now - last_report_cpu, 0.0)
                on_stats(
                    PipelineStats(
                        input_config=cam.negotiated,
                        output_config=vcam.negotiated,
                        frames_in=cur_in - last_report_in,
                        frames_out=frames_out - last_report_out,
                        dropped_in=cur_drop,
                        elapsed_s=dt,
                        input_fps=(cur_in - last_report_in) / dt,
                        output_fps=(frames_out - last_report_out) / dt,
                        avg_process_ms=statistics.fmean(proc_ms_samples[-30:]) if proc_ms_samples else 0.0,
                        cpu_percent=(d_cpu / dt) * 100.0,
                        avg_lag_ms=statistics.fmean(lag_ms_samples[-30:]) if lag_ms_samples else 0.0,
                    )
                )
                last_report_wall = frame_now
                last_report_cpu = cpu_now
                last_report_in = cur_in
                last_report_out = frames_out

            try:
                if emit_every_frame and self._effect is not None:
                    slot_cond = threading.Condition()
                    latest_item: tuple[Any, int] | None = None
                    reader_stop = threading.Event()
                    reader_done = False
                    capture_exc: BaseException | None = None

                    def _capture_loop() -> None:
                        nonlocal frames_in, dropped_in, latest_item, reader_done, capture_exc
                        try:
                            while not reader_stop.is_set() and not self._stop_event.is_set():
                                if duration_s is not None and (time.monotonic() - wall_start) >= duration_s:
                                    break
                                fr = cam.read_frame(decode=True, native=True, output_format=output_format)
                                with slot_cond:
                                    frames_in += 1
                                    dropped_in += fr.dropped_since_last
                                    latest_item = (fr, frames_in)
                                    slot_cond.notify()
                        except BaseException as exc:  # pragma: no cover - surfaced on main thread
                            capture_exc = exc
                        finally:
                            with slot_cond:
                                reader_done = True
                                slot_cond.notify_all()

                    reader_thread = threading.Thread(target=_capture_loop, daemon=True, name="lse-capture-decode")
                    reader_thread.start()
                    try:
                        while not self._stop_event.is_set():
                            now = time.monotonic()
                            if duration_s is not None and (now - wall_start) >= duration_s:
                                break
                            if max_frames is not None and frames_out >= max_frames:
                                break
                            with slot_cond:
                                while latest_item is None and not reader_done and not self._stop_event.is_set() and not reader_stop.is_set():
                                    slot_cond.wait(timeout=0.05)
                                    if duration_s is not None and (time.monotonic() - wall_start) >= duration_s:
                                        break
                                item = latest_item
                                latest_item = None
                                cur_frames_in = frames_in
                                cur_dropped_in = dropped_in
                            if item is None:
                                if reader_done or (duration_s is not None and (time.monotonic() - wall_start) >= duration_s):
                                    break
                                continue
                            frame, frame_idx = item
                            frame_now = frame.capture_monotonic_s
                            last_frame_now = frame_now
                            _render_and_write(frame, frame_idx)
                            _maybe_report_stats(frame_now, cur_frames_in, cur_dropped_in)
                    finally:
                        reader_stop.set()
                        with slot_cond:
                            slot_cond.notify_all()
                        reader_thread.join(timeout=2.0)
                        if capture_exc is not None:
                            raise capture_exc
                else:
                    while not self._stop_event.is_set():
                        now = time.monotonic()
                        if duration_s is not None and (now - wall_start) >= duration_s:
                            break
                        if max_frames is not None and frames_out >= max_frames:
                            break

                        # Emit every captured frame whenever target output_fps matches camera fps;
                        # only downsample when explicitly outputting slower than camera rate.
                        if emit_every_frame:
                            should_emit = True
                        else:
                            est_arrival = max(now, last_frame_now + cam_period)
                            should_emit = est_arrival >= next_out_deadline - 0.002
                        frame = cam.read_frame(decode=should_emit, native=True, output_format=output_format)
                        frames_in += 1
                        dropped_in += frame.dropped_since_last
                        frame_now = frame.capture_monotonic_s
                        last_frame_now = frame_now

                        if should_emit and frame.data is not None:
                            _render_and_write(frame, frames_in)
                            if not emit_every_frame and target_out_period > 0.0:
                                next_out_deadline += target_out_period
                                if next_out_deadline < frame_now:
                                    next_out_deadline = frame_now + target_out_period

                        _maybe_report_stats(frame_now, frames_in, dropped_in)
            except KeyboardInterrupt:
                self._stop_event.set()
            finally:
                if self._effect is not None:
                    self._effect.teardown()

            total_wall = max(time.monotonic() - wall_start, 1e-9)
            total_cpu = max(time.process_time() - cpu_start, 0.0)
            return PipelineStats(
                input_config=cam.negotiated,
                output_config=vcam.negotiated,
                frames_in=frames_in,
                frames_out=frames_out,
                dropped_in=dropped_in,
                elapsed_s=total_wall,
                input_fps=frames_in / total_wall,
                output_fps=frames_out / total_wall,
                avg_process_ms=statistics.fmean(proc_ms_samples) if proc_ms_samples else 0.0,
                cpu_percent=(total_cpu / total_wall) * 100.0,
                avg_lag_ms=statistics.fmean(lag_ms_samples) if lag_ms_samples else 0.0,
            )


def _parse_size(text: str) -> tuple[int, int]:
    parts = text.lower().split("x", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected WxH size, got {text!r}")
    return int(parts[0]), int(parts[1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run configurable V4L2 camera -> v4l2loopback pipeline.")
    parser.add_argument("--in-device", default="", help="Input camera path (default: auto-detect)")
    parser.add_argument("--in-size", type=_parse_size, default=(1280, 720), help="Input WxH (default: 1280x720)")
    parser.add_argument("--in-format", default="YUYV", help="Uncompressed native input FourCC (default: YUYV)")
    parser.add_argument("--in-fps", type=float, default=30.0, help="Input FPS (default: 30)")
    parser.add_argument("--out-device", default="", help="Output v4l2loopback path (default: auto-detect)")
    parser.add_argument("--out-size", type=_parse_size, default=(1280, 720), help="Output WxH (default: 1280x720)")
    parser.add_argument("--out-format", default="YUYV", help="Output FourCC; must match --in-format exactly")
    parser.add_argument("--out-fps", type=float, default=30.0, help="Output FPS (default: 30)")
    parser.add_argument("--resize-mode", choices=("cover", "stretch"), default="cover")
    parser.add_argument(
        "--effect",
        choices=tuple(BUILTIN_EFFECTS.keys()),
        default="none",
        help="Built-in CPU frame effect to apply (default: none)",
    )
    parser.add_argument("--duration", type=float, default=0.0, help="Run duration in seconds (0 = until Ctrl+C)")
    args = parser.parse_args(argv)

    cfg = PipelineConfig(
        input_device=args.in_device,
        input_width=args.in_size[0],
        input_height=args.in_size[1],
        input_format=args.in_format.upper(),
        input_fps=args.in_fps,
        output_device=args.out_device,
        output_width=args.out_size[0],
        output_height=args.out_size[1],
        output_format=args.out_format.upper(),
        output_fps=args.out_fps,
        resize_mode=args.resize_mode,
    )
    pipe = VideoPipeline(cfg, effect=BUILTIN_EFFECTS[args.effect]())

    def _print_tick(st: PipelineStats) -> None:
        print(
            f"[pipeline] in={st.input_fps:.1f} fps ({st.input_config.width}x{st.input_config.height} {st.input_config.pixel_format}) "
            f"-> out={st.output_fps:.1f} fps ({st.output_config.width}x{st.output_config.height} {st.output_config.pixel_format} @ {st.output_config.device}) "
            f"| proc+write={st.avg_process_ms:.2f} ms | CPU={st.cpu_percent:.1f}%",
            flush=True,
        )

    try:
        summary = pipe.run(
            duration_s=args.duration if args.duration > 0 else None,
            on_stats=_print_tick,
        )
    except KeyboardInterrupt:
        pipe.stop()
        return 0

    print(
        f"\n[summary] {summary.frames_in} frames in ({summary.input_fps:.2f} fps), "
        f"{summary.frames_out} frames out ({summary.output_fps:.2f} fps), "
        f"dropped_in={summary.dropped_in}, avg_proc={summary.avg_process_ms:.2f} ms, "
        f"CPU={summary.cpu_percent:.1f}% (100% = 1 full core)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
