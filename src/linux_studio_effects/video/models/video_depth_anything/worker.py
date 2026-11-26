"""Shared non-blocking background Video Depth Anything worker."""

from __future__ import annotations

import threading
import time

import numpy as np

from ...pixels import model_luma_input
from . import npu_ops
from .model import InferenceResult, VideoDepthAnythingModel


class AsyncDepthWorker:
    """Runs VideoDepthAnythingModel asynchronously so live 30fps effects never stall."""

    def __init__(self, interval_s: float = 0.05, *, thread_name: str = "vda-depth-worker") -> None:
        self.interval_s = float(interval_s)
        self._thread_name = thread_name
        self._lock = threading.Lock()
        self._latest_frame: tuple[np.ndarray, str, int, float] | None = None
        self._latest_result: InferenceResult | None = None
        self._updates: int = 0
        self._seen_frames: int = 0
        self._result_frame_idx: int = 0
        self._result_capture_t: float | None = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def updates(self) -> int:
        with self._lock:
            return self._updates

    @property
    def latest_result(self) -> InferenceResult | None:
        with self._lock:
            return self._latest_result

    @property
    def lag_frames(self) -> int:
        """Number of input video frames the active depth map lags behind the latest frame."""
        with self._lock:
            if self._result_frame_idx <= 0:
                return 0
            return max(0, self._seen_frames - self._result_frame_idx)

    @property
    def lag_ms(self) -> float:
        """Age (ms) from when the depth-source frame was received until now."""
        with self._lock:
            t = self._result_capture_t
        if t is None:
            return 0.0
        return max(0.0, (time.perf_counter() - t) * 1000.0)

    def start(self, model: VideoDepthAnythingModel) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._ready_event.clear()

        def _loop() -> None:
            npu_ops.set_npu_worker_thread(True)
            try:
                while not self._stop_event.is_set():
                    if not self._ready_event.wait(timeout=0.05):
                        continue
                    self._ready_event.clear()
                    with self._lock:
                        snap = self._latest_frame
                    if snap is None:
                        continue
                    raw_frame, raw_fmt, src_idx, src_t = snap
                    t0 = time.perf_counter()
                    res = model.infer(model_luma_input(raw_frame, raw_fmt))
                    with self._lock:
                        self._latest_result = res
                        self._result_frame_idx = src_idx
                        self._result_capture_t = src_t
                        self._updates += 1
                    rem = self.interval_s - (time.perf_counter() - t0)
                    if rem > 0.0:
                        self._stop_event.wait(rem)
            finally:
                npu_ops.set_npu_worker_thread(False)

        self._thread = threading.Thread(target=_loop, daemon=True, name=self._thread_name)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._ready_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        with self._lock:
            self._latest_frame = None
            self._latest_result = None
            self._updates = 0
            self._seen_frames = 0
            self._result_frame_idx = 0
            self._result_capture_t = None

    def step(
        self,
        model: VideoDepthAnythingModel,
        frame: np.ndarray,
        pixel_format: str,
        *,
        async_enabled: bool,
    ) -> InferenceResult:
        """Return latest depth result synchronously or via background worker."""
        now = time.perf_counter()
        if not async_enabled:
            res = model.infer(model_luma_input(frame, pixel_format))
            with self._lock:
                self._seen_frames += 1
                self._latest_result = res
                self._result_frame_idx = self._seen_frames
                self._result_capture_t = now
                self._updates += 1
            return res

        frame_copy = frame.copy()
        with self._lock:
            self._seen_frames += 1
            cur_idx = self._seen_frames
            cur = self._latest_result
            self._latest_frame = (frame_copy, pixel_format, cur_idx, now)
        if cur is None:
            res = model.infer(model_luma_input(frame_copy, pixel_format))
            with self._lock:
                self._latest_result = res
                self._result_frame_idx = cur_idx
                self._result_capture_t = now
                self._updates += 1
            self.start(model)
            return res
        self._ready_event.set()
        return cur
