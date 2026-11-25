"""Shared non-blocking background MODNet worker for segmentation and auto-framing effects."""

from __future__ import annotations

import threading
import time

import numpy as np

from ...pixels import model_luma_input
from .model import ModnetModel, SegmentationResult


class AsyncModnetWorker:
    """Runs ModnetModel inference asynchronously so live 30fps effects never stall."""

    def __init__(self, interval_s: float = 0.05, *, thread_name: str = "modnet-worker") -> None:
        self.interval_s = float(interval_s)
        self._thread_name = thread_name
        self._lock = threading.Lock()
        self._latest_frame: tuple[np.ndarray, str] | None = None
        self._latest_result: SegmentationResult | None = None
        self._updates: int = 0
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def updates(self) -> int:
        with self._lock:
            return self._updates

    @property
    def latest_result(self) -> SegmentationResult | None:
        with self._lock:
            return self._latest_result

    def start(self, model: ModnetModel) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._ready_event.clear()

        def _loop() -> None:
            while not self._stop_event.is_set():
                if not self._ready_event.wait(timeout=0.05):
                    continue
                self._ready_event.clear()
                with self._lock:
                    snap = self._latest_frame
                if snap is None:
                    continue
                raw_frame, raw_fmt = snap
                t0 = time.perf_counter()
                model_input = model_luma_input(raw_frame, raw_fmt)
                res = model.infer(model_input)
                with self._lock:
                    self._latest_result = res
                    self._updates += 1
                rem = self.interval_s - (time.perf_counter() - t0)
                if rem > 0.0:
                    self._stop_event.wait(rem)

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

    def step(
        self,
        model: ModnetModel,
        frame: np.ndarray,
        pixel_format: str,
        *,
        async_enabled: bool,
    ) -> SegmentationResult:
        """Return the latest segmentation result synchronously or via background worker."""
        if not async_enabled:
            model_input = model_luma_input(frame, pixel_format)
            res = model.infer(model_input)
            with self._lock:
                self._latest_result = res
                self._updates += 1
            return res

        frame_copy = frame.copy()
        with self._lock:
            cur = self._latest_result
            self._latest_frame = (frame_copy, pixel_format)
        if cur is None:
            model_input = model_luma_input(frame_copy, pixel_format)
            res = model.infer(model_input)
            with self._lock:
                self._latest_result = res
                self._updates += 1
            self.start(model)
            return res
        self._ready_event.set()
        return cur
