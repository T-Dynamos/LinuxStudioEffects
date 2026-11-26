"""V4L2 memory-mapped (MMAP) camera capture pipeline."""

from __future__ import annotations

import ctypes
import mmap
import os
import select
import time
from dataclasses import dataclass

import numpy as np

from .decoder import FrameDecoder
from .v4l2 import (
    V4L2_BUF_TYPE_VIDEO_CAPTURE,
    V4L2_CAP_DEVICE_CAPS,
    V4L2_CAP_STREAMING,
    V4L2_CAP_VIDEO_CAPTURE,
    V4L2_CID_EXPOSURE_AUTO_PRIORITY,
    V4L2_FIELD_ANY,
    V4L2_MEMORY_MMAP,
    VIDIOC_DQBUF,
    VIDIOC_G_PARM,
    VIDIOC_QBUF,
    VIDIOC_QUERYBUF,
    VIDIOC_QUERYCAP,
    VIDIOC_REQBUFS,
    VIDIOC_S_CTRL,
    VIDIOC_S_FMT,
    VIDIOC_S_PARM,
    VIDIOC_STREAMOFF,
    VIDIOC_STREAMON,
    fourcc_to_str,
    pick_default_camera,
    str_to_fourcc,
    v4l2_buffer,
    v4l2_capability,
    v4l2_control,
    v4l2_format,
    v4l2_requestbuffers,
    v4l2_streamparm,
    xioctl,
)


@dataclass(frozen=True)
class CameraConfig:
    """Requested or negotiated V4L2 camera stream parameters."""

    device: str = ""
    width: int = 1280
    height: int = 720
    pixel_format: str = "MJPG"
    fps: float = 30.0
    buffer_count: int = 4
    bytesperline: int = 0
    sizeimage: int = 0
    colorspace: int = 0
    ycbcr_enc: int = 0
    quantization: int = 0
    xfer_func: int = 0


@dataclass(frozen=True)
class CapturedFrame:
    """Single captured frame with timing and V4L2 metadata."""

    data: np.ndarray | None
    raw_size: int
    width: int
    height: int
    pixel_format: str
    sequence: int
    driver_timestamp_s: float
    capture_monotonic_s: float
    decode_time_s: float
    dropped_since_last: int = 0


class V4L2Camera:
    """Streaming V4L2 camera reader using kernel MMAP ring buffers."""

    def __init__(self, config: CameraConfig | None = None, *, fast_dct: bool = True) -> None:
        self._requested = config or CameraConfig()
        self._negotiated: CameraConfig | None = None
        self._fd: int | None = None
        self._mmaps: list[mmap.mmap] = []
        self._streaming = False
        self._decoder = FrameDecoder(fast_dct=fast_dct)
        self._last_sequence: int | None = None

    @property
    def negotiated(self) -> CameraConfig:
        """Return actual format/fps negotiated with the V4L2 driver."""
        if self._negotiated is None:
            raise RuntimeError("Camera is not open")
        return self._negotiated

    def open(self) -> CameraConfig:
        """Open device, configure format/fps, allocate MMAP buffers, and start streaming."""
        if self._fd is not None:
            return self.negotiated

        dev_path = self._requested.device
        if not dev_path:
            default_dev = pick_default_camera()
            if default_dev is None:
                raise RuntimeError("No accessible V4L2 capture camera found")
            dev_path = default_dev.path

        fd = os.open(dev_path, os.O_RDWR | os.O_NONBLOCK)
        try:
            cap = v4l2_capability()
            xioctl(fd, VIDIOC_QUERYCAP, cap)
            effective_caps = cap.device_caps if (cap.capabilities & V4L2_CAP_DEVICE_CAPS) else cap.capabilities
            if not (effective_caps & V4L2_CAP_VIDEO_CAPTURE) or not (effective_caps & V4L2_CAP_STREAMING):
                raise RuntimeError(f"{dev_path} does not support V4L2 streaming video capture")

            fmt = v4l2_format()
            fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            fmt.fmt.pix.width = self._requested.width
            fmt.fmt.pix.height = self._requested.height
            fmt.fmt.pix.pixelformat = str_to_fourcc(self._requested.pixel_format)
            fmt.fmt.pix.field = V4L2_FIELD_ANY
            xioctl(fd, VIDIOC_S_FMT, fmt)
            actual_format = fourcc_to_str(fmt.fmt.pix.pixelformat)
            if actual_format != self._requested.pixel_format.upper():
                raise RuntimeError(
                    f"{dev_path} negotiated {actual_format} instead of requested "
                    f"{self._requested.pixel_format.upper()}; format conversion is disabled"
                )
            if (fmt.fmt.pix.width, fmt.fmt.pix.height) != (self._requested.width, self._requested.height):
                raise RuntimeError(
                    f"{dev_path} negotiated {fmt.fmt.pix.width}x{fmt.fmt.pix.height} instead of requested "
                    f"{self._requested.width}x{self._requested.height}; resolution changes are disabled"
                )

            actual_fps = self._configure_fps(fd, self._requested.fps)

            req = v4l2_requestbuffers()
            req.count = self._requested.buffer_count
            req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            req.memory = V4L2_MEMORY_MMAP
            xioctl(fd, VIDIOC_REQBUFS, req)
            if req.count < 2:
                raise RuntimeError(f"Insufficient V4L2 MMAP buffers allocated ({req.count})")

            mmaps: list[mmap.mmap] = []
            for idx in range(req.count):
                buf = v4l2_buffer()
                buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
                buf.memory = V4L2_MEMORY_MMAP
                buf.index = idx
                xioctl(fd, VIDIOC_QUERYBUF, buf)

                mm = mmap.mmap(
                    fd,
                    buf.length,
                    flags=mmap.MAP_SHARED,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE,
                    offset=buf.m.offset,
                )
                mmaps.append(mm)
                xioctl(fd, VIDIOC_QBUF, buf)

            buf_type = ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE)
            xioctl(fd, VIDIOC_STREAMON, buf_type)

            self._fd = fd
            self._mmaps = mmaps
            self._streaming = True
            self._last_sequence = None
            self._negotiated = CameraConfig(
                device=dev_path,
                width=fmt.fmt.pix.width,
                height=fmt.fmt.pix.height,
                pixel_format=fourcc_to_str(fmt.fmt.pix.pixelformat),
                fps=actual_fps,
                buffer_count=req.count,
                bytesperline=fmt.fmt.pix.bytesperline,
                sizeimage=fmt.fmt.pix.sizeimage,
                colorspace=fmt.fmt.pix.colorspace,
                ycbcr_enc=fmt.fmt.pix.ycbcr_enc,
                quantization=fmt.fmt.pix.quantization,
                xfer_func=fmt.fmt.pix.xfer_func,
            )
            return self._negotiated
        except Exception:
            self._cleanup_fd_and_mmaps(fd, locals().get("mmaps", []))
            raise

    def _configure_fps(self, fd: int, target_fps: float) -> float:
        parm = v4l2_streamparm()
        parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        try:
            xioctl(fd, VIDIOC_G_PARM, parm)
        except OSError:
            return target_fps

        if target_fps > 0:
            parm.parm.capture.timeperframe.numerator = 1000
            parm.parm.capture.timeperframe.denominator = max(1, int(round(target_fps * 1000)))
            try:
                xioctl(fd, VIDIOC_S_PARM, parm)
            except OSError:
                pass
        if target_fps >= 20.0:
            # Prevent UVC dynamic exposure priority from silently halving sensor FPS in indoor light.
            ctrl = v4l2_control()
            ctrl.id = V4L2_CID_EXPOSURE_AUTO_PRIORITY
            ctrl.value = 0
            try:
                xioctl(fd, VIDIOC_S_CTRL, ctrl)
            except OSError:
                pass

        num = parm.parm.capture.timeperframe.numerator
        den = parm.parm.capture.timeperframe.denominator
        if num > 0 and den > 0:
            return round(den / num, 3)
        return target_fps

    def read_frame(
        self,
        *,
        decode: bool = True,
        native: bool = False,
        output_format: str | None = None,
        timeout_s: float = 2.0,
    ) -> CapturedFrame:
        """Wait for and return the next frame from the V4L2 MMAP queue."""
        if self._fd is None or not self._streaming or self._negotiated is None:
            raise RuntimeError("Camera stream is not open")

        poller = select.poll()
        poller.register(self._fd, select.POLLIN)
        events = poller.poll(int(timeout_s * 1000))
        if not events:
            raise TimeoutError(f"Timed out waiting for frame on {self._negotiated.device}")

        buf = v4l2_buffer()
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = V4L2_MEMORY_MMAP
        xioctl(self._fd, VIDIOC_DQBUF, buf)

        capture_t = time.monotonic()
        driver_ts = buf.timestamp.tv_sec + buf.timestamp.tv_usec * 1e-6
        seq = int(buf.sequence)
        dropped = 0
        if self._last_sequence is not None and seq > self._last_sequence + 1:
            dropped = seq - self._last_sequence - 1
        self._last_sequence = seq

        decode_t = 0.0
        frame_arr: np.ndarray | None = None
        raw_size = int(buf.bytesused)

        try:
            if decode and raw_size > 0:
                mm = self._mmaps[buf.index]
                t0 = time.perf_counter()
                with memoryview(mm)[:raw_size] as view:
                    decode_frame = self._decoder.decode_native if native else self._decoder.decode
                    kwargs = {"bytesperline": self._negotiated.bytesperline, "output_format": output_format} if native else {}
                    frame_arr = decode_frame(
                        view,
                        self._negotiated.width,
                        self._negotiated.height,
                        self._negotiated.pixel_format,
                        **kwargs,
                    )
                decode_t = time.perf_counter() - t0
        finally:
            xioctl(self._fd, VIDIOC_QBUF, buf)

        return CapturedFrame(
            data=frame_arr,
            raw_size=raw_size,
            width=self._negotiated.width,
            height=self._negotiated.height,
            pixel_format=self._negotiated.pixel_format,
            sequence=seq,
            driver_timestamp_s=driver_ts,
            capture_monotonic_s=capture_t,
            decode_time_s=decode_t,
            dropped_since_last=dropped,
        )

    def close(self) -> None:
        """Stop streaming, unmap kernel buffers, and close the device."""
        if self._fd is not None:
            if self._streaming:
                buf_type = ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE)
                try:
                    xioctl(self._fd, VIDIOC_STREAMOFF, buf_type)
                except OSError:
                    pass
                self._streaming = False
            self._cleanup_fd_and_mmaps(self._fd, self._mmaps)
            self._fd = None
            self._mmaps = []
        self._decoder.close()

    @staticmethod
    def _cleanup_fd_and_mmaps(fd: int, mmaps: list[mmap.mmap]) -> None:
        for mm in mmaps:
            try:
                mm.close()
            except BufferError:
                pass
        try:
            req = v4l2_requestbuffers()
            req.count = 0
            req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            req.memory = V4L2_MEMORY_MMAP
            xioctl(fd, VIDIOC_REQBUFS, req)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass

    def __enter__(self) -> V4L2Camera:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
