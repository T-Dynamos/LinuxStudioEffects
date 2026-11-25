"""Low-level Linux V4L2 ioctl bindings and device enumeration via ctypes."""

from __future__ import annotations

import ctypes
import fcntl
import glob
import os
from dataclasses import dataclass

# --- ioctl encoding (asm-generic/ioctl.h on Linux x86_64) ---
_IOC_NRBITS = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14
_IOC_DIRBITS = 2

_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS

_IOC_NONE = 0
_IOC_WRITE = 1
_IOC_READ = 2


def _ioc(direction: int, ioc_type: str, nr: int, size: int) -> int:
    return (
        (direction << _IOC_DIRSHIFT)
        | (ord(ioc_type) << _IOC_TYPESHIFT)
        | (nr << _IOC_NRSHIFT)
        | (size << _IOC_SIZESHIFT)
    )


def _ior(ioc_type: str, nr: int, struct_cls: type[ctypes.Structure]) -> int:
    return _ioc(_IOC_READ, ioc_type, nr, ctypes.sizeof(struct_cls))


def _iow(ioc_type: str, nr: int, struct_cls: type[ctypes.Structure | ctypes._SimpleCData]) -> int:
    return _ioc(_IOC_WRITE, ioc_type, nr, ctypes.sizeof(struct_cls))


def _iowr(ioc_type: str, nr: int, struct_cls: type[ctypes.Structure]) -> int:
    return _ioc(_IOC_READ | _IOC_WRITE, ioc_type, nr, ctypes.sizeof(struct_cls))


# --- V4L2 constants ---
V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_BUF_TYPE_VIDEO_OUTPUT = 2
V4L2_MEMORY_MMAP = 1
V4L2_FIELD_ANY = 0
V4L2_FIELD_NONE = 1
V4L2_COLORSPACE_SRGB = 8

V4L2_CAP_VIDEO_CAPTURE = 0x00000001
V4L2_CAP_VIDEO_OUTPUT = 0x00000002
V4L2_CAP_EXT_PIX_FORMAT = 0x00200000
V4L2_CAP_META_CAPTURE = 0x00800000
V4L2_CAP_READWRITE = 0x01000000
V4L2_CAP_STREAMING = 0x04000000
V4L2_CAP_DEVICE_CAPS = 0x80000000

V4L2_CTRL_FLAG_DISABLED = 0x0001
V4L2_CTRL_FLAG_READ_ONLY = 0x0004
V4L2_CTRL_FLAG_INACTIVE = 0x0010
V4L2_CTRL_FLAG_HAS_PAYLOAD = 0x0100
V4L2_CTRL_FLAG_NEXT_CTRL = 0x80000000

V4L2_CTRL_TYPE_NAMES = {
    1: "int",
    2: "bool",
    3: "menu",
    4: "button",
    5: "int64",
    6: "ctrl_class",
    7: "string",
    8: "bitmask",
    9: "int_menu",
}

V4L2_FRMSIZE_TYPE_DISCRETE = 1
V4L2_FRMSIZE_TYPE_CONTINUOUS = 2
V4L2_FRMSIZE_TYPE_STEPWISE = 3

V4L2_FRMIVAL_TYPE_DISCRETE = 1
V4L2_FRMIVAL_TYPE_CONTINUOUS = 2
V4L2_FRMIVAL_TYPE_STEPWISE = 3


def str_to_fourcc(code: str) -> int:
    """Convert a 4-character pixel format string (e.g. 'MJPG') to V4L2 fourcc uint32."""
    if len(code) != 4:
        raise ValueError(f"FourCC must be 4 characters, got {code!r}")
    b = code.encode("ascii")
    return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)


def fourcc_to_str(value: int) -> str:
    """Convert a V4L2 fourcc uint32 into a 4-character ASCII string."""
    chars = [
        chr(value & 0xFF),
        chr((value >> 8) & 0xFF),
        chr((value >> 16) & 0xFF),
        chr((value >> 24) & 0xFF),
    ]
    return "".join(chars)


# --- V4L2 structures matching Linux x86_64 <linux/videodev2.h> ---
class v4l2_fract(ctypes.Structure):
    _fields_ = [
        ("numerator", ctypes.c_uint32),
        ("denominator", ctypes.c_uint32),
    ]


class v4l2_capability(ctypes.Structure):
    _fields_ = [
        ("driver", ctypes.c_uint8 * 16),
        ("card", ctypes.c_uint8 * 32),
        ("bus_info", ctypes.c_uint8 * 32),
        ("version", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("device_caps", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
    ]


class v4l2_fmtdesc(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("description", ctypes.c_uint8 * 32),
        ("pixelformat", ctypes.c_uint32),
        ("mbus_code", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
    ]


class v4l2_frmsize_discrete(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
    ]


class v4l2_frmsize_stepwise(ctypes.Structure):
    _fields_ = [
        ("min_width", ctypes.c_uint32),
        ("max_width", ctypes.c_uint32),
        ("step_width", ctypes.c_uint32),
        ("min_height", ctypes.c_uint32),
        ("max_height", ctypes.c_uint32),
        ("step_height", ctypes.c_uint32),
    ]


class _v4l2_frmsize_union(ctypes.Union):
    _fields_ = [
        ("discrete", v4l2_frmsize_discrete),
        ("stepwise", v4l2_frmsize_stepwise),
    ]


class v4l2_frmsizeenum(ctypes.Structure):
    _anonymous_ = ("_size",)
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("pixel_format", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("_size", _v4l2_frmsize_union),
        ("reserved", ctypes.c_uint32 * 2),
    ]


class v4l2_frmival_stepwise(ctypes.Structure):
    _fields_ = [
        ("min", v4l2_fract),
        ("max", v4l2_fract),
        ("step", v4l2_fract),
    ]


class _v4l2_frmival_union(ctypes.Union):
    _fields_ = [
        ("discrete", v4l2_fract),
        ("stepwise", v4l2_frmival_stepwise),
    ]


class v4l2_frmivalenum(ctypes.Structure):
    _anonymous_ = ("_ival",)
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("pixel_format", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("_ival", _v4l2_frmival_union),
        ("reserved", ctypes.c_uint32 * 2),
    ]


class _v4l2_pix_enc_union(ctypes.Union):
    _fields_ = [
        ("ycbcr_enc", ctypes.c_uint32),
        ("hsv_enc", ctypes.c_uint32),
    ]


class v4l2_pix_format(ctypes.Structure):
    _anonymous_ = ("_enc",)
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("pixelformat", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("bytesperline", ctypes.c_uint32),
        ("sizeimage", ctypes.c_uint32),
        ("colorspace", ctypes.c_uint32),
        ("priv", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("_enc", _v4l2_pix_enc_union),
        ("quantization", ctypes.c_uint32),
        ("xfer_func", ctypes.c_uint32),
    ]


class _v4l2_fmt_union(ctypes.Union):
    _fields_ = [
        ("pix", v4l2_pix_format),
        # Pointer-containing union members in C (e.g. v4l2_window) force 8-byte alignment.
        ("_align", ctypes.c_void_p),
        ("raw_data", ctypes.c_uint8 * 200),
    ]


class v4l2_format(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("fmt", _v4l2_fmt_union),
    ]


class v4l2_captureparm(ctypes.Structure):
    _fields_ = [
        ("capability", ctypes.c_uint32),
        ("capturemode", ctypes.c_uint32),
        ("timeperframe", v4l2_fract),
        ("extendedmode", ctypes.c_uint32),
        ("readbuffers", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 4),
    ]


class _v4l2_parm_union(ctypes.Union):
    _fields_ = [
        ("capture", v4l2_captureparm),
        ("raw_data", ctypes.c_uint8 * 200),
    ]


class v4l2_streamparm(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("parm", _v4l2_parm_union),
    ]


class v4l2_requestbuffers(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("flags", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 3),
    ]


class timeval(ctypes.Structure):
    _fields_ = [
        ("tv_sec", ctypes.c_long),
        ("tv_usec", ctypes.c_long),
    ]


class v4l2_timecode(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("frames", ctypes.c_uint8),
        ("seconds", ctypes.c_uint8),
        ("minutes", ctypes.c_uint8),
        ("hours", ctypes.c_uint8),
        ("userbits", ctypes.c_uint8 * 4),
    ]


class _v4l2_buffer_m(ctypes.Union):
    _fields_ = [
        ("offset", ctypes.c_uint32),
        ("userptr", ctypes.c_ulong),
        ("planes", ctypes.c_void_p),
        ("fd", ctypes.c_int32),
    ]


class _v4l2_buffer_req(ctypes.Union):
    _fields_ = [
        ("request_fd", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


class v4l2_buffer(ctypes.Structure):
    _anonymous_ = ("_req",)
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("bytesused", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("timestamp", timeval),
        ("timecode", v4l2_timecode),
        ("sequence", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("m", _v4l2_buffer_m),
        ("length", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
        ("_req", _v4l2_buffer_req),
    ]


class v4l2_queryctrl(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("name", ctypes.c_uint8 * 32),
        ("minimum", ctypes.c_int32),
        ("maximum", ctypes.c_int32),
        ("step", ctypes.c_int32),
        ("default_value", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 2),
    ]


class v4l2_control(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("value", ctypes.c_int32),
    ]


# --- ioctl numbers ---
VIDIOC_QUERYCAP = _ior("V", 0, v4l2_capability)
VIDIOC_ENUM_FMT = _iowr("V", 2, v4l2_fmtdesc)
VIDIOC_G_FMT = _iowr("V", 4, v4l2_format)
VIDIOC_S_FMT = _iowr("V", 5, v4l2_format)
VIDIOC_REQBUFS = _iowr("V", 8, v4l2_requestbuffers)
VIDIOC_QUERYBUF = _iowr("V", 9, v4l2_buffer)
VIDIOC_QBUF = _iowr("V", 15, v4l2_buffer)
VIDIOC_DQBUF = _iowr("V", 17, v4l2_buffer)
VIDIOC_STREAMON = _iow("V", 18, ctypes.c_int)
VIDIOC_STREAMOFF = _iow("V", 19, ctypes.c_int)
VIDIOC_G_PARM = _iowr("V", 21, v4l2_streamparm)
VIDIOC_S_PARM = _iowr("V", 22, v4l2_streamparm)
VIDIOC_G_CTRL = _iowr("V", 27, v4l2_control)
VIDIOC_QUERYCTRL = _iowr("V", 36, v4l2_queryctrl)
VIDIOC_ENUM_FRAMESIZES = _iowr("V", 74, v4l2_frmsizeenum)
VIDIOC_ENUM_FRAMEINTERVALS = _iowr("V", 75, v4l2_frmivalenum)


def xioctl(fd: int, request: int, arg: ctypes.Structure | ctypes._SimpleCData) -> None:
    """Invoke ioctl while automatically retrying EINTR."""
    while True:
        try:
            fcntl.ioctl(fd, request, arg)
            return
        except InterruptedError:
            continue


def _cstr(buf: ctypes.Array[ctypes.c_uint8]) -> str:
    return bytes(buf).split(b"\x00", 1)[0].decode("utf-8", errors="replace")


@dataclass(frozen=True)
class VideoMode:
    """A discrete capture mode advertised by a V4L2 device."""

    pixel_format: str
    description: str
    width: int
    height: int
    fps_options: tuple[float, ...]


@dataclass(frozen=True)
class VideoControlInfo:
    """Metadata and current value for a single V4L2 camera control."""

    ctrl_id: int
    name: str
    ctrl_type: str
    minimum: int
    maximum: int
    step: int
    default_value: int
    current_value: int | None
    flags: int
    read_only: bool = False
    inactive: bool = False


@dataclass(frozen=True)
class ActiveStreamFormat:
    """Currently configured format and frame rate on a V4L2 capture node."""

    width: int
    height: int
    pixel_format: str
    fps: float | None
    bytesperline: int
    sizeimage: int


@dataclass(frozen=True)
class VideoDeviceInfo:
    """Capabilities and advertised modes for a /dev/video* node."""

    path: str
    driver: str
    card: str
    bus_info: str
    capabilities: int
    device_caps: int
    can_capture: bool
    can_stream: bool
    is_metadata_only: bool
    rw_accessible: bool
    modes: tuple[VideoMode, ...]
    driver_version: str = ""
    active_format: ActiveStreamFormat | None = None
    controls: tuple[VideoControlInfo, ...] = ()
    can_output: bool = False
    is_loopback: bool = False
    error: str = ""

    @property
    def is_color_camera(self) -> bool:
        """True if this physical capture device advertises a standard color pixel format."""
        color_formats = {"MJPG", "YUYV", "NV12", "UYVY", "RGB3", "BGR3"}
        return self.can_capture and not self.is_loopback and any(m.pixel_format in color_formats for m in self.modes)


def decode_driver_version(version: int) -> str:
    """Decode packed KERNEL_VERSION(a, b, c) integer into 'a.b.c'."""
    if version <= 0:
        return ""
    return f"{(version >> 16) & 0xFF}.{(version >> 8) & 0xFF}.{version & 0xFF}"


def decode_capability_flags(caps: int) -> tuple[str, ...]:
    """Return human-readable names for set V4L2 capability bits."""
    known = (
        (V4L2_CAP_VIDEO_CAPTURE, "VIDEO_CAPTURE"),
        (V4L2_CAP_VIDEO_OUTPUT, "VIDEO_OUTPUT"),
        (V4L2_CAP_META_CAPTURE, "META_CAPTURE"),
        (V4L2_CAP_STREAMING, "STREAMING"),
        (V4L2_CAP_READWRITE, "READWRITE"),
        (V4L2_CAP_EXT_PIX_FORMAT, "EXT_PIX_FORMAT"),
        (V4L2_CAP_DEVICE_CAPS, "DEVICE_CAPS"),
    )
    return tuple(label for bit, label in known if caps & bit)


def _enum_fps(fd: int, pixel_format: int, width: int, height: int) -> tuple[float, ...]:
    fps_list: list[float] = []
    idx = 0
    while True:
        ival = v4l2_frmivalenum()
        ival.index = idx
        ival.pixel_format = pixel_format
        ival.width = width
        ival.height = height
        try:
            xioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, ival)
        except OSError:
            break
        if ival.type == V4L2_FRMIVAL_TYPE_DISCRETE and ival.discrete.numerator > 0:
            fps_list.append(round(ival.discrete.denominator / ival.discrete.numerator, 3))
        elif ival.type in (V4L2_FRMIVAL_TYPE_CONTINUOUS, V4L2_FRMIVAL_TYPE_STEPWISE):
            if ival.stepwise.min.numerator > 0:
                fps_list.append(round(ival.stepwise.min.denominator / ival.stepwise.min.numerator, 3))
            break
        idx += 1
    return tuple(fps_list)


def _enum_modes(fd: int) -> tuple[VideoMode, ...]:
    modes: list[VideoMode] = []
    fmt_idx = 0
    while True:
        desc = v4l2_fmtdesc()
        desc.index = fmt_idx
        desc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        try:
            xioctl(fd, VIDIOC_ENUM_FMT, desc)
        except OSError:
            break

        fourcc = fourcc_to_str(desc.pixelformat)
        description = _cstr(desc.description)
        sz_idx = 0
        while True:
            frmsize = v4l2_frmsizeenum()
            frmsize.index = sz_idx
            frmsize.pixel_format = desc.pixelformat
            try:
                xioctl(fd, VIDIOC_ENUM_FRAMESIZES, frmsize)
            except OSError:
                break
            if frmsize.type == V4L2_FRMSIZE_TYPE_DISCRETE:
                w = frmsize.discrete.width
                h = frmsize.discrete.height
                modes.append(
                    VideoMode(
                        pixel_format=fourcc,
                        description=description,
                        width=w,
                        height=h,
                        fps_options=_enum_fps(fd, desc.pixelformat, w, h),
                    )
                )
            elif frmsize.type in (V4L2_FRMSIZE_TYPE_CONTINUOUS, V4L2_FRMSIZE_TYPE_STEPWISE):
                w = frmsize.stepwise.max_width
                h = frmsize.stepwise.max_height
                modes.append(
                    VideoMode(
                        pixel_format=fourcc,
                        description=description,
                        width=w,
                        height=h,
                        fps_options=_enum_fps(fd, desc.pixelformat, w, h),
                    )
                )
                break
            sz_idx += 1
        fmt_idx += 1
    return tuple(modes)


def _query_active_format(fd: int) -> ActiveStreamFormat | None:
    fmt = v4l2_format()
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
    try:
        xioctl(fd, VIDIOC_G_FMT, fmt)
    except OSError:
        return None

    fps: float | None = None
    parm = v4l2_streamparm()
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
    try:
        xioctl(fd, VIDIOC_G_PARM, parm)
        num = parm.parm.capture.timeperframe.numerator
        den = parm.parm.capture.timeperframe.denominator
        if num > 0 and den > 0:
            fps = round(den / num, 3)
    except OSError:
        pass

    return ActiveStreamFormat(
        width=fmt.fmt.pix.width,
        height=fmt.fmt.pix.height,
        pixel_format=fourcc_to_str(fmt.fmt.pix.pixelformat),
        fps=fps,
        bytesperline=fmt.fmt.pix.bytesperline,
        sizeimage=fmt.fmt.pix.sizeimage,
    )


def _query_controls(fd: int) -> tuple[VideoControlInfo, ...]:
    controls: list[VideoControlInfo] = []
    qctrl = v4l2_queryctrl()
    qctrl.id = V4L2_CTRL_FLAG_NEXT_CTRL
    while True:
        try:
            xioctl(fd, VIDIOC_QUERYCTRL, qctrl)
        except OSError:
            break

        ctrl_id = int(qctrl.id)
        if not (qctrl.flags & V4L2_CTRL_FLAG_DISABLED) and qctrl.type != 6:
            current_val: int | None = None
            if not (qctrl.flags & V4L2_CTRL_FLAG_HAS_PAYLOAD):
                ctrl = v4l2_control()
                ctrl.id = ctrl_id
                try:
                    xioctl(fd, VIDIOC_G_CTRL, ctrl)
                    current_val = int(ctrl.value)
                except OSError:
                    current_val = None

            controls.append(
                VideoControlInfo(
                    ctrl_id=ctrl_id,
                    name=_cstr(qctrl.name),
                    ctrl_type=V4L2_CTRL_TYPE_NAMES.get(int(qctrl.type), f"type_{qctrl.type}"),
                    minimum=int(qctrl.minimum),
                    maximum=int(qctrl.maximum),
                    step=int(qctrl.step),
                    default_value=int(qctrl.default_value),
                    current_value=current_val,
                    flags=int(qctrl.flags),
                    read_only=bool(qctrl.flags & V4L2_CTRL_FLAG_READ_ONLY),
                    inactive=bool(qctrl.flags & V4L2_CTRL_FLAG_INACTIVE),
                )
            )

        qctrl = v4l2_queryctrl()
        qctrl.id = ctrl_id | V4L2_CTRL_FLAG_NEXT_CTRL
    return tuple(controls)


def query_video_device(path: str) -> VideoDeviceInfo:
    """Query V4L2 capabilities and supported capture modes for a device path."""
    rw = os.access(path, os.R_OK | os.W_OK)
    try:
        fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    except OSError as exc:
        return VideoDeviceInfo(
            path=path,
            driver="",
            card="",
            bus_info="",
            capabilities=0,
            device_caps=0,
            can_capture=False,
            can_stream=False,
            is_metadata_only=False,
            rw_accessible=rw,
            modes=(),
            error=str(exc),
        )

    try:
        cap = v4l2_capability()
        xioctl(fd, VIDIOC_QUERYCAP, cap)
        driver_name = _cstr(cap.driver)
        effective_caps = cap.device_caps if (cap.capabilities & V4L2_CAP_DEVICE_CAPS) else cap.capabilities
        can_capture = bool(effective_caps & V4L2_CAP_VIDEO_CAPTURE)
        can_output = bool(effective_caps & V4L2_CAP_VIDEO_OUTPUT)
        can_stream = bool(effective_caps & V4L2_CAP_STREAMING)
        is_meta_only = bool(effective_caps & V4L2_CAP_META_CAPTURE) and not can_capture
        is_loopback = can_output or ("loopback" in driver_name.lower())
        modes = _enum_modes(fd) if (can_capture and not is_loopback) else ()
        active_fmt = _query_active_format(fd) if can_capture else None
        controls = _query_controls(fd) if (can_capture and not is_loopback) else ()
        return VideoDeviceInfo(
            path=path,
            driver=driver_name,
            card=_cstr(cap.card),
            bus_info=_cstr(cap.bus_info),
            capabilities=cap.capabilities,
            device_caps=effective_caps,
            can_capture=can_capture,
            can_stream=can_stream,
            is_metadata_only=is_meta_only,
            rw_accessible=rw,
            modes=modes,
            driver_version=decode_driver_version(cap.version),
            active_format=active_fmt,
            controls=controls,
            can_output=can_output,
            is_loopback=is_loopback,
        )
    except OSError as exc:
        return VideoDeviceInfo(
            path=path,
            driver="",
            card="",
            bus_info="",
            capabilities=0,
            device_caps=0,
            can_capture=False,
            can_stream=False,
            is_metadata_only=False,
            rw_accessible=rw,
            modes=(),
            error=str(exc),
        )
    finally:
        os.close(fd)


def list_video_devices() -> list[VideoDeviceInfo]:
    """Enumerate all /dev/video* nodes sorted naturally by index."""
    def _sort_key(p: str) -> tuple[int, str]:
        suffix = p.removeprefix("/dev/video")
        return (int(suffix), p) if suffix.isdigit() else (10**9, p)

    paths = sorted(glob.glob("/dev/video*"), key=_sort_key)
    return [query_video_device(p) for p in paths]


def pick_default_camera(devices: list[VideoDeviceInfo] | None = None) -> VideoDeviceInfo | None:
    """Select the preferred physical capture-capable color webcam device, if present."""
    devs = devices if devices is not None else list_video_devices()
    streamable = [d for d in devs if d.can_capture and d.can_stream and d.rw_accessible and not d.is_loopback]
    for d in streamable:
        if d.is_color_camera:
            return d
    return streamable[0] if streamable else None


def list_loopback_devices(devices: list[VideoDeviceInfo] | None = None) -> list[VideoDeviceInfo]:
    """Return all accessible V4L2 loopback output devices."""
    devs = devices if devices is not None else list_video_devices()
    return [d for d in devs if d.can_output and d.rw_accessible]


def pick_default_loopback(devices: list[VideoDeviceInfo] | None = None) -> VideoDeviceInfo | None:
    """Select the first accessible v4l2loopback output device, if present."""
    loopbacks = list_loopback_devices(devices)
    return loopbacks[0] if loopbacks else None
