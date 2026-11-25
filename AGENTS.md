# Project guidance

## Effect architecture

- Effects live in `src/linux_studio_effects/video/effects/<effect>/`.
- A backend exists only when its file exists (`cpu.py`, `gpu.py`, or `npu.py`).
  Backend availability is discovered by `video/backend.py`; do not add a
  `supported_devices` list or duplicate backend registry.
- Every backend for one effect defines the same public class name. For example,
  mirror backends both define `MirrorEffect`; the module path identifies the
  backend. Backend files should have only their effect's implementation, not
  shared generic CPU/GPU/NPU directories.
- Keep effect imports lazy. `video/effects/__init__.py` should not import every
  implementation at package import time.
- `FrameEffect.setup(width, height, fps)` is called before per-frame context is
  available. Native-format backend allocations should therefore be deferred
  until `process(frame, ctx)` when the implementation needs `ctx.pixel_format`.
- Preserve effect names (`effect_name`) across implementations so automatic
  backend selection and `FrameGraph` composition work consistently.

## Live video formats

- The live `VideoPipeline` path supports native `YUYV`, `UYVY`, `RGB3`, and
  `BGR3` capture and output. Native input/output FourCC, colorspace, YCbCr
  encoding, and quantization range must match after resolving V4L2 default
  values. Drivers may canonicalize transfer-function metadata to DEFAULT.
  Capture and output negotiation must preserve requested dimensions; fail
  instead of silently accepting a different size.
- Compressed `MJPG` camera modes are decoded and reformatted through PyAV
  (FFmpeg) into a supported native output format. Do not add hand-written
  MJPEG, YUV, or RGB conversion routines. Keep the requested capture and output
  dimensions equal unless a user explicitly requests a resize.
- Use `video/pixels.py` for packed-frame layout, YUYV/UYVY plane access, and
  same-format resizing. A YUYV/UYVY array has shape `(height, width // 2, 4)`;
  RGB3/BGR3 has shape `(height, width, 3)`. Packed 4:2:2 widths must be even.
- Effects must preserve their processing FourCC and packed layout. If a backend
  needs a model-friendly view of YUYV/UYVY, use luma directly instead of
  decoding to RGB. Keep optional RGB decode/encode helpers out of the live
  capture-to-loopback path.
- Keep examples and hardware tests from silently mismatching native input and
  output formats. For effects that only accept RGB, select RGB3 or use the
  PyAV path when the camera exposes MJPG instead.

## NPU and XRT

- NPU execution uses MLIR-AIE / IRON with the standalone XRT bundle installed
  by `setup_ironenv.py` into `.ironenv/xrt`; the expected runtime is XRT, not
  HRX or system XRT. The setup script's default bundle path is documented in
  `README.md` and can be overridden with `--xrt-bundle`.
- Use `aie.iron` tensors and `aie.utils.DefaultNPURuntime` as the existing NPU
  effect does. Do not silently fall back to CPU in an NPU implementation.
- IRON's `@iron.jit` inspects `In`, `Out`, and `CompileTime` annotations when
  decorating nested kernel functions. If those marker types are imported
  inside a kernel builder, avoid postponed annotations in that module so the
  markers are actual types rather than unresolved strings.
- Make NPU kernels preserve the complete native pixel layout. Account for row
  boundaries when partitioning image work across workers; format conversion
  is not a kernel's responsibility in the live pipeline.
- Hardware-dependent checks should probe for an accessible AMD NPU and skip
  cleanly when it is unavailable. Likewise, camera/loopback integration tests
  should skip if the required V4L2 devices are absent.

## In-progress model effects

- Video Depth and Portrait Blur are in progress. Keep changes to their model
  code and implementations scoped to an explicit request to work on those
  effects; do not use them as a reason to change the stable native-frame
  contract for other effects.

## Tests and examples

- Tests use Python `unittest` discovery. Run the suite with:
  `PYTHONPATH=src .ironenv/bin/python -m unittest discover -s tests -v`.
- The `.ironenv` is local and may not exist on every machine. A system Python
  can run the CPU-only tests if NumPy is installed, but the NPU tests require
  the configured environment and hardware.
- `examples/09_run_effect.py` is the general live-camera-to-loopback runner;
  `examples/04_virtual_camera_passthrough.py` is the format-preserving
  passthrough runner. Both should require or select matching native formats.
- `examples/10_benchmark_effect_backends.py` benchmarks discovered effect and
  backend pairs using real camera frames; keep its measurements on the effect
  processing path distinct from capture timing.
