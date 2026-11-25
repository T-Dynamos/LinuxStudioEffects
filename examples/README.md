# Examples

Run any example from the repository root using `iron-env/bin/python` (or any Python `>= 3.11` with `numpy` for the camera-only examples):

1. **[01_npu_probe_and_kernel.py](01_npu_probe_and_kernel.py)** — Probe `/dev/accel/accel*`, `amdxdna`, and run a simple `@iron.jit` vector kernel on the AMD Ryzen AI NPU.
   ```bash
   ./iron-env/bin/python examples/01_npu_probe_and_kernel.py
   ```

2. **[02_camera_deps_and_capabilities.py](02_camera_deps_and_capabilities.py)** — Check V4L2/TurboJPEG/`v4l2loopback` dependencies and print a detailed summary of every `/dev/video*` device, supported formats, and controls.
   ```bash
   ./iron-env/bin/python examples/02_camera_deps_and_capabilities.py
   ```

3. **[03_camera_fps_benchmark.py](03_camera_fps_benchmark.py)** — Measure raw V4L2 capture FPS and CPU-decoded RGB FPS at a chosen resolution/format.
   ```bash
   ./iron-env/bin/python examples/03_camera_fps_benchmark.py --width 1280 --height 720 --format MJPG --duration 3.0
   ```

4. **[04_virtual_camera_passthrough.py](04_virtual_camera_passthrough.py)** — Run a live camera -> `v4l2loopback` pipeline with interactive or CLI-configured input/output resolutions and FPS, running until `Ctrl+C` while reporting throughput and CPU usage %.
   ```bash
   # Ensure v4l2loopback is loaded first (creates /dev/video4 or next free node):
   # sudo modprobe v4l2loopback devices=1 card_label="LinuxStudioEffects" exclusive_caps=1

   # Interactive prompt (select input mode, output resolution, and FPS; runs until Ctrl+C):
   ./iron-env/bin/python examples/04_virtual_camera_passthrough.py

   # Or non-interactive with explicit flags:
   ./iron-env/bin/python examples/04_virtual_camera_passthrough.py \
     --in-size 1280x720 --in-fps 30 \
     --out-size 640x480 --out-fps 30
   ```
   While it runs, view the virtual webcam from any app (or `ffplay /dev/video4`).

5. **[05_frame_effect_pipeline.py](05_frame_effect_pipeline.py)** — Device-agnostic frame-in -> frame-out -> virtual camera demo combining `FrameEffect` subclasses, plain Python `fn(frame) -> frame` callables, and `EffectChain`.
   ```bash
   ./iron-env/bin/python examples/05_frame_effect_pipeline.py --effect studio_demo
   ```

6. **[06_npu_greyscale_camera.py](06_npu_greyscale_camera.py)** — Live webcam -> AMD Ryzen AI NPU (`GreyscaleEffect` via MLIR-AIE / IRON) -> `v4l2loopback` virtual webcam.
   ```bash
   ./iron-env/bin/python examples/06_npu_greyscale_camera.py
   ```
