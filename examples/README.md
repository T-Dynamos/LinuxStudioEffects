# Examples

Run any example from the repository root using `.ironenv/bin/python` (or Python `>= 3.11` with `numpy` for the camera-only examples):

1. **[01_npu_probe_and_kernel.py](01_npu_probe_and_kernel.py)** — Probe `/dev/accel/accel*`, `amdxdna`, and run a simple `@iron.jit` vector kernel on the AMD Ryzen AI NPU.
   ```bash
   ./.ironenv/bin/python examples/01_npu_probe_and_kernel.py
   ```

2. **[02_camera_deps_and_capabilities.py](02_camera_deps_and_capabilities.py)** — Check V4L2/PyAV/`v4l2loopback` dependencies and print a detailed summary of every `/dev/video*` device, supported formats, and controls.
   ```bash
   ./.ironenv/bin/python examples/02_camera_deps_and_capabilities.py
   ```

3. **[03_camera_fps_benchmark.py](03_camera_fps_benchmark.py)** — Measure raw V4L2 capture FPS and CPU-decoded RGB FPS at a chosen resolution/format.
   ```bash
   ./.ironenv/bin/python examples/03_camera_fps_benchmark.py --width 1280 --height 720 --format MJPG --duration 3.0
   ```

4. **[04_virtual_camera_passthrough.py](04_virtual_camera_passthrough.py)** — Run a live camera -> `v4l2loopback` pipeline with interactive or CLI-configured input/output resolutions and FPS, running until `Ctrl+C` while reporting throughput and CPU usage %.
   ```bash
   # Ensure v4l2loopback is loaded first (creates /dev/video4 or next free node):
   # sudo modprobe v4l2loopback devices=1 card_label="LinuxStudioEffects" exclusive_caps=1

   # Interactive prompt (select input mode, output resolution, and FPS; runs until Ctrl+C):
   ./.ironenv/bin/python examples/04_virtual_camera_passthrough.py

   # Or non-interactive with explicit flags:
   ./.ironenv/bin/python examples/04_virtual_camera_passthrough.py \
     --in-size 1280x720 --in-fps 30 \
     --out-size 640x480 --out-fps 30
   ```
   While it runs, view the virtual webcam from any app (or `ffplay /dev/video4`).

5. **[05_frame_effect_pipeline.py](05_frame_effect_pipeline.py)** — Device-agnostic frame graph demo combining `FrameEffect` subclasses and plain Python `fn(frame) -> frame` callables. Each named node can take the camera frame or an earlier effect's output.
   ```bash
   ./.ironenv/bin/python examples/05_frame_effect_pipeline.py --effect studio_demo
   ```

6. **[06_npu_greyscale_camera.py](06_npu_greyscale_camera.py)** — Live webcam -> AMD Ryzen AI NPU (`GreyscaleEffect` via MLIR-AIE / IRON) -> `v4l2loopback` virtual webcam.
   ```bash
   ./.ironenv/bin/python examples/06_npu_greyscale_camera.py
   ```

7. **[07_video_depth_benchmark.py](07_video_depth_benchmark.py)** — Streaming Video Depth Anything (`Small` / `Base` / `Large`) hardware feasibility matrix, NPU vs CPU vs GPU benchmark, and optional `--live` temporally consistent depth camera stream.
   ```bash
   ./.ironenv/bin/python examples/07_video_depth_benchmark.py --variants small --input-size 252 --real-weights
   ```

8. **[08_live_video_depth_side_by_side.py](08_live_video_depth_side_by_side.py)** — Fastest streaming Video Depth Anything `Small` side-by-side (`[Input Video | Depth Map]`) live preview / `/dev/video4` / MP4 output.
   ```bash
   ./.ironenv/bin/python examples/08_live_video_depth_side_by_side.py --input-size 140 --preview
   ```

9. **[09_run_effect.py](09_run_effect.py)** — Run any discovered effect/backend on the live camera using an advertised native 1920x1080 mode at 30 fps by default. Input and output sizes must match; use `--in-size` (and the same `--out-size`, if supplied) to choose another advertised native size. A missing `/dev/videoN` output is created for this run, assigned to the invoking user, and removed on exit; module changes may prompt for `sudo`. Use `--list` to inspect effect folders, or `--frames 5` for an optional synthetic-frame check. Depth effects use synthetic weights unless `--real-weights` is given.
   ```bash
   ./.ironenv/bin/python examples/09_run_effect.py --list
   ./.ironenv/bin/python examples/09_run_effect.py --effect mirror
   ./.ironenv/bin/python examples/09_run_effect.py --effect mirror --in-size 1920x1080 --out-size 1920x1080
   ./.ironenv/bin/python examples/09_run_effect.py --effect video_depth --device auto
   ```

10. **[10_benchmark_effect_backends.py](10_benchmark_effect_backends.py)** — Benchmark each discovered effect/backend on real camera frames. Depth effects use RGB3 modes; other effects use a matching native camera mode. Depth weights are synthetic by default.
    ```bash
    ./.ironenv/bin/python examples/10_benchmark_effect_backends.py --duration 3
    ```
