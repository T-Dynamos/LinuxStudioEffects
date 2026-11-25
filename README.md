# LinuxStudioEffects

An open-source Linux alternative to **Windows Studio Effects** designed to run real-time camera and microphone AI effects on **AMD Ryzen AI NPUs** (using [MLIR-AIE / IRON](https://github.com/Xilinx/mlir-aie) and the standalone HRX userspace runtime).

## Project Structure

```text
LinuxStudioEffects/
├── setup_ironenv.py                  # Self-contained NPU environment setup & verification
├── pyproject.toml
├── src/
│   └── linux_studio_effects/
│       ├── npu/                      # NPU hardware/driver probing & runtime helpers
│       ├── video/                    # Camera effects (blur, framing, eye contact, lighting, filters)
│       └── audio/                    # Microphone effects (Voice Focus)
└── tests/
    └── test_npu_smoke.py             # End-to-end NPU detection & IRON kernel smoke test
```

## Quickstart

### Prerequisites
- Linux (`x86_64`) with kernel >= 6.14 (`amdxdna` driver exposing `/dev/accel/accel*` with read/write permissions)
- Python 3.11–3.14 (or `uv`), `git`, and a C/C++ compiler (`gcc`/`g++` or `clang`/`clang++`)
- No system `xrt` or `xrt-plugin-amdxdna` packages required

### Setup `iron-env` and verify NPU execution

```bash
python3 setup_ironenv.py
```

Activate in your shell:

```bash
# bash / zsh
source iron-env/bin/activate

# fish
source iron-env/bin/activate.fish
```

Run the NPU smoke test:

```bash
python3 tests/test_npu_smoke.py
```

## TODO

### NPU Runtime & Core Pipeline
- [x] Self-contained `iron-env` setup (`mlir_aie`, `llvm-aie`, `libhrx.so`, standalone `hrx-xclbinutil`)
- [x] `/dev/accel/accel*` & `amdxdna` preflight detection and NPU smoke test
- [ ] Validate across NPU generations (Phoenix/Hawk Point NPU1, Strix Point/Krackan NPU2)
- [ ] Zero-copy / persistent DMA ring buffer pipeline for real-time video & audio frames
- [ ] Custom AIE2 / AIE2p vectorized C++ kernels for image convolutions, resize/warp, and FFT/DSP

### Video & Camera Effects
- [ ] **Background Blur (Standard Blur):** Fast segmentation + Gaussian background blur
- [ ] **Portrait Blur:** Smooth depth/edge-aware bokeh background blur
- [ ] **Automatic Framing:** Real-time face detection with smooth pan, crop, and zoom tracking
- [ ] **Eye Contact (Standard):** Subtle gaze redirection toward the camera
- [ ] **Eye Contact (Teleprompter):** Continuous gaze correction while reading across the screen
- [ ] **Portrait Light:** Low-light face illumination and exposure balancing
- [ ] **Creative Filters:** Stylized real-time filters (*Illustrated*, *Animated*, *Watercolor*)

### Audio Effects
- [ ] **Voice Focus:** Real-time microphone noise suppression and voice isolation on the NPU

### Linux Desktop & System Integration
- [ ] Virtual webcam output (PipeWire / `v4l2loopback`) compatible with OBS, Zoom, Google Meet, Teams, Discord
- [ ] Virtual microphone filter node (PipeWire audio filter chain)
- [ ] Support for both built-in laptop webcams and external USB cameras
- [ ] Live toggle CLI (`lse-ctl`) and desktop Quick Settings / system tray UI
