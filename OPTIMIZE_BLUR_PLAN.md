# Portrait Blur & Video Depth Optimization Plan

## Current Baseline vs Optimized Results (`1280x720`, `OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4`)

Measured on `1280x720` RGB portrait input before and after implementing Phases 1–5 (with `NpuPortraitBlurEffect` executing both NPU Gaussian blur pyramid and NPU 4-column bokeh blend with zero CPU fallback):

| Preset | Variant / Input Size | CPU Model (`bb / head`) | NPU Model (`bb / head`) | E2E Sync `process()` (NPU) | Depth Pearson `r` | Async Paced 30 FPS Stream |
| --- | --- | --- | --- | --- | --- | --- |
| `fast` (Before) | `small` @ `196` | `106.9 ms` (`55.0 / 52.0`) | `100.4 ms` (`50.9 / 49.5`) | `141.7 ms` | `0.999997` | — |
| `fast` (After) | `small` @ `196` | `78.2 ms` (`40.9 / 37.2`) | **`75.8 ms`** (`38.6 / 37.2`) | **`98.7 ms`** (`49` dispatches) | `0.999991` (`img_mae=0.10`) | **CPU `114.7 FPS` uncapped / `30.33 FPS` (`9.54 ms` avg)**; **NPU `37.1 FPS` uncapped / `30.13 FPS` (`24.61 ms` avg, `max=29.73 ms`, `8` depth updates/2s)** |
| `balanced` (Before) | `base` @ `252` | `476.9 ms` (`283.0 / 193.9`) | `391.5 ms` (`212.6 / 178.9`) | `416.9 ms` | `0.999992` | `6.5 FPS` (`153.98 ms` avg, `8` updates/30f) |
| `balanced` (After) | `base` @ `252` | `376.5 ms` (`237.6 / 138.9`) | **`303.1 ms`** (`167.8 / 135.3`) | **`336.1 ms`** (`65` dispatches) | `0.999996` (`img_mae=0.14`) | **CPU `103.7 FPS` uncapped / `30.33 FPS` (`10.66 ms` avg)**; **NPU `30.6 FPS` uncapped / `29.61 FPS` (`31.77 ms` avg)** |
| `high` (Before) | `base` @ `336` | `887.9 ms` (`526.2 / 361.7`) | `701.0 ms` (`371.0 / 330.0`) | `782.2 ms` | `0.999998` | `4.6 FPS` (`217.79 ms` avg) |
| `high` (After) | `base` @ `336` | `656.5 ms` (`385.7 / 270.8`) | **`507.6 ms`** (`290.9 / 216.7`) | **`543.4 ms`** (`77` dispatches) | `0.999996` (`img_mae=0.15`) | — |

Optional `LSE_NPU_BF16_OUT=1` (`balanced`, `base@252`) cuts host C DMA further to `npu_model = 283.2 ms` (`npu_e2e = 303.7 ms`, `copy = 47.7 ms`, `depth_r = 0.998711`).

NPU-only per-frame `apply_depth_bokeh` (`1280x720`, `balanced`) drops from `91.41 ms` idle (CPU baseline) to **`25.62 ms`** idle even with full NPU blur + NPU composite:
- CoC guided filter + luma extract/downsample: **`2.24 ms`** (down from `32.84 ms`)
- Downsample2x (`ds1`) + C fused unpremul & resize (`up`): **`0.41 + 1.69 = 2.10 ms`** (down from `~25 ms`)
- NPU `bfloat16` Gaussian pyramid (`g1+g2+g3`): **`12.92 ms`** (`6.51 + 4.27 + 2.14`)
- NPU SoA-32 vectorized 4-column bokeh blend (`comp`): **`8.37 ms`** (down from `34.10 ms`)

---

## Optimization Phases

### Phase 1 — Decouple Live Bokeh Rendering via Default `async_depth=True` *(Implemented)*
- **Files**: `src/linux_studio_effects/video/effects/portrait_blur/cpu.py` (inherited by `npu.py`)
- Enabled `async_depth=True` and `depth_interval_s=0.05` by default so depth updates run asynchronously on `vda-depth-worker`, deferring `model_luma_input` conversion into the worker thread.
- Added NPU main-thread render priority arbitration (`render_priority()`, `set_npu_worker_thread()`, `wait_for_npu_render_priority()` in `npu_ops.py` & `portrait_blur/cpu.py` / `npu.py`) so `vda-depth-worker` yields between NPU dispatches when a live 30 FPS render frame is active.

### Phase 1b — C/AVX-512 Bokeh Pyramid, CoC Guided Filter & NPU Blur Pyramid *(Implemented)*
- **Files**: `src/linux_studio_effects/video/models/video_depth_anything/_host_ops.c`, `host_ops.py`, `src/linux_studio_effects/video/effects/portrait_blur/cpu.py`, `portrait_blur/npu.py`
- Implemented AVX-512 & `-pthread` host kernels: `gaussian5_hw4_f32`, `downsample2x_hw4_f32`, `box_filter_2d_f32`, `extract_luma_and_downsample_f32`, `guided_upsample_coc_f32` (fused bilinear `mean_a * luma_full + mean_b` without full-res intermediate planes), `unpremul_resize_hw4_to_rgb_u8_f32`, and `pack_bokeh_npu_inputs_soa32_u8`.
- Moved NPU bokeh generation cleanly onto NPU1 in `portrait_blur/npu.py`:
  - Single-XCLBIN 4-column `DispatchTime` separable `bfloat16` blur (`lse_gaussian5_hw4_bf16`) supporting both 5-tap (`1` pass) and 9-tap binomial (`2` passes in one H+V round-trip).
  - Vectorized SoA-32 `lse_bokeh_blend4` with `padded_pixels` alignment (`128`-aligned) and `DispatchTime` XCLBIN reuse so **all resolutions** run bokeh blur and bokeh composite on NPU with zero silent CPU fallback.

### Phase 2 — Narrow-`N` (`N_align=32`) NPU `Conv2d` Tiling for Output Head *(Implemented)*
- **Files**: `src/linux_studio_effects/video/models/video_depth_anything/npu_ops.py`, `model.py`
- Added lazy `_build_narrow_n_design` (`M_align=512`, `N_align=32`, 4 columns × 4 rows M-sharded) triggered automatically for `n <= 96`, keeping normal ViT/MODNet runs on a single XCLBIN.
- Added spatial row-chunking (`y0:y1` strips with exact `p=1` top/bottom halos) inside `conv2d_3x3_s1_p1_nchw` and lowered `_conv2d` 3×3 NPU offload gate in `model.py` from `c_out >= 128` to `c_out >= 32`.

### Phase 3 — Persistent `insts` BO Allocation & Optional `bfloat16` C DMA *(Implemented)*
- **Files**: `src/linux_studio_effects/video/models/video_depth_anything/npu_ops.py`, `_host_ops.c`, `host_ops.py`
- Cached persistent `XrtTransport` instruction BOs keyed by shape & kernel variant, dispatching directly through `handle.kernel_run.dispatch([3, insts_bo, insts_bytes, *buffers])`.
- Added opt-in `bfloat16` C-output mode (`LSE_NPU_BF16_OUT=1` / `use_bf16_out=True`) with fp32 L1 accumulation and fused `bf16_unpad_bias_gelu` / `bf16_unpad_transpose_bias_relu` AVX-512 C kernels.

### Phase 4 — Zero-Copy Temporal Attention Ring Buffer & Persistent `-pthread` C Loops *(Implemented)*
- **Files**: `src/linux_studio_effects/video/models/video_depth_anything/_host_ops.c`, `host_ops.py`, `model.py`
- Added `temporal_attn_roll_ring_f32` and tracked `_kv_ring_cursor` in `VideoDepthAnythingModel` across anchors `[0, 1]` and rolling window `[2..31]` with zero `memmove` shifts.
- Added persistent `-pthread` thread pool (`lse_parallel_for`, `trylock` concurrency safety with async worker threads) parallelizing resize, transpose, im2col, composite, blur, attention, and preprocess kernels.

### Phase 5 — Fused Backbone Spatial Attention & Uint8-to-NCHW Preprocess *(Implemented)*
- **Files**: `src/linux_studio_effects/video/models/video_depth_anything/_host_ops.c`, `host_ops.py`, `model.py`
- Added `fused_mha_qkv_f32` computing per-head `softmax(Q @ K^T * scale) @ V` directly from packed `(n_tok, 3*D)` `qkv` in AVX-512 without allocating `qh`/`kh`/`vh`/`attn`/`ctx`.
- Added `preprocess_rgb_u8_to_nchw_f32` fusing uint8 bilinear resize (`align_corners=False`) and ImageNet normalization in a single parallel pass.
