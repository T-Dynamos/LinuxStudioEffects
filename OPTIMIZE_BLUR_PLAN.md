# Portrait Blur & Video Depth Optimization Plan

## Current Baseline (1280x720, `OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4`)

Synchronous single-frame inference + bokeh timings before this plan:

| Preset | Variant / Input Size | CPU Model | NPU Model | NPU Dispatches / Kernel / Host Copy | E2E Sync `process()` (NPU) | Depth Pearson `r` |
| --- | --- | --- | --- | --- | --- | --- |
| `fast` | `small` @ `196` | `106.9 ms` (`bb=55.0`, `head=52.0`) | `100.4 ms` (`bb=50.9`, `head=49.5`) | `48` / `28.6 ms` / `8.1 ms` | `141.7 ms` | `0.999997` |
| `balanced` | `base` @ `252` | `476.9 ms` (`bb=283.0`, `head=193.9`) | `391.5 ms` (`bb=212.6`, `head=178.9`) | `60` / `152.3 ms` / `54.5 ms` | `416.9 ms` | `0.999992` |
| `high` | `base` @ `336` | `887.9 ms` (`bb=526.2`, `head=361.7`) | `701.0 ms` (`bb=371.0`, `head=330.0`) | `70` / `232.8 ms` / `95.1 ms` | `782.2 ms` | `0.999998` |

Stage breakdown on `balanced` (`base@252`, `sum=396.3ms`: `preprocess=5.2ms`, `bb=206.8ms`, `head=184.4ms`):

- `npu_matmul_fn`: `178.41 ms` (`NpuMatMulExecutor.matmul` across backbone projections)
- `motion_module_total`: `67.04 ms` (of which `temporal_attn_total`: `47.64 ms`)
- `cpu_conv2d`: `66.31 ms` (residual / head convs with `c_out < 128` or below NPU crossover)
- `npu_conv_fn`: `21.33 ms` (`NpuMatMulExecutor.conv2d_3x3_s1_p1_nchw` on heavy `c_out >= 128` layers)
- `bilinear_resize`: `12.06 ms`
- `softmax`: `6.60 ms`
- `layer_norm`: `4.02 ms`
- `group_norm`: `0.30 ms`
- Bokeh pyramid + resize + composite (`e2e - model`): `~25.4 ms` (`~39 FPS` ceiling when decoupled)

---

## Optimization Phases

### Phase 1 — Decouple Live Bokeh Rendering via Default `async_depth=True` *(Implemented)*
- **Files**: `src/linux_studio_effects/video/effects/portrait_blur/cpu.py` (inherited by `npu.py`)
- **Problem**: Synchronous `async_depth=False` stalled every camera frame behind `391–701 ms` of depth model inference (`balanced=416.9 ms` / `high=782.2 ms` e2e).
- **Changes**:
  - Enable `async_depth: bool = True` and `depth_interval_s: float = 0.05` by default in `PortraitBlurEffect`, and expose `async_depth` in `set_params(...)`.
  - Keep frame 0 synchronous so `last_result` and the initial depth map are immediately valid, then run depth updates on `vda-depth-worker`.
  - Store `(frame.copy(), pixel_format)` on the worker slot and defer `model_luma_input(...)` conversion (`YUYV`/`UYVY`/`BGR3` -> RGB) into the background worker so dropped video frames do not waste YUV-to-RGB work on the main thread.
- **Measured Async Result (`1280x720`, 30 frames)**:
  - `balanced` (`base@252`): steady `avg = 153.98 ms` (`6.5 FPS`, `depth_updates = 8`) vs `416.9 ms` sync (`2.7x` faster).
  - `high` (`base@336`): steady `avg = 217.79 ms` (`4.6 FPS`, `depth_updates = 4`) vs `782.2 ms` sync (`3.6x` faster).
  - **Finding**: `pyxrt` (`self.run.wait()`) releases the GIL cleanly (`1ms` sleep jitter avg `1.85 ms`). Instead, the ceiling is `apply_depth_bokeh` itself taking `91.41 ms` idle (`_build_blur_pyramid = 53.10 ms` [`_gaussian5_hw4 = 30.02 ms`], `_compute_coc_full = 32.84 ms`, `_composite_bokeh = 2.93 ms`) plus CPU contention with the background depth worker (`91.4 ms` -> `154–218 ms`).

### Phase 1b — C/AVX-512 Bokeh Pyramid & CoC Guided Filter *(To Hit True 30 FPS)*
- **Files**: `src/linux_studio_effects/video/models/video_depth_anything/_host_ops.c`, `host_ops.py`, `src/linux_studio_effects/video/effects/portrait_blur/cpu.py`
- **Problem**: With `async_depth=True` live, steady-state frame rate is bottlenecked by Python/NumPy memory traffic inside `apply_depth_bokeh`: `30.02 ms` in `_gaussian5_hw4` (`blur_passes=(2,4,6)` allocates dozens of temporary 3.7 MB buffers) and `32.84 ms` in `_compute_coc_full` (NumPy uint8->float32 luma math + three `_resize_2d` fancy-indexing passes).
- **Plan**:
  - Port `_gaussian5_hw4` and separable `_box_filter_2d` to ping-pong C/AVX-512 kernels in `_host_ops.c` (eliminating intermediate `mmap` page faults).
  - Replace NumPy `luma_full` + `_resize_2d` in `_compute_coc_full` with fused C AVX-512 luma extraction (`RGB3`/`BGR3`/`YUYV`/`UYVY`) and bilinear `mean_a * luma_full + mean_b` upsample+apply.

### Phase 2 — Narrow-`N` (`N_align=32`) NPU `Conv2d` Tiling for Output Head (`cpu_conv2d = 66.31 ms`)
- **Files**: `src/linux_studio_effects/video/models/video_depth_anything/npu_ops.py`, `model.py`
- **Problem**: `_build_whole_array_design` tiles `(M, N)` across `4x4` cores with `_N_ALIGN = _NCOLS * _TN = 128`. Output head layers `output_conv1` (`144x144`, `128->64`) and `output_conv2_0` (`252x252`, `64->32`) have `c_out < 128`, leaving ~`2.7B` MACs on CPU because 2x–4x column padding made the `N_align=128` kernel slower than OpenBLAS.
- **Plan**:
  - Add a narrow-`N` dispatch mode (partitioning `M` across all rows/columns with `N_align = 32` or cascading K across columns cleanly without exceeding NPU1's 6-context XRT limit) or unify `M`-major dispatch tiling so `c_out=32/64` achieves full array utilization.
  - Extend `_conv2d` in `model.py` to offload `output_conv1` and `output_conv2_0` when `c_out >= 32` and MACs meet threshold.

### Phase 3 — Eliminate Per-Dispatch `insts` BO Allocation & Halve C DMA (`copy = 49.3 ms`)
- **Files**: `src/linux_studio_effects/video/models/video_depth_anything/npu_ops.py`, `_host_ops.c`, `host_ops.py`
- **Problem**:
  - `DefaultNPURuntime.run(..., dispatch_insts=insts)` allocates, copies, syncs `to_device`, and destroys a fresh `XrtTransport` BO for `insts` on every single dispatch (`60–70x` per frame).
  - `_build_whole_array_design` outputs `np.float32`, doubling C-matrix DMA bandwidth vs `bfloat16`.
  - Per-call `np.empty((m, n), dtype=np.float32)` triggers Linux kernel page faults for allocations above `M_MMAP_THRESHOLD` (`128 KB`).
- **Plan**:
  - Cache persistent `XrtTransport` instruction BOs keyed by `(m_pad, k_pad, n_pad)` and submit via `kernel_run.dispatch(...)`.
  - Add a `bfloat16` output MatMul xclbin option paired with fused AVX-512 `bf16_to_f32_unpad_bias_gelu` / `bf16_to_f32_unpad_transpose_bias_relu` to cut C DMA bytes by 2x.
  - Reuse scratch output buffers where downstream layers do not hold long-lived references.

### Phase 4 — Zero-Copy Temporal Attention Ring Buffer & Parallel Host C Loops (`temporal_attn_total = 47.64 ms`, `bilinear_resize = 12.06 ms`)
- **Files**: `src/linux_studio_effects/video/models/video_depth_anything/_host_ops.c`, `host_ops.py`, `model.py`
- **Problem**: `temporal_attn_roll_f32` shifts 28 rolling historical frames via `memmove` for every token on every frame (~`30 MB` copied across `m0`–`m3`), and all `_host_ops.c` kernels run single-threaded.
- **Plan**:
  - Track a per-slot rolling write cursor instead of shifting `k_buf`/`v_buf` with `memmove`, reading slots `[0, 1]` plus the modular window `[2..T-1]` directly with aligned positional embeddings.
  - Parallelize token/pixel loops across Zen4 threads (`bilinear_resize_nchw_f32`, `temporal_attn_roll_f32`, `im2col_3x3_s1_p1_kykxc_bf16_pad`, `bokeh_composite_u8`).

### Phase 5 — Fused Backbone Spatial Attention & Uint8-to-NCHW Preprocess (`~60 ms` CPU ops inside backbone)
- **Files**: `src/linux_studio_effects/video/models/video_depth_anything/_host_ops.c`, `host_ops.py`, `model.py`
- **Problem**: Inside `_forward_backbone`, 12 ViT blocks allocate transposed `qh`, `kh`, `vh`, `attn`, and `ctx` tensors for `Q @ K^T -> softmax -> Attn @ V`, and `preprocess` creates full-resolution `(1, 3, H, W)` float32 intermediates before resizing (`5.2 ms`).
- **Plan**:
  - Implement a fused AVX-512 multi-head spatial attention C kernel reading packed `qkv` directly (`q_stride = 3*C`) without intermediate transpose allocations.
  - Implement fused uint8 HWC `(H, W, 3)` bilinear resize + ImageNet normalization directly into `(1, 3, S, S)` float32 in `_host_ops.c`.
