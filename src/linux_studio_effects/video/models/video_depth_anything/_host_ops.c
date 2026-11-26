#define _POSIX_C_SOURCE 200809L
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/*
 * Fast multi-threaded AVX-512 / AVX-512-BF16 host kernels for Video Depth Anything.
 * Compiled with:
 *   gcc -O3 -mavx512f -mavx512bf16 -mavx512bw -mavx512vl -mfma -pthread -shared -fPIC
 */

#define LSE_MAX_THREADS 16

typedef void (*lse_task_fn)(int64_t start, int64_t end, void *arg);

typedef struct {
    int tid;
} lse_worker_arg_t;

typedef struct {
    pthread_mutex_t dispatch_mu;
    pthread_mutex_t mu;
    pthread_cond_t cv_work;
    pthread_cond_t cv_done;
    pthread_t threads[LSE_MAX_THREADS - 1];
    lse_worker_arg_t worker_args[LSE_MAX_THREADS - 1];
    lse_task_fn fn;
    void *arg;
    int64_t starts[LSE_MAX_THREADS];
    int64_t ends[LSE_MAX_THREADS];
    int n_threads;
    int active_workers;
    int remaining;
    uint64_t generation;
    int stop;
    int initialized;
} lse_pool_t;

static pthread_once_t g_pool_once = PTHREAD_ONCE_INIT;
static lse_pool_t g_pool = {
    .dispatch_mu = PTHREAD_MUTEX_INITIALIZER,
    .mu = PTHREAD_MUTEX_INITIALIZER,
    .cv_work = PTHREAD_COND_INITIALIZER,
    .cv_done = PTHREAD_COND_INITIALIZER,
};

static void *lse_worker_main(void *p) {
    int tid = ((const lse_worker_arg_t *)p)->tid;
    uint64_t seen_gen = 0;
    for (;;) {
        pthread_mutex_lock(&g_pool.mu);
        while (!g_pool.stop && (g_pool.generation == seen_gen || tid > g_pool.active_workers)) {
            pthread_cond_wait(&g_pool.cv_work, &g_pool.mu);
        }
        if (g_pool.stop) {
            pthread_mutex_unlock(&g_pool.mu);
            return NULL;
        }
        seen_gen = g_pool.generation;
        lse_task_fn fn = g_pool.fn;
        void *arg = g_pool.arg;
        int64_t s = g_pool.starts[tid];
        int64_t e = g_pool.ends[tid];
        pthread_mutex_unlock(&g_pool.mu);

        fn(s, e, arg);

        pthread_mutex_lock(&g_pool.mu);
        if (--g_pool.remaining == 0) {
            pthread_cond_signal(&g_pool.cv_done);
        }
        pthread_mutex_unlock(&g_pool.mu);
    }
    return NULL;
}

static void lse_atfork_child(void) {
    pthread_mutex_init(&g_pool.dispatch_mu, NULL);
    pthread_mutex_init(&g_pool.mu, NULL);
    pthread_cond_init(&g_pool.cv_work, NULL);
    pthread_cond_init(&g_pool.cv_done, NULL);
    g_pool.n_threads = 0;
    g_pool.active_workers = 0;
    g_pool.remaining = 0;
    g_pool.generation = 0;
    g_pool.stop = 0;
    g_pool.initialized = 0;
    g_pool_once = PTHREAD_ONCE_INIT;
}

static int lse_resolve_thread_count(void) {
    const char *env = getenv("LSE_THREADS");
    if (!env || !*env) {
        env = getenv("OMP_NUM_THREADS");
    }
    long online = sysconf(_SC_NPROCESSORS_ONLN);
    int max_hw = (online > 0) ? (int)online : 1;
    if (max_hw > LSE_MAX_THREADS) {
        max_hw = LSE_MAX_THREADS;
    }
    if (env && *env) {
        char *endptr = NULL;
        long req = strtol(env, &endptr, 10);
        if (endptr != env && req > 0) {
            if (req > LSE_MAX_THREADS) req = LSE_MAX_THREADS;
            return (int)req;
        }
    }
    return 1;
}

static void lse_pool_init(void) {
    pthread_atfork(NULL, NULL, lse_atfork_child);
    int n = lse_resolve_thread_count();
    g_pool.stop = 0;
    g_pool.generation = 0;
    g_pool.n_threads = n;
    for (int i = 0; i < n - 1; ++i) {
        g_pool.worker_args[i].tid = i + 1;
        if (pthread_create(&g_pool.threads[i], NULL, lse_worker_main, &g_pool.worker_args[i]) != 0) {
            g_pool.n_threads = i + 1;
            break;
        }
    }
    g_pool.initialized = 1;
}

__attribute__((destructor))
static void lse_pool_destroy(void) {
    if (!g_pool.initialized) {
        return;
    }
    pthread_mutex_lock(&g_pool.mu);
    g_pool.stop = 1;
    pthread_cond_broadcast(&g_pool.cv_work);
    pthread_mutex_unlock(&g_pool.mu);
    for (int i = 0; i < g_pool.n_threads - 1; ++i) {
        pthread_join(g_pool.threads[i], NULL);
    }
    g_pool.initialized = 0;
}

static void lse_parallel_for(int64_t total, int64_t min_chunk, lse_task_fn fn, void *arg) {
    if (total <= 0) {
        return;
    }
    pthread_once(&g_pool_once, lse_pool_init);
    if (g_pool.n_threads <= 1 || total <= min_chunk || pthread_mutex_trylock(&g_pool.dispatch_mu) != 0) {
        fn(0, total, arg);
        return;
    }
    int active = g_pool.n_threads;
    int64_t max_parts = (total + min_chunk - 1) / min_chunk;
    if (max_parts < active) {
        active = (int)max_parts;
    }
    if (active <= 1) {
        pthread_mutex_unlock(&g_pool.dispatch_mu);
        fn(0, total, arg);
        return;
    }
    int64_t base = total / active;
    int64_t rem = total % active;
    int64_t cur = 0;
    for (int t = 0; t < active; ++t) {
        int64_t len = base + (t < rem ? 1 : 0);
        g_pool.starts[t] = cur;
        g_pool.ends[t] = cur + len;
        cur += len;
    }
    pthread_mutex_lock(&g_pool.mu);
    g_pool.fn = fn;
    g_pool.arg = arg;
    g_pool.active_workers = active - 1;
    g_pool.remaining = active - 1;
    g_pool.generation++;
    pthread_cond_broadcast(&g_pool.cv_work);
    pthread_mutex_unlock(&g_pool.mu);

    fn(g_pool.starts[0], g_pool.ends[0], arg);

    pthread_mutex_lock(&g_pool.mu);
    while (g_pool.remaining > 0) {
        pthread_cond_wait(&g_pool.cv_done, &g_pool.mu);
    }
    pthread_mutex_unlock(&g_pool.mu);
    pthread_mutex_unlock(&g_pool.dispatch_mu);
}

static inline __m512 exp_ps_avx512(__m512 v) {
    /* Clamp to prevent overflow/underflow traps; <= -104 underflows cleanly to 0. */
    const __m512 min_val = _mm512_set1_ps(-104.0f);
    const __m512 max_val = _mm512_set1_ps(88.3762626647949f);
    v = _mm512_max_ps(v, min_val);
    v = _mm512_min_ps(v, max_val);

    const __m512 log2e = _mm512_set1_ps(1.44269504088896341f);
    const __m512 ln2_hi = _mm512_set1_ps(0.693145751953125f);
    const __m512 ln2_lo = _mm512_set1_ps(1.428606765330187045e-6f);

    const __m512 c1 = _mm512_set1_ps(1.0f);
    const __m512 c2 = _mm512_set1_ps(0.5f);
    const __m512 c3 = _mm512_set1_ps(0.1666666716337204f);
    const __m512 c4 = _mm512_set1_ps(0.0416666679084301f);
    const __m512 c5 = _mm512_set1_ps(0.008333333767950535f);
    const __m512 c6 = _mm512_set1_ps(0.0013888889225199819f);

    __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(v, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(n, ln2_hi, v);
    r = _mm512_fnmadd_ps(n, ln2_lo, r);

    __m512 p = _mm512_fmadd_ps(c6, r, c5);
    p = _mm512_fmadd_ps(p, r, c4);
    p = _mm512_fmadd_ps(p, r, c3);
    p = _mm512_fmadd_ps(p, r, c2);
    p = _mm512_fmadd_ps(p, r, c1);
    p = _mm512_fmadd_ps(p, r, c1);

    return _mm512_scalef_ps(p, n);
}

void f32_to_bf16_pad(
    const float *src,
    int64_t src_stride_elems,
    uint16_t *dst,
    int64_t dst_stride_elems,
    int64_t dst_rows,
    int64_t dst_cols,
    int64_t rows,
    int64_t cols,
    int zero_tail
) {
    for (int64_t r = 0; r < rows; ++r) {
        const float *srow = src + r * src_stride_elems;
        uint16_t *drow = dst + r * dst_stride_elems;
        int64_t c = 0;
        for (; c + 16 <= cols; c += 16) {
            __m512 vf = _mm512_loadu_ps(srow + c);
            __m256bh vb = _mm512_cvtneps_pbh(vf);
            _mm256_storeu_si256((__m256i *)(drow + c), (__m256i)vb);
        }
        if (c < cols) {
            __mmask16 km = (__mmask16)((1u << (cols - c)) - 1u);
            __m512 vf = _mm512_maskz_loadu_ps(km, srow + c);
            __m256bh vb = _mm512_cvtneps_pbh(vf);
            _mm256_mask_storeu_epi16(drow + c, km, (__m256i)vb);
        }
        if (dst_cols > cols) {
            memset(drow + cols, 0, (size_t)(dst_cols - cols) * sizeof(uint16_t));
        }
    }
    if (zero_tail && dst_rows > rows) {
        if (dst_stride_elems == dst_cols) {
            memset(
                dst + rows * dst_stride_elems,
                0,
                (size_t)(dst_rows - rows) * (size_t)dst_cols * sizeof(uint16_t)
            );
        } else {
            for (int64_t r = rows; r < dst_rows; ++r) {
                memset(dst + r * dst_stride_elems, 0, (size_t)dst_cols * sizeof(uint16_t));
            }
        }
    }
}

void bf16_to_f32_unpad(
    const uint16_t *src,
    int64_t src_stride_elems,
    float *dst,
    int64_t dst_stride_elems,
    int64_t rows,
    int64_t cols
) {
    for (int64_t r = 0; r < rows; ++r) {
        const uint16_t *srow = src + r * src_stride_elems;
        float *drow = dst + r * dst_stride_elems;
        int64_t c = 0;
        for (; c + 16 <= cols; c += 16) {
            __m256bh vb = (__m256bh)_mm256_loadu_si256((const __m256i *)(srow + c));
            __m512 vf = _mm512_cvtpbh_ps(vb);
            _mm512_storeu_ps(drow + c, vf);
        }
        if (c < cols) {
            __mmask16 km = (__mmask16)((1u << (cols - c)) - 1u);
            __m256bh vb = (__m256bh)_mm256_maskz_loadu_epi16(km, srow + c);
            __m512 vf = _mm512_cvtpbh_ps(vb);
            _mm512_mask_storeu_ps(drow + c, km, vf);
        }
    }
}

static inline void interp_row_x_f32(
    const float *in_row,
    float *dst,
    int out_w,
    const int32_t *x0,
    const int32_t *x1,
    const float *wx
) {
    int ox = 0;
    for (; ox + 16 <= out_w; ox += 16) {
        __m512i idx0 = _mm512_loadu_si512((const __m512i *)(x0 + ox));
        __m512i idx1 = _mm512_loadu_si512((const __m512i *)(x1 + ox));
        __m512 v0 = _mm512_i32gather_ps(idx0, in_row, 4);
        __m512 v1 = _mm512_i32gather_ps(idx1, in_row, 4);
        __m512 vw = _mm512_loadu_ps(wx + ox);
        __m512 res = _mm512_fmadd_ps(vw, _mm512_sub_ps(v1, v0), v0);
        _mm512_storeu_ps(dst + ox, res);
    }
    for (; ox < out_w; ++ox) {
        float v0 = in_row[x0[ox]];
        float v1 = in_row[x1[ox]];
        dst[ox] = v0 + wx[ox] * (v1 - v0);
    }
}

typedef struct {
    const float *x;
    float *out;
    const int32_t *x0;
    const int32_t *x1;
    const int32_t *y0;
    const int32_t *y1;
    const float *wx;
    const float *wy;
    int in_h;
    int in_w;
    int out_h;
    int out_w;
} bilinear_nchw_ctx_t;

static void bilinear_nchw_worker(int64_t c_start, int64_t c_end, void *arg) {
    const bilinear_nchw_ctx_t *ctx = (const bilinear_nchw_ctx_t *)arg;
    const int in_h = ctx->in_h;
    const int in_w = ctx->in_w;
    const int out_h = ctx->out_h;
    const int out_w = ctx->out_w;
    const int32_t *x0 = ctx->x0;
    const int32_t *x1 = ctx->x1;
    const int32_t *y0 = ctx->y0;
    const int32_t *y1 = ctx->y1;
    const float *wx = ctx->wx;
    const float *wy = ctx->wy;

    float stack_rows[4096];
    float *heap_rows = (out_w > 2048) ? (float *)malloc((size_t)out_w * 2 * sizeof(float)) : NULL;
    float *row_a = heap_rows ? heap_rows : stack_rows;
    float *row_b = row_a + out_w;

    const size_t in_plane = (size_t)in_h * (size_t)in_w;
    const size_t out_plane = (size_t)out_h * (size_t)out_w;

    for (int64_t c = c_start; c < c_end; ++c) {
        const float *in_ch = ctx->x + (size_t)c * in_plane;
        float *out_ch = ctx->out + (size_t)c * out_plane;

        int cached_id[2] = {-1, -1};
        float *slots[2] = {row_a, row_b};

        for (int oy = 0; oy < out_h; ++oy) {
            int r0 = y0[oy];
            int r1 = y1[oy];
            float w = wy[oy];

            float *p0;
            if (r0 == cached_id[0]) {
                p0 = slots[0];
            } else if (r0 == cached_id[1]) {
                p0 = slots[1];
            } else {
                int slot = (cached_id[0] == r1) ? 1 : 0;
                interp_row_x_f32(in_ch + (size_t)r0 * in_w, slots[slot], out_w, x0, x1, wx);
                cached_id[slot] = r0;
                p0 = slots[slot];
            }

            float *out_row = out_ch + (size_t)oy * out_w;
            if (r1 == r0 || w == 0.0f) {
                memcpy(out_row, p0, (size_t)out_w * sizeof(float));
                continue;
            }

            float *p1;
            if (r1 == cached_id[0]) {
                p1 = slots[0];
            } else if (r1 == cached_id[1]) {
                p1 = slots[1];
            } else {
                int slot = (p0 == slots[0]) ? 1 : 0;
                interp_row_x_f32(in_ch + (size_t)r1 * in_w, slots[slot], out_w, x0, x1, wx);
                cached_id[slot] = r1;
                p1 = slots[slot];
            }

            __m512 vwy = _mm512_set1_ps(w);
            int ox = 0;
            for (; ox + 16 <= out_w; ox += 16) {
                __m512 va = _mm512_loadu_ps(p0 + ox);
                __m512 vb = _mm512_loadu_ps(p1 + ox);
                _mm512_storeu_ps(out_row + ox, _mm512_fmadd_ps(vwy, _mm512_sub_ps(vb, va), va));
            }
            if (ox < out_w) {
                __mmask16 km = (__mmask16)((1u << (out_w - ox)) - 1u);
                __m512 va = _mm512_maskz_loadu_ps(km, p0 + ox);
                __m512 vb = _mm512_maskz_loadu_ps(km, p1 + ox);
                _mm512_mask_storeu_ps(
                    out_row + ox, km, _mm512_fmadd_ps(vwy, _mm512_sub_ps(vb, va), va)
                );
            }
        }
    }

    if (heap_rows) {
        free(heap_rows);
    }
}

int bilinear_resize_nchw_f32(
    const float *x,
    float *out,
    int nc,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int align_corners
) {
    if (nc <= 0 || in_h <= 0 || in_w <= 0 || out_h <= 0 || out_w <= 0) {
        return -1;
    }
    if (in_h == out_h && in_w == out_w) {
        memcpy(out, x, (size_t)nc * (size_t)in_h * (size_t)in_w * sizeof(float));
        return 0;
    }

    size_t int_count = (size_t)out_w * 2 + (size_t)out_h * 2;
    size_t flt_count = (size_t)out_w + (size_t)out_h;
    int32_t *ibuf = (int32_t *)malloc(int_count * sizeof(int32_t));
    float *fbuf = (float *)malloc(flt_count * sizeof(float));
    if (!ibuf || !fbuf) {
        free(ibuf);
        free(fbuf);
        return -1;
    }

    int32_t *x0 = ibuf;
    int32_t *x1 = x0 + out_w;
    int32_t *y0 = x1 + out_w;
    int32_t *y1 = y0 + out_h;

    float *wx = fbuf;
    float *wy = wx + out_w;

    if (out_w == 1) {
        x0[0] = 0;
        x1[0] = 0;
        wx[0] = 0.0f;
    } else if (align_corners) {
        double step = (double)(in_w - 1) / (double)(out_w - 1);
        for (int i = 0; i < out_w; ++i) {
            float xc = (i == out_w - 1) ? (float)(in_w - 1) : (float)((double)i * step);
            int i0 = (int)floorf(xc);
            if (i0 < 0) i0 = 0;
            if (i0 >= in_w) i0 = in_w - 1;
            int i1 = (i0 + 1 < in_w) ? i0 + 1 : in_w - 1;
            x0[i] = i0;
            x1[i] = i1;
            wx[i] = xc - (float)i0;
        }
    } else {
        float scale = (float)((double)in_w / (double)out_w);
        for (int i = 0; i < out_w; ++i) {
            float prod = ((float)i + 0.5f) * scale;
            __asm__("" : "+x"(prod));
            float xc = prod - 0.5f;
            if (xc < 0.0f) xc = 0.0f;
            if (xc > (float)(in_w - 1)) xc = (float)(in_w - 1);
            int i0 = (int)floorf(xc);
            int i1 = (i0 + 1 < in_w) ? i0 + 1 : in_w - 1;
            x0[i] = i0;
            x1[i] = i1;
            wx[i] = xc - (float)i0;
        }
    }

    if (out_h == 1) {
        y0[0] = 0;
        y1[0] = 0;
        wy[0] = 0.0f;
    } else if (align_corners) {
        double step = (double)(in_h - 1) / (double)(out_h - 1);
        for (int i = 0; i < out_h; ++i) {
            float yc = (i == out_h - 1) ? (float)(in_h - 1) : (float)((double)i * step);
            int i0 = (int)floorf(yc);
            if (i0 < 0) i0 = 0;
            if (i0 >= in_h) i0 = in_h - 1;
            int i1 = (i0 + 1 < in_h) ? i0 + 1 : in_h - 1;
            y0[i] = i0;
            y1[i] = i1;
            wy[i] = yc - (float)i0;
        }
    } else {
        float scale = (float)((double)in_h / (double)out_h);
        for (int i = 0; i < out_h; ++i) {
            float prod = ((float)i + 0.5f) * scale;
            __asm__("" : "+x"(prod));
            float yc = prod - 0.5f;
            if (yc < 0.0f) yc = 0.0f;
            if (yc > (float)(in_h - 1)) yc = (float)(in_h - 1);
            int i0 = (int)floorf(yc);
            int i1 = (i0 + 1 < in_h) ? i0 + 1 : in_h - 1;
            y0[i] = i0;
            y1[i] = i1;
            wy[i] = yc - (float)i0;
        }
    }

    bilinear_nchw_ctx_t ctx = {
        .x = x,
        .out = out,
        .x0 = x0,
        .x1 = x1,
        .y0 = y0,
        .y1 = y1,
        .wx = wx,
        .wy = wy,
        .in_h = in_h,
        .in_w = in_w,
        .out_h = out_h,
        .out_w = out_w,
    };

    const size_t work = (size_t)nc * (size_t)out_h * (size_t)out_w;
    if (nc >= 2 && work >= 16384) {
        lse_parallel_for((int64_t)nc, 1, bilinear_nchw_worker, &ctx);
    } else {
        bilinear_nchw_worker(0, (int64_t)nc, &ctx);
    }

    free(ibuf);
    free(fbuf);
    return 0;
}

void layer_norm_f32(
    const float *x,
    const float *weight,
    const float *bias,
    float *out,
    int64_t M,
    int64_t D,
    float eps
) {
    if (M <= 0 || D <= 0) {
        return;
    }
    const float inv_d = 1.0f / (float)D;
    for (int64_t r = 0; r < M; ++r) {
        const float *xrow = x + r * D;
        float *orow = out + r * D;

        __m512 s0 = _mm512_setzero_ps();
        __m512 s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps();
        __m512 s3 = _mm512_setzero_ps();
        int64_t j = 0;
        for (; j + 64 <= D; j += 64) {
            s0 = _mm512_add_ps(s0, _mm512_loadu_ps(xrow + j));
            s1 = _mm512_add_ps(s1, _mm512_loadu_ps(xrow + j + 16));
            s2 = _mm512_add_ps(s2, _mm512_loadu_ps(xrow + j + 32));
            s3 = _mm512_add_ps(s3, _mm512_loadu_ps(xrow + j + 48));
        }
        s0 = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
        for (; j + 16 <= D; j += 16) {
            s0 = _mm512_add_ps(s0, _mm512_loadu_ps(xrow + j));
        }
        if (j < D) {
            __mmask16 km = (__mmask16)((1u << (D - j)) - 1u);
            s0 = _mm512_add_ps(s0, _mm512_maskz_loadu_ps(km, xrow + j));
        }
        float mean = _mm512_reduce_add_ps(s0) * inv_d;
        __m512 vmean = _mm512_set1_ps(mean);

        __m512 v0 = _mm512_setzero_ps();
        __m512 v1 = _mm512_setzero_ps();
        __m512 v2 = _mm512_setzero_ps();
        __m512 v3 = _mm512_setzero_ps();
        j = 0;
        for (; j + 64 <= D; j += 64) {
            __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(xrow + j), vmean);
            __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(xrow + j + 16), vmean);
            __m512 d2 = _mm512_sub_ps(_mm512_loadu_ps(xrow + j + 32), vmean);
            __m512 d3 = _mm512_sub_ps(_mm512_loadu_ps(xrow + j + 48), vmean);
            v0 = _mm512_fmadd_ps(d0, d0, v0);
            v1 = _mm512_fmadd_ps(d1, d1, v1);
            v2 = _mm512_fmadd_ps(d2, d2, v2);
            v3 = _mm512_fmadd_ps(d3, d3, v3);
        }
        v0 = _mm512_add_ps(_mm512_add_ps(v0, v1), _mm512_add_ps(v2, v3));
        for (; j + 16 <= D; j += 16) {
            __m512 d = _mm512_sub_ps(_mm512_loadu_ps(xrow + j), vmean);
            v0 = _mm512_fmadd_ps(d, d, v0);
        }
        if (j < D) {
            __mmask16 km = (__mmask16)((1u << (D - j)) - 1u);
            __m512 d = _mm512_maskz_sub_ps(km, _mm512_maskz_loadu_ps(km, xrow + j), vmean);
            v0 = _mm512_fmadd_ps(d, d, v0);
        }
        float var = _mm512_reduce_add_ps(v0) * inv_d;
        float inv_std = 1.0f / sqrtf(var + eps);
        __m512 vinv = _mm512_set1_ps(inv_std);

        j = 0;
        for (; j + 16 <= D; j += 16) {
            __m512 d = _mm512_sub_ps(_mm512_loadu_ps(xrow + j), vmean);
            __m512 normed = _mm512_mul_ps(d, vinv);
            __m512 w = _mm512_loadu_ps(weight + j);
            __m512 b = _mm512_loadu_ps(bias + j);
            _mm512_storeu_ps(orow + j, _mm512_fmadd_ps(normed, w, b));
        }
        if (j < D) {
            __mmask16 km = (__mmask16)((1u << (D - j)) - 1u);
            __m512 d = _mm512_sub_ps(_mm512_maskz_loadu_ps(km, xrow + j), vmean);
            __m512 normed = _mm512_mul_ps(d, vinv);
            __m512 w = _mm512_maskz_loadu_ps(km, weight + j);
            __m512 b = _mm512_maskz_loadu_ps(km, bias + j);
            _mm512_mask_storeu_ps(orow + j, km, _mm512_fmadd_ps(normed, w, b));
        }
    }
}

void softmax_rows_f32(const float *x, float *out, int64_t M, int64_t D) {
    if (M <= 0 || D <= 0) {
        return;
    }
    if (D == 1) {
        for (int64_t r = 0; r < M; ++r) {
            out[r] = 1.0f;
        }
        return;
    }
    for (int64_t r = 0; r < M; ++r) {
        const float *xrow = x + r * D;
        float *orow = out + r * D;

        __m512 vmax = _mm512_set1_ps(-INFINITY);
        int64_t j = 0;
        for (; j + 16 <= D; j += 16) {
            vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(xrow + j));
        }
        if (j < D) {
            __mmask16 km = (__mmask16)((1u << (D - j)) - 1u);
            __m512 vtail = _mm512_mask_loadu_ps(vmax, km, xrow + j);
            vmax = _mm512_max_ps(vmax, vtail);
        }
        float max_val = _mm512_reduce_max_ps(vmax);
        vmax = _mm512_set1_ps(max_val);

        __m512 vsum = _mm512_setzero_ps();
        j = 0;
        for (; j + 16 <= D; j += 16) {
            __m512 e = exp_ps_avx512(_mm512_sub_ps(_mm512_loadu_ps(xrow + j), vmax));
            _mm512_storeu_ps(orow + j, e);
            vsum = _mm512_add_ps(vsum, e);
        }
        if (j < D) {
            __mmask16 km = (__mmask16)((1u << (D - j)) - 1u);
            __m512 vdiff = _mm512_sub_ps(_mm512_maskz_loadu_ps(km, xrow + j), vmax);
            __m512 e = exp_ps_avx512(vdiff);
            _mm512_mask_storeu_ps(orow + j, km, e);
            vsum = _mm512_mask_add_ps(vsum, km, vsum, e);
        }
        float inv_sum = 1.0f / _mm512_reduce_add_ps(vsum);
        __m512 vinv = _mm512_set1_ps(inv_sum);

        j = 0;
        for (; j + 16 <= D; j += 16) {
            _mm512_storeu_ps(orow + j, _mm512_mul_ps(_mm512_loadu_ps(orow + j), vinv));
        }
        if (j < D) {
            __mmask16 km = (__mmask16)((1u << (D - j)) - 1u);
            __m512 v = _mm512_maskz_loadu_ps(km, orow + j);
            _mm512_mask_storeu_ps(orow + j, km, _mm512_mul_ps(v, vinv));
        }
    }
}

static inline __m512 gelu_vec_avx512(__m512 vx) {
    const __m512 c_044715 = _mm512_set1_ps(0.044715f);
    const __m512 c_sqrt_2_pi = _mm512_set1_ps(0.7978845608028654f);
    const __m512 c_neg2 = _mm512_set1_ps(-2.0f);
    const __m512 c_one = _mm512_set1_ps(1.0f);
    const __m512i abs_mask = _mm512_set1_epi32(0x7fffffff);

    __m512 vx2 = _mm512_mul_ps(vx, vx);
    __m512 vx3 = _mm512_mul_ps(vx2, vx);
    __m512 inner = _mm512_fmadd_ps(c_044715, vx3, vx);
    __m512 z = _mm512_mul_ps(c_sqrt_2_pi, inner);

    __m512 az = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(z), abs_mask));
    __m512 e = exp_ps_avx512(_mm512_mul_ps(az, c_neg2));
    __m512 denom = _mm512_add_ps(c_one, e);
    __mmask16 is_neg = _mm512_cmp_ps_mask(z, _mm512_setzero_ps(), _CMP_LT_OQ);
    __m512 num = _mm512_mask_mov_ps(c_one, is_neg, e);
    __m512 sig = _mm512_div_ps(num, denom);
    return _mm512_mul_ps(vx, sig);
}

void gelu_inplace_f32(float *x, int64_t rows, int64_t cols, int64_t row_stride_elems) {
    for (int64_t r = 0; r < rows; ++r) {
        float *row = x + r * row_stride_elems;
        int64_t c = 0;
        for (; c + 16 <= cols; c += 16) {
            __m512 vx = _mm512_loadu_ps(row + c);
            _mm512_storeu_ps(row + c, gelu_vec_avx512(vx));
        }
        if (c < cols) {
            __mmask16 km = (__mmask16)((1u << (cols - c)) - 1u);
            __m512 vx = _mm512_maskz_loadu_ps(km, row + c);
            _mm512_mask_storeu_ps(row + c, km, gelu_vec_avx512(vx));
        }
    }
}

void group_norm_nchw_f32(
    const float *x,
    const float *weight,
    const float *bias,
    float *out,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t num_groups,
    float eps
) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0 || num_groups <= 0) {
        return;
    }
    const int64_t c_per_g = C / num_groups;
    const int64_t hw = H * W;
    const int64_t g_size = c_per_g * hw;
    const double inv_g = 1.0 / (double)g_size;

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t g = 0; g < num_groups; ++g) {
            const float *gx = x + (n * C + g * c_per_g) * hw;
            float *go = out + (n * C + g * c_per_g) * hw;

            __m512 s0 = _mm512_setzero_ps();
            __m512 s1 = _mm512_setzero_ps();
            __m512 s2 = _mm512_setzero_ps();
            __m512 s3 = _mm512_setzero_ps();
            int64_t i = 0;
            for (; i + 64 <= g_size; i += 64) {
                s0 = _mm512_add_ps(s0, _mm512_loadu_ps(gx + i));
                s1 = _mm512_add_ps(s1, _mm512_loadu_ps(gx + i + 16));
                s2 = _mm512_add_ps(s2, _mm512_loadu_ps(gx + i + 32));
                s3 = _mm512_add_ps(s3, _mm512_loadu_ps(gx + i + 48));
            }
            s0 = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
            for (; i + 16 <= g_size; i += 16) {
                s0 = _mm512_add_ps(s0, _mm512_loadu_ps(gx + i));
            }
            if (i < g_size) {
                __mmask16 km = (__mmask16)((1u << (g_size - i)) - 1u);
                s0 = _mm512_add_ps(s0, _mm512_maskz_loadu_ps(km, gx + i));
            }
            float mean = (float)((double)_mm512_reduce_add_ps(s0) * inv_g);
            __m512 vmean = _mm512_set1_ps(mean);

            __m512 v0 = _mm512_setzero_ps();
            __m512 v1 = _mm512_setzero_ps();
            __m512 v2 = _mm512_setzero_ps();
            __m512 v3 = _mm512_setzero_ps();
            i = 0;
            for (; i + 64 <= g_size; i += 64) {
                __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(gx + i), vmean);
                __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(gx + i + 16), vmean);
                __m512 d2 = _mm512_sub_ps(_mm512_loadu_ps(gx + i + 32), vmean);
                __m512 d3 = _mm512_sub_ps(_mm512_loadu_ps(gx + i + 48), vmean);
                v0 = _mm512_fmadd_ps(d0, d0, v0);
                v1 = _mm512_fmadd_ps(d1, d1, v1);
                v2 = _mm512_fmadd_ps(d2, d2, v2);
                v3 = _mm512_fmadd_ps(d3, d3, v3);
            }
            v0 = _mm512_add_ps(_mm512_add_ps(v0, v1), _mm512_add_ps(v2, v3));
            for (; i + 16 <= g_size; i += 16) {
                __m512 d = _mm512_sub_ps(_mm512_loadu_ps(gx + i), vmean);
                v0 = _mm512_fmadd_ps(d, d, v0);
            }
            if (i < g_size) {
                __mmask16 km = (__mmask16)((1u << (g_size - i)) - 1u);
                __m512 d = _mm512_maskz_sub_ps(km, _mm512_maskz_loadu_ps(km, gx + i), vmean);
                v0 = _mm512_fmadd_ps(d, d, v0);
            }
            double var = (double)_mm512_reduce_add_ps(v0) * inv_g;
            float inv_std = (float)(1.0 / sqrt(var + (double)eps));

            for (int64_t cl = 0; cl < c_per_g; ++cl) {
                int64_t c = g * c_per_g + cl;
                const float *cx = gx + cl * hw;
                float *co = go + cl * hw;
                __m512 vscale = _mm512_set1_ps(inv_std * weight[c]);
                __m512 vbias = _mm512_set1_ps(bias[c]);
                int64_t j = 0;
                for (; j + 16 <= hw; j += 16) {
                    __m512 d = _mm512_sub_ps(_mm512_loadu_ps(cx + j), vmean);
                    _mm512_storeu_ps(co + j, _mm512_fmadd_ps(d, vscale, vbias));
                }
                if (j < hw) {
                    __mmask16 km = (__mmask16)((1u << (hw - j)) - 1u);
                    __m512 d = _mm512_sub_ps(_mm512_maskz_loadu_ps(km, cx + j), vmean);
                    _mm512_mask_storeu_ps(co + j, km, _mm512_fmadd_ps(d, vscale, vbias));
                }
            }
        }
    }
}

void im2col_3x3_s1_p1_f32(
    const float *x,
    float *out_cols,
    int64_t C,
    int64_t H,
    int64_t W
) {
    const int64_t hw = H * W;
    const size_t w_bytes = (size_t)W * sizeof(float);
    const size_t w_minus_1_bytes = (W > 1) ? (size_t)(W - 1) * sizeof(float) : 0;

    for (int64_t c = 0; c < C; ++c) {
        const float *in_c = x + c * hw;
        float *base = out_cols + c * 9 * hw;

        /* k=0: ky=-1, kx=-1 */
        float *d0 = base;
        memset(d0, 0, w_bytes);
        for (int64_t y = 1; y < H; ++y) {
            const float *srow = in_c + (y - 1) * W;
            float *drow = d0 + y * W;
            drow[0] = 0.0f;
            if (w_minus_1_bytes) memcpy(drow + 1, srow, w_minus_1_bytes);
        }

        /* k=1: ky=-1, kx=0 */
        float *d1 = base + hw;
        memset(d1, 0, w_bytes);
        if (H > 1) memcpy(d1 + W, in_c, (size_t)(H - 1) * w_bytes);

        /* k=2: ky=-1, kx=+1 */
        float *d2 = base + 2 * hw;
        memset(d2, 0, w_bytes);
        for (int64_t y = 1; y < H; ++y) {
            const float *srow = in_c + (y - 1) * W;
            float *drow = d2 + y * W;
            if (w_minus_1_bytes) memcpy(drow, srow + 1, w_minus_1_bytes);
            drow[W - 1] = 0.0f;
        }

        /* k=3: ky=0, kx=-1 */
        float *d3 = base + 3 * hw;
        for (int64_t y = 0; y < H; ++y) {
            const float *srow = in_c + y * W;
            float *drow = d3 + y * W;
            drow[0] = 0.0f;
            if (w_minus_1_bytes) memcpy(drow + 1, srow, w_minus_1_bytes);
        }

        /* k=4: ky=0, kx=0 */
        memcpy(base + 4 * hw, in_c, (size_t)hw * sizeof(float));

        /* k=5: ky=0, kx=+1 */
        float *d5 = base + 5 * hw;
        for (int64_t y = 0; y < H; ++y) {
            const float *srow = in_c + y * W;
            float *drow = d5 + y * W;
            if (w_minus_1_bytes) memcpy(drow, srow + 1, w_minus_1_bytes);
            drow[W - 1] = 0.0f;
        }

        /* k=6: ky=+1, kx=-1 */
        float *d6 = base + 6 * hw;
        for (int64_t y = 0; y < H - 1; ++y) {
            const float *srow = in_c + (y + 1) * W;
            float *drow = d6 + y * W;
            drow[0] = 0.0f;
            if (w_minus_1_bytes) memcpy(drow + 1, srow, w_minus_1_bytes);
        }
        memset(d6 + (H - 1) * W, 0, w_bytes);

        /* k=7: ky=+1, kx=0 */
        float *d7 = base + 7 * hw;
        if (H > 1) memcpy(d7, in_c + W, (size_t)(H - 1) * w_bytes);
        memset(d7 + (H - 1) * W, 0, w_bytes);

        /* k=8: ky=+1, kx=+1 */
        float *d8 = base + 8 * hw;
        for (int64_t y = 0; y < H - 1; ++y) {
            const float *srow = in_c + (y + 1) * W;
            float *drow = d8 + y * W;
            if (w_minus_1_bytes) memcpy(drow, srow + 1, w_minus_1_bytes);
            drow[W - 1] = 0.0f;
        }
        memset(d8 + (H - 1) * W, 0, w_bytes);
    }
}

typedef struct {
    const uint8_t *frame_u8;
    const float *w0;
    float *out_hw4;
    int in_h;
    int in_w;
    int out_w;
} downsample2x_bg_ctx_t;

static void downsample2x_bg_worker(int64_t oy_start, int64_t oy_end, void *arg) {
    const downsample2x_bg_ctx_t *ctx = (const downsample2x_bg_ctx_t *)arg;
    const int in_h = ctx->in_h;
    const int in_w = ctx->in_w;
    const int out_w = ctx->out_w;
    for (int oy = (int)oy_start; oy < (int)oy_end; ++oy) {
        const int y0 = oy * 2;
        const int y1 = (y0 + 1 < in_h) ? (y0 + 1) : y0;
        const uint8_t *f_row0 = ctx->frame_u8 + (size_t)y0 * (size_t)in_w * 3;
        const uint8_t *f_row1 = ctx->frame_u8 + (size_t)y1 * (size_t)in_w * 3;
        const float *w_row0 = ctx->w0 + (size_t)y0 * (size_t)in_w;
        const float *w_row1 = ctx->w0 + (size_t)y1 * (size_t)in_w;
        float *dst = ctx->out_hw4 + (size_t)oy * (size_t)out_w * 4;

        for (int ox = 0; ox < out_w; ++ox) {
            const int x0 = ox * 2;
            const int x1 = (x0 + 1 < in_w) ? (x0 + 1) : x0;
            const float wa = w_row0[x0];
            const float wb = w_row0[x1];
            const float wc = w_row1[x0];
            const float wd = w_row1[x1];

            const uint8_t *pa = f_row0 + (size_t)x0 * 3;
            const uint8_t *pb = f_row0 + (size_t)x1 * 3;
            const uint8_t *pc = f_row1 + (size_t)x0 * 3;
            const uint8_t *pd = f_row1 + (size_t)x1 * 3;

            dst[ox * 4 + 0] = 0.25f * ((float)pa[0] * wa + (float)pb[0] * wb + (float)pc[0] * wc + (float)pd[0] * wd);
            dst[ox * 4 + 1] = 0.25f * ((float)pa[1] * wa + (float)pb[1] * wb + (float)pc[1] * wc + (float)pd[1] * wd);
            dst[ox * 4 + 2] = 0.25f * ((float)pa[2] * wa + (float)pb[2] * wb + (float)pc[2] * wc + (float)pd[2] * wd);
            dst[ox * 4 + 3] = 0.25f * (wa + wb + wc + wd);
        }
    }
}

void downsample2x_bg_hw4_f32(
    const uint8_t *frame_u8,
    const float *w0,
    float *out_hw4,
    int in_h,
    int in_w
) {
    const int out_h = (in_h + 1) >> 1;
    const int out_w = (in_w + 1) >> 1;
    downsample2x_bg_ctx_t ctx = {
        .frame_u8 = frame_u8,
        .w0 = w0,
        .out_hw4 = out_hw4,
        .in_h = in_h,
        .in_w = in_w,
        .out_w = out_w,
    };
    if (out_h >= 16 && (int64_t)out_h * out_w >= 8192) {
        lse_parallel_for((int64_t)out_h, 8, downsample2x_bg_worker, &ctx);
    } else {
        downsample2x_bg_worker(0, (int64_t)out_h, &ctx);
    }
}

int bilinear_resize_hwc_f32_to_u8(
    const float *x,
    uint8_t *out,
    int in_h,
    int in_w,
    int out_h,
    int out_w
) {
    if (in_h <= 0 || in_w <= 0 || out_h <= 0 || out_w <= 0) {
        return -1;
    }

    size_t int_count = (size_t)out_w * 2 + (size_t)out_h * 2;
    size_t flt_count = (size_t)out_w * 7 + (size_t)out_h;
    int32_t *ibuf = (int32_t *)malloc(int_count * sizeof(int32_t));
    float *fbuf = (float *)malloc(flt_count * sizeof(float));
    if (!ibuf || !fbuf) {
        free(ibuf);
        free(fbuf);
        return -1;
    }

    int32_t *x0 = ibuf;
    int32_t *x1 = x0 + out_w;
    int32_t *y0 = x1 + out_w;
    int32_t *y1 = y0 + out_h;

    float *wx = fbuf;
    float *wy = wx + out_w;
    float *row_a = wy + out_h;
    float *row_b = row_a + (size_t)out_w * 3;

    if (in_w == 1 || out_w == 1) {
        for (int i = 0; i < out_w; ++i) {
            x0[i] = 0;
            x1[i] = 0;
            wx[i] = 0.0f;
        }
    } else {
        float scale = (float)((double)in_w / (double)out_w);
        for (int i = 0; i < out_w; ++i) {
            float prod = ((float)i + 0.5f) * scale;
            __asm__("" : "+x"(prod));
            float xc = prod - 0.5f;
            if (xc < 0.0f) xc = 0.0f;
            if (xc > (float)(in_w - 1)) xc = (float)(in_w - 1);
            int i0 = (int)floorf(xc);
            int i1 = (i0 + 1 < in_w) ? i0 + 1 : in_w - 1;
            x0[i] = i0;
            x1[i] = i1;
            wx[i] = xc - (float)i0;
        }
    }

    if (in_h == 1 || out_h == 1) {
        for (int i = 0; i < out_h; ++i) {
            y0[i] = 0;
            y1[i] = 0;
            wy[i] = 0.0f;
        }
    } else {
        float scale = (float)((double)in_h / (double)out_h);
        for (int i = 0; i < out_h; ++i) {
            float prod = ((float)i + 0.5f) * scale;
            __asm__("" : "+x"(prod));
            float yc = prod - 0.5f;
            if (yc < 0.0f) yc = 0.0f;
            if (yc > (float)(in_h - 1)) yc = (float)(in_h - 1);
            int i0 = (int)floorf(yc);
            int i1 = (i0 + 1 < in_h) ? i0 + 1 : in_h - 1;
            y0[i] = i0;
            y1[i] = i1;
            wy[i] = yc - (float)i0;
        }
    }

    const int out_w3 = out_w * 3;
    const size_t in_stride = (size_t)in_w * 3;
    int cached_id[2] = {-1, -1};
    float *slots[2] = {row_a, row_b};
    const __m512 vzero = _mm512_setzero_ps();
    const __m512 v255 = _mm512_set1_ps(255.0f);

    for (int oy = 0; oy < out_h; ++oy) {
        const int r0 = y0[oy];
        const int r1 = y1[oy];
        const float wyv = wy[oy];

        float *p0;
        if (r0 == cached_id[0]) {
            p0 = slots[0];
        } else if (r0 == cached_id[1]) {
            p0 = slots[1];
        } else {
            int slot = (cached_id[0] == r1) ? 1 : 0;
            const float *in_row = x + (size_t)r0 * in_stride;
            float *dst = slots[slot];
            for (int ox = 0; ox < out_w; ++ox) {
                const float *s0 = in_row + (size_t)x0[ox] * 3;
                const float *s1 = in_row + (size_t)x1[ox] * 3;
                const float w = wx[ox];
                dst[ox * 3 + 0] = s0[0] + w * (s1[0] - s0[0]);
                dst[ox * 3 + 1] = s0[1] + w * (s1[1] - s0[1]);
                dst[ox * 3 + 2] = s0[2] + w * (s1[2] - s0[2]);
            }
            cached_id[slot] = r0;
            p0 = slots[slot];
        }

        uint8_t *out_row = out + (size_t)oy * (size_t)out_w3;
        if (r1 == r0 || wyv == 0.0f) {
            int k = 0;
            for (; k + 16 <= out_w3; k += 16) {
                __m512 vf = _mm512_loadu_ps(p0 + k);
                vf = _mm512_min_ps(_mm512_max_ps(vf, vzero), v255);
                __m512i vi = _mm512_cvtps_epi32(
                    _mm512_roundscale_ps(vf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
                );
                _mm_storeu_si128((__m128i *)(out_row + k), _mm512_cvtusepi32_epi8(vi));
            }
            for (; k < out_w3; ++k) {
                float v = p0[k];
                if (v < 0.0f) v = 0.0f;
                if (v > 255.0f) v = 255.0f;
                out_row[k] = (uint8_t)lrintf(v);
            }
            continue;
        }

        float *p1;
        if (r1 == cached_id[0]) {
            p1 = slots[0];
        } else if (r1 == cached_id[1]) {
            p1 = slots[1];
        } else {
            int slot = (p0 == slots[0]) ? 1 : 0;
            const float *in_row = x + (size_t)r1 * in_stride;
            float *dst = slots[slot];
            for (int ox = 0; ox < out_w; ++ox) {
                const float *s0 = in_row + (size_t)x0[ox] * 3;
                const float *s1 = in_row + (size_t)x1[ox] * 3;
                const float w = wx[ox];
                dst[ox * 3 + 0] = s0[0] + w * (s1[0] - s0[0]);
                dst[ox * 3 + 1] = s0[1] + w * (s1[1] - s0[1]);
                dst[ox * 3 + 2] = s0[2] + w * (s1[2] - s0[2]);
            }
            cached_id[slot] = r1;
            p1 = slots[slot];
        }

        const __m512 vwy = _mm512_set1_ps(wyv);
        int k = 0;
        for (; k + 16 <= out_w3; k += 16) {
            __m512 va = _mm512_loadu_ps(p0 + k);
            __m512 vb = _mm512_loadu_ps(p1 + k);
            __m512 vf = _mm512_fmadd_ps(vwy, _mm512_sub_ps(vb, va), va);
            vf = _mm512_min_ps(_mm512_max_ps(vf, vzero), v255);
            __m512i vi = _mm512_cvtps_epi32(
                _mm512_roundscale_ps(vf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
            );
            _mm_storeu_si128((__m128i *)(out_row + k), _mm512_cvtusepi32_epi8(vi));
        }
        for (; k < out_w3; ++k) {
            float v = p0[k] + wyv * (p1[k] - p0[k]);
            if (v < 0.0f) v = 0.0f;
            if (v > 255.0f) v = 255.0f;
            out_row[k] = (uint8_t)lrintf(v);
        }
    }

    free(ibuf);
    free(fbuf);
    return 0;
}

typedef struct {
    const uint8_t *l0;
    const uint8_t *l1;
    const uint8_t *l2;
    const uint8_t *l3;
    const uint8_t *coc_u8;
    uint8_t *out;
} bokeh_composite_ctx_t;

static void bokeh_composite_worker(int64_t start, int64_t end, void *arg) {
    const bokeh_composite_ctx_t *ctx = (const bokeh_composite_ctx_t *)arg;
    const uint8_t *l0 = ctx->l0;
    const uint8_t *l1 = ctx->l1;
    const uint8_t *l2 = ctx->l2;
    const uint8_t *l3 = ctx->l3;
    const uint8_t *coc_u8 = ctx->coc_u8;
    uint8_t *out = ctx->out;

    const __m256i c85 = _mm256_set1_epi16(85);
    const __m256i c169 = _mm256_set1_epi16(169);
    const __m256i c170 = _mm256_set1_epi16(170);
    const __m256i c256 = _mm256_set1_epi16(256);
    const __m256i c3 = _mm256_set1_epi16(3);
    const __m512i v128 = _mm512_set1_epi16(128);
    const __m256i v128_256 = _mm256_set1_epi16(128);

    static const int16_t idx_a_arr[32] = {
        0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5,
        5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10
    };
    static const int16_t idx_b_arr[16] = {
        10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15
    };
    const __m512i idx_a = _mm512_loadu_si512((const __m512i *)idx_a_arr);
    const __m256i idx_b = _mm256_loadu_si256((const __m256i *)idx_b_arr);

    int64_t i = start;
    for (; i + 16 <= end; i += 16) {
        __m256i vc = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(coc_u8 + i)));
        __mmask16 m0 = _mm256_cmpgt_epi16_mask(c85, vc);
        __mmask16 m_ge170 = _mm256_cmpgt_epi16_mask(vc, c169);
        __mmask16 m1 = (__mmask16)(~m0 & ~m_ge170);

        __m256i t0 = _mm256_mullo_epi16(vc, c3);
        __m256i d1 = _mm256_sub_epi16(vc, c85);
        __m256i t1 = _mm256_mullo_epi16(d1, c3);
        __m256i d2 = _mm256_sub_epi16(vc, c170);
        __m256i t2 = _mm256_add_epi16(_mm256_mullo_epi16(d2, c3), _mm256_srli_epi16(d2, 6));

        __m256i w0 = _mm256_maskz_sub_epi16(m0, c256, t0);
        __m256i w1 = _mm256_mask_mov_epi16(_mm256_maskz_sub_epi16(m1, c256, t1), m0, t0);
        __m256i w2 = _mm256_mask_mov_epi16(_mm256_maskz_sub_epi16(m_ge170, c256, t2), m1, t1);
        __m256i w3 = _mm256_maskz_mov_epi16(m_ge170, t2);

        __m512i w0_a = _mm512_permutexvar_epi16(idx_a, _mm512_castsi256_si512(w0));
        __m512i w1_a = _mm512_permutexvar_epi16(idx_a, _mm512_castsi256_si512(w1));
        __m512i w2_a = _mm512_permutexvar_epi16(idx_a, _mm512_castsi256_si512(w2));
        __m512i w3_a = _mm512_permutexvar_epi16(idx_a, _mm512_castsi256_si512(w3));

        __m256i w0_b = _mm256_permutexvar_epi16(idx_b, w0);
        __m256i w1_b = _mm256_permutexvar_epi16(idx_b, w1);
        __m256i w2_b = _mm256_permutexvar_epi16(idx_b, w2);
        __m256i w3_b = _mm256_permutexvar_epi16(idx_b, w3);

        const int64_t idx = i * 3;
        __m512i acc_a = v128;
        acc_a = _mm512_add_epi16(acc_a, _mm512_mullo_epi16(_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)(l0 + idx))), w0_a));
        acc_a = _mm512_add_epi16(acc_a, _mm512_mullo_epi16(_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)(l1 + idx))), w1_a));
        acc_a = _mm512_add_epi16(acc_a, _mm512_mullo_epi16(_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)(l2 + idx))), w2_a));
        acc_a = _mm512_add_epi16(acc_a, _mm512_mullo_epi16(_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)(l3 + idx))), w3_a));
        _mm256_storeu_si256((__m256i *)(out + idx), _mm512_cvtepi16_epi8(_mm512_srli_epi16(acc_a, 8)));

        __m256i acc_b = v128_256;
        acc_b = _mm256_add_epi16(acc_b, _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(l0 + idx + 32))), w0_b));
        acc_b = _mm256_add_epi16(acc_b, _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(l1 + idx + 32))), w1_b));
        acc_b = _mm256_add_epi16(acc_b, _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(l2 + idx + 32))), w2_b));
        acc_b = _mm256_add_epi16(acc_b, _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(l3 + idx + 32))), w3_b));
        _mm_storeu_si128((__m128i *)(out + idx + 32), _mm256_cvtepi16_epi8(_mm256_srli_epi16(acc_b, 8)));
    }

    for (; i < end; ++i) {
        int32_t c = (int32_t)coc_u8[i];
        int32_t w0 = 0, w1 = 0, w2 = 0, w3 = 0;
        if (c < 85) {
            w1 = c * 3;
            w0 = 256 - w1;
        } else if (c < 170) {
            w2 = (c - 85) * 3;
            w1 = 256 - w2;
        } else {
            int32_t d = c - 170;
            w3 = d * 3 + (d >> 6);
            w2 = 256 - w3;
        }
        const int64_t idx = i * 3;
        out[idx + 0] = (uint8_t)((128 + l0[idx + 0] * w0 + l1[idx + 0] * w1 + l2[idx + 0] * w2 + l3[idx + 0] * w3) >> 8);
        out[idx + 1] = (uint8_t)((128 + l0[idx + 1] * w0 + l1[idx + 1] * w1 + l2[idx + 1] * w2 + l3[idx + 1] * w3) >> 8);
        out[idx + 2] = (uint8_t)((128 + l0[idx + 2] * w0 + l1[idx + 2] * w1 + l2[idx + 2] * w2 + l3[idx + 2] * w3) >> 8);
    }
}

void bokeh_composite_u8(
    const uint8_t *l0,
    const uint8_t *l1,
    const uint8_t *l2,
    const uint8_t *l3,
    const uint8_t *coc_u8,
    uint8_t *out,
    int64_t num_pixels
) {
    bokeh_composite_ctx_t ctx = {
        .l0 = l0,
        .l1 = l1,
        .l2 = l2,
        .l3 = l3,
        .coc_u8 = coc_u8,
        .out = out,
    };
    if (num_pixels >= 32768) {
        lse_parallel_for(num_pixels, 16384, bokeh_composite_worker, &ctx);
    } else {
        bokeh_composite_worker(0, num_pixels, &ctx);
    }
}

void f32_unpad_bias_gelu(
    const float *src,
    int64_t src_stride_elems,
    float *dst,
    int64_t dst_stride_elems,
    const float *bias,
    int64_t rows,
    int64_t cols,
    int apply_gelu
) {
    for (int64_t r = 0; r < rows; ++r) {
        const float *srow = src + r * src_stride_elems;
        float *drow = dst + r * dst_stride_elems;
        int64_t c = 0;
        for (; c + 16 <= cols; c += 16) {
            __m512 v = _mm512_loadu_ps(srow + c);
            if (bias) {
                v = _mm512_add_ps(v, _mm512_loadu_ps(bias + c));
            }
            if (apply_gelu) {
                v = gelu_vec_avx512(v);
            }
            _mm512_storeu_ps(drow + c, v);
        }
        if (c < cols) {
            __mmask16 km = (__mmask16)((1u << (cols - c)) - 1u);
            __m512 v = _mm512_maskz_loadu_ps(km, srow + c);
            if (bias) {
                v = _mm512_add_ps(v, _mm512_maskz_loadu_ps(km, bias + c));
            }
            if (apply_gelu) {
                v = gelu_vec_avx512(v);
            }
            _mm512_mask_storeu_ps(drow + c, km, v);
        }
    }
}

typedef struct {
    const uint16_t *src_bf16;
    int64_t src_stride_elems;
    float *dst;
    int64_t dst_stride_elems;
    const float *bias;
    int64_t cols;
    int apply_gelu;
} bf16_unpad_bias_gelu_ctx_t;

static void bf16_unpad_bias_gelu_worker(int64_t r_start, int64_t r_end, void *arg) {
    const bf16_unpad_bias_gelu_ctx_t *ctx = (const bf16_unpad_bias_gelu_ctx_t *)arg;
    const int64_t cols = ctx->cols;
    const float *bias = ctx->bias;
    const int apply_gelu = ctx->apply_gelu;
    for (int64_t r = r_start; r < r_end; ++r) {
        const uint16_t *srow = ctx->src_bf16 + r * ctx->src_stride_elems;
        float *drow = ctx->dst + r * ctx->dst_stride_elems;
        int64_t c = 0;
        for (; c + 16 <= cols; c += 16) {
            __m256bh vb = (__m256bh)_mm256_loadu_si256((const __m256i *)(srow + c));
            __m512 v = _mm512_cvtpbh_ps(vb);
            if (bias) {
                v = _mm512_add_ps(v, _mm512_loadu_ps(bias + c));
            }
            if (apply_gelu) {
                v = gelu_vec_avx512(v);
            }
            _mm512_storeu_ps(drow + c, v);
        }
        if (c < cols) {
            __mmask16 km = (__mmask16)((1u << (cols - c)) - 1u);
            __m256bh vb = (__m256bh)_mm256_maskz_loadu_epi16(km, srow + c);
            __m512 v = _mm512_cvtpbh_ps(vb);
            if (bias) {
                v = _mm512_add_ps(v, _mm512_maskz_loadu_ps(km, bias + c));
            }
            if (apply_gelu) {
                v = gelu_vec_avx512(v);
            }
            _mm512_mask_storeu_ps(drow + c, km, v);
        }
    }
}

void bf16_unpad_bias_gelu(
    const uint16_t *src_bf16,
    int64_t src_stride_elems,
    float *dst,
    int64_t dst_stride_elems,
    const float *bias,
    int64_t rows,
    int64_t cols,
    int apply_gelu
) {
    bf16_unpad_bias_gelu_ctx_t ctx = {
        .src_bf16 = src_bf16,
        .src_stride_elems = src_stride_elems,
        .dst = dst,
        .dst_stride_elems = dst_stride_elems,
        .bias = bias,
        .cols = cols,
        .apply_gelu = apply_gelu,
    };
    if (rows >= 16 && rows * cols >= 16384) {
        lse_parallel_for(rows, 8, bf16_unpad_bias_gelu_worker, &ctx);
    } else {
        bf16_unpad_bias_gelu_worker(0, rows, &ctx);
    }
}

static inline void transpose_8x16_f32_to_bf16_strided(
    const float *src,
    int64_t src_stride_elems,
    uint16_t *dst,
    int64_t dst_stride_elems
) {
    /*
     * Reads 8 rows (r=0..7) x 16 cols (c=0..15) of float32, converts to bf16, and stores
     * transposed 16 rows x 8 cols of uint16 into dst[c * dst_stride_elems + 0..7].
     */
    __m256i v0 = (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(src + 0 * src_stride_elems));
    __m256i v1 = (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(src + 1 * src_stride_elems));
    __m256i v2 = (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(src + 2 * src_stride_elems));
    __m256i v3 = (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(src + 3 * src_stride_elems));
    __m256i v4 = (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(src + 4 * src_stride_elems));
    __m256i v5 = (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(src + 5 * src_stride_elems));
    __m256i v6 = (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(src + 6 * src_stride_elems));
    __m256i v7 = (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(src + 7 * src_stride_elems));

    __m256i t0 = _mm256_unpacklo_epi16(v0, v1);
    __m256i t1 = _mm256_unpackhi_epi16(v0, v1);
    __m256i t2 = _mm256_unpacklo_epi16(v2, v3);
    __m256i t3 = _mm256_unpackhi_epi16(v2, v3);
    __m256i t4 = _mm256_unpacklo_epi16(v4, v5);
    __m256i t5 = _mm256_unpackhi_epi16(v4, v5);
    __m256i t6 = _mm256_unpacklo_epi16(v6, v7);
    __m256i t7 = _mm256_unpackhi_epi16(v6, v7);

    __m256i u0 = _mm256_unpacklo_epi32(t0, t2);
    __m256i u1 = _mm256_unpackhi_epi32(t0, t2);
    __m256i u2 = _mm256_unpacklo_epi32(t1, t3);
    __m256i u3 = _mm256_unpackhi_epi32(t1, t3);
    __m256i u4 = _mm256_unpacklo_epi32(t4, t6);
    __m256i u5 = _mm256_unpackhi_epi32(t4, t6);
    __m256i u6 = _mm256_unpacklo_epi32(t5, t7);
    __m256i u7 = _mm256_unpackhi_epi32(t5, t7);

    __m256i w0 = _mm256_unpacklo_epi64(u0, u4);
    __m256i w1 = _mm256_unpackhi_epi64(u0, u4);
    __m256i w2 = _mm256_unpacklo_epi64(u1, u5);
    __m256i w3 = _mm256_unpackhi_epi64(u1, u5);
    __m256i w4 = _mm256_unpacklo_epi64(u2, u6);
    __m256i w5 = _mm256_unpackhi_epi64(u2, u6);
    __m256i w6 = _mm256_unpacklo_epi64(u3, u7);
    __m256i w7 = _mm256_unpackhi_epi64(u3, u7);

    _mm_storeu_si128((__m128i *)(dst + 0 * dst_stride_elems), _mm256_castsi256_si128(w0));
    _mm_storeu_si128((__m128i *)(dst + 1 * dst_stride_elems), _mm256_castsi256_si128(w1));
    _mm_storeu_si128((__m128i *)(dst + 2 * dst_stride_elems), _mm256_castsi256_si128(w2));
    _mm_storeu_si128((__m128i *)(dst + 3 * dst_stride_elems), _mm256_castsi256_si128(w3));
    _mm_storeu_si128((__m128i *)(dst + 4 * dst_stride_elems), _mm256_castsi256_si128(w4));
    _mm_storeu_si128((__m128i *)(dst + 5 * dst_stride_elems), _mm256_castsi256_si128(w5));
    _mm_storeu_si128((__m128i *)(dst + 6 * dst_stride_elems), _mm256_castsi256_si128(w6));
    _mm_storeu_si128((__m128i *)(dst + 7 * dst_stride_elems), _mm256_castsi256_si128(w7));
    _mm_storeu_si128((__m128i *)(dst + 8 * dst_stride_elems), _mm256_extracti128_si256(w0, 1));
    _mm_storeu_si128((__m128i *)(dst + 9 * dst_stride_elems), _mm256_extracti128_si256(w1, 1));
    _mm_storeu_si128((__m128i *)(dst + 10 * dst_stride_elems), _mm256_extracti128_si256(w2, 1));
    _mm_storeu_si128((__m128i *)(dst + 11 * dst_stride_elems), _mm256_extracti128_si256(w3, 1));
    _mm_storeu_si128((__m128i *)(dst + 12 * dst_stride_elems), _mm256_extracti128_si256(w4, 1));
    _mm_storeu_si128((__m128i *)(dst + 13 * dst_stride_elems), _mm256_extracti128_si256(w5, 1));
    _mm_storeu_si128((__m128i *)(dst + 14 * dst_stride_elems), _mm256_extracti128_si256(w6, 1));
    _mm_storeu_si128((__m128i *)(dst + 15 * dst_stride_elems), _mm256_extracti128_si256(w7, 1));
}

typedef struct {
    const float *src;
    int64_t src_stride_elems;
    uint16_t *dst;
    int64_t dst_stride_elems;
    int64_t out_rows;
    int64_t out_cols;
    size_t tail_bytes;
} f32_transpose_bf16_ctx_t;

static void f32_transpose_bf16_worker(int64_t blk_start, int64_t blk_end, void *arg) {
    const f32_transpose_bf16_ctx_t *ctx = (const f32_transpose_bf16_ctx_t *)arg;
    const float *src = ctx->src;
    const int64_t src_stride_elems = ctx->src_stride_elems;
    uint16_t *dst = ctx->dst;
    const int64_t dst_stride_elems = ctx->dst_stride_elems;
    const int64_t out_rows = ctx->out_rows;
    const int64_t out_cols = ctx->out_cols;
    const size_t tail_bytes = ctx->tail_bytes;

    for (int64_t blk = blk_start; blk < blk_end; ++blk) {
        const int64_t m0 = blk * 16;
        if (m0 + 16 <= out_rows) {
            int64_t k0 = 0;
            for (; k0 + 8 <= out_cols; k0 += 8) {
                transpose_8x16_f32_to_bf16_strided(
                    src + k0 * src_stride_elems + m0,
                    src_stride_elems,
                    dst + m0 * dst_stride_elems + k0,
                    dst_stride_elems
                );
            }
            if (k0 < out_cols) {
                const int64_t bk = out_cols - k0;
                uint16_t tile[8][16];
                for (int64_t kk = 0; kk < bk; ++kk) {
                    __m512 vf = _mm512_loadu_ps(src + (k0 + kk) * src_stride_elems + m0);
                    __m256bh vb = _mm512_cvtneps_pbh(vf);
                    _mm256_storeu_si256((__m256i *)tile[kk], (__m256i)vb);
                }
                for (int64_t mm = 0; mm < 16; ++mm) {
                    uint16_t *drow = dst + (m0 + mm) * dst_stride_elems + k0;
                    for (int64_t kk = 0; kk < bk; ++kk) {
                        drow[kk] = tile[kk][mm];
                    }
                }
            }
            if (tail_bytes) {
                for (int64_t mm = 0; mm < 16; ++mm) {
                    memset(dst + (m0 + mm) * dst_stride_elems + out_cols, 0, tail_bytes);
                }
            }
        } else {
            const int64_t bm = out_rows - m0;
            uint16_t tile[16][16];
            for (int64_t k0 = 0; k0 < out_cols; k0 += 16) {
                const int64_t bk = (out_cols - k0 < 16) ? (out_cols - k0) : 16;
                const __mmask16 km = (__mmask16)((1u << bm) - 1u);
                for (int64_t kk = 0; kk < bk; ++kk) {
                    __m512 vf = _mm512_maskz_loadu_ps(km, src + (k0 + kk) * src_stride_elems + m0);
                    __m256bh vb = _mm512_cvtneps_pbh(vf);
                    _mm256_mask_storeu_epi16(tile[kk], km, (__m256i)vb);
                }
                for (int64_t mm = 0; mm < bm; ++mm) {
                    uint16_t *drow = dst + (m0 + mm) * dst_stride_elems + k0;
                    for (int64_t kk = 0; kk < bk; ++kk) {
                        drow[kk] = tile[kk][mm];
                    }
                }
            }
            if (tail_bytes) {
                for (int64_t mm = 0; mm < bm; ++mm) {
                    memset(dst + (m0 + mm) * dst_stride_elems + out_cols, 0, tail_bytes);
                }
            }
        }
    }
}

void f32_transpose_to_bf16_pad(
    const float *src,
    int64_t src_stride_elems,
    uint16_t *dst,
    int64_t dst_stride_elems,
    int64_t dst_rows,
    int64_t dst_cols,
    int64_t src_rows,
    int64_t src_cols,
    int zero_tail
) {
    /*
     * src has shape (src_rows=K, src_cols=M); dst has shape (dst_rows=M_pad, dst_cols=K_pad).
     * Writes dst[m, k] = bf16(src[k, m]) using AVX-512 8x16 register transposes.
     */
    const int64_t out_rows = src_cols;
    const int64_t out_cols = src_rows;
    const size_t tail_bytes = (dst_cols > out_cols) ? (size_t)(dst_cols - out_cols) * sizeof(uint16_t) : 0;
    const int64_t num_blocks = (out_rows + 15) / 16;

    f32_transpose_bf16_ctx_t ctx = {
        .src = src,
        .src_stride_elems = src_stride_elems,
        .dst = dst,
        .dst_stride_elems = dst_stride_elems,
        .out_rows = out_rows,
        .out_cols = out_cols,
        .tail_bytes = tail_bytes,
    };
    if (num_blocks >= 4 && out_rows * out_cols >= 16384) {
        lse_parallel_for(num_blocks, 2, f32_transpose_bf16_worker, &ctx);
    } else {
        f32_transpose_bf16_worker(0, num_blocks, &ctx);
    }

    if (zero_tail && dst_rows > out_rows) {
        if (dst_stride_elems == dst_cols) {
            memset(
                dst + out_rows * dst_stride_elems,
                0,
                (size_t)(dst_rows - out_rows) * (size_t)dst_cols * sizeof(uint16_t)
            );
        } else {
            for (int64_t r = out_rows; r < dst_rows; ++r) {
                memset(dst + r * dst_stride_elems, 0, (size_t)dst_cols * sizeof(uint16_t));
            }
        }
    }
}

static inline void bf16_copy_vec(uint16_t *dst, const uint16_t *src, size_t n_u16) {
    size_t i = 0;
    for (; i + 32 <= n_u16; i += 32) {
        _mm512_storeu_si512(dst + i, _mm512_loadu_si512(src + i));
    }
    if (i + 16 <= n_u16) {
        _mm256_storeu_si256((__m256i *)(dst + i), _mm256_loadu_si256((const __m256i *)(src + i)));
        i += 16;
    }
    if (i + 8 <= n_u16) {
        _mm_storeu_si128((__m128i *)(dst + i), _mm_loadu_si128((const __m128i *)(src + i)));
        i += 8;
    }
    for (; i < n_u16; ++i) {
        dst[i] = src[i];
    }
}

static inline void bf16_zero_vec(uint16_t *dst, size_t n_u16) {
    const __m512i z512 = _mm512_setzero_si512();
    const __m256i z256 = _mm256_setzero_si256();
    const __m128i z128 = _mm_setzero_si128();
    size_t i = 0;
    for (; i + 32 <= n_u16; i += 32) {
        _mm512_storeu_si512(dst + i, z512);
    }
    if (i + 16 <= n_u16) {
        _mm256_storeu_si256((__m256i *)(dst + i), z256);
        i += 16;
    }
    if (i + 8 <= n_u16) {
        _mm_storeu_si128((__m128i *)(dst + i), z128);
        i += 8;
    }
    for (; i < n_u16; ++i) {
        dst[i] = 0;
    }
}

static void convert_nchw_row_to_hwc_bf16(
    const float *x_nchw,
    int64_t hw,
    int64_t c_in,
    int64_t w,
    int64_t sy,
    uint16_t *dst_row
) {
    uint16_t tile[16][16];
    const size_t row_off = (size_t)sy * (size_t)w;
    for (int64_t c0 = 0; c0 < c_in; c0 += 16) {
        const int64_t bc = (c_in - c0 < 16) ? (c_in - c0) : 16;
        for (int64_t x0 = 0; x0 < w; x0 += 16) {
            const int64_t bx = (w - x0 < 16) ? (w - x0) : 16;
            const __mmask16 km = (bx == 16) ? (__mmask16)0xFFFFu : (__mmask16)((1u << bx) - 1u);
            for (int64_t cc = 0; cc < bc; ++cc) {
                const float *s = x_nchw + (size_t)(c0 + cc) * (size_t)hw + row_off + x0;
                __m512 vf = (bx == 16) ? _mm512_loadu_ps(s) : _mm512_maskz_loadu_ps(km, s);
                __m256bh vb = _mm512_cvtneps_pbh(vf);
                _mm256_mask_storeu_epi16(tile[cc], km, (__m256i)vb);
            }
            for (int64_t xx = 0; xx < bx; ++xx) {
                uint16_t *d = dst_row + (x0 + xx) * c_in + c0;
                for (int64_t cc = 0; cc < bc; ++cc) {
                    d[cc] = tile[cc][xx];
                }
            }
        }
    }
}

typedef struct {
    const float *x_nchw;
    uint16_t *dst_mk;
    int64_t dst_stride_elems;
    int64_t c_in;
    int64_t h;
    int64_t w;
    int64_t hw;
    int64_t k_elems;
    int64_t k_tail_elems;
} im2col_kykxc_bf16_ctx_t;

static __thread uint16_t *g_im2col_ring_buf = NULL;
static __thread size_t g_im2col_ring_cap = 0;

static void im2col_kykxc_bf16_worker(int64_t y_start, int64_t y_end, void *arg) {
    const im2col_kykxc_bf16_ctx_t *ctx = (const im2col_kykxc_bf16_ctx_t *)arg;
    const float *x_nchw = ctx->x_nchw;
    uint16_t *dst_mk = ctx->dst_mk;
    const int64_t dst_stride_elems = ctx->dst_stride_elems;
    const int64_t c_in = ctx->c_in;
    const int64_t h = ctx->h;
    const int64_t w = ctx->w;
    const int64_t hw = ctx->hw;
    const int64_t k_elems = ctx->k_elems;
    const int64_t k_tail_elems = ctx->k_tail_elems;
    const size_t row_elems = (size_t)w * (size_t)c_in;
    const size_t span3_elems = (size_t)(3 * c_in);
    const size_t need_cap = 3 * row_elems;

    if (need_cap > g_im2col_ring_cap) {
        uint16_t *nbuf = (uint16_t *)realloc(g_im2col_ring_buf, need_cap * sizeof(uint16_t));
        if (!nbuf) {
            return;
        }
        g_im2col_ring_buf = nbuf;
        g_im2col_ring_cap = need_cap;
    }

    uint16_t *ring[3] = {
        g_im2col_ring_buf,
        g_im2col_ring_buf + row_elems,
        g_im2col_ring_buf + 2 * row_elems,
    };

    if (y_start - 1 >= 0) {
        convert_nchw_row_to_hwc_bf16(x_nchw, hw, c_in, w, y_start - 1, ring[(y_start - 1) % 3]);
    }
    if (y_start < h) {
        convert_nchw_row_to_hwc_bf16(x_nchw, hw, c_in, w, y_start, ring[y_start % 3]);
    }

    for (int64_t y = y_start; y < y_end; ++y) {
        if (y + 1 < h) {
            convert_nchw_row_to_hwc_bf16(x_nchw, hw, c_in, w, y + 1, ring[(y + 1) % 3]);
        }
        const uint16_t *r_top = (y > 0) ? ring[(y - 1) % 3] : NULL;
        const uint16_t *r_mid = ring[y % 3];
        const uint16_t *r_bot = (y + 1 < h) ? ring[(y + 1) % 3] : NULL;
        const uint16_t *src_rows[3] = {r_top, r_mid, r_bot};

        for (int64_t x = 0; x < w; ++x) {
            uint16_t *drow = dst_mk + (y * w + x) * dst_stride_elems;
            const int interior = (x > 0 && x + 1 < w);
            for (int64_t ky = 0; ky < 3; ++ky) {
                const uint16_t *s_row = src_rows[ky];
                uint16_t *dk = drow + (size_t)ky * span3_elems;
                if (!s_row) {
                    bf16_zero_vec(dk, span3_elems);
                } else if (interior) {
                    bf16_copy_vec(dk, s_row + (size_t)(x - 1) * (size_t)c_in, span3_elems);
                } else {
                    if (x > 0) {
                        bf16_copy_vec(dk, s_row + (size_t)(x - 1) * (size_t)c_in, (size_t)c_in);
                    } else {
                        bf16_zero_vec(dk, (size_t)c_in);
                    }
                    bf16_copy_vec(dk + c_in, s_row + (size_t)x * (size_t)c_in, (size_t)c_in);
                    if (x + 1 < w) {
                        bf16_copy_vec(dk + 2 * c_in, s_row + (size_t)(x + 1) * (size_t)c_in, (size_t)c_in);
                    } else {
                        bf16_zero_vec(dk + 2 * c_in, (size_t)c_in);
                    }
                }
            }
            if (k_tail_elems > 0) {
                bf16_zero_vec(drow + k_elems, (size_t)k_tail_elems);
            }
        }
    }
}

int im2col_3x3_s1_p1_kykxc_bf16_pad(
    const float *x_nchw,
    uint16_t *dst_mk,
    int64_t dst_stride_elems,
    int64_t dst_rows,
    int64_t dst_cols,
    int64_t c_in,
    int64_t h,
    int64_t w,
    int zero_tail
) {
    /*
     * Single-pass rolling 3-row L1/L2 cache-hot im2col: converts (1, c_in, h, w) float32
     * directly into dst_mk[(h*w)_pad, (9*c_in)_pad] bf16 without full-tensor HWC allocation.
     */
    const int64_t hw = h * w;
    const int64_t k_elems = 9 * c_in;
    const int64_t k_tail_elems = (dst_cols > k_elems) ? (dst_cols - k_elems) : 0;

    im2col_kykxc_bf16_ctx_t ctx = {
        .x_nchw = x_nchw,
        .dst_mk = dst_mk,
        .dst_stride_elems = dst_stride_elems,
        .c_in = c_in,
        .h = h,
        .w = w,
        .hw = hw,
        .k_elems = k_elems,
        .k_tail_elems = k_tail_elems,
    };
    if (h >= 8 && hw * k_elems >= 32768) {
        lse_parallel_for(h, 4, im2col_kykxc_bf16_worker, &ctx);
    } else {
        im2col_kykxc_bf16_worker(0, h, &ctx);
    }

    if (zero_tail && dst_rows > hw) {
        if (dst_stride_elems == dst_cols) {
            bf16_zero_vec(
                dst_mk + hw * dst_stride_elems,
                (size_t)(dst_rows - hw) * (size_t)dst_cols
            );
        } else {
            for (int64_t r = hw; r < dst_rows; ++r) {
                bf16_zero_vec(dst_mk + r * dst_stride_elems, (size_t)dst_cols);
            }
        }
    }

    return 0;
}

void f32_unpad_transpose_bias_relu(
    const float *src,
    int64_t src_stride_elems,
    float *dst_nchw,
    const float *bias,
    int64_t m_hw,
    int64_t c_out,
    int apply_relu
) {
    /*
     * Transposes (M, C_out) -> (C_out, M) float32 in 64x16 cache blocks with optional
     * vector bias addition and ReLU before storing contiguous channel rows.
     */
    const __m512 vzero = _mm512_setzero_ps();
    float tile[64][16];

    for (int64_t c0 = 0; c0 < c_out; c0 += 16) {
        const int64_t bc = (c_out - c0 < 16) ? (c_out - c0) : 16;
        __mmask16 km = (bc == 16) ? (__mmask16)0xFFFFu : (__mmask16)((1u << bc) - 1u);
        __m512 vbias = bias ? ((bc == 16) ? _mm512_loadu_ps(bias + c0) : _mm512_maskz_loadu_ps(km, bias + c0)) : vzero;

        for (int64_t m0 = 0; m0 < m_hw; m0 += 64) {
            const int64_t bm = (m_hw - m0 < 64) ? (m_hw - m0) : 64;
            for (int64_t mm = 0; mm < bm; ++mm) {
                const float *srow = src + (m0 + mm) * src_stride_elems + c0;
                __m512 v = (bc == 16) ? _mm512_loadu_ps(srow) : _mm512_maskz_loadu_ps(km, srow);
                if (bias) {
                    v = _mm512_add_ps(v, vbias);
                }
                if (apply_relu) {
                    v = _mm512_max_ps(v, vzero);
                }
                _mm512_storeu_ps(tile[mm], v);
            }
            for (int64_t cc = 0; cc < bc; ++cc) {
                float *drow = dst_nchw + (c0 + cc) * m_hw + m0;
                for (int64_t mm = 0; mm < bm; ++mm) {
                    drow[mm] = tile[mm][cc];
                }
            }
        }
    }
}

typedef struct {
    const uint16_t *src_bf16;
    int64_t src_stride_elems;
    float *dst_nchw;
    const float *bias;
    int64_t m_hw;
    int64_t c_out;
    int apply_relu;
} bf16_unpad_tr_ctx_t;

static void bf16_unpad_tr_worker(int64_t blk_start, int64_t blk_end, void *arg) {
    const bf16_unpad_tr_ctx_t *ctx = (const bf16_unpad_tr_ctx_t *)arg;
    const uint16_t *src = ctx->src_bf16;
    const int64_t src_stride_elems = ctx->src_stride_elems;
    float *dst_nchw = ctx->dst_nchw;
    const float *bias = ctx->bias;
    const int64_t m_hw = ctx->m_hw;
    const int64_t c_out = ctx->c_out;
    const int apply_relu = ctx->apply_relu;
    const __m512 vzero = _mm512_setzero_ps();
    float tile[64][16];

    for (int64_t blk = blk_start; blk < blk_end; ++blk) {
        const int64_t c0 = blk * 16;
        const int64_t bc = (c_out - c0 < 16) ? (c_out - c0) : 16;
        __mmask16 km = (bc == 16) ? (__mmask16)0xFFFFu : (__mmask16)((1u << bc) - 1u);
        __m512 vbias = bias ? ((bc == 16) ? _mm512_loadu_ps(bias + c0) : _mm512_maskz_loadu_ps(km, bias + c0)) : vzero;

        for (int64_t m0 = 0; m0 < m_hw; m0 += 64) {
            const int64_t bm = (m_hw - m0 < 64) ? (m_hw - m0) : 64;
            for (int64_t mm = 0; mm < bm; ++mm) {
                const uint16_t *srow = src + (m0 + mm) * src_stride_elems + c0;
                __m256bh vb = (__m256bh)((bc == 16) ? _mm256_loadu_si256((const __m256i *)srow) : _mm256_maskz_loadu_epi16(km, srow));
                __m512 v = _mm512_cvtpbh_ps(vb);
                if (bias) {
                    v = _mm512_add_ps(v, vbias);
                }
                if (apply_relu) {
                    v = _mm512_max_ps(v, vzero);
                }
                _mm512_storeu_ps(tile[mm], v);
            }
            for (int64_t cc = 0; cc < bc; ++cc) {
                float *drow = dst_nchw + (c0 + cc) * m_hw + m0;
                for (int64_t mm = 0; mm < bm; ++mm) {
                    drow[mm] = tile[mm][cc];
                }
            }
        }
    }
}

void bf16_unpad_transpose_bias_relu(
    const uint16_t *src_bf16,
    int64_t src_stride_elems,
    float *dst_nchw,
    const float *bias,
    int64_t m_hw,
    int64_t c_out,
    int apply_relu
) {
    const int64_t num_blocks = (c_out + 15) / 16;
    bf16_unpad_tr_ctx_t ctx = {
        .src_bf16 = src_bf16,
        .src_stride_elems = src_stride_elems,
        .dst_nchw = dst_nchw,
        .bias = bias,
        .m_hw = m_hw,
        .c_out = c_out,
        .apply_relu = apply_relu,
    };
    if (num_blocks >= 4 && m_hw * c_out >= 16384) {
        lse_parallel_for(num_blocks, 1, bf16_unpad_tr_worker, &ctx);
    } else {
        bf16_unpad_tr_worker(0, num_blocks, &ctx);
    }
}

typedef struct {
    const float *q;
    const float *k_cur;
    const float *v_cur;
    float *k_buf;
    float *v_buf;
    const float *k_pe;
    const float *v_pe;
    float *out;
    int64_t t_len;
    int64_t n_heads;
    int64_t dh;
    int64_t ch;
    float scale;
    int has_history;
    int cursor;
    const int64_t *slot_map;
} temporal_attn_ctx_t;

static void temporal_attn_roll_worker(int64_t tok_start, int64_t tok_end, void *arg) {
    const temporal_attn_ctx_t *ctx = (const temporal_attn_ctx_t *)arg;
    const float *q = ctx->q;
    const float *k_cur = ctx->k_cur;
    const float *v_cur = ctx->v_cur;
    float *k_buf = ctx->k_buf;
    float *v_buf = ctx->v_buf;
    const float *k_pe = ctx->k_pe;
    const float *v_pe = ctx->v_pe;
    float *out = ctx->out;
    const int64_t t_len = ctx->t_len;
    const int64_t n_heads = ctx->n_heads;
    const int64_t dh = ctx->dh;
    const int64_t ch = ctx->ch;
    const float scale = ctx->scale;
    const size_t ch_bytes = (size_t)ch * sizeof(float);

    if (!ctx->has_history) {
        for (int64_t tok = tok_start; tok < tok_end; ++tok) {
            const float *kc = k_cur + tok * ch;
            const float *vc = v_cur + tok * ch;
            float *out_tok = out + tok * ch;
            int64_t c = 0;
            for (; c + 16 <= ch; c += 16) {
                __m512 vv = _mm512_add_ps(_mm512_loadu_ps(vc + c), _mm512_loadu_ps(v_pe + c));
                _mm512_storeu_ps(out_tok + c, vv);
            }
            if (c < ch) {
                __mmask16 km = (__mmask16)((1u << (ch - c)) - 1u);
                __m512 vv = _mm512_add_ps(_mm512_maskz_loadu_ps(km, vc + c), _mm512_maskz_loadu_ps(km, v_pe + c));
                _mm512_mask_storeu_ps(out_tok + c, km, vv);
            }
            float *kb = k_buf + tok * t_len * ch;
            float *vb = v_buf + tok * t_len * ch;
            for (int64_t t = 0; t < t_len - 1; ++t) {
                memcpy(kb + t * ch, kc, ch_bytes);
                memcpy(vb + t * ch, vc, ch_bytes);
            }
        }
        return;
    }

    float stack_scores[128];
    float *scores = (t_len <= 128) ? NULL : (float *)malloc((size_t)t_len * sizeof(float));
    float *sc = scores ? scores : stack_scores;
    const size_t shift_bytes = (t_len > 4) ? (size_t)(t_len - 4) * ch_bytes : 0;

    for (int64_t tok = tok_start; tok < tok_end; ++tok) {
        const float *q_tok = q + tok * ch;
        const float *kc = k_cur + tok * ch;
        const float *vc = v_cur + tok * ch;
        float *kb = k_buf + tok * t_len * ch;
        float *vb = v_buf + tok * t_len * ch;
        float *out_tok = out + tok * ch;

        memcpy(kb + (t_len - 1) * ch, kc, ch_bytes);
        memcpy(vb + (t_len - 1) * ch, vc, ch_bytes);

        for (int64_t h = 0; h < n_heads; ++h) {
            const int64_t h_off = h * dh;
            const float *qh = q_tok + h_off;
            float *oh = out_tok + h_off;

            float max_sc = -3.402823466e+38f;
            for (int64_t t = 0; t < t_len; ++t) {
                const float *kt = kb + t * ch + h_off;
                const float *kpet = k_pe + t * ch + h_off;
                __m512 vacc = _mm512_setzero_ps();
                int64_t i = 0;
                for (; i + 16 <= dh; i += 16) {
                    __m512 vq = _mm512_loadu_ps(qh + i);
                    __m512 vk = _mm512_add_ps(_mm512_loadu_ps(kt + i), _mm512_loadu_ps(kpet + i));
                    vacc = _mm512_fmadd_ps(vq, vk, vacc);
                }
                if (i < dh) {
                    __mmask16 km = (__mmask16)((1u << (dh - i)) - 1u);
                    __m512 vq = _mm512_maskz_loadu_ps(km, qh + i);
                    __m512 vk = _mm512_add_ps(_mm512_maskz_loadu_ps(km, kt + i), _mm512_maskz_loadu_ps(km, kpet + i));
                    vacc = _mm512_fmadd_ps(vq, vk, vacc);
                }
                float s = _mm512_reduce_add_ps(vacc) * scale;
                sc[t] = s;
                if (s > max_sc) max_sc = s;
            }

            const __m512 vmax = _mm512_set1_ps(max_sc);
            __m512 vsum = _mm512_setzero_ps();
            int64_t t = 0;
            for (; t + 16 <= t_len; t += 16) {
                __m512 ve = exp_ps_avx512(_mm512_sub_ps(_mm512_loadu_ps(sc + t), vmax));
                _mm512_storeu_ps(sc + t, ve);
                vsum = _mm512_add_ps(vsum, ve);
            }
            if (t < t_len) {
                __mmask16 km = (__mmask16)((1u << (t_len - t)) - 1u);
                __m512 ve = exp_ps_avx512(_mm512_sub_ps(_mm512_maskz_loadu_ps(km, sc + t), vmax));
                _mm512_mask_storeu_ps(sc + t, km, ve);
                vsum = _mm512_mask_add_ps(vsum, km, vsum, ve);
            }
            float inv_sum = 1.0f / _mm512_reduce_add_ps(vsum);
            const __m512 vinv = _mm512_set1_ps(inv_sum);
            for (t = 0; t + 16 <= t_len; t += 16) {
                _mm512_storeu_ps(sc + t, _mm512_mul_ps(_mm512_loadu_ps(sc + t), vinv));
            }
            if (t < t_len) {
                __mmask16 km = (__mmask16)((1u << (t_len - t)) - 1u);
                _mm512_mask_storeu_ps(sc + t, km, _mm512_mul_ps(_mm512_maskz_loadu_ps(km, sc + t), vinv));
            }

            int64_t i = 0;
            for (; i + 16 <= dh; i += 16) {
                __m512 oacc = _mm512_setzero_ps();
                for (int64_t tt = 0; tt < t_len; ++tt) {
                    const float *vt = vb + tt * ch + h_off + i;
                    const float *vpet = v_pe + tt * ch + h_off + i;
                    __m512 vw = _mm512_set1_ps(sc[tt]);
                    __m512 vv = _mm512_add_ps(_mm512_loadu_ps(vt), _mm512_loadu_ps(vpet));
                    oacc = _mm512_fmadd_ps(vw, vv, oacc);
                }
                _mm512_storeu_ps(oh + i, oacc);
            }
            if (i < dh) {
                __mmask16 km = (__mmask16)((1u << (dh - i)) - 1u);
                __m512 oacc = _mm512_setzero_ps();
                for (int64_t tt = 0; tt < t_len; ++tt) {
                    const float *vt = vb + tt * ch + h_off + i;
                    const float *vpet = v_pe + tt * ch + h_off + i;
                    __m512 vw = _mm512_set1_ps(sc[tt]);
                    __m512 vv = _mm512_add_ps(_mm512_maskz_loadu_ps(km, vt), _mm512_maskz_loadu_ps(km, vpet));
                    oacc = _mm512_fmadd_ps(vw, vv, oacc);
                }
                _mm512_mask_storeu_ps(oh + i, km, oacc);
            }
        }

        if (t_len >= 4) {
            if (shift_bytes) {
                memmove(kb + 2 * ch, kb + 3 * ch, shift_bytes);
                memmove(vb + 2 * ch, vb + 3 * ch, shift_bytes);
            }
            memcpy(kb + (t_len - 2) * ch, kc, ch_bytes);
            memcpy(vb + (t_len - 2) * ch, vc, ch_bytes);
        }
    }

    if (scores) {
        free(scores);
    }
}

void temporal_attn_roll_f32(
    const float *q,
    const float *k_cur,
    const float *v_cur,
    float *k_buf,
    float *v_buf,
    const float *k_pe,
    const float *v_pe,
    float *out,
    int64_t d,
    int64_t t_len,
    int64_t n_heads,
    int64_t dh,
    float scale,
    int has_history
) {
    const int64_t ch = n_heads * dh;
    temporal_attn_ctx_t ctx = {
        .q = q,
        .k_cur = k_cur,
        .v_cur = v_cur,
        .k_buf = k_buf,
        .v_buf = v_buf,
        .k_pe = k_pe,
        .v_pe = v_pe,
        .out = out,
        .t_len = t_len,
        .n_heads = n_heads,
        .dh = dh,
        .ch = ch,
        .scale = scale,
        .has_history = has_history,
        .cursor = 0,
        .slot_map = NULL,
    };
    if (d >= 16 && d * ch * t_len >= 16384) {
        lse_parallel_for(d, 4, temporal_attn_roll_worker, &ctx);
    } else {
        temporal_attn_roll_worker(0, d, &ctx);
    }
}

static void temporal_attn_roll_ring_worker(int64_t tok_start, int64_t tok_end, void *arg) {
    const temporal_attn_ctx_t *ctx = (const temporal_attn_ctx_t *)arg;
    const float *q = ctx->q;
    const float *k_cur = ctx->k_cur;
    const float *v_cur = ctx->v_cur;
    float *k_buf = ctx->k_buf;
    float *v_buf = ctx->v_buf;
    const float *k_pe = ctx->k_pe;
    const float *v_pe = ctx->v_pe;
    float *out = ctx->out;
    const int64_t t_len = ctx->t_len;
    const int64_t n_heads = ctx->n_heads;
    const int64_t dh = ctx->dh;
    const int64_t ch = ctx->ch;
    const float scale = ctx->scale;
    const int64_t *slot_map = ctx->slot_map;
    const size_t ch_bytes = (size_t)ch * sizeof(float);

    float stack_scores[128];
    float *scores = (t_len <= 128) ? NULL : (float *)malloc((size_t)t_len * sizeof(float));
    float *sc = scores ? scores : stack_scores;

    for (int64_t tok = tok_start; tok < tok_end; ++tok) {
        const float *q_tok = q + tok * ch;
        const float *kc = k_cur + tok * ch;
        const float *vc = v_cur + tok * ch;
        float *kb = k_buf + tok * t_len * ch;
        float *vb = v_buf + tok * t_len * ch;
        float *out_tok = out + tok * ch;

        for (int64_t h = 0; h < n_heads; ++h) {
            const int64_t h_off = h * dh;
            const float *qh = q_tok + h_off;
            float *oh = out_tok + h_off;

            float max_sc = -3.402823466e+38f;
            for (int64_t t = 0; t < t_len; ++t) {
                const int64_t sl = slot_map[t];
                const float *kt = (sl < 0) ? (kc + h_off) : (kb + sl * ch + h_off);
                const float *kpet = k_pe + t * ch + h_off;
                __m512 vacc = _mm512_setzero_ps();
                int64_t i = 0;
                for (; i + 16 <= dh; i += 16) {
                    __m512 vq = _mm512_loadu_ps(qh + i);
                    __m512 vk = _mm512_add_ps(_mm512_loadu_ps(kt + i), _mm512_loadu_ps(kpet + i));
                    vacc = _mm512_fmadd_ps(vq, vk, vacc);
                }
                if (i < dh) {
                    __mmask16 km = (__mmask16)((1u << (dh - i)) - 1u);
                    __m512 vq = _mm512_maskz_loadu_ps(km, qh + i);
                    __m512 vk = _mm512_add_ps(_mm512_maskz_loadu_ps(km, kt + i), _mm512_maskz_loadu_ps(km, kpet + i));
                    vacc = _mm512_fmadd_ps(vq, vk, vacc);
                }
                float s = _mm512_reduce_add_ps(vacc) * scale;
                sc[t] = s;
                if (s > max_sc) max_sc = s;
            }

            const __m512 vmax = _mm512_set1_ps(max_sc);
            __m512 vsum = _mm512_setzero_ps();
            int64_t t = 0;
            for (; t + 16 <= t_len; t += 16) {
                __m512 ve = exp_ps_avx512(_mm512_sub_ps(_mm512_loadu_ps(sc + t), vmax));
                _mm512_storeu_ps(sc + t, ve);
                vsum = _mm512_add_ps(vsum, ve);
            }
            if (t < t_len) {
                __mmask16 km = (__mmask16)((1u << (t_len - t)) - 1u);
                __m512 ve = exp_ps_avx512(_mm512_sub_ps(_mm512_maskz_loadu_ps(km, sc + t), vmax));
                _mm512_mask_storeu_ps(sc + t, km, ve);
                vsum = _mm512_mask_add_ps(vsum, km, vsum, ve);
            }
            float inv_sum = 1.0f / _mm512_reduce_add_ps(vsum);
            const __m512 vinv = _mm512_set1_ps(inv_sum);
            for (t = 0; t + 16 <= t_len; t += 16) {
                _mm512_storeu_ps(sc + t, _mm512_mul_ps(_mm512_loadu_ps(sc + t), vinv));
            }
            if (t < t_len) {
                __mmask16 km = (__mmask16)((1u << (t_len - t)) - 1u);
                _mm512_mask_storeu_ps(sc + t, km, _mm512_mul_ps(_mm512_maskz_loadu_ps(km, sc + t), vinv));
            }

            int64_t i = 0;
            for (; i + 16 <= dh; i += 16) {
                __m512 oacc = _mm512_setzero_ps();
                for (int64_t tt = 0; tt < t_len; ++tt) {
                    const int64_t sl = slot_map[tt];
                    const float *vt = (sl < 0) ? (vc + h_off + i) : (vb + sl * ch + h_off + i);
                    const float *vpet = v_pe + tt * ch + h_off + i;
                    __m512 vw = _mm512_set1_ps(sc[tt]);
                    __m512 vv = _mm512_add_ps(_mm512_loadu_ps(vt), _mm512_loadu_ps(vpet));
                    oacc = _mm512_fmadd_ps(vw, vv, oacc);
                }
                _mm512_storeu_ps(oh + i, oacc);
            }
            if (i < dh) {
                __mmask16 km = (__mmask16)((1u << (dh - i)) - 1u);
                __m512 oacc = _mm512_setzero_ps();
                for (int64_t tt = 0; tt < t_len; ++tt) {
                    const int64_t sl = slot_map[tt];
                    const float *vt = (sl < 0) ? (vc + h_off + i) : (vb + sl * ch + h_off + i);
                    const float *vpet = v_pe + tt * ch + h_off + i;
                    __m512 vw = _mm512_set1_ps(sc[tt]);
                    __m512 vv = _mm512_add_ps(_mm512_maskz_loadu_ps(km, vt), _mm512_maskz_loadu_ps(km, vpet));
                    oacc = _mm512_fmadd_ps(vw, vv, oacc);
                }
                _mm512_mask_storeu_ps(oh + i, km, oacc);
            }
        }

        if (t_len >= 4) {
            const int64_t write_slot = ctx->cursor;
            memcpy(kb + write_slot * ch, kc, ch_bytes);
            memcpy(vb + write_slot * ch, vc, ch_bytes);
        }
    }

    if (scores) {
        free(scores);
    }
}

void temporal_attn_roll_ring_f32(
    const float *q,
    const float *k_cur,
    const float *v_cur,
    float *k_buf,
    float *v_buf,
    const float *k_pe,
    const float *v_pe,
    float *out,
    int64_t d,
    int64_t t_len,
    int64_t n_heads,
    int64_t dh,
    float scale,
    int has_history,
    int cursor
) {
    const int64_t ch = n_heads * dh;
    if (!has_history) {
        temporal_attn_roll_f32(
            q, k_cur, v_cur, k_buf, v_buf, k_pe, v_pe, out, d, t_len, n_heads, dh, scale, 0
        );
        return;
    }

    int64_t stack_map[128];
    int64_t *heap_map = (t_len > 128) ? (int64_t *)malloc((size_t)t_len * sizeof(int64_t)) : NULL;
    int64_t *slot_map = heap_map ? heap_map : stack_map;

    int resolved_cursor = cursor;
    if (t_len >= 4) {
        const int64_t ring = t_len - 3;
        const int64_t base_off = (((int64_t)cursor - 2) % ring + ring) % ring;
        resolved_cursor = (int)(2 + base_off);
        slot_map[0] = 0;
        slot_map[1] = 1;
        for (int64_t t = 2; t < t_len - 1; ++t) {
            slot_map[t] = 2 + (base_off + (t - 2)) % ring;
        }
        slot_map[t_len - 1] = -1;
    } else {
        for (int64_t t = 0; t < t_len - 1; ++t) {
            slot_map[t] = t;
        }
        if (t_len > 0) {
            slot_map[t_len - 1] = -1;
        }
    }

    temporal_attn_ctx_t ctx = {
        .q = q,
        .k_cur = k_cur,
        .v_cur = v_cur,
        .k_buf = k_buf,
        .v_buf = v_buf,
        .k_pe = k_pe,
        .v_pe = v_pe,
        .out = out,
        .t_len = t_len,
        .n_heads = n_heads,
        .dh = dh,
        .ch = ch,
        .scale = scale,
        .has_history = 1,
        .cursor = resolved_cursor,
        .slot_map = slot_map,
    };
    if (d >= 16 && d * ch * t_len >= 16384) {
        lse_parallel_for(d, 4, temporal_attn_roll_ring_worker, &ctx);
    } else {
        temporal_attn_roll_ring_worker(0, d, &ctx);
    }

    if (heap_map) {
        free(heap_map);
    }
}

/* ---------------- Phase 1b: Bokeh Pyramid & Guided CoC Filter ---------------- */

typedef struct {
    const float *src;
    float *dst;
    int h;
    int w;
} gaussian5_hw4_ctx_t;

static void gaussian5_hw4_h_worker(int64_t y_start, int64_t y_end, void *arg) {
    const gaussian5_hw4_ctx_t *ctx = (const gaussian5_hw4_ctx_t *)arg;
    const int w = ctx->w;
    const int stride = w * 4;
    const __m512 v4 = _mm512_set1_ps(4.0f);
    const __m512 v6 = _mm512_set1_ps(6.0f);
    const __m512 vscale = _mm512_set1_ps(1.0f / 16.0f);
    const __m128 v4_128 = _mm_set1_ps(4.0f);
    const __m128 v6_128 = _mm_set1_ps(6.0f);
    const __m128 vscale_128 = _mm_set1_ps(1.0f / 16.0f);

    for (int y = (int)y_start; y < (int)y_end; ++y) {
        const float *srow = ctx->src + (size_t)y * (size_t)stride;
        float *drow = ctx->dst + (size_t)y * (size_t)stride;

        int x = 0;
        const int left_end = (w < 2) ? w : 2;
        for (; x < left_end; ++x) {
            int xm2 = (x - 2 < 0) ? 0 : (x - 2);
            int xm1 = (x - 1 < 0) ? 0 : (x - 1);
            int xp1 = (x + 1 >= w) ? (w - 1) : (x + 1);
            int xp2 = (x + 2 >= w) ? (w - 1) : (x + 2);
            __m128 a0 = _mm_loadu_ps(srow + xm2 * 4);
            __m128 a1 = _mm_loadu_ps(srow + xm1 * 4);
            __m128 a2 = _mm_loadu_ps(srow + x * 4);
            __m128 a3 = _mm_loadu_ps(srow + xp1 * 4);
            __m128 a4 = _mm_loadu_ps(srow + xp2 * 4);
            __m128 sum = _mm_add_ps(a0, a4);
            sum = _mm_fmadd_ps(v4_128, _mm_add_ps(a1, a3), sum);
            sum = _mm_fmadd_ps(v6_128, a2, sum);
            _mm_storeu_ps(drow + x * 4, _mm_mul_ps(sum, vscale_128));
        }

        const int mid_end = w - 2;
        for (; x + 4 <= mid_end; x += 4) {
            __m512 a0 = _mm512_loadu_ps(srow + (x - 2) * 4);
            __m512 a1 = _mm512_loadu_ps(srow + (x - 1) * 4);
            __m512 a2 = _mm512_loadu_ps(srow + x * 4);
            __m512 a3 = _mm512_loadu_ps(srow + (x + 1) * 4);
            __m512 a4 = _mm512_loadu_ps(srow + (x + 2) * 4);
            __m512 sum = _mm512_add_ps(a0, a4);
            sum = _mm512_fmadd_ps(v4, _mm512_add_ps(a1, a3), sum);
            sum = _mm512_fmadd_ps(v6, a2, sum);
            _mm512_storeu_ps(drow + x * 4, _mm512_mul_ps(sum, vscale));
        }

        for (; x < w; ++x) {
            int xm2 = (x - 2 < 0) ? 0 : (x - 2);
            int xm1 = (x - 1 < 0) ? 0 : (x - 1);
            int xp1 = (x + 1 >= w) ? (w - 1) : (x + 1);
            int xp2 = (x + 2 >= w) ? (w - 1) : (x + 2);
            __m128 a0 = _mm_loadu_ps(srow + xm2 * 4);
            __m128 a1 = _mm_loadu_ps(srow + xm1 * 4);
            __m128 a2 = _mm_loadu_ps(srow + x * 4);
            __m128 a3 = _mm_loadu_ps(srow + xp1 * 4);
            __m128 a4 = _mm_loadu_ps(srow + xp2 * 4);
            __m128 sum = _mm_add_ps(a0, a4);
            sum = _mm_fmadd_ps(v4_128, _mm_add_ps(a1, a3), sum);
            sum = _mm_fmadd_ps(v6_128, a2, sum);
            _mm_storeu_ps(drow + x * 4, _mm_mul_ps(sum, vscale_128));
        }
    }
}

static void gaussian5_hw4_v_worker(int64_t y_start, int64_t y_end, void *arg) {
    const gaussian5_hw4_ctx_t *ctx = (const gaussian5_hw4_ctx_t *)arg;
    const int h = ctx->h;
    const int w4 = ctx->w * 4;
    const __m512 v4 = _mm512_set1_ps(4.0f);
    const __m512 v6 = _mm512_set1_ps(6.0f);
    const __m512 vscale = _mm512_set1_ps(1.0f / 16.0f);

    for (int y = (int)y_start; y < (int)y_end; ++y) {
        int ym2 = (y - 2 < 0) ? 0 : (y - 2);
        int ym1 = (y - 1 < 0) ? 0 : (y - 1);
        int yp1 = (y + 1 >= h) ? (h - 1) : (y + 1);
        int yp2 = (y + 2 >= h) ? (h - 1) : (y + 2);

        const float *r0 = ctx->src + (size_t)ym2 * (size_t)w4;
        const float *r1 = ctx->src + (size_t)ym1 * (size_t)w4;
        const float *r2 = ctx->src + (size_t)y * (size_t)w4;
        const float *r3 = ctx->src + (size_t)yp1 * (size_t)w4;
        const float *r4 = ctx->src + (size_t)yp2 * (size_t)w4;
        float *drow = ctx->dst + (size_t)y * (size_t)w4;

        int k = 0;
        for (; k + 16 <= w4; k += 16) {
            __m512 a0 = _mm512_loadu_ps(r0 + k);
            __m512 a1 = _mm512_loadu_ps(r1 + k);
            __m512 a2 = _mm512_loadu_ps(r2 + k);
            __m512 a3 = _mm512_loadu_ps(r3 + k);
            __m512 a4 = _mm512_loadu_ps(r4 + k);
            __m512 sum = _mm512_add_ps(a0, a4);
            sum = _mm512_fmadd_ps(v4, _mm512_add_ps(a1, a3), sum);
            sum = _mm512_fmadd_ps(v6, a2, sum);
            _mm512_storeu_ps(drow + k, _mm512_mul_ps(sum, vscale));
        }
        if (k < w4) {
            __mmask16 km = (__mmask16)((1u << (w4 - k)) - 1u);
            __m512 a0 = _mm512_maskz_loadu_ps(km, r0 + k);
            __m512 a1 = _mm512_maskz_loadu_ps(km, r1 + k);
            __m512 a2 = _mm512_maskz_loadu_ps(km, r2 + k);
            __m512 a3 = _mm512_maskz_loadu_ps(km, r3 + k);
            __m512 a4 = _mm512_maskz_loadu_ps(km, r4 + k);
            __m512 sum = _mm512_add_ps(a0, a4);
            sum = _mm512_fmadd_ps(v4, _mm512_add_ps(a1, a3), sum);
            sum = _mm512_fmadd_ps(v6, a2, sum);
            _mm512_mask_storeu_ps(drow + k, km, _mm512_mul_ps(sum, vscale));
        }
    }
}

static __thread float *g_gaussian5_tmp = NULL;
static __thread size_t g_gaussian5_tmp_cap = 0;

int gaussian5_hw4_f32(const float *src, float *dst, int h, int w, int passes) {
    if (h <= 0 || w <= 0) {
        return -1;
    }
    const size_t total_floats = (size_t)h * (size_t)w * 4;
    if (passes <= 0) {
        if (dst != src) {
            memcpy(dst, src, total_floats * sizeof(float));
        }
        return 0;
    }
    if (total_floats > g_gaussian5_tmp_cap) {
        float *ntmp = (float *)realloc(g_gaussian5_tmp, total_floats * sizeof(float));
        if (!ntmp) {
            return -1;
        }
        g_gaussian5_tmp = ntmp;
        g_gaussian5_tmp_cap = total_floats;
    }
    float *tmp = g_gaussian5_tmp;

    const int use_mt = (h >= 16 && (int64_t)h * w >= 8192);
    for (int p = 0; p < passes; ++p) {
        gaussian5_hw4_ctx_t h_ctx = {
            .src = (p == 0) ? src : dst,
            .dst = tmp,
            .h = h,
            .w = w,
        };
        if (use_mt) {
            lse_parallel_for((int64_t)h, 8, gaussian5_hw4_h_worker, &h_ctx);
        } else {
            gaussian5_hw4_h_worker(0, (int64_t)h, &h_ctx);
        }

        gaussian5_hw4_ctx_t v_ctx = {
            .src = tmp,
            .dst = dst,
            .h = h,
            .w = w,
        };
        if (use_mt) {
            lse_parallel_for((int64_t)h, 8, gaussian5_hw4_v_worker, &v_ctx);
        } else {
            gaussian5_hw4_v_worker(0, (int64_t)h, &v_ctx);
        }
    }

    return 0;
}

typedef struct {
    const float *src;
    float *dst;
    int in_h;
    int in_w;
    int out_w;
} downsample2x_hw4_ctx_t;

static void downsample2x_hw4_worker(int64_t oy_start, int64_t oy_end, void *arg) {
    const downsample2x_hw4_ctx_t *ctx = (const downsample2x_hw4_ctx_t *)arg;
    const int in_h = ctx->in_h;
    const int in_w = ctx->in_w;
    const int out_w = ctx->out_w;
    const __m512 v025 = _mm512_set1_ps(0.25f);
    const __m128 v025_128 = _mm_set1_ps(0.25f);

    for (int oy = (int)oy_start; oy < (int)oy_end; ++oy) {
        const int y0 = oy * 2;
        const int y1 = (y0 + 1 < in_h) ? (y0 + 1) : y0;
        const float *r0 = ctx->src + (size_t)y0 * (size_t)in_w * 4;
        const float *r1 = ctx->src + (size_t)y1 * (size_t)in_w * 4;
        float *dst = ctx->dst + (size_t)oy * (size_t)out_w * 4;

        int ox = 0;
        for (; ox + 4 <= out_w && (ox + 3) * 2 + 1 < in_w; ox += 4) {
            const int x0 = ox * 2;
            __m512 s0 = _mm512_add_ps(_mm512_loadu_ps(r0 + x0 * 4), _mm512_loadu_ps(r1 + x0 * 4));
            __m512 s1 = _mm512_add_ps(_mm512_loadu_ps(r0 + (x0 + 4) * 4), _mm512_loadu_ps(r1 + (x0 + 4) * 4));
            __m512 even = _mm512_shuffle_f32x4(s0, s1, _MM_SHUFFLE(2, 0, 2, 0));
            __m512 odd = _mm512_shuffle_f32x4(s0, s1, _MM_SHUFFLE(3, 1, 3, 1));
            _mm512_storeu_ps(dst + ox * 4, _mm512_mul_ps(_mm512_add_ps(even, odd), v025));
        }
        for (; ox < out_w; ++ox) {
            const int x0 = ox * 2;
            const int x1 = (x0 + 1 < in_w) ? (x0 + 1) : x0;
            __m128 a = _mm_loadu_ps(r0 + x0 * 4);
            __m128 b = _mm_loadu_ps(r0 + x1 * 4);
            __m128 c = _mm_loadu_ps(r1 + x0 * 4);
            __m128 d = _mm_loadu_ps(r1 + x1 * 4);
            __m128 s = _mm_add_ps(_mm_add_ps(a, b), _mm_add_ps(c, d));
            _mm_storeu_ps(dst + ox * 4, _mm_mul_ps(s, v025_128));
        }
    }
}

void downsample2x_hw4_f32(const float *src, float *dst, int in_h, int in_w) {
    if (in_h <= 0 || in_w <= 0) {
        return;
    }
    const int out_h = (in_h + 1) >> 1;
    const int out_w = (in_w + 1) >> 1;
    downsample2x_hw4_ctx_t ctx = {
        .src = src,
        .dst = dst,
        .in_h = in_h,
        .in_w = in_w,
        .out_w = out_w,
    };
    if (out_h >= 16 && (int64_t)out_h * out_w >= 8192) {
        lse_parallel_for((int64_t)out_h, 8, downsample2x_hw4_worker, &ctx);
    } else {
        downsample2x_hw4_worker(0, (int64_t)out_h, &ctx);
    }
}

typedef struct {
    const float *src;
    float *dst;
    int h;
    int w;
    int radius;
} box_filter_ctx_t;

static void box_filter_h_worker(int64_t y_start, int64_t y_end, void *arg) {
    const box_filter_ctx_t *ctx = (const box_filter_ctx_t *)arg;
    const int w = ctx->w;
    const int r = ctx->radius;
    const double inv_win = 1.0 / (double)(2 * r + 1);

    for (int y = (int)y_start; y < (int)y_end; ++y) {
        const float *row = ctx->src + (size_t)y * (size_t)w;
        float *drow = ctx->dst + (size_t)y * (size_t)w;
        double acc = (double)(r + 1) * (double)row[0];
        for (int k = 1; k <= r; ++k) {
            int idx = (k < w) ? k : (w - 1);
            acc += (double)row[idx];
        }
        drow[0] = (float)(acc * inv_win);
        for (int x = 1; x < w; ++x) {
            int x_out = x - 1 - r;
            if (x_out < 0) x_out = 0;
            int x_in = x + r;
            if (x_in >= w) x_in = w - 1;
            acc += (double)row[x_in] - (double)row[x_out];
            drow[x] = (float)(acc * inv_win);
        }
    }
}

static void box_filter_v_worker(int64_t x_start, int64_t x_end, void *arg) {
    const box_filter_ctx_t *ctx = (const box_filter_ctx_t *)arg;
    const int h = ctx->h;
    const int w = ctx->w;
    const int r = ctx->radius;
    const int sw = (int)(x_end - x_start);
    const double inv_win = 1.0 / (double)(2 * r + 1);
    const __m512d v_inv_win = _mm512_set1_pd(inv_win);
    const __m512d v_r1 = _mm512_set1_pd((double)(r + 1));

    double stack_acc[512];
    double *heap_acc = (sw > 512) ? (double *)malloc((size_t)sw * sizeof(double)) : NULL;
    double *acc = heap_acc ? heap_acc : stack_acc;

    const float *row0 = ctx->src + x_start;
    int j = 0;
    for (; j + 8 <= sw; j += 8) {
        __m512d va = _mm512_mul_pd(_mm512_cvtps_pd(_mm256_loadu_ps(row0 + j)), v_r1);
        _mm512_storeu_pd(acc + j, va);
    }
    for (; j < sw; ++j) {
        acc[j] = (double)(r + 1) * (double)row0[j];
    }

    for (int k = 1; k <= r; ++k) {
        int yk = (k < h) ? k : (h - 1);
        const float *rk = ctx->src + (size_t)yk * (size_t)w + x_start;
        j = 0;
        for (; j + 8 <= sw; j += 8) {
            __m512d va = _mm512_loadu_pd(acc + j);
            va = _mm512_add_pd(va, _mm512_cvtps_pd(_mm256_loadu_ps(rk + j)));
            _mm512_storeu_pd(acc + j, va);
        }
        for (; j < sw; ++j) {
            acc[j] += (double)rk[j];
        }
    }

    float *drow0 = ctx->dst + x_start;
    j = 0;
    for (; j + 8 <= sw; j += 8) {
        __m512d va = _mm512_loadu_pd(acc + j);
        _mm256_storeu_ps(drow0 + j, _mm512_cvtpd_ps(_mm512_mul_pd(va, v_inv_win)));
    }
    for (; j < sw; ++j) {
        drow0[j] = (float)(acc[j] * inv_win);
    }

    for (int y = 1; y < h; ++y) {
        int y_out = y - 1 - r;
        if (y_out < 0) y_out = 0;
        int y_in = y + r;
        if (y_in >= h) y_in = h - 1;
        const float *rin = ctx->src + (size_t)y_in * (size_t)w + x_start;
        const float *rout = ctx->src + (size_t)y_out * (size_t)w + x_start;
        float *drow = ctx->dst + (size_t)y * (size_t)w + x_start;

        j = 0;
        for (; j + 8 <= sw; j += 8) {
            __m512d va = _mm512_loadu_pd(acc + j);
            __m512d vin = _mm512_cvtps_pd(_mm256_loadu_ps(rin + j));
            __m512d vout = _mm512_cvtps_pd(_mm256_loadu_ps(rout + j));
            va = _mm512_add_pd(va, _mm512_sub_pd(vin, vout));
            _mm512_storeu_pd(acc + j, va);
            _mm256_storeu_ps(drow + j, _mm512_cvtpd_ps(_mm512_mul_pd(va, v_inv_win)));
        }
        for (; j < sw; ++j) {
            acc[j] += (double)rin[j] - (double)rout[j];
            drow[j] = (float)(acc[j] * inv_win);
        }
    }

    if (heap_acc) {
        free(heap_acc);
    }
}

static void box_filter_2d_with_tmp(const float *src, float *dst, float *tmp, int h, int w, int radius) {
    if (radius <= 0) {
        if (dst != src) {
            memcpy(dst, src, (size_t)h * (size_t)w * sizeof(float));
        }
        return;
    }
    const int use_mt = ((int64_t)h * w >= 32768);
    box_filter_ctx_t h_ctx = { .src = src, .dst = tmp, .h = h, .w = w, .radius = radius };
    if (use_mt && h >= 16) {
        lse_parallel_for((int64_t)h, 8, box_filter_h_worker, &h_ctx);
    } else {
        box_filter_h_worker(0, (int64_t)h, &h_ctx);
    }

    box_filter_ctx_t v_ctx = { .src = tmp, .dst = dst, .h = h, .w = w, .radius = radius };
    if (use_mt && w >= 128) {
        lse_parallel_for((int64_t)w, 64, box_filter_v_worker, &v_ctx);
    } else {
        box_filter_v_worker(0, (int64_t)w, &v_ctx);
    }
}

int box_filter_2d_f32(const float *src, float *dst, int h, int w, int radius) {
    if (h <= 0 || w <= 0) {
        return -1;
    }
    if (radius <= 0) {
        if (dst != src) {
            memcpy(dst, src, (size_t)h * (size_t)w * sizeof(float));
        }
        return 0;
    }
    float *tmp = (float *)malloc((size_t)h * (size_t)w * sizeof(float));
    if (!tmp) {
        return -1;
    }
    box_filter_2d_with_tmp(src, dst, tmp, h, w, radius);
    free(tmp);
    return 0;
}

typedef struct {
    const uint8_t *frame;
    float *luma_full;
    int fmt_code;
    int h;
    int w;
} extract_luma_ctx_t;

static void extract_luma_worker(int64_t y_start, int64_t y_end, void *arg) {
    const extract_luma_ctx_t *ctx = (const extract_luma_ctx_t *)arg;
    const int fmt = ctx->fmt_code;
    const int w = ctx->w;
    const __m512 vinv255 = _mm512_set1_ps(1.0f / 255.0f);

    if (fmt == 2 || fmt == 3) {
        const __m256i vmask_lo = _mm256_set1_epi16(0x00FF);
        const int pair_w = w >> 1;
        for (int y = (int)y_start; y < (int)y_end; ++y) {
            const uint8_t *srow = ctx->frame + (size_t)y * (size_t)pair_w * 4;
            float *drow = ctx->luma_full + (size_t)y * (size_t)w;
            int x = 0;
            for (; x + 16 <= w; x += 16) {
                __m256i v = _mm256_loadu_si256((const __m256i *)(srow + x * 2));
                __m256i y16 = (fmt == 2) ? _mm256_and_si256(v, vmask_lo) : _mm256_srli_epi16(v, 8);
                __m512 yf = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(y16));
                _mm512_storeu_ps(drow + x, _mm512_mul_ps(yf, vinv255));
            }
            const int y_off0 = (fmt == 2) ? 0 : 1;
            const int y_off1 = (fmt == 2) ? 2 : 3;
            for (; x < w; x += 2) {
                const uint8_t *p = srow + (size_t)x * 2;
                drow[x + 0] = (float)p[y_off0] * (1.0f / 255.0f);
                drow[x + 1] = (float)p[y_off1] * (1.0f / 255.0f);
            }
        }
        return;
    }

    const float w0 = (fmt == 0) ? (0.299f / 255.0f) : (0.114f / 255.0f);
    const float w1 = 0.587f / 255.0f;
    const float w2 = (fmt == 0) ? (0.114f / 255.0f) : (0.299f / 255.0f);
    const __m512 vw0 = _mm512_set1_ps(w0);
    const __m512 vw1 = _mm512_set1_ps(w1);
    const __m512 vw2 = _mm512_set1_ps(w2);
    const __m512i vmask8 = _mm512_set1_epi32(0xFF);
    const __m128i shuf_rgb = _mm_setr_epi8(0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1);

    for (int y = (int)y_start; y < (int)y_end; ++y) {
        const uint8_t *srow = ctx->frame + (size_t)y * (size_t)w * 3;
        float *drow = ctx->luma_full + (size_t)y * (size_t)w;
        int x = 0;
        for (; x + 17 <= w; x += 16) {
            const uint8_t *p = srow + (size_t)x * 3;
            __m128i q0 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(p + 0)), shuf_rgb);
            __m128i q1 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(p + 12)), shuf_rgb);
            __m128i q2 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(p + 24)), shuf_rgb);
            __m128i q3 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(p + 36)), shuf_rgb);
            __m512i v = _mm512_castsi128_si512(q0);
            v = _mm512_inserti32x4(v, q1, 1);
            v = _mm512_inserti32x4(v, q2, 2);
            v = _mm512_inserti32x4(v, q3, 3);

            __m512 c0 = _mm512_cvtepi32_ps(_mm512_and_si512(v, vmask8));
            __m512 c1 = _mm512_cvtepi32_ps(_mm512_and_si512(_mm512_srli_epi32(v, 8), vmask8));
            __m512 c2 = _mm512_cvtepi32_ps(_mm512_and_si512(_mm512_srli_epi32(v, 16), vmask8));
            __m512 luma = _mm512_fmadd_ps(c0, vw0, _mm512_fmadd_ps(c1, vw1, _mm512_mul_ps(c2, vw2)));
            _mm512_storeu_ps(drow + x, luma);
        }
        for (; x < w; ++x) {
            const uint8_t *p = srow + (size_t)x * 3;
            drow[x] = (float)p[0] * w0 + (float)p[1] * w1 + (float)p[2] * w2;
        }
    }
}

int extract_luma_and_downsample_f32(
    const uint8_t *frame,
    int fmt_code,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    float *luma_full,
    float *luma_low
) {
    if (in_h <= 0 || in_w <= 0 || out_h <= 0 || out_w <= 0 || fmt_code < 0 || fmt_code > 3) {
        return -1;
    }
    if ((fmt_code == 2 || fmt_code == 3) && (in_w & 1)) {
        return -1;
    }

    extract_luma_ctx_t ctx = {
        .frame = frame,
        .luma_full = luma_full,
        .fmt_code = fmt_code,
        .h = in_h,
        .w = in_w,
    };
    if (in_h >= 16 && (int64_t)in_h * in_w >= 8192) {
        lse_parallel_for((int64_t)in_h, 8, extract_luma_worker, &ctx);
    } else {
        extract_luma_worker(0, (int64_t)in_h, &ctx);
    }

    return bilinear_resize_nchw_f32(luma_full, luma_low, 1, in_h, in_w, out_h, out_w, 0);
}

typedef struct {
    const float *luma_full;
    const float *mean_a_low;
    const float *mean_b_low;
    float *coc_full;
    const int32_t *x0;
    const int32_t *x1;
    const int32_t *y0;
    const int32_t *y1;
    const float *wx;
    const float *wy;
    int full_h;
    int full_w;
    int low_w;
} guided_upsample_ctx_t;

static void guided_upsample_worker(int64_t oy_start, int64_t oy_end, void *arg) {
    const guided_upsample_ctx_t *ctx = (const guided_upsample_ctx_t *)arg;
    const int full_w = ctx->full_w;
    const int low_w = ctx->low_w;
    const int32_t *x0 = ctx->x0;
    const int32_t *x1 = ctx->x1;
    const int32_t *y0 = ctx->y0;
    const int32_t *y1 = ctx->y1;
    const float *wx = ctx->wx;
    const float *wy = ctx->wy;
    const __m512 vzero = _mm512_setzero_ps();
    const __m512 vone = _mm512_set1_ps(1.0f);

    float stack_rows[8192];
    float *heap_rows = (full_w > 2048) ? (float *)malloc((size_t)full_w * 4 * sizeof(float)) : NULL;
    float *buf = heap_rows ? heap_rows : stack_rows;
    float *a_slots[2] = { buf, buf + full_w };
    float *b_slots[2] = { buf + 2 * full_w, buf + 3 * full_w };
    int cached_id[2] = {-1, -1};

    for (int oy = (int)oy_start; oy < (int)oy_end; ++oy) {
        const int r0 = y0[oy];
        const int r1 = y1[oy];
        const float w = wy[oy];

        int s0;
        if (r0 == cached_id[0]) {
            s0 = 0;
        } else if (r0 == cached_id[1]) {
            s0 = 1;
        } else {
            s0 = (cached_id[0] == r1) ? 1 : 0;
            interp_row_x_f32(ctx->mean_a_low + (size_t)r0 * low_w, a_slots[s0], full_w, x0, x1, wx);
            interp_row_x_f32(ctx->mean_b_low + (size_t)r0 * low_w, b_slots[s0], full_w, x0, x1, wx);
            cached_id[s0] = r0;
        }

        const float *lf_row = ctx->luma_full + (size_t)oy * (size_t)full_w;
        float *out_row = ctx->coc_full + (size_t)oy * (size_t)full_w;
        const float *a0 = a_slots[s0];
        const float *b0 = b_slots[s0];

        if (r1 == r0 || w == 0.0f) {
            int ox = 0;
            for (; ox + 16 <= full_w; ox += 16) {
                __m512 ma = _mm512_loadu_ps(a0 + ox);
                __m512 mb = _mm512_loadu_ps(b0 + ox);
                __m512 lf = _mm512_loadu_ps(lf_row + ox);
                __m512 v = _mm512_fmadd_ps(ma, lf, mb);
                v = _mm512_min_ps(_mm512_max_ps(v, vzero), vone);
                _mm512_storeu_ps(out_row + ox, v);
            }
            if (ox < full_w) {
                __mmask16 km = (__mmask16)((1u << (full_w - ox)) - 1u);
                __m512 ma = _mm512_maskz_loadu_ps(km, a0 + ox);
                __m512 mb = _mm512_maskz_loadu_ps(km, b0 + ox);
                __m512 lf = _mm512_maskz_loadu_ps(km, lf_row + ox);
                __m512 v = _mm512_fmadd_ps(ma, lf, mb);
                v = _mm512_min_ps(_mm512_max_ps(v, vzero), vone);
                _mm512_mask_storeu_ps(out_row + ox, km, v);
            }
            continue;
        }

        int s1;
        if (r1 == cached_id[0]) {
            s1 = 0;
        } else if (r1 == cached_id[1]) {
            s1 = 1;
        } else {
            s1 = (s0 == 0) ? 1 : 0;
            interp_row_x_f32(ctx->mean_a_low + (size_t)r1 * low_w, a_slots[s1], full_w, x0, x1, wx);
            interp_row_x_f32(ctx->mean_b_low + (size_t)r1 * low_w, b_slots[s1], full_w, x0, x1, wx);
            cached_id[s1] = r1;
        }

        const float *a1 = a_slots[s1];
        const float *b1 = b_slots[s1];
        const __m512 vwy = _mm512_set1_ps(w);

        int ox = 0;
        for (; ox + 16 <= full_w; ox += 16) {
            __m512 va0 = _mm512_loadu_ps(a0 + ox);
            __m512 va1 = _mm512_loadu_ps(a1 + ox);
            __m512 vb0 = _mm512_loadu_ps(b0 + ox);
            __m512 vb1 = _mm512_loadu_ps(b1 + ox);
            __m512 ma = _mm512_fmadd_ps(vwy, _mm512_sub_ps(va1, va0), va0);
            __m512 mb = _mm512_fmadd_ps(vwy, _mm512_sub_ps(vb1, vb0), vb0);
            __m512 lf = _mm512_loadu_ps(lf_row + ox);
            __m512 v = _mm512_fmadd_ps(ma, lf, mb);
            v = _mm512_min_ps(_mm512_max_ps(v, vzero), vone);
            _mm512_storeu_ps(out_row + ox, v);
        }
        if (ox < full_w) {
            __mmask16 km = (__mmask16)((1u << (full_w - ox)) - 1u);
            __m512 va0 = _mm512_maskz_loadu_ps(km, a0 + ox);
            __m512 va1 = _mm512_maskz_loadu_ps(km, a1 + ox);
            __m512 vb0 = _mm512_maskz_loadu_ps(km, b0 + ox);
            __m512 vb1 = _mm512_maskz_loadu_ps(km, b1 + ox);
            __m512 ma = _mm512_fmadd_ps(vwy, _mm512_sub_ps(va1, va0), va0);
            __m512 mb = _mm512_fmadd_ps(vwy, _mm512_sub_ps(vb1, vb0), vb0);
            __m512 lf = _mm512_maskz_loadu_ps(km, lf_row + ox);
            __m512 v = _mm512_fmadd_ps(ma, lf, mb);
            v = _mm512_min_ps(_mm512_max_ps(v, vzero), vone);
            _mm512_mask_storeu_ps(out_row + ox, km, v);
        }
    }

    if (heap_rows) {
        free(heap_rows);
    }
}

int guided_upsample_coc_f32(
    const float *luma_full,
    const float *luma_low,
    const float *coc_low,
    float *coc_full,
    int full_h,
    int full_w,
    int low_h,
    int low_w,
    int radius,
    float eps
) {
    if (full_h <= 0 || full_w <= 0 || low_h <= 0 || low_w <= 0) {
        return -1;
    }

    const size_t low_n = (size_t)low_h * (size_t)low_w;
    float *scratch = (float *)malloc(low_n * 6 * sizeof(float));
    int32_t *ibuf = (int32_t *)malloc(((size_t)full_w * 2 + (size_t)full_h * 2) * sizeof(int32_t));
    float *fbuf = (float *)malloc(((size_t)full_w + (size_t)full_h) * sizeof(float));
    if (!scratch || !ibuf || !fbuf) {
        free(scratch);
        free(ibuf);
        free(fbuf);
        return -1;
    }

    float *mean_i = scratch;
    float *mean_p = mean_i + low_n;
    float *buf_in = mean_p + low_n;
    float *corr_i = buf_in + low_n;
    float *corr_ip = corr_i + low_n;
    float *box_tmp = corr_ip + low_n;

    box_filter_2d_with_tmp(luma_low, mean_i, box_tmp, low_h, low_w, radius);
    box_filter_2d_with_tmp(coc_low, mean_p, box_tmp, low_h, low_w, radius);

    size_t k = 0;
    for (; k + 16 <= low_n; k += 16) {
        __m512 vi = _mm512_loadu_ps(luma_low + k);
        _mm512_storeu_ps(buf_in + k, _mm512_mul_ps(vi, vi));
    }
    for (; k < low_n; ++k) {
        buf_in[k] = luma_low[k] * luma_low[k];
    }
    box_filter_2d_with_tmp(buf_in, corr_i, box_tmp, low_h, low_w, radius);

    k = 0;
    for (; k + 16 <= low_n; k += 16) {
        __m512 vi = _mm512_loadu_ps(luma_low + k);
        __m512 vp = _mm512_loadu_ps(coc_low + k);
        _mm512_storeu_ps(buf_in + k, _mm512_mul_ps(vi, vp));
    }
    for (; k < low_n; ++k) {
        buf_in[k] = luma_low[k] * coc_low[k];
    }
    box_filter_2d_with_tmp(buf_in, corr_ip, box_tmp, low_h, low_w, radius);

    const __m512 vzero = _mm512_setzero_ps();
    const __m512 veps = _mm512_set1_ps(eps);
    float *a_low = buf_in;
    float *b_low = corr_i;
    k = 0;
    for (; k + 16 <= low_n; k += 16) {
        __m512 mi = _mm512_loadu_ps(mean_i + k);
        __m512 mp = _mm512_loadu_ps(mean_p + k);
        __m512 ci = _mm512_loadu_ps(corr_i + k);
        __m512 cip = _mm512_loadu_ps(corr_ip + k);
        __m512 var_i = _mm512_max_ps(_mm512_fnmadd_ps(mi, mi, ci), vzero);
        __m512 cov_ip = _mm512_fnmadd_ps(mi, mp, cip);
        __m512 a = _mm512_div_ps(cov_ip, _mm512_add_ps(var_i, veps));
        __m512 b = _mm512_fnmadd_ps(a, mi, mp);
        _mm512_storeu_ps(a_low + k, a);
        _mm512_storeu_ps(b_low + k, b);
    }
    for (; k < low_n; ++k) {
        float mi = mean_i[k];
        float mp = mean_p[k];
        float var_i = corr_i[k] - mi * mi;
        if (var_i < 0.0f) var_i = 0.0f;
        float cov_ip = corr_ip[k] - mi * mp;
        float a = cov_ip / (var_i + eps);
        a_low[k] = a;
        b_low[k] = mp - a * mi;
    }

    float *mean_a_low = mean_i;
    float *mean_b_low = mean_p;
    box_filter_2d_with_tmp(a_low, mean_a_low, box_tmp, low_h, low_w, radius);
    box_filter_2d_with_tmp(b_low, mean_b_low, box_tmp, low_h, low_w, radius);

    int32_t *x0 = ibuf;
    int32_t *x1 = x0 + full_w;
    int32_t *y0 = x1 + full_w;
    int32_t *y1 = y0 + full_h;
    float *wx = fbuf;
    float *wy = wx + full_w;

    if (low_w == 1 || full_w == 1) {
        for (int i = 0; i < full_w; ++i) {
            x0[i] = 0;
            x1[i] = 0;
            wx[i] = 0.0f;
        }
    } else {
        float scale = (float)((double)low_w / (double)full_w);
        for (int i = 0; i < full_w; ++i) {
            float prod = ((float)i + 0.5f) * scale;
            __asm__("" : "+x"(prod));
            float xc = prod - 0.5f;
            if (xc < 0.0f) xc = 0.0f;
            if (xc > (float)(low_w - 1)) xc = (float)(low_w - 1);
            int i0 = (int)floorf(xc);
            int i1 = (i0 + 1 < low_w) ? i0 + 1 : low_w - 1;
            x0[i] = i0;
            x1[i] = i1;
            wx[i] = xc - (float)i0;
        }
    }

    if (low_h == 1 || full_h == 1) {
        for (int i = 0; i < full_h; ++i) {
            y0[i] = 0;
            y1[i] = 0;
            wy[i] = 0.0f;
        }
    } else {
        float scale = (float)((double)low_h / (double)full_h);
        for (int i = 0; i < full_h; ++i) {
            float prod = ((float)i + 0.5f) * scale;
            __asm__("" : "+x"(prod));
            float yc = prod - 0.5f;
            if (yc < 0.0f) yc = 0.0f;
            if (yc > (float)(low_h - 1)) yc = (float)(low_h - 1);
            int i0 = (int)floorf(yc);
            int i1 = (i0 + 1 < low_h) ? i0 + 1 : low_h - 1;
            y0[i] = i0;
            y1[i] = i1;
            wy[i] = yc - (float)i0;
        }
    }

    guided_upsample_ctx_t ctx = {
        .luma_full = luma_full,
        .mean_a_low = mean_a_low,
        .mean_b_low = mean_b_low,
        .coc_full = coc_full,
        .x0 = x0,
        .x1 = x1,
        .y0 = y0,
        .y1 = y1,
        .wx = wx,
        .wy = wy,
        .full_h = full_h,
        .full_w = full_w,
        .low_w = low_w,
    };
    if (full_h >= 16 && (int64_t)full_h * full_w >= 8192) {
        lse_parallel_for((int64_t)full_h, 8, guided_upsample_worker, &ctx);
    } else {
        guided_upsample_worker(0, (int64_t)full_h, &ctx);
    }

    free(scratch);
    free(ibuf);
    free(fbuf);
    return 0;
}

/* ---------------- Phase 5a: Fused Preprocess & Spatial MHA ---------------- */

typedef struct {
    const uint8_t *rgb;
    float *out_nchw;
    const int32_t *x0;
    const int32_t *x1;
    const int32_t *y0;
    const int32_t *y1;
    const float *wx;
    const float *wy;
    int in_h;
    int in_w;
    int out_h;
    int out_w;
} preprocess_rgb_ctx_t;

static inline void interp_rgb_u8_row_x(
    const uint8_t *in_row,
    float *dst_3_planes,
    int out_w,
    const int32_t *x0,
    const int32_t *x1,
    const float *wx
) {
    float *d0 = dst_3_planes;
    float *d1 = dst_3_planes + out_w;
    float *d2 = dst_3_planes + 2 * out_w;
    for (int ox = 0; ox < out_w; ++ox) {
        const uint8_t *p0 = in_row + (size_t)x0[ox] * 3;
        const uint8_t *p1 = in_row + (size_t)x1[ox] * 3;
        const float w = wx[ox];
        const float f00 = (float)p0[0];
        const float f01 = (float)p0[1];
        const float f02 = (float)p0[2];
        d0[ox] = f00 + w * ((float)p1[0] - f00);
        d1[ox] = f01 + w * ((float)p1[1] - f01);
        d2[ox] = f02 + w * ((float)p1[2] - f02);
    }
}

static void preprocess_rgb_worker(int64_t oy_start, int64_t oy_end, void *arg) {
    const preprocess_rgb_ctx_t *ctx = (const preprocess_rgb_ctx_t *)arg;
    const int in_w = ctx->in_w;
    const int out_h = ctx->out_h;
    const int out_w = ctx->out_w;
    const size_t out_hw = (size_t)out_h * (size_t)out_w;
    const size_t in_stride = (size_t)in_w * 3;

    const float scales[3] = {
        1.0f / (255.0f * 0.229f),
        1.0f / (255.0f * 0.224f),
        1.0f / (255.0f * 0.225f),
    };
    const float offsets[3] = {
        -0.485f / 0.229f,
        -0.456f / 0.224f,
        -0.406f / 0.225f,
    };

    if (ctx->in_h == out_h && in_w == out_w) {
        for (int oy = (int)oy_start; oy < (int)oy_end; ++oy) {
            const uint8_t *srow = ctx->rgb + (size_t)oy * in_stride;
            float *out0 = ctx->out_nchw + 0 * out_hw + (size_t)oy * out_w;
            float *out1 = ctx->out_nchw + 1 * out_hw + (size_t)oy * out_w;
            float *out2 = ctx->out_nchw + 2 * out_hw + (size_t)oy * out_w;
            for (int ox = 0; ox < out_w; ++ox) {
                const uint8_t *p = srow + (size_t)ox * 3;
                out0[ox] = (float)p[0] * scales[0] + offsets[0];
                out1[ox] = (float)p[1] * scales[1] + offsets[1];
                out2[ox] = (float)p[2] * scales[2] + offsets[2];
            }
        }
        return;
    }

    float stack_rows[6144];
    float *heap_rows = (out_w > 1024) ? (float *)malloc((size_t)out_w * 6 * sizeof(float)) : NULL;
    float *buf = heap_rows ? heap_rows : stack_rows;
    float *slots[2] = { buf, buf + (size_t)out_w * 3 };
    int cached_id[2] = {-1, -1};

    const int32_t *x0 = ctx->x0;
    const int32_t *x1 = ctx->x1;
    const int32_t *y0 = ctx->y0;
    const int32_t *y1 = ctx->y1;
    const float *wx = ctx->wx;
    const float *wy = ctx->wy;

    for (int oy = (int)oy_start; oy < (int)oy_end; ++oy) {
        const int r0 = y0[oy];
        const int r1 = y1[oy];
        const float w = wy[oy];

        int s0;
        if (r0 == cached_id[0]) {
            s0 = 0;
        } else if (r0 == cached_id[1]) {
            s0 = 1;
        } else {
            s0 = (cached_id[0] == r1) ? 1 : 0;
            interp_rgb_u8_row_x(ctx->rgb + (size_t)r0 * in_stride, slots[s0], out_w, x0, x1, wx);
            cached_id[s0] = r0;
        }

        const float *p0 = slots[s0];
        if (r1 == r0 || w == 0.0f) {
            for (int c = 0; c < 3; ++c) {
                const float *a = p0 + (size_t)c * out_w;
                float *dst = ctx->out_nchw + (size_t)c * out_hw + (size_t)oy * out_w;
                const __m512 vsc = _mm512_set1_ps(scales[c]);
                const __m512 vof = _mm512_set1_ps(offsets[c]);
                int ox = 0;
                for (; ox + 16 <= out_w; ox += 16) {
                    _mm512_storeu_ps(dst + ox, _mm512_fmadd_ps(_mm512_loadu_ps(a + ox), vsc, vof));
                }
                if (ox < out_w) {
                    __mmask16 km = (__mmask16)((1u << (out_w - ox)) - 1u);
                    _mm512_mask_storeu_ps(dst + ox, km, _mm512_fmadd_ps(_mm512_maskz_loadu_ps(km, a + ox), vsc, vof));
                }
            }
            continue;
        }

        int s1;
        if (r1 == cached_id[0]) {
            s1 = 0;
        } else if (r1 == cached_id[1]) {
            s1 = 1;
        } else {
            s1 = (s0 == 0) ? 1 : 0;
            interp_rgb_u8_row_x(ctx->rgb + (size_t)r1 * in_stride, slots[s1], out_w, x0, x1, wx);
            cached_id[s1] = r1;
        }

        const float *p1 = slots[s1];
        const __m512 vwy = _mm512_set1_ps(w);
        for (int c = 0; c < 3; ++c) {
            const float *a = p0 + (size_t)c * out_w;
            const float *b = p1 + (size_t)c * out_w;
            float *dst = ctx->out_nchw + (size_t)c * out_hw + (size_t)oy * out_w;
            const __m512 vsc = _mm512_set1_ps(scales[c]);
            const __m512 vof = _mm512_set1_ps(offsets[c]);
            int ox = 0;
            for (; ox + 16 <= out_w; ox += 16) {
                __m512 va = _mm512_loadu_ps(a + ox);
                __m512 vb = _mm512_loadu_ps(b + ox);
                __m512 interp = _mm512_fmadd_ps(vwy, _mm512_sub_ps(vb, va), va);
                _mm512_storeu_ps(dst + ox, _mm512_fmadd_ps(interp, vsc, vof));
            }
            if (ox < out_w) {
                __mmask16 km = (__mmask16)((1u << (out_w - ox)) - 1u);
                __m512 va = _mm512_maskz_loadu_ps(km, a + ox);
                __m512 vb = _mm512_maskz_loadu_ps(km, b + ox);
                __m512 interp = _mm512_fmadd_ps(vwy, _mm512_sub_ps(vb, va), va);
                _mm512_mask_storeu_ps(dst + ox, km, _mm512_fmadd_ps(interp, vsc, vof));
            }
        }
    }

    if (heap_rows) {
        free(heap_rows);
    }
}

int preprocess_rgb_u8_to_nchw_f32(
    const uint8_t *rgb,
    float *out_nchw,
    int in_h,
    int in_w,
    int out_h,
    int out_w
) {
    if (in_h <= 0 || in_w <= 0 || out_h <= 0 || out_w <= 0) {
        return -1;
    }
    if (in_h == out_h && in_w == out_w) {
        preprocess_rgb_ctx_t ctx = {
            .rgb = rgb,
            .out_nchw = out_nchw,
            .x0 = NULL, .x1 = NULL, .y0 = NULL, .y1 = NULL, .wx = NULL, .wy = NULL,
            .in_h = in_h, .in_w = in_w, .out_h = out_h, .out_w = out_w,
        };
        if (out_h >= 16 && (int64_t)out_h * out_w >= 8192) {
            lse_parallel_for((int64_t)out_h, 8, preprocess_rgb_worker, &ctx);
        } else {
            preprocess_rgb_worker(0, (int64_t)out_h, &ctx);
        }
        return 0;
    }

    int32_t *ibuf = (int32_t *)malloc(((size_t)out_w * 2 + (size_t)out_h * 2) * sizeof(int32_t));
    float *fbuf = (float *)malloc(((size_t)out_w + (size_t)out_h) * sizeof(float));
    if (!ibuf || !fbuf) {
        free(ibuf);
        free(fbuf);
        return -1;
    }

    int32_t *x0 = ibuf;
    int32_t *x1 = x0 + out_w;
    int32_t *y0 = x1 + out_w;
    int32_t *y1 = y0 + out_h;
    float *wx = fbuf;
    float *wy = wx + out_w;

    if (in_w == 1 || out_w == 1) {
        for (int i = 0; i < out_w; ++i) {
            x0[i] = 0; x1[i] = 0; wx[i] = 0.0f;
        }
    } else {
        float scale = (float)((double)in_w / (double)out_w);
        for (int i = 0; i < out_w; ++i) {
            float prod = ((float)i + 0.5f) * scale;
            __asm__("" : "+x"(prod));
            float xc = prod - 0.5f;
            if (xc < 0.0f) xc = 0.0f;
            if (xc > (float)(in_w - 1)) xc = (float)(in_w - 1);
            int i0 = (int)floorf(xc);
            int i1 = (i0 + 1 < in_w) ? i0 + 1 : in_w - 1;
            x0[i] = i0; x1[i] = i1; wx[i] = xc - (float)i0;
        }
    }

    if (in_h == 1 || out_h == 1) {
        for (int i = 0; i < out_h; ++i) {
            y0[i] = 0; y1[i] = 0; wy[i] = 0.0f;
        }
    } else {
        float scale = (float)((double)in_h / (double)out_h);
        for (int i = 0; i < out_h; ++i) {
            float prod = ((float)i + 0.5f) * scale;
            __asm__("" : "+x"(prod));
            float yc = prod - 0.5f;
            if (yc < 0.0f) yc = 0.0f;
            if (yc > (float)(in_h - 1)) yc = (float)(in_h - 1);
            int i0 = (int)floorf(yc);
            int i1 = (i0 + 1 < in_h) ? i0 + 1 : in_h - 1;
            y0[i] = i0; y1[i] = i1; wy[i] = yc - (float)i0;
        }
    }

    preprocess_rgb_ctx_t ctx = {
        .rgb = rgb,
        .out_nchw = out_nchw,
        .x0 = x0, .x1 = x1, .y0 = y0, .y1 = y1, .wx = wx, .wy = wy,
        .in_h = in_h, .in_w = in_w, .out_h = out_h, .out_w = out_w,
    };
    if (out_h >= 16 && (int64_t)out_h * out_w >= 8192) {
        lse_parallel_for((int64_t)out_h, 8, preprocess_rgb_worker, &ctx);
    } else {
        preprocess_rgb_worker(0, (int64_t)out_h, &ctx);
    }

    free(ibuf);
    free(fbuf);
    return 0;
}

typedef struct {
    const float *qkv;
    float *out;
    int64_t n_tok;
    int64_t n_heads;
    int64_t head_dim;
    float scale;
} fused_mha_ctx_t;

static inline void softmax_row_inplace_avx512(float *sc, int64_t n, float max_sc) {
    const __m512 vmax = _mm512_set1_ps(max_sc);
    __m512 vsum = _mm512_setzero_ps();
    int64_t j = 0;
    for (; j + 16 <= n; j += 16) {
        __m512 ve = exp_ps_avx512(_mm512_sub_ps(_mm512_loadu_ps(sc + j), vmax));
        _mm512_storeu_ps(sc + j, ve);
        vsum = _mm512_add_ps(vsum, ve);
    }
    if (j < n) {
        __mmask16 km = (__mmask16)((1u << (n - j)) - 1u);
        __m512 ve = exp_ps_avx512(_mm512_sub_ps(_mm512_maskz_loadu_ps(km, sc + j), vmax));
        _mm512_mask_storeu_ps(sc + j, km, ve);
        vsum = _mm512_mask_add_ps(vsum, km, vsum, ve);
    }
    float inv_sum = 1.0f / _mm512_reduce_add_ps(vsum);
    const __m512 vinv = _mm512_set1_ps(inv_sum);
    for (j = 0; j + 16 <= n; j += 16) {
        _mm512_storeu_ps(sc + j, _mm512_mul_ps(_mm512_loadu_ps(sc + j), vinv));
    }
    if (j < n) {
        __mmask16 km = (__mmask16)((1u << (n - j)) - 1u);
        _mm512_mask_storeu_ps(sc + j, km, _mm512_mul_ps(_mm512_maskz_loadu_ps(km, sc + j), vinv));
    }
}

static void fused_mha_qkv_worker(int64_t h_start, int64_t h_end, void *arg) {
    const fused_mha_ctx_t *ctx = (const fused_mha_ctx_t *)arg;
    const float *qkv = ctx->qkv;
    float *out = ctx->out;
    const int64_t n_tok = ctx->n_tok;
    const int64_t dh = ctx->head_dim;
    const int64_t d_total = ctx->n_heads * dh;
    const int64_t qkv_stride = 3 * d_total;
    const float scale = ctx->scale;

    const size_t kv_elems = (size_t)n_tok * (size_t)dh;
    const size_t sc_elems = (size_t)n_tok * 4;
    float *scratch = (float *)malloc((kv_elems * 2 + sc_elems) * sizeof(float));
    if (!scratch) {
        return;
    }
    float *k_h = scratch;
    float *v_h = k_h + kv_elems;
    float *sc0 = v_h + kv_elems;
    float *sc1 = sc0 + n_tok;
    float *sc2 = sc1 + n_tok;
    float *sc3 = sc2 + n_tok;

    for (int64_t h = h_start; h < h_end; ++h) {
        const int64_t h_off = h * dh;
        for (int64_t j = 0; j < n_tok; ++j) {
            const float *src_row = qkv + j * qkv_stride + h_off;
            memcpy(k_h + j * dh, src_row + d_total, (size_t)dh * sizeof(float));
            memcpy(v_h + j * dh, src_row + 2 * d_total, (size_t)dh * sizeof(float));
        }

        int64_t i = 0;
        if (dh == 64) {
            for (; i + 4 <= n_tok; i += 4) {
                const float *q0 = qkv + (i + 0) * qkv_stride + h_off;
                const float *q1 = qkv + (i + 1) * qkv_stride + h_off;
                const float *q2 = qkv + (i + 2) * qkv_stride + h_off;
                const float *q3 = qkv + (i + 3) * qkv_stride + h_off;

                __m512 q0a = _mm512_loadu_ps(q0 + 0),  q0b = _mm512_loadu_ps(q0 + 16), q0c = _mm512_loadu_ps(q0 + 32), q0d = _mm512_loadu_ps(q0 + 48);
                __m512 q1a = _mm512_loadu_ps(q1 + 0),  q1b = _mm512_loadu_ps(q1 + 16), q1c = _mm512_loadu_ps(q1 + 32), q1d = _mm512_loadu_ps(q1 + 48);
                __m512 q2a = _mm512_loadu_ps(q2 + 0),  q2b = _mm512_loadu_ps(q2 + 16), q2c = _mm512_loadu_ps(q2 + 32), q2d = _mm512_loadu_ps(q2 + 48);
                __m512 q3a = _mm512_loadu_ps(q3 + 0),  q3b = _mm512_loadu_ps(q3 + 16), q3c = _mm512_loadu_ps(q3 + 32), q3d = _mm512_loadu_ps(q3 + 48);

                float mx0 = -3.402823466e+38f, mx1 = -3.402823466e+38f, mx2 = -3.402823466e+38f, mx3 = -3.402823466e+38f;
                for (int64_t j = 0; j < n_tok; ++j) {
                    const float *kj = k_h + j * 64;
                    __m512 ka = _mm512_loadu_ps(kj + 0);
                    __m512 kb = _mm512_loadu_ps(kj + 16);
                    __m512 kc = _mm512_loadu_ps(kj + 32);
                    __m512 kd = _mm512_loadu_ps(kj + 48);

                    __m512 a0 = _mm512_fmadd_ps(q0d, kd, _mm512_fmadd_ps(q0c, kc, _mm512_fmadd_ps(q0b, kb, _mm512_mul_ps(q0a, ka))));
                    __m512 a1 = _mm512_fmadd_ps(q1d, kd, _mm512_fmadd_ps(q1c, kc, _mm512_fmadd_ps(q1b, kb, _mm512_mul_ps(q1a, ka))));
                    __m512 a2 = _mm512_fmadd_ps(q2d, kd, _mm512_fmadd_ps(q2c, kc, _mm512_fmadd_ps(q2b, kb, _mm512_mul_ps(q2a, ka))));
                    __m512 a3 = _mm512_fmadd_ps(q3d, kd, _mm512_fmadd_ps(q3c, kc, _mm512_fmadd_ps(q3b, kb, _mm512_mul_ps(q3a, ka))));

                    float s0 = _mm512_reduce_add_ps(a0) * scale;
                    float s1 = _mm512_reduce_add_ps(a1) * scale;
                    float s2 = _mm512_reduce_add_ps(a2) * scale;
                    float s3 = _mm512_reduce_add_ps(a3) * scale;
                    sc0[j] = s0; if (s0 > mx0) mx0 = s0;
                    sc1[j] = s1; if (s1 > mx1) mx1 = s1;
                    sc2[j] = s2; if (s2 > mx2) mx2 = s2;
                    sc3[j] = s3; if (s3 > mx3) mx3 = s3;
                }

                softmax_row_inplace_avx512(sc0, n_tok, mx0);
                softmax_row_inplace_avx512(sc1, n_tok, mx1);
                softmax_row_inplace_avx512(sc2, n_tok, mx2);
                softmax_row_inplace_avx512(sc3, n_tok, mx3);

                __m512 o0a = _mm512_setzero_ps(), o0b = _mm512_setzero_ps(), o0c = _mm512_setzero_ps(), o0d = _mm512_setzero_ps();
                __m512 o1a = _mm512_setzero_ps(), o1b = _mm512_setzero_ps(), o1c = _mm512_setzero_ps(), o1d = _mm512_setzero_ps();
                __m512 o2a = _mm512_setzero_ps(), o2b = _mm512_setzero_ps(), o2c = _mm512_setzero_ps(), o2d = _mm512_setzero_ps();
                __m512 o3a = _mm512_setzero_ps(), o3b = _mm512_setzero_ps(), o3c = _mm512_setzero_ps(), o3d = _mm512_setzero_ps();

                for (int64_t j = 0; j < n_tok; ++j) {
                    const float *vj = v_h + j * 64;
                    __m512 va = _mm512_loadu_ps(vj + 0);
                    __m512 vb = _mm512_loadu_ps(vj + 16);
                    __m512 vc = _mm512_loadu_ps(vj + 32);
                    __m512 vd = _mm512_loadu_ps(vj + 48);

                    __m512 w0 = _mm512_set1_ps(sc0[j]);
                    __m512 w1 = _mm512_set1_ps(sc1[j]);
                    __m512 w2 = _mm512_set1_ps(sc2[j]);
                    __m512 w3 = _mm512_set1_ps(sc3[j]);

                    o0a = _mm512_fmadd_ps(w0, va, o0a); o0b = _mm512_fmadd_ps(w0, vb, o0b); o0c = _mm512_fmadd_ps(w0, vc, o0c); o0d = _mm512_fmadd_ps(w0, vd, o0d);
                    o1a = _mm512_fmadd_ps(w1, va, o1a); o1b = _mm512_fmadd_ps(w1, vb, o1b); o1c = _mm512_fmadd_ps(w1, vc, o1c); o1d = _mm512_fmadd_ps(w1, vd, o1d);
                    o2a = _mm512_fmadd_ps(w2, va, o2a); o2b = _mm512_fmadd_ps(w2, vb, o2b); o2c = _mm512_fmadd_ps(w2, vc, o2c); o2d = _mm512_fmadd_ps(w2, vd, o2d);
                    o3a = _mm512_fmadd_ps(w3, va, o3a); o3b = _mm512_fmadd_ps(w3, vb, o3b); o3c = _mm512_fmadd_ps(w3, vc, o3c); o3d = _mm512_fmadd_ps(w3, vd, o3d);
                }

                float *dst0 = out + (i + 0) * d_total + h_off;
                float *dst1 = out + (i + 1) * d_total + h_off;
                float *dst2 = out + (i + 2) * d_total + h_off;
                float *dst3 = out + (i + 3) * d_total + h_off;
                _mm512_storeu_ps(dst0 + 0, o0a); _mm512_storeu_ps(dst0 + 16, o0b); _mm512_storeu_ps(dst0 + 32, o0c); _mm512_storeu_ps(dst0 + 48, o0d);
                _mm512_storeu_ps(dst1 + 0, o1a); _mm512_storeu_ps(dst1 + 16, o1b); _mm512_storeu_ps(dst1 + 32, o1c); _mm512_storeu_ps(dst1 + 48, o1d);
                _mm512_storeu_ps(dst2 + 0, o2a); _mm512_storeu_ps(dst2 + 16, o2b); _mm512_storeu_ps(dst2 + 32, o2c); _mm512_storeu_ps(dst2 + 48, o2d);
                _mm512_storeu_ps(dst3 + 0, o3a); _mm512_storeu_ps(dst3 + 16, o3b); _mm512_storeu_ps(dst3 + 32, o3c); _mm512_storeu_ps(dst3 + 48, o3d);
            }
        }

        for (; i < n_tok; ++i) {
            const float *qi = qkv + i * qkv_stride + h_off;
            float mx = -3.402823466e+38f;
            for (int64_t j = 0; j < n_tok; ++j) {
                const float *kj = k_h + j * dh;
                __m512 vacc = _mm512_setzero_ps();
                int64_t d = 0;
                for (; d + 16 <= dh; d += 16) {
                    vacc = _mm512_fmadd_ps(_mm512_loadu_ps(qi + d), _mm512_loadu_ps(kj + d), vacc);
                }
                if (d < dh) {
                    __mmask16 km = (__mmask16)((1u << (dh - d)) - 1u);
                    vacc = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(km, qi + d), _mm512_maskz_loadu_ps(km, kj + d), vacc);
                }
                float s = _mm512_reduce_add_ps(vacc) * scale;
                sc0[j] = s;
                if (s > mx) mx = s;
            }
            softmax_row_inplace_avx512(sc0, n_tok, mx);

            float *dst_i = out + i * d_total + h_off;
            int64_t d = 0;
            for (; d + 16 <= dh; d += 16) {
                __m512 oacc = _mm512_setzero_ps();
                for (int64_t j = 0; j < n_tok; ++j) {
                    oacc = _mm512_fmadd_ps(_mm512_set1_ps(sc0[j]), _mm512_loadu_ps(v_h + j * dh + d), oacc);
                }
                _mm512_storeu_ps(dst_i + d, oacc);
            }
            if (d < dh) {
                __mmask16 km = (__mmask16)((1u << (dh - d)) - 1u);
                __m512 oacc = _mm512_setzero_ps();
                for (int64_t j = 0; j < n_tok; ++j) {
                    oacc = _mm512_fmadd_ps(_mm512_set1_ps(sc0[j]), _mm512_maskz_loadu_ps(km, v_h + j * dh + d), oacc);
                }
                _mm512_mask_storeu_ps(dst_i + d, km, oacc);
            }
        }
    }

    free(scratch);
}

int fused_mha_qkv_f32(
    const float *qkv,
    float *out,
    int64_t n_tok,
    int64_t n_heads,
    int64_t head_dim,
    float scale
) {
    if (n_tok < 0 || n_heads <= 0 || head_dim <= 0) {
        return -1;
    }
    if (n_tok == 0) {
        return 0;
    }
    fused_mha_ctx_t ctx = {
        .qkv = qkv,
        .out = out,
        .n_tok = n_tok,
        .n_heads = n_heads,
        .head_dim = head_dim,
        .scale = scale,
    };
    if (n_heads >= 2 && n_tok * n_heads * head_dim >= 4096) {
        lse_parallel_for(n_heads, 1, fused_mha_qkv_worker, &ctx);
    } else {
        fused_mha_qkv_worker(0, n_heads, &ctx);
    }
    return 0;
}

typedef struct {
    const uint8_t *l0;
    const uint8_t *coc;
    const uint8_t *l1;
    const uint8_t *l2;
    const uint8_t *l3;
    uint8_t *dst_a;
    uint8_t *dst_b;
    int64_t num_pixels;
} pack_bokeh_soa32_ctx_t;

static inline void deinterleave3_32_u8(const uint8_t *src, uint8_t *r, uint8_t *g, uint8_t *b) {
    for (int k = 0; k < 32; ++k) {
        r[k] = src[k * 3 + 0];
        g[k] = src[k * 3 + 1];
        b[k] = src[k * 3 + 2];
    }
}

static void pack_bokeh_soa32_worker(int64_t c_start, int64_t c_end, void *arg) {
    const pack_bokeh_soa32_ctx_t *ctx = (const pack_bokeh_soa32_ctx_t *)arg;
    const uint8_t *l0 = ctx->l0;
    const uint8_t *coc = ctx->coc;
    const uint8_t *l1 = ctx->l1;
    const uint8_t *l2 = ctx->l2;
    const uint8_t *l3 = ctx->l3;
    uint8_t *dst_a = ctx->dst_a;
    uint8_t *dst_b = ctx->dst_b;

    for (int64_t c = c_start; c < c_end; ++c) {
        const int64_t px = c * 32;
        uint8_t *pa = dst_a + c * 128;
        uint8_t *pb = dst_b + c * 288;
        deinterleave3_32_u8(l0 + px * 3, pa + 0, pa + 32, pa + 64);
        memcpy(pa + 96, coc + px, 32);
        deinterleave3_32_u8(l1 + px * 3, pb + 0, pb + 32, pb + 64);
        deinterleave3_32_u8(l2 + px * 3, pb + 96, pb + 128, pb + 160);
        deinterleave3_32_u8(l3 + px * 3, pb + 192, pb + 224, pb + 256);
    }
}

int pack_bokeh_npu_inputs_soa32_u8(
    const uint8_t *l0,
    const uint8_t *coc_u8,
    const uint8_t *l1,
    const uint8_t *l2,
    const uint8_t *l3,
    uint8_t *dst_a,
    uint8_t *dst_b,
    int64_t num_pixels,
    int64_t padded_pixels
) {
    if (num_pixels < 0 || padded_pixels < num_pixels || (padded_pixels & 31) != 0) {
        return -1;
    }
    const int64_t full_chunks = num_pixels / 32;
    pack_bokeh_soa32_ctx_t ctx = {
        .l0 = l0,
        .coc = coc_u8,
        .l1 = l1,
        .l2 = l2,
        .l3 = l3,
        .dst_a = dst_a,
        .dst_b = dst_b,
        .num_pixels = num_pixels,
    };
    if (full_chunks >= 64) {
        lse_parallel_for(full_chunks, 16, pack_bokeh_soa32_worker, &ctx);
    } else if (full_chunks > 0) {
        pack_bokeh_soa32_worker(0, full_chunks, &ctx);
    }

    const int64_t total_chunks = padded_pixels / 32;
    const int64_t rem = num_pixels - full_chunks * 32;
    if (rem > 0) {
        uint8_t *pa = dst_a + full_chunks * 128;
        uint8_t *pb = dst_b + full_chunks * 288;
        memset(pa, 0, 128);
        memset(pb, 0, 288);
        const int64_t base = full_chunks * 32;
        for (int64_t k = 0; k < rem; ++k) {
            const int64_t p = base + k;
            pa[0 + k] = l0[p * 3 + 0];
            pa[32 + k] = l0[p * 3 + 1];
            pa[64 + k] = l0[p * 3 + 2];
            pa[96 + k] = coc_u8[p];
            pb[0 + k] = l1[p * 3 + 0];
            pb[32 + k] = l1[p * 3 + 1];
            pb[64 + k] = l1[p * 3 + 2];
            pb[96 + k] = l2[p * 3 + 0];
            pb[128 + k] = l2[p * 3 + 1];
            pb[160 + k] = l2[p * 3 + 2];
            pb[192 + k] = l3[p * 3 + 0];
            pb[224 + k] = l3[p * 3 + 1];
            pb[256 + k] = l3[p * 3 + 2];
        }
        if (total_chunks > full_chunks + 1) {
            const int64_t tail_c = total_chunks - (full_chunks + 1);
            memset(dst_a + (full_chunks + 1) * 128, 0, (size_t)tail_c * 128);
            memset(dst_b + (full_chunks + 1) * 288, 0, (size_t)tail_c * 288);
        }
    } else if (total_chunks > full_chunks) {
        const int64_t tail_c = total_chunks - full_chunks;
        memset(dst_a + full_chunks * 128, 0, (size_t)tail_c * 128);
        memset(dst_b + full_chunks * 288, 0, (size_t)tail_c * 288);
    }
    return 0;
}

typedef struct {
    const float *src_hw4;
    uint8_t *out;
    int in_h;
    int in_w;
    int out_h;
    int out_w;
    const int32_t *x0;
    const int32_t *x1;
    const int32_t *y0;
    const int32_t *y1;
    const float *wx;
    const float *wy;
} unpremul_resize_ctx_t;

static void unpremul_row_to_rgb_f32(
    const float *src_row4,
    float *dst_row3,
    const int32_t *x0,
    const int32_t *x1,
    const float *wx,
    int out_w
) {
    for (int ox = 0; ox < out_w; ++ox) {
        const float *p0 = src_row4 + (size_t)x0[ox] * 4;
        const float *p1 = src_row4 + (size_t)x1[ox] * 4;
        const float inv0 = 1.0f / (p0[3] > 1e-5f ? p0[3] : 1e-5f);
        const float inv1 = 1.0f / (p1[3] > 1e-5f ? p1[3] : 1e-5f);
        const float r0 = p0[0] * inv0, g0 = p0[1] * inv0, b0 = p0[2] * inv0;
        const float r1 = p1[0] * inv1, g1 = p1[1] * inv1, b1 = p1[2] * inv1;
        const float w = wx[ox];
        dst_row3[ox * 3 + 0] = r0 + w * (r1 - r0);
        dst_row3[ox * 3 + 1] = g0 + w * (g1 - g0);
        dst_row3[ox * 3 + 2] = b0 + w * (b1 - b0);
    }
}

static void unpremul_resize_worker(int64_t oy_start, int64_t oy_end, void *arg) {
    const unpremul_resize_ctx_t *ctx = (const unpremul_resize_ctx_t *)arg;
    const float *src = ctx->src_hw4;
    uint8_t *out = ctx->out;
    const size_t in_stride = (size_t)ctx->in_w * 4;
    const int out_w = ctx->out_w;
    const int out_w3 = out_w * 3;
    const int32_t *x0 = ctx->x0;
    const int32_t *x1 = ctx->x1;
    const int32_t *y0 = ctx->y0;
    const int32_t *y1 = ctx->y1;
    const float *wx = ctx->wx;
    const float *wy = ctx->wy;

    float *row_a = (float *)malloc((size_t)out_w3 * 2 * sizeof(float));
    if (!row_a) {
        return;
    }
    float *row_b = row_a + out_w3;
    int cached_id[2] = {-1, -1};
    float *slots[2] = {row_a, row_b};
    const __m512 vzero = _mm512_setzero_ps();
    const __m512 v255 = _mm512_set1_ps(255.0f);

    for (int oy = (int)oy_start; oy < (int)oy_end; ++oy) {
        const int r0 = y0[oy];
        const int r1 = y1[oy];
        const float wyv = wy[oy];

        float *p0;
        if (r0 == cached_id[0]) {
            p0 = slots[0];
        } else if (r0 == cached_id[1]) {
            p0 = slots[1];
        } else {
            int slot = (cached_id[0] == r1) ? 1 : 0;
            unpremul_row_to_rgb_f32(src + (size_t)r0 * in_stride, slots[slot], x0, x1, wx, out_w);
            cached_id[slot] = r0;
            p0 = slots[slot];
        }

        uint8_t *out_row = out + (size_t)oy * (size_t)out_w3;
        if (r1 == r0 || wyv == 0.0f) {
            int k = 0;
            for (; k + 16 <= out_w3; k += 16) {
                __m512 vf = _mm512_min_ps(_mm512_max_ps(_mm512_loadu_ps(p0 + k), vzero), v255);
                __m512i vi = _mm512_cvtps_epi32(_mm512_roundscale_ps(vf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                _mm_storeu_si128((__m128i *)(out_row + k), _mm512_cvtusepi32_epi8(vi));
            }
            for (; k < out_w3; ++k) {
                float v = p0[k];
                if (v < 0.0f) v = 0.0f;
                if (v > 255.0f) v = 255.0f;
                out_row[k] = (uint8_t)lrintf(v);
            }
            continue;
        }

        float *p1;
        if (r1 == cached_id[0]) {
            p1 = slots[0];
        } else if (r1 == cached_id[1]) {
            p1 = slots[1];
        } else {
            int slot = (p0 == slots[0]) ? 1 : 0;
            unpremul_row_to_rgb_f32(src + (size_t)r1 * in_stride, slots[slot], x0, x1, wx, out_w);
            cached_id[slot] = r1;
            p1 = slots[slot];
        }

        const __m512 vwy = _mm512_set1_ps(wyv);
        int k = 0;
        for (; k + 16 <= out_w3; k += 16) {
            __m512 va = _mm512_loadu_ps(p0 + k);
            __m512 vb = _mm512_loadu_ps(p1 + k);
            __m512 vf = _mm512_fmadd_ps(vwy, _mm512_sub_ps(vb, va), va);
            vf = _mm512_min_ps(_mm512_max_ps(vf, vzero), v255);
            __m512i vi = _mm512_cvtps_epi32(_mm512_roundscale_ps(vf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            _mm_storeu_si128((__m128i *)(out_row + k), _mm512_cvtusepi32_epi8(vi));
        }
        for (; k < out_w3; ++k) {
            float v = p0[k] + wyv * (p1[k] - p0[k]);
            if (v < 0.0f) v = 0.0f;
            if (v > 255.0f) v = 255.0f;
            out_row[k] = (uint8_t)lrintf(v);
        }
    }
    free(row_a);
}

int unpremul_resize_hw4_to_rgb_u8_f32(
    const float *src_hw4,
    uint8_t *out,
    int in_h,
    int in_w,
    int out_h,
    int out_w
) {
    if (in_h <= 0 || in_w <= 0 || out_h <= 0 || out_w <= 0) {
        return -1;
    }
    int32_t *ibuf = (int32_t *)malloc(((size_t)out_w * 2 + (size_t)out_h * 2) * sizeof(int32_t));
    float *fbuf = (float *)malloc(((size_t)out_w + (size_t)out_h) * sizeof(float));
    if (!ibuf || !fbuf) {
        free(ibuf);
        free(fbuf);
        return -1;
    }
    int32_t *x0 = ibuf;
    int32_t *x1 = x0 + out_w;
    int32_t *y0 = x1 + out_w;
    int32_t *y1 = y0 + out_h;
    float *wx = fbuf;
    float *wy = wx + out_w;

    if (in_w == 1 || out_w == 1) {
        for (int i = 0; i < out_w; ++i) { x0[i] = 0; x1[i] = 0; wx[i] = 0.0f; }
    } else {
        float scale = (float)((double)in_w / (double)out_w);
        for (int i = 0; i < out_w; ++i) {
            float prod = ((float)i + 0.5f) * scale;
            __asm__("" : "+x"(prod));
            float xc = prod - 0.5f;
            if (xc < 0.0f) xc = 0.0f;
            if (xc > (float)(in_w - 1)) xc = (float)(in_w - 1);
            int i0 = (int)floorf(xc);
            int i1 = (i0 + 1 < in_w) ? i0 + 1 : in_w - 1;
            x0[i] = i0; x1[i] = i1; wx[i] = xc - (float)i0;
        }
    }
    if (in_h == 1 || out_h == 1) {
        for (int i = 0; i < out_h; ++i) { y0[i] = 0; y1[i] = 0; wy[i] = 0.0f; }
    } else {
        float scale = (float)((double)in_h / (double)out_h);
        for (int i = 0; i < out_h; ++i) {
            float prod = ((float)i + 0.5f) * scale;
            __asm__("" : "+x"(prod));
            float yc = prod - 0.5f;
            if (yc < 0.0f) yc = 0.0f;
            if (yc > (float)(in_h - 1)) yc = (float)(in_h - 1);
            int i0 = (int)floorf(yc);
            int i1 = (i0 + 1 < in_h) ? i0 + 1 : in_h - 1;
            y0[i] = i0; y1[i] = i1; wy[i] = yc - (float)i0;
        }
    }

    unpremul_resize_ctx_t ctx = {
        .src_hw4 = src_hw4,
        .out = out,
        .in_h = in_h,
        .in_w = in_w,
        .out_h = out_h,
        .out_w = out_w,
        .x0 = x0,
        .x1 = x1,
        .y0 = y0,
        .y1 = y1,
        .wx = wx,
        .wy = wy,
    };
    if (out_h >= 16 && (int64_t)out_h * out_w >= 8192) {
        lse_parallel_for((int64_t)out_h, 8, unpremul_resize_worker, &ctx);
    } else {
        unpremul_resize_worker(0, (int64_t)out_h, &ctx);
    }
    free(ibuf);
    free(fbuf);
    return 0;
}

typedef struct {
    const uint8_t *src;
    uint8_t *dst;
    int h;
    int pair_w;
    int is_uyvy;
} yuv422_conv_ctx_t;

static void yuv422_to_ch_worker(int64_t y_start, int64_t y_end, void *arg) {
    const yuv422_conv_ctx_t *ctx = (const yuv422_conv_ctx_t *)arg;
    const int pw = ctx->pair_w;
    const size_t row_in_bytes = (size_t)pw * 4;
    const size_t row_out_bytes = (size_t)pw * 6;
    const int y0_off = ctx->is_uyvy ? 1 : 0;
    const int u_off = ctx->is_uyvy ? 0 : 1;
    const int y1_off = ctx->is_uyvy ? 3 : 2;
    const int v_off = ctx->is_uyvy ? 2 : 3;

    for (int64_t y = y_start; y < y_end; ++y) {
        const uint8_t *in_row = ctx->src + (size_t)y * row_in_bytes;
        uint8_t *out_row = ctx->dst + (size_t)y * row_out_bytes;
        for (int p = 0; p < pw; ++p) {
            const uint8_t *ip = in_row + (size_t)p * 4;
            uint8_t *op = out_row + (size_t)p * 6;
            const uint8_t u = ip[u_off];
            const uint8_t v = ip[v_off];
            op[0] = ip[y0_off];
            op[1] = u;
            op[2] = v;
            op[3] = ip[y1_off];
            op[4] = u;
            op[5] = v;
        }
    }
}

void yuv422_to_channels_u8(const uint8_t *packed, uint8_t *out, int h, int pair_w, int is_uyvy) {
    if (h <= 0 || pair_w <= 0) return;
    yuv422_conv_ctx_t ctx = {packed, out, h, pair_w, is_uyvy};
    if (h >= 16 && (int64_t)h * pair_w >= 8192) {
        lse_parallel_for((int64_t)h, 16, yuv422_to_ch_worker, &ctx);
    } else {
        yuv422_to_ch_worker(0, (int64_t)h, &ctx);
    }
}

static void ch_to_yuv422_worker(int64_t y_start, int64_t y_end, void *arg) {
    const yuv422_conv_ctx_t *ctx = (const yuv422_conv_ctx_t *)arg;
    const int pw = ctx->pair_w;
    const size_t row_in_bytes = (size_t)pw * 6;
    const size_t row_out_bytes = (size_t)pw * 4;
    const int y0_off = ctx->is_uyvy ? 1 : 0;
    const int u_off = ctx->is_uyvy ? 0 : 1;
    const int y1_off = ctx->is_uyvy ? 3 : 2;
    const int v_off = ctx->is_uyvy ? 2 : 3;

    for (int64_t y = y_start; y < y_end; ++y) {
        const uint8_t *in_row = ctx->src + (size_t)y * row_in_bytes;
        uint8_t *out_row = ctx->dst + (size_t)y * row_out_bytes;
        for (int p = 0; p < pw; ++p) {
            const uint8_t *ip = in_row + (size_t)p * 6;
            uint8_t *op = out_row + (size_t)p * 4;
            op[y0_off] = ip[0];
            op[u_off] = (uint8_t)(((unsigned)ip[1] + (unsigned)ip[4]) >> 1);
            op[y1_off] = ip[3];
            op[v_off] = (uint8_t)(((unsigned)ip[2] + (unsigned)ip[5]) >> 1);
        }
    }
}

void channels_to_yuv422_u8(const uint8_t *ch, uint8_t *out, int h, int pair_w, int is_uyvy) {
    if (h <= 0 || pair_w <= 0) return;
    yuv422_conv_ctx_t ctx = {ch, out, h, pair_w, is_uyvy};
    if (h >= 16 && (int64_t)h * pair_w >= 8192) {
        lse_parallel_for((int64_t)h, 16, ch_to_yuv422_worker, &ctx);
    } else {
        ch_to_yuv422_worker(0, (int64_t)h, &ctx);
    }
}

static void _build_crop_lut_q8(
    int origin,
    int crop_len,
    int max_idx,
    int out_len,
    int32_t *i0_out,
    int32_t *i1_out,
    int32_t *w_q8_out
) {
    if (crop_len <= 1 || out_len <= 1) {
        int32_t c = (origin < 0) ? 0 : (origin > max_idx ? max_idx : origin);
        for (int i = 0; i < out_len; ++i) {
            i0_out[i] = c;
            i1_out[i] = c;
            w_q8_out[i] = 0;
        }
        return;
    }
    const double scale = (double)crop_len / (double)out_len;
    const double hi = (double)(crop_len - 1);
    for (int i = 0; i < out_len; ++i) {
        double c = ((double)i + 0.5) * scale - 0.5;
        if (c < 0.0) c = 0.0;
        if (c > hi) c = hi;
        int i0 = (int)floor(c);
        int i1 = (i0 + 1 < crop_len) ? (i0 + 1) : (crop_len - 1);
        int32_t wq = (int32_t)llround((c - (double)i0) * 256.0);
        if (wq < 0) wq = 0;
        if (wq > 256) wq = 256;
        int32_t a0 = origin + i0;
        int32_t a1 = origin + i1;
        if (a0 < 0) a0 = 0; else if (a0 > max_idx) a0 = max_idx;
        if (a1 < 0) a1 = 0; else if (a1 > max_idx) a1 = max_idx;
        i0_out[i] = a0;
        i1_out[i] = a1;
        w_q8_out[i] = wq;
    }
}

typedef struct {
    const uint8_t *src;
    uint8_t *dst;
    int in_stride;
    int out_h;
    int out_w;
    const int32_t *x0;
    const int32_t *x1;
    const int32_t *wx;
    const int32_t *y0;
    const int32_t *y1;
    const int32_t *wy;
} crop_resize_rgb_ctx_t;

static void crop_resize_rgb_worker(int64_t oy_start, int64_t oy_end, void *arg) {
    const crop_resize_rgb_ctx_t *ctx = (const crop_resize_rgb_ctx_t *)arg;
    const int ow = ctx->out_w;
    for (int64_t oy = oy_start; oy < oy_end; ++oy) {
        const uint8_t *r0 = ctx->src + (size_t)ctx->y0[oy] * (size_t)ctx->in_stride;
        const uint8_t *r1 = ctx->src + (size_t)ctx->y1[oy] * (size_t)ctx->in_stride;
        uint8_t *out_row = ctx->dst + (size_t)oy * (size_t)ow * 3;
        const int32_t wy1 = ctx->wy[oy];
        const int32_t wy0 = 256 - wy1;

        for (int ox = 0; ox < ow; ++ox) {
            const int32_t ix0 = ctx->x0[ox] * 3;
            const int32_t ix1 = ctx->x1[ox] * 3;
            const int32_t wx1 = ctx->wx[ox];
            const int32_t wx0 = 256 - wx1;
            const uint8_t *p00 = r0 + ix0;
            const uint8_t *p01 = r0 + ix1;
            const uint8_t *p10 = r1 + ix0;
            const uint8_t *p11 = r1 + ix1;
            uint8_t *dp = out_row + (size_t)ox * 3;
            for (int c = 0; c < 3; ++c) {
                const int32_t top = (int32_t)p00[c] * wx0 + (int32_t)p01[c] * wx1;
                const int32_t bot = (int32_t)p10[c] * wx0 + (int32_t)p11[c] * wx1;
                const int32_t val = (top * wy0 + bot * wy1 + 32768) >> 16;
                dp[c] = (uint8_t)(val < 0 ? 0 : (val > 255 ? 255 : val));
            }
        }
    }
}

int crop_bilinear_resize_rgb_u8(
    const uint8_t *src,
    uint8_t *dst,
    int in_h,
    int in_w,
    int x0,
    int y0,
    int crop_w,
    int crop_h,
    int out_h,
    int out_w
) {
    if (in_h <= 0 || in_w <= 0 || crop_w <= 0 || crop_h <= 0 || out_h <= 0 || out_w <= 0) {
        return -1;
    }
    int32_t *lut = (int32_t *)malloc(((size_t)out_w * 3 + (size_t)out_h * 3) * sizeof(int32_t));
    if (!lut) return -1;
    int32_t *lx0 = lut;
    int32_t *lx1 = lx0 + out_w;
    int32_t *lwx = lx1 + out_w;
    int32_t *ly0 = lwx + out_w;
    int32_t *ly1 = ly0 + out_h;
    int32_t *lwy = ly1 + out_h;

    _build_crop_lut_q8(x0, crop_w, in_w - 1, out_w, lx0, lx1, lwx);
    _build_crop_lut_q8(y0, crop_h, in_h - 1, out_h, ly0, ly1, lwy);

    crop_resize_rgb_ctx_t ctx = {src, dst, in_w * 3, out_h, out_w, lx0, lx1, lwx, ly0, ly1, lwy};
    if (out_h >= 16 && (int64_t)out_h * out_w >= 8192) {
        lse_parallel_for((int64_t)out_h, 16, crop_resize_rgb_worker, &ctx);
    } else {
        crop_resize_rgb_worker(0, (int64_t)out_h, &ctx);
    }
    free(lut);
    return 0;
}

typedef struct {
    const uint8_t *src;
    uint8_t *dst;
    int in_stride;
    int out_h;
    int out_pair_w;
    int is_uyvy;
    const int32_t *x0;
    const int32_t *x1;
    const int32_t *wx;
    const int32_t *cx0;
    const int32_t *cx1;
    const int32_t *cwx;
    const int32_t *y0;
    const int32_t *y1;
    const int32_t *wy;
} crop_resize_yuyv_ctx_t;

static void crop_resize_yuyv_worker(int64_t oy_start, int64_t oy_end, void *arg) {
    const crop_resize_yuyv_ctx_t *ctx = (const crop_resize_yuyv_ctx_t *)arg;
    const int opw = ctx->out_pair_w;
    const int y0_off = ctx->is_uyvy ? 1 : 0;
    const int u_off = ctx->is_uyvy ? 0 : 1;
    const int y1_off = ctx->is_uyvy ? 3 : 2;
    const int v_off = ctx->is_uyvy ? 2 : 3;
    const int lum_base = ctx->is_uyvy ? 1 : 0;

    for (int64_t oy = oy_start; oy < oy_end; ++oy) {
        const uint8_t *r0 = ctx->src + (size_t)ctx->y0[oy] * (size_t)ctx->in_stride;
        const uint8_t *r1 = ctx->src + (size_t)ctx->y1[oy] * (size_t)ctx->in_stride;
        uint8_t *out_row = ctx->dst + (size_t)oy * (size_t)opw * 4;
        const int32_t wy1 = ctx->wy[oy];
        const int32_t wy0 = 256 - wy1;

        for (int p = 0; p < opw; ++p) {
            const int ox0 = 2 * p;
            const int ox1 = ox0 + 1;

            const int32_t l_ix0 = (ctx->x0[ox0] >> 1) * 4 + (ctx->x0[ox0] & 1) * 2 + lum_base;
            const int32_t l_ix1 = (ctx->x1[ox0] >> 1) * 4 + (ctx->x1[ox0] & 1) * 2 + lum_base;
            const int32_t l_wx1 = ctx->wx[ox0];
            const int32_t l_wx0 = 256 - l_wx1;
            const int32_t yl_top = (int32_t)r0[l_ix0] * l_wx0 + (int32_t)r0[l_ix1] * l_wx1;
            const int32_t yl_bot = (int32_t)r1[l_ix0] * l_wx0 + (int32_t)r1[l_ix1] * l_wx1;
            const int32_t yl = (yl_top * wy0 + yl_bot * wy1 + 32768) >> 16;

            const int32_t r_ix0 = (ctx->x0[ox1] >> 1) * 4 + (ctx->x0[ox1] & 1) * 2 + lum_base;
            const int32_t r_ix1 = (ctx->x1[ox1] >> 1) * 4 + (ctx->x1[ox1] & 1) * 2 + lum_base;
            const int32_t r_wx1 = ctx->wx[ox1];
            const int32_t r_wx0 = 256 - r_wx1;
            const int32_t yr_top = (int32_t)r0[r_ix0] * r_wx0 + (int32_t)r0[r_ix1] * r_wx1;
            const int32_t yr_bot = (int32_t)r1[r_ix0] * r_wx0 + (int32_t)r1[r_ix1] * r_wx1;
            const int32_t yr = (yr_top * wy0 + yr_bot * wy1 + 32768) >> 16;

            const int32_t cp0 = ctx->cx0[p] * 4;
            const int32_t cp1 = ctx->cx1[p] * 4;
            const int32_t cwx1 = ctx->cwx[p];
            const int32_t cwx0 = 256 - cwx1;
            const int32_t u_top = (int32_t)r0[cp0 + u_off] * cwx0 + (int32_t)r0[cp1 + u_off] * cwx1;
            const int32_t u_bot = (int32_t)r1[cp0 + u_off] * cwx0 + (int32_t)r1[cp1 + u_off] * cwx1;
            const int32_t u_val = (u_top * wy0 + u_bot * wy1 + 32768) >> 16;

            const int32_t v_top = (int32_t)r0[cp0 + v_off] * cwx0 + (int32_t)r0[cp1 + v_off] * cwx1;
            const int32_t v_bot = (int32_t)r1[cp0 + v_off] * cwx0 + (int32_t)r1[cp1 + v_off] * cwx1;
            const int32_t v_val = (v_top * wy0 + v_bot * wy1 + 32768) >> 16;

            uint8_t *op = out_row + (size_t)p * 4;
            op[y0_off] = (uint8_t)(yl < 0 ? 0 : (yl > 255 ? 255 : yl));
            op[u_off] = (uint8_t)(u_val < 0 ? 0 : (u_val > 255 ? 255 : u_val));
            op[y1_off] = (uint8_t)(yr < 0 ? 0 : (yr > 255 ? 255 : yr));
            op[v_off] = (uint8_t)(v_val < 0 ? 0 : (v_val > 255 ? 255 : v_val));
        }
    }
}

int crop_bilinear_resize_yuyv_u8(
    const uint8_t *src,
    uint8_t *dst,
    int in_h,
    int in_pair_w,
    int x0,
    int y0,
    int crop_w,
    int crop_h,
    int out_h,
    int out_pair_w,
    int is_uyvy
) {
    if (in_h <= 0 || in_pair_w <= 0 || crop_w <= 0 || crop_h <= 0 || out_h <= 0 || out_pair_w <= 0) {
        return -1;
    }
    const int in_w = in_pair_w * 2;
    const int out_w = out_pair_w * 2;
    int32_t *lut = (int32_t *)malloc(((size_t)out_w * 3 + (size_t)out_pair_w * 3 + (size_t)out_h * 3) * sizeof(int32_t));
    if (!lut) return -1;
    int32_t *lx0 = lut;
    int32_t *lx1 = lx0 + out_w;
    int32_t *lwx = lx1 + out_w;
    int32_t *cx0 = lwx + out_w;
    int32_t *cx1 = cx0 + out_pair_w;
    int32_t *cwx = cx1 + out_pair_w;
    int32_t *ly0 = cwx + out_pair_w;
    int32_t *ly1 = ly0 + out_h;
    int32_t *lwy = ly1 + out_h;

    _build_crop_lut_q8(x0, crop_w, in_w - 1, out_w, lx0, lx1, lwx);
    _build_crop_lut_q8(x0 / 2, (crop_w + 1) / 2, in_pair_w - 1, out_pair_w, cx0, cx1, cwx);
    _build_crop_lut_q8(y0, crop_h, in_h - 1, out_h, ly0, ly1, lwy);

    crop_resize_yuyv_ctx_t ctx = {
        src, dst, in_pair_w * 4, out_h, out_pair_w, is_uyvy, lx0, lx1, lwx, cx0, cx1, cwx, ly0, ly1, lwy
    };
    if (out_h >= 16 && (int64_t)out_h * out_pair_w >= 4096) {
        lse_parallel_for((int64_t)out_h, 16, crop_resize_yuyv_worker, &ctx);
    } else {
        crop_resize_yuyv_worker(0, (int64_t)out_h, &ctx);
    }
    free(lut);
    return 0;
}
