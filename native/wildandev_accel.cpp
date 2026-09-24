#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <thread>
#include <vector>

extern "C" {

// C = A * B + bias (AVX-512 standard GEMM)
// A: (m x k), B: (k x n), C: (m x n)
void wildandev_gemm_f32_slice(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ bias,
    float* __restrict__ c,
    size_t row_start, size_t row_end,
    size_t k, size_t n)
{
    for (size_t i = row_start; i < row_end; ++i) {
        const float* a_row = a + i * k;
        float* c_row = c + i * n;

        if (bias) {
            size_t j = 0;
            for (; j + 16 <= n; j += 16) {
                _mm512_storeu_ps(c_row + j, _mm512_loadu_ps(bias + j));
            }
            for (; j < n; ++j) c_row[j] = bias[j];
        } else {
            std::memset(c_row, 0, n * sizeof(float));
        }

        for (size_t p = 0; p < k; ++p) {
            float aval = a_row[p];
            __m512 va = _mm512_set1_ps(aval);
            const float* b_row = b + p * n;
            size_t j = 0;
            for (; j + 16 <= n; j += 16) {
                __m512 vb = _mm512_loadu_ps(b_row + j);
                __m512 vc = _mm512_loadu_ps(c_row + j);
                vc = _mm512_fmadd_ps(va, vb, vc);
                _mm512_storeu_ps(c_row + j, vc);
            }
            for (; j < n; ++j) {
                c_row[j] += aval * b_row[j];
            }
        }
    }
}

void wildandev_gemm_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ bias,
    float* __restrict__ c,
    size_t m, size_t k, size_t n)
{
    wildandev_gemm_f32_slice(a, b, bias, c, 0, m, k, n);
}

// Multithreaded GEMM: uses 4 cores for big matrices
void wildandev_gemm_f32_mt(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ bias,
    float* __restrict__ c,
    size_t m, size_t k, size_t n,
    size_t num_threads)
{
    if (m < 8 || num_threads <= 1) {
        wildandev_gemm_f32_slice(a, b, bias, c, 0, m, k, n);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    size_t rows_per_th = (m + num_threads - 1) / num_threads;
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * rows_per_th;
        if (start >= m) break;
        size_t end = std::min(start + rows_per_th, m);
        workers.emplace_back(wildandev_gemm_f32_slice, a, b, bias, c, start, end, k, n);
    }
    for (auto& w : workers) {
        w.join();
    }
}

// F32 Transposed B
void wildandev_gemm_f32_tn(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ bias,
    float* __restrict__ c,
    size_t m, size_t k, size_t n)
{
    for (size_t i = 0; i < m; ++i) {
        const float* a_row = a + i * k;
        float* c_row = c + i * n;
        for (size_t j = 0; j < n; ++j) {
            const float* b_row = b + j * k;
            __m512 acc512 = _mm512_setzero_ps();
            size_t p = 0;
            for (; p + 16 <= k; p += 16) {
                __m512 va = _mm512_loadu_ps(a_row + p);
                __m512 vb = _mm512_loadu_ps(b_row + p);
                acc512 = _mm512_fmadd_ps(va, vb, acc512);
            }
            float acc = _mm512_reduce_add_ps(acc512);
            for (; p < k; ++p) acc += a_row[p] * b_row[p];
            if (bias) acc += bias[j];
            c_row[j] = acc;
        }
    }
}

// INT8 GEMM with AVX-512 VNNI/BW
void wildandev_gemm_i8(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,   // (n x k)
    float* __restrict__ c,
    float scale_a,
    float scale_b,
    const float* __restrict__ bias,
    size_t m, size_t k, size_t n)
{
    float total_scale = scale_a * scale_b;
    for (size_t i = 0; i < m; ++i) {
        const int8_t* a_row = a + i * k;
        float* c_row = c + i * n;
        for (size_t j = 0; j < n; ++j) {
            const int8_t* b_row = b + j * k;
            int32_t acc = 0;
            size_t p = 0;
            for (; p + 32 <= k; p += 32) {
                __m256i va = _mm256_loadu_si256((const __m256i*)(a_row + p));
                __m256i vb = _mm256_loadu_si256((const __m256i*)(b_row + p));
                __m512i va16 = _mm512_cvtepi8_epi16(va);
                __m512i vb16 = _mm512_cvtepi8_epi16(vb);
                __m512i prod = _mm512_madd_epi16(va16, vb16);
                acc += (int32_t)_mm512_reduce_add_epi32(prod);
            }
            for (; p < k; ++p) acc += (int32_t)a_row[p] * (int32_t)b_row[p];
            float val = (float)acc * total_scale;
            if (bias) val += bias[j];
            c_row[j] = val;
        }
    }
}

// In-place RMSNorm via AVX-512
void wildandev_rmsnorm(
    float* __restrict__ x,
    const float* __restrict__ g,
    size_t n,
    float eps)
{
    __m512 sum512 = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        sum512 = _mm512_fmadd_ps(v, v, sum512);
    }
    float ss = _mm512_reduce_add_ps(sum512);
    for (; i < n; ++i) ss += x[i] * x[i];

    float rms = 1.0f / std::sqrt(ss / (float)n + eps);
    __m512 vrms = _mm512_set1_ps(rms);

    i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 vg = _mm512_loadu_ps(g + i);
        __m512 out = _mm512_mul_ps(_mm512_mul_ps(vx, vrms), vg);
        _mm512_storeu_ps(x + i, out);
    }
    for (; i < n; ++i) {
        x[i] = x[i] * rms * g[i];
    }
}

// In-place SwiGLU: gate = silu(gate) * up via AVX-512
void wildandev_swiglu(
    float* __restrict__ gate,
    const float* __restrict__ up,
    size_t len)
{
    for (size_t i = 0; i < len; ++i) {
        float g = gate[i];
        float silu = g / (1.0f + std::exp(-g));
        gate[i] = silu * up[i];
    }
}

// Backward-compatible symbol
void wildandev_elis_gemm_forward(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ bias,
    float* __restrict__ c,
    size_t m, size_t k, size_t n)
{
    wildandev_gemm_f32(a, b, bias, c, m, k, n);
}

} // extern C
