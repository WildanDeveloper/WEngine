#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <thread>
#include <vector>

extern "C" {


// GEMM kernels


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

// dW += X^T @ dY   (accumulate outer products)
// X: (m x k), dY: (m x n), dW: (k x n)
void wildandev_gemm_acc_atb(
    const float* __restrict__ x,
    const float* __restrict__ dy,
    float* __restrict__ dw,
    size_t m, size_t k, size_t n)
{
    for (size_t i = 0; i < m; ++i) {
        const float* x_row = x + i * k;
        const float* dy_row = dy + i * n;
        for (size_t p = 0; p < k; ++p) {
            float xv = x_row[p];
            if (xv == 0.0f) continue;
            __m512 vx = _mm512_set1_ps(xv);
            float* dw_row = dw + p * n;
            size_t j = 0;
            for (; j + 16 <= n; j += 16) {
                __m512 acc = _mm512_loadu_ps(dw_row + j);
                acc = _mm512_fmadd_ps(vx, _mm512_loadu_ps(dy_row + j), acc);
                _mm512_storeu_ps(dw_row + j, acc);
            }
            for (; j < n; ++j) {
                dw_row[j] += xv * dy_row[j];
            }
        }
    }
}


// Fused activations backward


// SwiGLU backward: given gate pre-act g, up u, and dH:
//   dg = dH * u * silu'(g),  du = dH * silu(g)
// silu(g) = g * sigmoid(g); silu'(g) = sigmoid(g) * (1 + g*(1-sigmoid(g)))
void wildandev_swiglu_bwd(
    const float* __restrict__ g,
    const float* __restrict__ u,
    const float* __restrict__ dh,
    float* __restrict__ dg,
    float* __restrict__ du,
    size_t len)
{
    size_t i = 0;
    for (; i + 16 <= len; i += 16) {
        __m512 vg = _mm512_loadu_ps(g + i);
        __m512 vu = _mm512_loadu_ps(u + i);
        __m512 vd = _mm512_loadu_ps(dh + i);

        // sigmoid(g) = 1 / (1 + exp(-g)); AVX-512 exp via exponential identity:
        // exp(-x) = 2^(-x * log2(e)); use ex2.approx via _mm512_exp? Not available
        // in base AVX-512F -> compute scalar-free via polynomial-free approach:
        // use fast vectorized exp through bit manipulation of 2^k * poly.
        __m512 neg = _mm512_sub_ps(_mm512_setzero_ps(), vg);
        // exp via SVML-style: r = round(n), frac = x - r*ln2
        __m512 log2e = _mm512_set1_ps(1.4426950408889634f);
        __m512 t = _mm512_mul_ps(neg, log2e);
        __m512 rn = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m512 frac = _mm512_sub_ps(t, rn);
        // poly approx of 2^frac on [-0.5, 0.5]
        __m512 p = _mm512_set1_ps(1.0f);
        p = _mm512_fmadd_ps(p, frac, _mm512_set1_ps(0.6931472f));
        p = _mm512_fmadd_ps(p, frac, _mm512_set1_ps(0.2402265f));
        p = _mm512_fmadd_ps(p, frac, _mm512_set1_ps(0.0555041f));
        p = _mm512_fmadd_ps(p, frac, _mm512_set1_ps(0.0096181f));
        p = _mm512_fmadd_ps(p, frac, _mm512_set1_ps(0.0013333f));
        p = _mm512_fmadd_ps(p, frac, _mm512_set1_ps(0.0001540f));
        p = _mm512_fmadd_ps(p, frac, _mm512_set1_ps(0.0000152f));
        // scale by 2^rn via int bits
        __m512i irn = _mm512_cvtps_epi32(rn);
        __m512 scale = _mm512_castsi512_ps(_mm512_slli_epi32(irn, 23));
        __m512 expp = _mm512_mul_ps(p, scale);
        __m512 sig = _mm512_div_ps(_mm512_set1_ps(1.0f), _mm512_add_ps(_mm512_set1_ps(1.0f), expp));

        // silu(g) = g * sig
        __m512 silu = _mm512_mul_ps(vg, sig);

        // silu'(g) = sig * (1 + g*(1-sig))
        __m512 silu_p = _mm512_mul_ps(sig,
            _mm512_add_ps(_mm512_set1_ps(1.0f),
                _mm512_mul_ps(vg, _mm512_sub_ps(_mm512_set1_ps(1.0f), sig))));

        _mm512_storeu_ps(dg + i, _mm512_mul_ps(vd, _mm512_mul_ps(vu, silu_p)));
        _mm512_storeu_ps(du + i, _mm512_mul_ps(vd, silu));
    }
    for (; i < len; ++i) {
        float gv = g[i];
        float sig = 1.0f / (1.0f + std::exp(-gv));
        float silu = gv * sig;
        float silu_p = sig * (1.0f + gv * (1.0f - sig));
        dg[i] = dh[i] * u[i] * silu_p;
        du[i] = dh[i] * silu;
    }
}

// RMSNorm backward (single row, in-place accumulate into dx and dg)
// y = x * rms * g,  rms = 1/sqrt(mean(x^2)+eps)
// dx += g*rms*dy - x*rms^3*sum(dy*g*x)/n
// dg += dy * (x*rms)
void wildandev_rmsnorm_bwd(
    const float* __restrict__ x,
    const float* __restrict__ g,
    const float* __restrict__ dy,
    float* __restrict__ dx,   // accumulated in-place
    float* __restrict__ dg,   // accumulated in-place
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
    float rms3 = rms * rms * rms;

    // sum(dy * g * x)
    __m512 s512 = _mm512_setzero_ps();
    i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 vdy = _mm512_loadu_ps(dy + i);
        __m512 vg = _mm512_loadu_ps(g + i);
        __m512 vx = _mm512_loadu_ps(x + i);
        s512 = _mm512_fmadd_ps(vdy, _mm512_mul_ps(vg, vx), s512);
    }
    float sum_dgx = _mm512_reduce_add_ps(s512);
    for (; i < n; ++i) sum_dgx += dy[i] * g[i] * x[i];
    float c = rms3 * sum_dgx / (float)n;

    __m512 vrms = _mm512_set1_ps(rms);
    __m512 vc = _mm512_set1_ps(c);

    i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 vdy = _mm512_loadu_ps(dy + i);
        __m512 vg = _mm512_loadu_ps(g + i);
        __m512 vx = _mm512_loadu_ps(x + i);

        __m512 ddx = _mm512_sub_ps(
            _mm512_mul_ps(_mm512_mul_ps(vg, vrms), vdy),
            _mm512_mul_ps(vx, vc));
        _mm512_storeu_ps(dx + i, _mm512_add_ps(_mm512_loadu_ps(dx + i), ddx));

        __m512 ddg = _mm512_mul_ps(vdy, _mm512_mul_ps(vx, vrms));
        _mm512_storeu_ps(dg + i, _mm512_add_ps(_mm512_loadu_ps(dg + i), ddg));
    }
    for (; i < n; ++i) {
        dx[i] += g[i] * rms * dy[i] - x[i] * c;
        dg[i] += dy[i] * x[i] * rms;
    }
}

// Fused softmax + cross-entropy forward AND backward
// logits: (m x v), targets: (m), dlogits: (m x v) out
// returns mean CE loss
float wildandev_ce_fwd_bwd(
    const float* __restrict__ logits,
    const int* __restrict__ targets,
    float* __restrict__ dlogits,
    size_t m, size_t v)
{
    float total = 0.0f;
    float inv_m = 1.0f / (float)m;
    for (size_t i = 0; i < m; ++i) {
        const float* row = logits + i * v;
        float mx = row[0];
        for (size_t c = 1; c < v; ++c) if (row[c] > mx) mx = row[c];

        float sum = 0.0f;
        for (size_t c = 0; c < v; ++c) sum += std::exp(row[c] - mx);
        float inv_sum = 1.0f / sum;

        int t = targets[i];
        total -= (row[t] - mx - std::log(sum));

        float* drow = dlogits + i * v;
        for (size_t c = 0; c < v; ++c) {
            float p = std::exp(row[c] - mx) * inv_sum;
            drow[c] = (p - (c == (size_t)t ? 1.0f : 0.0f)) * inv_m;
        }
    }
    return total * inv_m;
}


// Optimizer: vectorized AdamW

void wildandev_adamw(
    float* __restrict__ p,
    const float* __restrict__ g,
    float* __restrict__ m,
    float* __restrict__ v,
    float lr, float beta1, float beta2,
    float eps, float wd,
    float bc1, float bc2,
    size_t n)
{
    __m512 vlr = _mm512_set1_ps(lr);
    __m512 vb1 = _mm512_set1_ps(beta1);
    __m512 vone_mb1 = _mm512_set1_ps(1.0f - beta1);
    __m512 vb2 = _mm512_set1_ps(beta2);
    __m512 vone_mb2 = _mm512_set1_ps(1.0f - beta2);
    __m512 veps = _mm512_set1_ps(eps);
    __m512 vwd = _mm512_set1_ps(wd);
    __m512 vbc1 = _mm512_set1_ps(bc1);
    __m512 vbc2 = _mm512_set1_ps(bc2);

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 vp = _mm512_loadu_ps(p + i);
        __m512 vg = _mm512_loadu_ps(g + i);
        __m512 vm = _mm512_loadu_ps(m + i);
        __m512 vv = _mm512_loadu_ps(v + i);

        vm = _mm512_add_ps(_mm512_mul_ps(vb1, vm), _mm512_mul_ps(vone_mb1, vg));
        vv = _mm512_add_ps(_mm512_mul_ps(vb2, vv),
            _mm512_mul_ps(vone_mb2, _mm512_mul_ps(vg, vg)));

        __m512 mhat = _mm512_div_ps(vm, vbc1);
        __m512 vhat = _mm512_div_ps(vv, vbc2);

        __m512 upd = _mm512_add_ps(
            _mm512_mul_ps(vwd, vp),
            _mm512_div_ps(mhat, _mm512_add_ps(_mm512_sqrt_ps(vhat), veps)));

        _mm512_storeu_ps(p + i, _mm512_sub_ps(vp, _mm512_mul_ps(vlr, upd)));
        _mm512_storeu_ps(m + i, vm);
        _mm512_storeu_ps(v + i, vv);
    }
    for (; i < n; ++i) {
        m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * g[i] * g[i];
        float mhat = m[i] / bc1;
        float vhat = v[i] / bc2;
        p[i] -= lr * (wd * p[i] + mhat / (std::sqrt(vhat) + eps));
    }
}


// Inference helpers (existing)


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
