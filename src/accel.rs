extern "C" {
    fn wildandev_gemm_f32(
        a: *const f32,
        b: *const f32,
        bias: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    );

    fn wildandev_gemm_f32_mt(
        a: *const f32,
        b: *const f32,
        bias: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
        n: usize,
        num_threads: usize,
    );

    fn wildandev_gemm_f32_tn(
        a: *const f32,
        b: *const f32,
        bias: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    );

    fn wildandev_gemm_i8(
        a: *const i8,
        b: *const i8,
        c: *mut f32,
        scale_a: f32,
        scale_b: f32,
        bias: *const f32,
        m: usize,
        k: usize,
        n: usize,
    );

    fn wildandev_rmsnorm(
        x: *mut f32,
        g: *const f32,
        n: usize,
        eps: f32,
    );

    fn wildandev_swiglu(
        gate: *mut f32,
        up: *const f32,
        len: usize,
    );
}

#[inline(always)]
pub fn gemm_forward(a: &[f32], b: &[f32], bias: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    let bias_ptr = if bias.is_empty() {
        std::ptr::null()
    } else {
        debug_assert_eq!(bias.len(), n);
        bias.as_ptr()
    };
    unsafe {
        wildandev_gemm_f32(a.as_ptr(), b.as_ptr(), bias_ptr, c.as_mut_ptr(), m, k, n);
    }
}

#[inline(always)]
pub fn gemm_forward_mt(a: &[f32], b: &[f32], bias: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, threads: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    let bias_ptr = if bias.is_empty() { std::ptr::null() } else { bias.as_ptr() };
    unsafe {
        wildandev_gemm_f32_mt(a.as_ptr(), b.as_ptr(), bias_ptr, c.as_mut_ptr(), m, k, n, threads);
    }
}

#[inline(always)]
pub fn gemm_transposed_b(a: &[f32], b_t: &[f32], bias: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b_t.len(), n * k);
    debug_assert_eq!(c.len(), m * n);
    let bias_ptr = if bias.is_empty() { std::ptr::null() } else { bias.as_ptr() };
    unsafe {
        wildandev_gemm_f32_tn(a.as_ptr(), b_t.as_ptr(), bias_ptr, c.as_mut_ptr(), m, k, n);
    }
}

#[inline(always)]
pub fn gemm_i8(a: &[i8], b_t: &[i8], c: &mut [f32], scale_a: f32, scale_b: f32, bias: &[f32], m: usize, k: usize, n: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b_t.len(), n * k);
    debug_assert_eq!(c.len(), m * n);
    let bias_ptr = if bias.is_empty() { std::ptr::null() } else { bias.as_ptr() };
    unsafe {
        wildandev_gemm_i8(a.as_ptr(), b_t.as_ptr(), c.as_mut_ptr(), scale_a, scale_b, bias_ptr, m, k, n);
    }
}

#[inline(always)]
pub fn rmsnorm_inplace(x: &mut [f32], g: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), g.len());
    unsafe {
        wildandev_rmsnorm(x.as_mut_ptr(), g.as_ptr(), x.len(), eps);
    }
}

#[inline(always)]
pub fn swiglu_inplace(gate: &mut [f32], up: &[f32]) {
    debug_assert_eq!(gate.len(), up.len());
    unsafe {
        wildandev_swiglu(gate.as_mut_ptr(), up.as_ptr(), gate.len());
    }
}
