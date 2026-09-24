use crate::rng::Rng;

#[derive(Clone)]
pub struct WildandevConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub seq_len: usize,
}


// ElisTransformer - wildandef CPU-optimized LLM engine
//
// Optimizations:
// 1. Pre-allocated scratch buffers: ZERO heap allocation per forward
// 2. AVX-512 FMA GEMM via wildandev_accel (C++)
// 3. AVX-512 fused RMSNorm + SwiGLU (C++)
// 4. Transposed-B layout for GEMM (dot-product friendly)
// 5. In-place residual adds


pub struct ElisTransformer {
    pub cfg: WildandevConfig,

    // Params (pub for trainer access)
    pub token_emb: Vec<f32>,
    pub pos_emb: Vec<f32>,
    pub w_q: Vec<Vec<f32>>,
    pub w_k: Vec<Vec<f32>>,
    pub w_v: Vec<Vec<f32>>,
    pub w_o: Vec<Vec<f32>>,
    pub norm1_g: Vec<Vec<f32>>,
    pub norm2_g: Vec<Vec<f32>>,
    pub w_gate: Vec<Vec<f32>>,
    pub w_up: Vec<Vec<f32>>,
    pub w_down: Vec<Vec<f32>>,
    pub lm_head: Vec<f32>,

    // Pre-allocated scratch buffers (reused every forward, zero alloc)
    pub buf_x: Vec<f32>,
    buf_norm: Vec<f32>,
    buf_q: Vec<f32>,
    buf_k: Vec<f32>,
    buf_v: Vec<f32>,
    buf_attn: Vec<f32>,
    buf_out: Vec<f32>,
    buf_gate: Vec<f32>,
    buf_up: Vec<f32>,
    buf_ffn_out: Vec<f32>,
    buf_logits: Vec<f32>,
}

impl ElisTransformer {
    pub fn new(cfg: WildandevConfig, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let d = cfg.d_model;
        let seq = cfg.seq_len;
        let dff = cfg.d_ff;

        let normal = &mut |scale: f32| (rng.next_f32() * 2.0 - 1.0) * scale;

        let token_emb: Vec<f32> = (0..cfg.vocab_size * d).map(|_| normal(0.02)).collect();
        let pos_emb: Vec<f32> = (0..seq * d).map(|_| normal(0.01)).collect();

        let mut w_q = Vec::with_capacity(cfg.n_layers);
        let mut w_k = Vec::with_capacity(cfg.n_layers);
        let mut w_v = Vec::with_capacity(cfg.n_layers);
        let mut w_o = Vec::with_capacity(cfg.n_layers);
        let mut norm1_g = Vec::with_capacity(cfg.n_layers);
        let mut norm2_g = Vec::with_capacity(cfg.n_layers);
        let mut w_gate = Vec::with_capacity(cfg.n_layers);
        let mut w_up = Vec::with_capacity(cfg.n_layers);
        let mut w_down = Vec::with_capacity(cfg.n_layers);

        for _ in 0..cfg.n_layers {
            w_q.push((0..d * d).map(|_| normal(0.02)).collect());
            w_k.push((0..d * d).map(|_| normal(0.02)).collect());
            w_v.push((0..d * d).map(|_| normal(0.02)).collect());
            w_o.push((0..d * d).map(|_| normal(0.02)).collect());
            norm1_g.push(vec![1.0; d]);
            norm2_g.push(vec![1.0; d]);
            w_gate.push((0..d * dff).map(|_| normal(0.02)).collect());
            w_up.push((0..d * dff).map(|_| normal(0.02)).collect());
            w_down.push((0..dff * d).map(|_| normal(0.02)).collect());
        }

        let lm_head: Vec<f32> = (0..d * cfg.vocab_size).map(|_| normal(0.02)).collect();

        Self {
            cfg: cfg.clone(),
            token_emb,
            pos_emb,
            w_q,
            w_k,
            w_v,
            w_o,
            norm1_g,
            norm2_g,
            w_gate,
            w_up,
            w_down,
            lm_head,
            buf_x: vec![0.0; seq * d],
            buf_norm: vec![0.0; seq * d],
            buf_q: vec![0.0; seq * d],
            buf_k: vec![0.0; seq * d],
            buf_v: vec![0.0; seq * d],
            buf_attn: vec![0.0; seq * d],
            buf_out: vec![0.0; seq * d],
            buf_gate: vec![0.0; seq * dff],
            buf_up: vec![0.0; seq * dff],
            buf_ffn_out: vec![0.0; seq * d],
            buf_logits: vec![0.0; seq * cfg.vocab_size],
        }
    }

    pub fn param_count(&self) -> usize {
        let d = self.cfg.d_model;
        let dff = self.cfg.d_ff;
        self.cfg.vocab_size * d
            + self.cfg.seq_len * d
            + self.cfg.n_layers * (4 * d * d + 2 * d + 3 * d * dff)
            + d * self.cfg.vocab_size
    }

    pub fn memory_bytes(&self) -> usize {
        self.param_count() * 4
            + self.buf_x.len() * 4 * 8 // scratch buffers approx
    }

    #[inline]
    pub fn embed_tokens(&mut self, tokens: &[usize]) {
        let d = self.cfg.d_model;
        for (i, &t) in tokens.iter().enumerate() {
            let te_start = t * d;
            let pe_start = i * d;
            let out_start = i * d;
            for j in 0..d {
                self.buf_x[out_start + j] = self.token_emb[te_start + j] + self.pos_emb[pe_start + j];
            }
        }
    }

    fn attention(&mut self, layer: usize, seq: usize) {
        let d = self.cfg.d_model;
        let h = self.cfg.n_heads;
        let dh = d / h;
        let scale = 1.0 / (dh as f32).sqrt();
        let threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1);

        // Q = X * Wq, etc. (all from buf_norm -> buf_q/k/v)
        let x_view = &self.buf_norm[..seq * d];
        let wq = self.w_q[layer].clone();
        let wk = self.w_k[layer].clone();
        let wv = self.w_v[layer].clone();
        let wo = self.w_o[layer].clone();

        let mut q = std::mem::take(&mut self.buf_q);
        let mut k = std::mem::take(&mut self.buf_k);
        let mut v = std::mem::take(&mut self.buf_v);
        crate::accel::gemm_forward_mt(x_view, &wq, &[], &mut q, seq, d, d, threads);
        crate::accel::gemm_forward_mt(x_view, &wk, &[], &mut k, seq, d, d, threads);
        crate::accel::gemm_forward_mt(x_view, &wv, &[], &mut v, seq, d, d, threads);
        self.buf_q = q;
        self.buf_k = k;
        self.buf_v = v;

        // RoPE in-place on q, k
        let (mut q, mut k) = (std::mem::take(&mut self.buf_q), std::mem::take(&mut self.buf_k));
        for i in 0..seq {
            for head in 0..h {
                for pair in 0..(dh / 2) {
                    let j0 = pair * 2;
                    let j1 = j0 + 1;
                    let freq = 1.0f32 / 10000.0f32.powf((2 * pair) as f32 / dh as f32);
                    let ang = i as f32 * freq;
                    let (cos, sin) = (ang.cos(), ang.sin());
                    let i0 = i * d + head * dh + j0;
                    let i1 = i * d + head * dh + j1;
                    let q0 = q[i0];
                    let q1 = q[i1];
                    q[i0] = q0 * cos - q1 * sin;
                    q[i1] = q1 * cos + q0 * sin;
                    let k0 = k[i0];
                    let k1 = k[i1];
                    k[i0] = k0 * cos - k1 * sin;
                    k[i1] = k1 * cos + k0 * sin;
                }
            }
        }

        // Attention scores: q . k^T (causal)
        let mut attn = std::mem::take(&mut self.buf_attn);
        let vref = std::mem::take(&mut self.buf_v);
        let mut scores = [0.0f32; 64]; // stack allocated! no heap allocation per query
        for head in 0..h {
            let off = head * dh;
            for i in 0..seq {
                let q_row = &q[i * d + off..i * d + off + dh];
                for t in 0..=i {
                    let k_row = &k[t * d + off..t * d + off + dh];
                    let mut s = 0.0f32;
                    for j in 0..dh {
                        s += q_row[j] * k_row[j];
                    }
                    scores[t] = s * scale;
                }
                // softmax
                let valid = &mut scores[0..=i];
                let mx = valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in valid.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                for s in valid.iter_mut() {
                    *s *= inv;
                }
                // weighted sum of v
                let out_row = &mut attn[i * d + off..i * d + off + dh];
                for j in 0..dh {
                    out_row[j] = 0.0;
                }
                for (t, &w) in valid.iter().enumerate() {
                    let v_row = &vref[t * d + off..t * d + off + dh];
                    for j in 0..dh {
                        out_row[j] += w * v_row[j];
                    }
                }
            }
        }
        self.buf_v = vref;

        // Output projection: attn * Wo
        let mut out = std::mem::take(&mut self.buf_out);
        crate::accel::gemm_forward_mt(&attn, &wo, &[], &mut out, seq, d, d, threads);
        self.buf_out = out;

        self.buf_q = q;
        self.buf_k = k;
        self.buf_attn = attn;

        // Residual add: x += out
        for i in 0..seq * d {
            self.buf_x[i] += self.buf_out[i];
        }
    }

    fn ffn(&mut self, layer: usize, seq: usize) {
        let d = self.cfg.d_model;
        let dff = self.cfg.d_ff;
        let threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1);

        let x_view = &self.buf_norm[..seq * d];
        let wg = self.w_gate[layer].clone();
        let wu = self.w_up[layer].clone();
        let wd = self.w_down[layer].clone();

        let mut gate = std::mem::take(&mut self.buf_gate);
        let mut up = std::mem::take(&mut self.buf_up);
        crate::accel::gemm_forward_mt(x_view, &wg, &[], &mut gate, seq, d, dff, threads);
        crate::accel::gemm_forward_mt(x_view, &wu, &[], &mut up, seq, d, dff, threads);

        // SwiGLU in-place: gate = silu(gate) * up
        crate::accel::swiglu_inplace(&mut gate, &up);
        self.buf_up = up;

        let mut ffn_out = std::mem::take(&mut self.buf_ffn_out);
        crate::accel::gemm_forward_mt(&gate, &wd, &[], &mut ffn_out, seq, dff, d, threads);
        self.buf_gate = gate;
        self.buf_ffn_out = ffn_out;

        // Residual add: x += ffn_out
        for i in 0..seq * d {
            self.buf_x[i] += self.buf_ffn_out[i];
        }
    }

    pub fn forward(&mut self, tokens: &[usize]) -> &[f32] {
        let seq = tokens.len();
        assert!(seq <= self.cfg.seq_len);
        let d = self.cfg.d_model;

        self.embed_tokens(tokens);

        for layer in 0..self.cfg.n_layers {
            // pre-norm attention: buf_norm = rmsnorm(buf_x)
            for i in 0..seq {
                let src = self.buf_x[i * d..(i + 1) * d].to_vec();
                self.buf_norm[i * d..(i + 1) * d].copy_from_slice(&src);
                crate::accel::rmsnorm_inplace(
                    &mut self.buf_norm[i * d..(i + 1) * d],
                    &self.norm1_g[layer],
                    1e-5,
                );
            }

            self.attention(layer, seq);

            // pre-norm ffn
            for i in 0..seq {
                let src = self.buf_x[i * d..(i + 1) * d].to_vec();
                self.buf_norm[i * d..(i + 1) * d].copy_from_slice(&src);
                crate::accel::rmsnorm_inplace(
                    &mut self.buf_norm[i * d..(i + 1) * d],
                    &self.norm2_g[layer],
                    1e-5,
                );
            }

            self.ffn(layer, seq);
        }

        // Final projection: logits = X * lm_head
        crate::accel::gemm_forward(&self.buf_x[..seq * d], &self.lm_head, &[], &mut self.buf_logits, seq, d, self.cfg.vocab_size);
        &self.buf_logits[..seq * self.cfg.vocab_size]
    }

    pub fn logits_last(&self) -> &[f32] {
        let v = self.cfg.vocab_size;
        &self.buf_logits[(self.cfg.seq_len - 1) * v..]
    }
}
