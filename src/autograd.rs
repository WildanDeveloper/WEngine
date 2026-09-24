use crate::grads::ElisGrads;


// Layer-by-layer forward activation snapshot for exact backprop
pub struct LayerCache {
    pub x_in: Vec<f32>,       // input to layer (residual base)
    pub norm1: Vec<f32>,      // after norm1
    pub q: Vec<f32>,          // after Wq + RoPE
    pub k: Vec<f32>,          // after Wk + RoPE
    pub v: Vec<f32>,          // after Wv
    pub attn_weights: Vec<Vec<f32>>, // per-head attention weights (seq x seq)
    pub attn_out: Vec<f32>,   // after Wo
    pub x_mid: Vec<f32>,      // x_in + attn_out
    pub norm2: Vec<f32>,      // after norm2
    pub gate_pre: Vec<f32>,   // after Wgate (pre-silu)
    pub up: Vec<f32>,         // after Wup
    pub ffn_out: Vec<f32>,    // after Wdown
}

pub struct ForwardCache {
    pub input_tokens: Vec<usize>,
    pub x_emb: Vec<f32>,      // token_emb + pos_emb
    pub layers: Vec<LayerCache>,
    pub x_final: Vec<f32>,    // output of final layer
    pub logits: Vec<f32>,     // x_final @ lm_head
}

impl crate::ElisTransformer {
    // Full forward with complete activation caching for exact analytic backprop
    pub fn forward_train(&mut self, tokens: &[usize]) -> ForwardCache {
        let seq = tokens.len();
        assert!(seq <= self.cfg.seq_len);
        let d = self.cfg.d_model;
        let h = self.cfg.n_heads;
        let dh = d / h;
        let dff = self.cfg.d_ff;
        let scale = 1.0 / (dh as f32).sqrt();

        // 1. Embeddings
        self.embed_tokens(tokens);
        let x_emb = self.buf_x[..seq * d].to_vec();
        let mut cur_x = x_emb.clone();

        let mut layer_caches = Vec::with_capacity(self.cfg.n_layers);

        for l in 0..self.cfg.n_layers {
            let x_in = cur_x.clone();

            // Norm 1
            let mut norm1 = vec![0.0f32; seq * d];
            for i in 0..seq {
                norm1[i * d..(i + 1) * d].copy_from_slice(&cur_x[i * d..(i + 1) * d]);
                crate::accel::rmsnorm_inplace(&mut norm1[i * d..(i + 1) * d], &self.norm1_g[l], 1e-5);
            }

            // Q, K, V
            let mut q = vec![0.0f32; seq * d];
            let mut k = vec![0.0f32; seq * d];
            let mut v = vec![0.0f32; seq * d];
            crate::accel::gemm_forward(&norm1, &self.w_q[l], &[], &mut q, seq, d, d);
            crate::accel::gemm_forward(&norm1, &self.w_k[l], &[], &mut k, seq, d, d);
            crate::accel::gemm_forward(&norm1, &self.w_v[l], &[], &mut v, seq, d, d);

            // RoPE
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
                        let (q0, q1) = (q[i0], q[i1]);
                        let (k0, k1) = (k[i0], k[i1]);
                        q[i0] = q0 * cos - q1 * sin;
                        q[i1] = q1 * cos + q0 * sin;
                        k[i0] = k0 * cos - k1 * sin;
                        k[i1] = k1 * cos + k0 * sin;
                    }
                }
            }

            // Attention + weights cache
            let mut attn_out_unproj = vec![0.0f32; seq * d];
            let mut attn_weights = Vec::with_capacity(h);
            for head in 0..h {
                let off = head * dh;
                let mut head_w = vec![0.0f32; seq * seq];
                for i in 0..seq {
                    let q_row = &q[i * d + off..i * d + off + dh];
                    let mut scores = vec![0.0f32; i + 1];
                    for t in 0..=i {
                        let k_row = &k[t * d + off..t * d + off + dh];
                        let mut s = 0.0f32;
                        for j in 0..dh { s += q_row[j] * k_row[j]; }
                        scores[t] = s * scale;
                    }
                    let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for s in scores.iter_mut() {
                        *s = (*s - mx).exp();
                        sum += *s;
                    }
                    let inv = 1.0 / sum;
                    for (t, s) in scores.iter().enumerate() {
                        let w = s * inv;
                        head_w[i * seq + t] = w;
                        let v_row = &v[t * d + off..t * d + off + dh];
                        for j in 0..dh {
                            attn_out_unproj[i * d + off + j] += w * v_row[j];
                        }
                    }
                }
                attn_weights.push(head_w);
            }

            // Wo
            let mut attn_out = vec![0.0f32; seq * d];
            crate::accel::gemm_forward(&attn_out_unproj, &self.w_o[l], &[], &mut attn_out, seq, d, d);

            // Residual 1
            let mut x_mid = vec![0.0f32; seq * d];
            for i in 0..seq * d { x_mid[i] = x_in[i] + attn_out[i]; }

            // Norm 2
            let mut norm2 = vec![0.0f32; seq * d];
            for i in 0..seq {
                norm2[i * d..(i + 1) * d].copy_from_slice(&x_mid[i * d..(i + 1) * d]);
                crate::accel::rmsnorm_inplace(&mut norm2[i * d..(i + 1) * d], &self.norm2_g[l], 1e-5);
            }

            // FFN: gate & up
            let mut gate_pre = vec![0.0f32; seq * dff];
            let mut up = vec![0.0f32; seq * dff];
            crate::accel::gemm_forward(&norm2, &self.w_gate[l], &[], &mut gate_pre, seq, d, dff);
            crate::accel::gemm_forward(&norm2, &self.w_up[l], &[], &mut up, seq, d, dff);

            // SwiGLU activation
            let mut activated = gate_pre.clone();
            crate::accel::swiglu_inplace(&mut activated, &up);

            // Down
            let mut ffn_out = vec![0.0f32; seq * d];
            crate::accel::gemm_forward(&activated, &self.w_down[l], &[], &mut ffn_out, seq, dff, d);

            // Residual 2
            for i in 0..seq * d { cur_x[i] = x_mid[i] + ffn_out[i]; }

            layer_caches.push(LayerCache {
                x_in, norm1, q, k, v, attn_weights, attn_out,
                x_mid, norm2, gate_pre, up, ffn_out,
            });
        }

        // Final logits
        let mut logits = vec![0.0f32; seq * self.cfg.vocab_size];
        crate::accel::gemm_forward(&cur_x, &self.lm_head, &[], &mut logits, seq, d, self.cfg.vocab_size);

        ForwardCache {
            input_tokens: tokens.to_vec(),
            x_emb,
            layers: layer_caches,
            x_final: cur_x,
            logits,
        }
    }

    // Full exact analytic backpropagation through all layers
    pub fn backward_train(&self, cache: &ForwardCache, targets: &[usize], grads: &mut ElisGrads) -> f32 {
        let seq = cache.input_tokens.len();
        let d = self.cfg.d_model;
        let h = self.cfg.n_heads;
        let dh = d / h;
        let dff = self.cfg.d_ff;
        let v = self.cfg.vocab_size;
        let scale = 1.0 / (dh as f32).sqrt();

        // 1. Loss & dlogits via fused C++ kernel
        let mut dlogits = vec![0.0f32; seq * v];
        let targets_i32: Vec<i32> = targets.iter().map(|&t| t as i32).collect();
        let loss = unsafe {
            extern "C" {
                fn wildandev_ce_fwd_bwd(
                    logits: *const f32,
                    targets: *const i32,
                    dlogits: *mut f32,
                    m: usize,
                    v: usize,
                ) -> f32;
            }
            wildandev_ce_fwd_bwd(cache.logits.as_ptr(), targets_i32.as_ptr(), dlogits.as_mut_ptr(), seq, v)
        };

        // 2. lm_head backward:
        //    d_lm_head += x_final^T @ dlogits
        //    dx = dlogits @ lm_head^T
        extern "C" {
            fn wildandev_gemm_acc_atb(x: *const f32, dy: *const f32, dw: *mut f32, m: usize, k: usize, n: usize);
            fn wildandev_swiglu_bwd(g: *const f32, u: *const f32, dh: *const f32, dg: *mut f32, du: *mut f32, len: usize);
            fn wildandev_rmsnorm_bwd(x: *const f32, g: *const f32, dy: *const f32, dx: *mut f32, dg: *mut f32, n: usize, eps: f32);
        }
        unsafe {
            wildandev_gemm_acc_atb(cache.x_final.as_ptr(), dlogits.as_ptr(), grads.d_lm_head.as_mut_ptr(), seq, d, v);
        }

        // dx_final = dlogits @ lm_head^T
        let mut dx = vec![0.0f32; seq * d];
        for i in 0..seq {
            let dl = &dlogits[i * v..(i + 1) * v];
            for p in 0..d {
                let mut acc = 0.0f32;
                for c in 0..v { acc += dl[c] * self.lm_head[p * v + c]; }
                dx[i * d + p] = acc;
            }
        }

        // 3. Backprop through transformer layers in reverse
        for l in (0..self.cfg.n_layers).rev() {
            let lc = &cache.layers[l];

            // --- FFN backward ---
            // dx splits to residual and ffn_out: dx_mid = dx, d_ffn_out = dx
            let d_ffn_out = dx.clone();

            // d_w_down += activated^T @ d_ffn_out
            let mut activated = lc.gate_pre.clone();
            crate::accel::swiglu_inplace(&mut activated, &lc.up);
            unsafe {
                wildandev_gemm_acc_atb(activated.as_ptr(), d_ffn_out.as_ptr(), grads.d_w_down[l].as_mut_ptr(), seq, dff, d);
            }

            // d_activated = d_ffn_out @ w_down^T
            let mut d_act = vec![0.0f32; seq * dff];
            for i in 0..seq {
                let df = &d_ffn_out[i * d..(i + 1) * d];
                for p in 0..dff {
                    let mut acc = 0.0f32;
                    for c in 0..d { acc += df[c] * self.w_down[l][p * d + c]; }
                    d_act[i * dff + p] = acc;
                }
            }

            // SwiGLU backward -> dgate, dup
            let mut dgate = vec![0.0f32; seq * dff];
            let mut dup = vec![0.0f32; seq * dff];
            unsafe {
                wildandev_swiglu_bwd(lc.gate_pre.as_ptr(), lc.up.as_ptr(), d_act.as_ptr(), dgate.as_mut_ptr(), dup.as_mut_ptr(), seq * dff);
            }

            // d_w_gate += norm2^T @ dgate;  d_w_up += norm2^T @ dup
            unsafe {
                wildandev_gemm_acc_atb(lc.norm2.as_ptr(), dgate.as_ptr(), grads.d_w_gate[l].as_mut_ptr(), seq, d, dff);
                wildandev_gemm_acc_atb(lc.norm2.as_ptr(), dup.as_ptr(), grads.d_w_up[l].as_mut_ptr(), seq, d, dff);
            }

            // dnorm2 = dgate @ w_gate^T + dup @ w_up^T
            let mut dnorm2 = vec![0.0f32; seq * d];
            for i in 0..seq {
                let dg = &dgate[i * dff..(i + 1) * dff];
                let du = &dup[i * dff..(i + 1) * dff];
                for p in 0..d {
                    let mut acc = 0.0f32;
                    for c in 0..dff {
                        acc += dg[c] * self.w_gate[l][p * dff + c];
                        acc += du[c] * self.w_up[l][p * dff + c];
                    }
                    dnorm2[i * d + p] = acc;
                }
            }

            // RMSNorm 2 backward: dnorm2 -> dx_mid, d_norm2_g
            let mut dx_mid = dx.clone(); // start with residual from before FFN
            for i in 0..seq {
                unsafe {
                    wildandev_rmsnorm_bwd(
                        lc.x_mid[i * d..].as_ptr(),
                        self.norm2_g[l].as_ptr(),
                        dnorm2[i * d..].as_ptr(),
                        dx_mid[i * d..].as_mut_ptr(),
                        grads.d_norm2_g[l].as_mut_ptr(),
                        d, 1e-5);
                }
            }

            // --- Attention backward ---
            // dx_mid splits to residual (dx_in) and d_attn_out
            let d_attn_out = &dx_mid;

            // d_w_o += attn_out_unproj^T @ d_attn_out
            // Recompute attn_out_unproj from cache
            let mut attn_out_unproj = vec![0.0f32; seq * d];
            for head in 0..h {
                let off = head * dh;
                let head_w = &lc.attn_weights[head];
                for i in 0..seq {
                    for t in 0..=i {
                        let w = head_w[i * seq + t];
                        for j in 0..dh {
                            attn_out_unproj[i * d + off + j] += w * lc.v[t * d + off + j];
                        }
                    }
                }
            }
            unsafe {
                wildandev_gemm_acc_atb(attn_out_unproj.as_ptr(), d_attn_out.as_ptr(), grads.d_w_o[l].as_mut_ptr(), seq, d, d);
            }

            // d_attn_unproj = d_attn_out @ w_o^T
            let mut d_unproj = vec![0.0f32; seq * d];
            for i in 0..seq {
                let da = &d_attn_out[i * d..(i + 1) * d];
                for p in 0..d {
                    let mut acc = 0.0f32;
                    for c in 0..d { acc += da[c] * self.w_o[l][p * d + c]; }
                    d_unproj[i * d + p] = acc;
                }
            }

            // Backward through attention pooling
            let mut dq = vec![0.0f32; seq * d];
            let mut dk = vec![0.0f32; seq * d];
            let mut dv = vec![0.0f32; seq * d];

            for head in 0..h {
                let off = head * dh;
                let head_w = &lc.attn_weights[head];
                for i in 0..seq {
                    let du_i = &d_unproj[i * d + off..i * d + off + dh];

                    // 1. dv contribution and dscores
                    let mut dscores = vec![0.0f32; i + 1];
                    for t in 0..=i {
                        let w = head_w[i * seq + t];
                        let v_t = &lc.v[t * d + off..t * d + off + dh];
                        // dscores[t] = w * (du_i dot v_t)
                        let mut dot = 0.0f32;
                        for j in 0..dh {
                            dot += du_i[j] * v_t[j];
                            dv[t * d + off + j] += w * du_i[j];
                        }
                        dscores[t] = dot;
                    }

                    // Softmax backward: dS = W * (dscores - sum(W * dscores))
                    let mut sum_wd = 0.0f32;
                    for t in 0..=i { sum_wd += head_w[i * seq + t] * dscores[t]; }
                    for t in 0..=i {
                        let ds = head_w[i * seq + t] * (dscores[t] - sum_wd) * scale;
                        // S = q_i dot k_t * scale -> dq_i += ds * k_t,  dk_t += ds * q_i
                        let q_i = &lc.q[i * d + off..i * d + off + dh];
                        let k_t = &lc.k[t * d + off..t * d + off + dh];
                        for j in 0..dh {
                            dq[i * d + off + j] += ds * k_t[j];
                            dk[t * d + off + j] += ds * q_i[j];
                        }
                    }
                }
            }

            // RoPE backward (unitary rotation: transpose rotation matrix = reverse angle)
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

                        // Inverse rotation: [cos, sin; -sin, cos]
                        let (dq0, dq1) = (dq[i0], dq[i1]);
                        let (dk0, dk1) = (dk[i0], dk[i1]);
                        dq[i0] = dq0 * cos + dq1 * sin;
                        dq[i1] = dq1 * cos - dq0 * sin;
                        dk[i0] = dk0 * cos + dk1 * sin;
                        dk[i1] = dk1 * cos - dk0 * sin;
                    }
                }
            }

            // Accumulate d_w_q, d_w_k, d_w_v
            unsafe {
                wildandev_gemm_acc_atb(lc.norm1.as_ptr(), dq.as_ptr(), grads.d_w_q[l].as_mut_ptr(), seq, d, d);
                wildandev_gemm_acc_atb(lc.norm1.as_ptr(), dk.as_ptr(), grads.d_w_k[l].as_mut_ptr(), seq, d, d);
                wildandev_gemm_acc_atb(lc.norm1.as_ptr(), dv.as_ptr(), grads.d_w_v[l].as_mut_ptr(), seq, d, d);
            }

            // dnorm1 = dq @ w_q^T + dk @ w_k^T + dv @ w_v^T
            let mut dnorm1 = vec![0.0f32; seq * d];
            for i in 0..seq {
                let dqi = &dq[i * d..(i + 1) * d];
                let dki = &dk[i * d..(i + 1) * d];
                let dvi = &dv[i * d..(i + 1) * d];
                for p in 0..d {
                    let mut acc = 0.0f32;
                    for c in 0..d {
                        acc += dqi[c] * self.w_q[l][p * d + c];
                        acc += dki[c] * self.w_k[l][p * d + c];
                        acc += dvi[c] * self.w_v[l][p * d + c];
                    }
                    dnorm1[i * d + p] = acc;
                }
            }

            // RMSNorm 1 backward: dnorm1 -> dx_in, d_norm1_g
            let mut dx_in = dx_mid; // residual
            for i in 0..seq {
                unsafe {
                    wildandev_rmsnorm_bwd(
                        lc.x_in[i * d..].as_ptr(),
                        self.norm1_g[l].as_ptr(),
                        dnorm1[i * d..].as_ptr(),
                        dx_in[i * d..].as_mut_ptr(),
                        grads.d_norm1_g[l].as_mut_ptr(),
                        d, 1e-5);
                }
            }

            dx = dx_in;
        }

        // 4. Embedding backward: dx -> d_token_emb, d_pos_emb
        for (i, &t) in cache.input_tokens.iter().enumerate() {
            let dx_row = &dx[i * d..(i + 1) * d];
            let te_row = &mut grads.d_token_emb[t * d..(t + 1) * d];
            let pe_row = &mut grads.d_pos_emb[i * d..(i + 1) * d];
            for j in 0..d {
                te_row[j] += dx_row[j];
                pe_row[j] += dx_row[j];
            }
        }

        loss
    }
}
