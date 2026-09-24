use crate::transformer::WildandevConfig;

// AdamW Optimizer untuk bobot transformer
pub struct WildandevAdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step: usize,
    m: Vec<f32>,
    v: Vec<f32>,
}

impl WildandevAdamW {
    pub fn new(len: usize, lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            step: 0,
            m: vec![0.0; len],
            v: vec![0.0; len],
        }
    }

    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.step += 1;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let bc1 = 1.0 - beta1.powi(self.step as i32);
        let bc2 = 1.0 - beta2.powi(self.step as i32);

        for i in 0..params.len() {
            let g = grads[i];
            let p = params[i];
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g;
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * g * g;

            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;

            params[i] -= self.lr * self.weight_decay * p;
            params[i] -= self.lr * (m_hat / (v_hat.sqrt() + self.eps));
        }
    }
}

// Checkpoint I/O
impl crate::ElisTransformer {
    pub fn save_checkpoint(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        f.write_all(b"WENG")?;
        f.write_all(&(self.cfg.vocab_size as u32).to_le_bytes())?;
        f.write_all(&(self.cfg.d_model as u32).to_le_bytes())?;
        f.write_all(&(self.cfg.n_heads as u32).to_le_bytes())?;
        f.write_all(&(self.cfg.n_layers as u32).to_le_bytes())?;
        f.write_all(&(self.cfg.d_ff as u32).to_le_bytes())?;
        f.write_all(&(self.cfg.seq_len as u32).to_le_bytes())?;

        let write_slice = |f: &mut std::fs::File, s: &[f32]| -> std::io::Result<()> {
            let bytes = unsafe {
                std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4)
            };
            f.write_all(bytes)
        };

        write_slice(&mut f, &self.token_emb)?;
        write_slice(&mut f, &self.pos_emb)?;
        for l in 0..self.cfg.n_layers {
            write_slice(&mut f, &self.w_q[l])?;
            write_slice(&mut f, &self.w_k[l])?;
            write_slice(&mut f, &self.w_v[l])?;
            write_slice(&mut f, &self.w_o[l])?;
            write_slice(&mut f, &self.norm1_g[l])?;
            write_slice(&mut f, &self.norm2_g[l])?;
            write_slice(&mut f, &self.w_gate[l])?;
            write_slice(&mut f, &self.w_up[l])?;
            write_slice(&mut f, &self.w_down[l])?;
        }
        write_slice(&mut f, &self.lm_head)?;
        Ok(())
    }

    pub fn load_checkpoint(path: &str) -> std::io::Result<Self> {
        use std::io::Read;
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"WENG" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid magic"));
        }

        let mut read_u32 = || -> std::io::Result<usize> {
            let mut b = [0u8; 4];
            f.read_exact(&mut b)?;
            Ok(u32::from_le_bytes(b) as usize)
        };

        let cfg = WildandevConfig {
            vocab_size: read_u32()?,
            d_model: read_u32()?,
            n_heads: read_u32()?,
            n_layers: read_u32()?,
            d_ff: read_u32()?,
            seq_len: read_u32()?,
        };

        let mut model = Self::new(cfg.clone(), 0);

        let read_slice = |f: &mut std::fs::File, s: &mut [f32]| -> std::io::Result<()> {
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut u8, s.len() * 4)
            };
            f.read_exact(bytes)
        };

        read_slice(&mut f, &mut model.token_emb)?;
        read_slice(&mut f, &mut model.pos_emb)?;
        for l in 0..cfg.n_layers {
            read_slice(&mut f, &mut model.w_q[l])?;
            read_slice(&mut f, &mut model.w_k[l])?;
            read_slice(&mut f, &mut model.w_v[l])?;
            read_slice(&mut f, &mut model.w_o[l])?;
            read_slice(&mut f, &mut model.norm1_g[l])?;
            read_slice(&mut f, &mut model.norm2_g[l])?;
            read_slice(&mut f, &mut model.w_gate[l])?;
            read_slice(&mut f, &mut model.w_up[l])?;
            read_slice(&mut f, &mut model.w_down[l])?;
        }
        read_slice(&mut f, &mut model.lm_head)?;
        Ok(model)
    }

    // Autoregressive Cross-Entropy Loss computation for training
    pub fn compute_loss_and_grads(
        &mut self,
        input_tokens: &[usize],
        target_tokens: &[usize],
    ) -> f32 {
        let logits = self.forward(input_tokens).to_vec();
        let seq = input_tokens.len();
        let v = self.cfg.vocab_size;

        let mut total_loss = 0.0f32;
        let mut dlogits = vec![0.0f32; seq * v];

        for i in 0..seq {
            let row = &logits[i * v..(i + 1) * v];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for &val in row {
                sum_exp += (val - mx).exp();
            }
            let target = target_tokens[i];
            let p_target = ((row[target] - mx).exp() / sum_exp).max(1e-10);
            total_loss -= p_target.ln();

            // Softmax grad: (p - 1(y=target)) / seq
            for c in 0..v {
                let p = (row[c] - mx).exp() / sum_exp;
                let ind = if c == target { 1.0 } else { 0.0 };
                dlogits[i * v + c] = (p - ind) / seq as f32;
            }
        }

        // Backward through lm_head: d_lm_head = X^T * dlogits
        // (Fast single-step CPU head training)
        let d = self.cfg.d_model;
        let lr = 0.01f32;
        for i in 0..seq {
            let x_row = &self.buf_x[i * d..(i + 1) * d];
            let dl_row = &dlogits[i * v..(i + 1) * v];
            for k in 0..d {
                let xk = x_row[k];
                for c in 0..v {
                    self.lm_head[k * v + c] -= lr * xk * dl_row[c];
                }
            }
        }

        total_loss / seq as f32
    }
}
