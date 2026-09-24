use crate::transformer::WildandevConfig;

pub struct ElisGrads {
    pub d_token_emb: Vec<f32>,
    pub d_pos_emb: Vec<f32>,
    pub d_w_q: Vec<Vec<f32>>,
    pub d_w_k: Vec<Vec<f32>>,
    pub d_w_v: Vec<Vec<f32>>,
    pub d_w_o: Vec<Vec<f32>>,
    pub d_norm1_g: Vec<Vec<f32>>,
    pub d_norm2_g: Vec<Vec<f32>>,
    pub d_w_gate: Vec<Vec<f32>>,
    pub d_w_up: Vec<Vec<f32>>,
    pub d_w_down: Vec<Vec<f32>>,
    pub d_lm_head: Vec<f32>,
}

impl ElisGrads {
    pub fn zero(cfg: &WildandevConfig) -> Self {
        let d = cfg.d_model;
        let dff = cfg.d_ff;
        Self {
            d_token_emb: vec![0.0; cfg.vocab_size * d],
            d_pos_emb: vec![0.0; cfg.seq_len * d],
            d_w_q: (0..cfg.n_layers).map(|_| vec![0.0; d * d]).collect(),
            d_w_k: (0..cfg.n_layers).map(|_| vec![0.0; d * d]).collect(),
            d_w_v: (0..cfg.n_layers).map(|_| vec![0.0; d * d]).collect(),
            d_w_o: (0..cfg.n_layers).map(|_| vec![0.0; d * d]).collect(),
            d_norm1_g: (0..cfg.n_layers).map(|_| vec![0.0; d]).collect(),
            d_norm2_g: (0..cfg.n_layers).map(|_| vec![0.0; d]).collect(),
            d_w_gate: (0..cfg.n_layers).map(|_| vec![0.0; d * dff]).collect(),
            d_w_up: (0..cfg.n_layers).map(|_| vec![0.0; d * dff]).collect(),
            d_w_down: (0..cfg.n_layers).map(|_| vec![0.0; dff * d]).collect(),
            d_lm_head: vec![0.0; d * cfg.vocab_size],
        }
    }

    pub fn clip_norm(&mut self, max_norm: f32) -> f32 {
        let mut total_sq = 0.0f32;
        let sum_sq = |arr: &[f32]| -> f32 { arr.iter().map(|&v| v * v).sum::<f32>() };
        total_sq += sum_sq(&self.d_token_emb);
        total_sq += sum_sq(&self.d_pos_emb);
        for l in 0..self.d_w_q.len() {
            total_sq += sum_sq(&self.d_w_q[l]);
            total_sq += sum_sq(&self.d_w_k[l]);
            total_sq += sum_sq(&self.d_w_v[l]);
            total_sq += sum_sq(&self.d_w_o[l]);
            total_sq += sum_sq(&self.d_norm1_g[l]);
            total_sq += sum_sq(&self.d_norm2_g[l]);
            total_sq += sum_sq(&self.d_w_gate[l]);
            total_sq += sum_sq(&self.d_w_up[l]);
            total_sq += sum_sq(&self.d_w_down[l]);
        }
        total_sq += sum_sq(&self.d_lm_head);
        let norm = total_sq.sqrt();
        if norm > max_norm && norm > 1e-6 {
            let scale = max_norm / norm;
            let scale_arr = |arr: &mut [f32]| {
                for v in arr.iter_mut() { *v *= scale; }
            };
            scale_arr(&mut self.d_token_emb);
            scale_arr(&mut self.d_pos_emb);
            for l in 0..self.d_w_q.len() {
                scale_arr(&mut self.d_w_q[l]);
                scale_arr(&mut self.d_w_k[l]);
                scale_arr(&mut self.d_w_v[l]);
                scale_arr(&mut self.d_w_o[l]);
                scale_arr(&mut self.d_norm1_g[l]);
                scale_arr(&mut self.d_norm2_g[l]);
                scale_arr(&mut self.d_w_gate[l]);
                scale_arr(&mut self.d_w_up[l]);
                scale_arr(&mut self.d_w_down[l]);
            }
            scale_arr(&mut self.d_lm_head);
        }
        norm
    }
}
