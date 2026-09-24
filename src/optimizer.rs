use crate::grads::ElisGrads;
use crate::trainer::WildandevAdamW;
use crate::transformer::ElisTransformer;

pub struct TransformerOptimizers {
    pub opt_token_emb: WildandevAdamW,
    pub opt_pos_emb: WildandevAdamW,
    pub opt_w_q: Vec<WildandevAdamW>,
    pub opt_w_k: Vec<WildandevAdamW>,
    pub opt_w_v: Vec<WildandevAdamW>,
    pub opt_w_o: Vec<WildandevAdamW>,
    pub opt_norm1_g: Vec<WildandevAdamW>,
    pub opt_norm2_g: Vec<WildandevAdamW>,
    pub opt_w_gate: Vec<WildandevAdamW>,
    pub opt_w_up: Vec<WildandevAdamW>,
    pub opt_w_down: Vec<WildandevAdamW>,
    pub opt_lm_head: WildandevAdamW,
}

impl TransformerOptimizers {
    pub fn new(model: &ElisTransformer, lr: f32) -> Self {
        let cfg = &model.cfg;
        let d = cfg.d_model;
        let dff = cfg.d_ff;

        Self {
            opt_token_emb: WildandevAdamW::new(cfg.vocab_size * d, lr),
            opt_pos_emb: WildandevAdamW::new(cfg.seq_len * d, lr),
            opt_w_q: (0..cfg.n_layers).map(|_| WildandevAdamW::new(d * d, lr)).collect(),
            opt_w_k: (0..cfg.n_layers).map(|_| WildandevAdamW::new(d * d, lr)).collect(),
            opt_w_v: (0..cfg.n_layers).map(|_| WildandevAdamW::new(d * d, lr)).collect(),
            opt_w_o: (0..cfg.n_layers).map(|_| WildandevAdamW::new(d * d, lr)).collect(),
            opt_norm1_g: (0..cfg.n_layers).map(|_| WildandevAdamW::new(d, lr)).collect(),
            opt_norm2_g: (0..cfg.n_layers).map(|_| WildandevAdamW::new(d, lr)).collect(),
            opt_w_gate: (0..cfg.n_layers).map(|_| WildandevAdamW::new(d * dff, lr)).collect(),
            opt_w_up: (0..cfg.n_layers).map(|_| WildandevAdamW::new(d * dff, lr)).collect(),
            opt_w_down: (0..cfg.n_layers).map(|_| WildandevAdamW::new(dff * d, lr)).collect(),
            opt_lm_head: WildandevAdamW::new(d * cfg.vocab_size, lr),
        }
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.opt_token_emb.lr = lr;
        self.opt_pos_emb.lr = lr;
        for o in self.opt_w_q.iter_mut() { o.lr = lr; }
        for o in self.opt_w_k.iter_mut() { o.lr = lr; }
        for o in self.opt_w_v.iter_mut() { o.lr = lr; }
        for o in self.opt_w_o.iter_mut() { o.lr = lr; }
        for o in self.opt_norm1_g.iter_mut() { o.lr = lr; }
        for o in self.opt_norm2_g.iter_mut() { o.lr = lr; }
        for o in self.opt_w_gate.iter_mut() { o.lr = lr; }
        for o in self.opt_w_up.iter_mut() { o.lr = lr; }
        for o in self.opt_w_down.iter_mut() { o.lr = lr; }
        self.opt_lm_head.lr = lr;
    }

    pub fn step(&mut self, model: &mut ElisTransformer, grads: &ElisGrads) {
        self.opt_token_emb.step(&mut model.token_emb, &grads.d_token_emb);
        self.opt_pos_emb.step(&mut model.pos_emb, &grads.d_pos_emb);
        for l in 0..model.cfg.n_layers {
            self.opt_w_q[l].step(&mut model.w_q[l], &grads.d_w_q[l]);
            self.opt_w_k[l].step(&mut model.w_k[l], &grads.d_w_k[l]);
            self.opt_w_v[l].step(&mut model.w_v[l], &grads.d_w_v[l]);
            self.opt_w_o[l].step(&mut model.w_o[l], &grads.d_w_o[l]);
            self.opt_norm1_g[l].step(&mut model.norm1_g[l], &grads.d_norm1_g[l]);
            self.opt_norm2_g[l].step(&mut model.norm2_g[l], &grads.d_norm2_g[l]);
            self.opt_w_gate[l].step(&mut model.w_gate[l], &grads.d_w_gate[l]);
            self.opt_w_up[l].step(&mut model.w_up[l], &grads.d_w_up[l]);
            self.opt_w_down[l].step(&mut model.w_down[l], &grads.d_w_down[l]);
        }
        self.opt_lm_head.step(&mut model.lm_head, &grads.d_lm_head);
    }
}
