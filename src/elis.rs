use crate::rng::Rng;
use crate::tensor::Tensor;

pub struct Elis {
    dims: Vec<usize>,
    weights: Vec<Tensor>,
    grad_weights: Vec<Tensor>,
    biases: Vec<Vec<f32>>,
    grad_biases: Vec<Vec<f32>>,
    activations: Vec<Tensor>,
    pre_acts: Vec<Tensor>,
    last_probs: Option<Tensor>,
    last_labels: Option<Vec<usize>>,
    velocities: Vec<Vec<f32>>,
    bias_velocities: Vec<Vec<f32>>,
}

impl Elis {
    pub fn new(dims: &[usize], seed: u64) -> Self {
        assert!(dims.len() >= 2);
        let mut rng = Rng::new(seed);
        let mut weights = Vec::new();
        let mut grad_weights = Vec::new();
        let mut biases = Vec::new();
        let mut grad_biases = Vec::new();

        for i in 0..dims.len() - 1 {
            let din = dims[i];
            let dout = dims[i + 1];
            // Glorot / Xavier uniform init
            let limit = (6.0 / (din + dout) as f32).sqrt();
            let data: Vec<f32> = (0..din * dout)
                .map(|_| (rng.next_f32() * 2.0 - 1.0) * limit)
                .collect();
            weights.push(Tensor::new(din, dout, data));
            grad_weights.push(Tensor::new(din, dout, vec![0.0; din * dout]));
            biases.push(vec![0.0f32; dout]);
            grad_biases.push(vec![0.0f32; dout]);
        }

        Self {
            dims: dims.to_vec(),
            weights,
            grad_weights,
            biases,
            grad_biases,
            activations: Vec::new(),
            pre_acts: Vec::new(),
            last_probs: None,
            last_labels: None,
            velocities: Vec::new(),
            bias_velocities: Vec::new(),
        }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn weight_shape(&self, layer: usize) -> (usize, usize) {
        (self.weights[layer].rows, self.weights[layer].cols)
    }

    pub fn grad_weight(&self, layer: usize, r: usize, c: usize) -> f32 {
        self.grad_weights[layer].get(r, c)
    }

    pub fn perturb_weight(&mut self, layer: usize, r: usize, c: usize, delta: f32) {
        let cols = self.weights[layer].cols;
        self.weights[layer].data[r * cols + c] += delta;
    }

    pub fn forward(&mut self, x: &Tensor, labels: &[usize]) -> f32 {
        let m = x.rows;
        assert_eq!(x.cols, self.dims[0]);
        assert_eq!(labels.len(), m);

        self.activations.clear();
        self.pre_acts.clear();
        self.activations.push(x.clone());

        let num_layers = self.weights.len();
        for l in 0..num_layers {
            let cur = &self.activations[l];
            let w = &self.weights[l];
            let b = &self.biases[l];
            let din = w.rows;
            let dout = w.cols;

            let mut z = vec![0.0f32; m * dout];
            crate::accel::gemm_forward(&cur.data, &w.data, b, &mut z, m, din, dout);

            self.pre_acts.push(Tensor::new(m, dout, z.clone()));

            if l + 1 < num_layers {
                // tanh activation for hidden layers
                let a: Vec<f32> = z.iter().map(|&v| v.tanh()).collect();
                self.activations.push(Tensor::new(m, dout, a));
            } else {
                // Logits layer
                self.activations.push(Tensor::new(m, dout, z));
            }
        }

        // Softmax + Cross Entropy Loss
        let logits = self.activations.last().unwrap();
        let num_classes = *self.dims.last().unwrap();
        let mut probs = vec![0.0f32; m * num_classes];
        let mut total_loss = 0.0f32;

        for i in 0..m {
            let row = &logits.data[i * num_classes..(i + 1) * num_classes];
            let max_logit = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for j in 0..num_classes {
                let p = (row[j] - max_logit).exp();
                probs[i * num_classes + j] = p;
                sum_exp += p;
            }
            for j in 0..num_classes {
                probs[i * num_classes + j] /= sum_exp;
            }
            let target = labels[i];
            let p_target = probs[i * num_classes + target].max(1e-12);
            total_loss -= p_target.ln();
        }

        self.last_probs = Some(Tensor::new(m, num_classes, probs));
        self.last_labels = Some(labels.to_vec());

        total_loss / (m as f32)
    }

    pub fn backward(&mut self) {
        let m = self.activations[0].rows;
        let num_layers = self.weights.len();
        let probs = self.last_probs.as_ref().unwrap();
        let labels = self.last_labels.as_ref().unwrap();
        let num_classes = *self.dims.last().unwrap();

        // dL/dz for output logits: (p - y) / m
        let mut dz = probs.data.clone();
        for i in 0..m {
            let t = labels[i];
            dz[i * num_classes + t] -= 1.0;
            for j in 0..num_classes {
                dz[i * num_classes + j] /= m as f32;
            }
        }

        let mut delta = dz;

        for l in (0..num_layers).rev() {
            let a_prev = &self.activations[l];
            let din = self.weights[l].rows;
            let dout = self.weights[l].cols;

            // dW[k, j] = sum_i ( a_prev[i, k] * delta[i, j] )
            for k in 0..din {
                for j in 0..dout {
                    let mut sum = 0.0f32;
                    for i in 0..m {
                        sum += a_prev.data[i * din + k] * delta[i * dout + j];
                    }
                    self.grad_weights[l].data[k * dout + j] = sum;
                }
            }

            // db[j] = sum_i delta[i, j]
            for j in 0..dout {
                let mut sum = 0.0f32;
                for i in 0..m {
                    sum += delta[i * dout + j];
                }
                self.grad_biases[l][j] = sum;
            }

            if l > 0 {
                let din_prev = self.weights[l - 1].cols; // == din
                let mut prev_delta = vec![0.0f32; m * din_prev];
                let w = &self.weights[l];

                for i in 0..m {
                    for k in 0..din_prev {
                        let mut sum = 0.0f32;
                        for j in 0..dout {
                            sum += delta[i * dout + j] * w.data[k * dout + j];
                        }
                        // tanh derivative: 1 - tanh(z)^2 = 1 - a^2
                        let a_val = self.activations[l].data[i * din + k];
                        prev_delta[i * din_prev + k] = sum * (1.0 - a_val * a_val);
                    }
                }
                delta = prev_delta;
            }
        }
    }

    pub fn sgd_step(&mut self, lr: f32) {
        for l in 0..self.weights.len() {
            for idx in 0..self.weights[l].data.len() {
                self.weights[l].data[idx] -= lr * self.grad_weights[l].data[idx];
            }
            for j in 0..self.biases[l].len() {
                self.biases[l][j] -= lr * self.grad_biases[l][j];
            }
        }
    }

    pub fn sgd_step_momentum(&mut self, lr: f32) {
        if self.velocities.is_empty() {
            self.velocities = self
                .weights
                .iter()
                .map(|w| vec![0.0f32; w.data.len()])
                .collect();
            self.bias_velocities = self
                .biases
                .iter()
                .map(|b| vec![0.0f32; b.len()])
                .collect();
        }
        let beta = 0.9f32;
        for l in 0..self.weights.len() {
            for idx in 0..self.weights[l].data.len() {
                let g = self.grad_weights[l].data[idx];
                let v = beta * self.velocities[l][idx] + (1.0 - beta) * g;
                self.velocities[l][idx] = v;
                self.weights[l].data[idx] -= lr * v;
            }
            for j in 0..self.biases[l].len() {
                let g = self.grad_biases[l][j];
                let v = beta * self.bias_velocities[l][j] + (1.0 - beta) * g;
                self.bias_velocities[l][j] = v;
                self.biases[l][j] -= lr * v;
            }
        }
    }

    pub fn predict(&mut self, x: &Tensor) -> Vec<usize> {
        let m = x.rows;
        let dummy_labels = vec![0usize; m];
        self.forward(x, &dummy_labels);
        let probs = self.last_probs.as_ref().unwrap();
        let num_classes = *self.dims.last().unwrap();

        let mut preds = Vec::with_capacity(m);
        for i in 0..m {
            let row = &probs.data[i * num_classes..(i + 1) * num_classes];
            let mut best_class = 0;
            let mut max_val = row[0];
            for (c, &val) in row.iter().enumerate().skip(1) {
                if val > max_val {
                    max_val = val;
                    best_class = c;
                }
            }
            preds.push(best_class);
        }
        preds
    }
}
