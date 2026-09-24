use wildandev_core::{ElisGrads, ElisTransformer, WildandevConfig};

#[test]
fn test_transformer_loss_computation() {
    let cfg = WildandevConfig {
        vocab_size: 16,
        d_model: 32,
        n_heads: 2,
        n_layers: 1,
        d_ff: 64,
        seq_len: 8,
    };
    let mut model = ElisTransformer::new(cfg.clone(), 1234);
    let inputs = vec![1, 2, 3, 4];
    let targets = vec![2, 3, 4, 5];

    let cache = model.forward_train(&inputs);
    let mut grads = ElisGrads::zero(&cfg);
    let loss = model.backward_train(&cache, &targets, &mut grads);

    // Initial cross entropy for 16 classes is approx -ln(1/16) = 2.77
    assert!(loss > 2.0 && loss < 4.0, "Unexpected loss: {loss}");

    // Check that grads are actually populated (non-zero)
    let sum_abs = |arr: &[f32]| arr.iter().map(|v| v.abs()).sum::<f32>();
    assert!(sum_abs(&grads.d_lm_head) > 0.0, "d_lm_head is all zeros!");
    assert!(sum_abs(&grads.d_w_down[0]) > 0.0, "d_w_down is all zeros!");
    assert!(sum_abs(&grads.d_w_gate[0]) > 0.0, "d_w_gate is all zeros!");
    assert!(sum_abs(&grads.d_w_o[0]) > 0.0, "d_w_o is all zeros!");
    assert!(sum_abs(&grads.d_w_q[0]) > 0.0, "d_w_q is all zeros!");
    assert!(sum_abs(&grads.d_token_emb) > 0.0, "d_token_emb is all zeros!");
}

#[test]
fn test_transformer_convergence() {
    // Overfit on single 4-token sequence: loss must drop strictly
    let cfg = WildandevConfig {
        vocab_size: 8,
        d_model: 32,
        n_heads: 2,
        n_layers: 1,
        d_ff: 64,
        seq_len: 8,
    };
    let mut model = ElisTransformer::new(cfg.clone(), 777);
    let mut opt = wildandev_core::TransformerOptimizers::new(&model, 0.05);

    let inputs = vec![1, 2, 3, 4];
    let targets = vec![2, 3, 4, 5];

    let cache0 = model.forward_train(&inputs);
    let mut grads0 = ElisGrads::zero(&cfg);
    let initial_loss = model.backward_train(&cache0, &targets, &mut grads0);

    let mut last_loss = initial_loss;
    for _ in 0..100 {
        let cache = model.forward_train(&inputs);
        let mut grads = ElisGrads::zero(&cfg);
        last_loss = model.backward_train(&cache, &targets, &mut grads);
        grads.clip_norm(1.0);
        opt.step(&mut model, &grads);
    }

    assert!(
        last_loss < initial_loss * 0.2,
        "Transformer did not overfit: initial {initial_loss} vs final {last_loss}"
    );
}
