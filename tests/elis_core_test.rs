use wildandev_core::rng::Rng;
use wildandev_core::tensor::Tensor;
use wildandev_core::Elis;

#[test]
fn test_gradient_check() {
    let mut rng = Rng::new(42);
    let (m, din, dout) = (5usize, 3usize, 2usize);
    let mut net = Elis::new(&[din, 4, dout], 7);
    let x = Tensor::new(m, din, (0..m * din).map(|_| rng.next_f32() - 0.5).collect());
    let labels: Vec<usize> = (0..m).map(|i| (rng.next_u64() as usize + i) % dout).collect();

    net.forward(&x, &labels);
    net.backward();

    let h = 1e-2;
    for layer in 0..net.dims().len() - 1 {
        let (rows, cols) = net.weight_shape(layer);
        for r in 0..rows {
            for c in 0..cols {
                let g = net.grad_weight(layer, r, c);
                net.perturb_weight(layer, r, c, h);
                let lp = net.forward(&x, &labels);
                net.perturb_weight(layer, r, c, -2.0 * h);
                let lm = net.forward(&x, &labels);
                net.perturb_weight(layer, r, c, h);
                let num = (lp - lm) / (2.0 * h);
                let err = (num - g).abs() / (num.abs() + g.abs() + 1e-8);
                assert!(
                    err < 2e-2,
                    "gradient mismatch: layer {layer} ({r},{c}) analytic {g} vs numeric {num}"
                );
            }
        }
    }
}

#[test]
fn test_xor_convergence() {
    let x = Tensor::new(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
    let labels = vec![0usize, 1, 1, 0];
    let mut elis = Elis::new(&[2, 8, 2], 1234);

    let mut loss = f32::INFINITY;
    for _ in 0..3000 {
        loss = elis.forward(&x, &labels);
        elis.backward();
        elis.sgd_step(0.5);
    }
    assert!(loss < 0.05, "XOR training failed: final loss {loss}");
    assert_eq!(elis.predict(&x), vec![0, 1, 1, 0]);
}
