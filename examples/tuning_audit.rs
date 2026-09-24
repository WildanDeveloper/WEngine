use std::time::Instant;
use wildandev_core::rng::Rng;
use wildandev_core::tensor::Tensor;
use wildandev_core::Elis;

fn main() {
    // 1. Profiling Elis MLP murni - cek overhead per-step
    println!("=== Audit 1: Elis MLP overhead ===");
    let x = Tensor::new(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
    let labels = vec![0usize, 1, 1, 0];
    let mut elis = Elis::new(&[2, 8, 2], 1234);
    let _ = elis.forward(&x, &labels); // warmup
    let t0 = Instant::now();
    let mut loss = 0.0;
    for _ in 0..100000 {
        loss = elis.forward(&x, &labels);
    }
    println!("100k forwards (m=4, 2-8-2): {:?} = {:.2} us/op", t0.elapsed(), t0.elapsed().as_secs_f64() * 1e6 / 100000.0);

    let t0 = Instant::now();
    for _ in 0..100000 {
        elis.backward();
    }
    println!("100k backwards:              {:?} = {:.2} us/op", t0.elapsed(), t0.elapsed().as_secs_f64() * 1e6 / 100000.0);

    // 2. Cek alokasi per-forward di ElisTransformer
    println!("\n=== Audit 2: ElisTransformer alokasi ===");
    let cfg = wildandev_core::WildandevConfig {
        vocab_size: 256,
        d_model: 128,
        n_heads: 4,
        n_layers: 2,
        d_ff: 256,
        seq_len: 32,
    };
    let mut tr = wildandev_core::ElisTransformer::new(cfg, 42);
    let prompt: Vec<usize> = (0..32).map(|i| i % 256).collect();
    let _ = tr.forward(&prompt);
    let t0 = Instant::now();
    for _ in 0..1000 {
        let _ = tr.forward(&prompt);
    }
    let dt = t0.elapsed();
    println!("1000 forwards (seq=32, d=128, 2L): {:?} = {:.1} us/op", dt, dt.as_secs_f64() * 1e6 / 1000.0);
    println!("final MLP loss sanity: {:.6}", loss);

    // 3. Memory footprint kasar
    let params = 256*128 + 32*128 + 2*(128*128*4 + 128*256*3 + 256*128) + 128*256;
    println!("\n=== Audit 3: ukuran model ===");
    println!("~{} parameter f32 = {:.1} KB RAM", params, params as f64 * 4.0 / 1024.0);
}
