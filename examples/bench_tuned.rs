use std::time::Instant;
use wildandev_core::{ElisTransformer, WildandevConfig};

fn main() {
    println!("=== wildandev ElisTransformer Tuned Benchmark ===");
    let cfg = WildandevConfig {
        vocab_size: 256,
        d_model: 128,
        n_heads: 4,
        n_layers: 2,
        d_ff: 256,
        seq_len: 32,
    };

    let mut model = ElisTransformer::new(cfg, 42);
    println!("Model params:        {}", model.param_count());
    println!("Memory footprint:    {:.1} KB", model.memory_bytes() as f64 / 1024.0);

    let prompt: Vec<usize> = (0..32).map(|i| (i * 7 + 13) % 256).collect();

    // Warmup
    for _ in 0..100 {
        let _ = model.forward(&prompt);
    }

    let iters = 5000;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = model.forward(&prompt);
    }
    let elapsed = t0.elapsed();
    let per_forward_us = elapsed.as_secs_f64() * 1e6 / iters as f64;
    let tokens_per_sec = (iters * 32) as f64 / elapsed.as_secs_f64();

    println!("Throughput:          {:.1} us/forward", per_forward_us);
    println!("Speed (seq=32):      {:.0} tokens/sec", tokens_per_sec);
}
