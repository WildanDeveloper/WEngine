use std::time::Instant;
use wildandev_core::{ElisTransformer, WildandevConfig};

fn main() {
    println!("============================================================");
    println!("   WEngine: Elis-15M Scale-Up Benchmark (AVX-512 + 4 Cores)");
    println!("============================================================");

    // Elis-15M: d_model=512, 8 heads, 6 layers, d_ff=1024, seq_len=64
    let cfg = WildandevConfig {
        vocab_size: 256,
        d_model: 512,
        n_heads: 8,
        n_layers: 6,
        d_ff: 1024,
        seq_len: 64,
    };

    println!("Initializing model Elis-15M...");
    let mut model = ElisTransformer::new(cfg, 42);
    let params = model.param_count();
    let ram_mb = model.memory_bytes() as f64 / (1024.0 * 1024.0);

    println!("Parameter count:     {:.2} Juta ({})", params as f64 / 1e6, params);
    println!("RAM footprint:       {:.2} MB", ram_mb);
    println!("Active CPU threads:  {}", std::thread::available_parallelism().unwrap().get());

    let prompt: Vec<usize> = (0..64).map(|i| (i * 11 + 5) % 256).collect();

    // Warmup
    print!("Warming up...");
    for _ in 0..10 {
        let _ = model.forward(&prompt);
    }
    println!(" done.\n");

    let iters = 200;
    println!("Running benchmark ({} iterations, seq=64)...", iters);
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = model.forward(&prompt);
    }
    let elapsed = t0.elapsed();

    let per_forward_ms = (elapsed.as_secs_f64() * 1000.0) / iters as f64;
    let tokens_per_sec = (iters * 64) as f64 / elapsed.as_secs_f64();
    let gflops_approx = (iters as f64 * (2.0 * params as f64 * 64.0)) / (elapsed.as_secs_f64() * 1e9);

    println!("------------------------------------------------------------");
    println!("Latency per forward: {:.2} ms", per_forward_ms);
    println!("Throughput:          {:.0} tokens / detik di CPU!", tokens_per_sec);
    println!("Compute performance: {:.2} GFLOPS", gflops_approx);
    println!("------------------------------------------------------------");
}
