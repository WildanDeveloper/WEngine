use std::time::Instant;
use wildandev_core::{ElisTransformer, WildandevConfig};

fn main() {
    println!("=== wildandev ElisTransformer Engine ===");

    // Arsitektur mini LLM (128 d_model, 4 heads, 2 layers, RoPE, RMSNorm, SwiGLU)
    let cfg = WildandevConfig {
        vocab_size: 256, // byte-level tokenizer
        d_model: 128,
        n_heads: 4,
        n_layers: 2,
        d_ff: 256,
        seq_len: 32,
    };

    let mut model = ElisTransformer::new(cfg, 42);
    let v = model.cfg.vocab_size;

    let prompt: Vec<usize> = "Halo wildandev!".bytes().map(|b| b as usize).collect();
    println!("Input tokens: {:?}", prompt);

    let t0 = Instant::now();
    let logits = model.forward(&prompt);
    let dt = t0.elapsed();

    println!("Forward pass finish in {:?}", dt);
    println!("Total output logits shape: ({} x {})", prompt.len(), v);
    println!("First 5 logits of first token: {:?}", &logits[..5]);
}
