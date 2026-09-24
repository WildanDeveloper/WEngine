use wildandev_core::{ElisTransformer, WildandevConfig};

fn main() {
    
    println!("     WEngine - wildandev Elis AI Text Generator");
    

    let cfg = WildandevConfig {
        vocab_size: 256,
        d_model: 128,
        n_heads: 4,
        n_layers: 2,
        d_ff: 256,
        seq_len: 64,
    };

    let mut model = ElisTransformer::new(cfg, 1234);

    let prompt = "wildandev elis ";
    let mut tokens: Vec<usize> = prompt.bytes().map(|b| b as usize).collect();

    print!("Prompt: \"{}\"\nGenerating: {}", prompt, prompt);

    let v = model.cfg.vocab_size;
    // Greedy autoregressive sampling
    for _ in 0..30 {
        if tokens.len() >= 64 { break; }
        let logits = model.forward(&tokens);
        let last_idx = tokens.len() - 1;
        let row = &logits[last_idx * v..(last_idx + 1) * v];

        // argmax sampling (greedy)
        let mut best_t = 0;
        let mut best_val = row[0];
        for (t, &val) in row.iter().enumerate().skip(1) {
            if val > best_val {
                best_val = val;
                best_t = t;
            }
        }
        tokens.push(best_t);
        let ch = if best_t >= 32 && best_t <= 126 { best_t as u8 as char } else { '.' };
        print!("{ch}");
    }
    println!("\nDone.");
}
