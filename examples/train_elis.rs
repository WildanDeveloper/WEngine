use std::time::Instant;
use wildandev_core::optimizer::TransformerOptimizers;
use wildandev_core::tokenizer::WildandevTokenizer;
use wildandev_core::{ElisGrads, ElisTransformer, WildandevConfig};

fn main() {
    println!("WEngine: WildanDev Elis AI Model Trainer (full backprop + AdamW)");

    let corpus = r#"
wildandev membuat kecerdasan buatan bernama elis.
elis adalah ai engine yang sangat ringan dan cepat di cpu.
wengine dirancang dengan rust dan c++ menggunakan instruksi avx-512.
elis bisa berpikir dan menghasilkan teks secara mandiri.
wildandev mengembangkan elis agar hemat memori dan bisa berjalan di server manapun.
arsitektur elis berbasis transformer rope swiglu dan rmsnorm.
kecepatan inferensi elis mencapai ribuan token per detik di cpu.
wildandev elis ai engine masa depan.
"#;

    println!("[1/4] Training BPE Tokenizer...");
    let tokenizer = WildandevTokenizer::train(corpus, 320);
    println!("      Vocab: {} tokens | Merges: {}", tokenizer.vocab_size(), tokenizer.merges_count());

    let token_ids = tokenizer.encode(corpus);
    println!("      Corpus: {} tokens", token_ids.len());

    println!("[2/4] ElisTransformer init...");
    let cfg = WildandevConfig {
        vocab_size: tokenizer.vocab_size(),
        d_model: 128,
        n_heads: 4,
        n_layers: 2,
        d_ff: 256,
        seq_len: 24,
    };
    let mut model = ElisTransformer::new(cfg.clone(), 42);
    let mut opt = TransformerOptimizers::new(&model, 1e-3);
    println!("      Params: {} | RAM: {:.2} KB", model.param_count(), model.memory_bytes() as f64 / 1024.0);

    let seq_len = 16;
    let epochs = 500;

    // Data windows
    let mut windows: Vec<(usize, usize)> = (0..token_ids.len().saturating_sub(seq_len + 1))
        .step_by(seq_len / 2)
        .map(|s| (s, s + seq_len))
        .collect();
    if windows.is_empty() {
        windows.push((0, seq_len));
    }

    let total_steps = epochs * windows.len();
    let warmup = (total_steps as f32 * 0.05).max(20.0) as usize;
    let base_lr = 3e-3f32;

    println!("[3/4] Training {} epochs x {} windows (AdamW + cosine schedule)...", epochs, windows.len());
    let t0 = Instant::now();
    let mut global_step = 0usize;
    let mut shown_loss = 0.0f32;
    let mut current_lr = base_lr;

    for epoch in 1..=epochs {
        let mut epoch_loss = 0.0f32;
        let mut nb = 0usize;

        for &(start, end) in &windows {
            let inputs: Vec<usize> = token_ids[start..end].to_vec();
            let targets: Vec<usize> = token_ids[start + 1..end + 1.min(token_ids.len())].to_vec();
            if targets.len() < inputs.len() { continue; }

            // LR schedule: linear warmup + cosine decay
            let lr = if global_step < warmup {
                base_lr * (global_step as f32 + 1.0) / warmup as f32
            } else {
                let progress = (global_step - warmup) as f32 / ((total_steps - warmup).max(1) as f32);
                base_lr * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos()) + 1e-5
            };
            current_lr = lr;
            opt.set_lr(lr);

            let cache = model.forward_train(&inputs);
            let mut grads = ElisGrads::zero(&cfg);
            let loss = model.backward_train(&cache, &targets, &mut grads);
            grads.clip_norm(1.0);
            opt.step(&mut model, &grads);

            epoch_loss += loss;
            nb += 1;
            global_step += 1;
        }

        if epoch % 100 == 0 || epoch == 1 {
            shown_loss = epoch_loss / nb.max(1) as f32;
            println!("      Epoch {:4}/{} | Loss: {:.4} | LR: {:.5}", epoch, epochs, shown_loss, current_lr);
        }
    }
    let dt = t0.elapsed();
    println!("      Selesai dalam {:.1}s | Final loss: {:.4}", dt.as_secs_f64(), shown_loss);

    println!("[4/4] Checkpoint...");
    model.save_checkpoint("/root/WEngine/elis_checkpoint.bin").expect("save gagal");
    println!("      Saved: elis_checkpoint.bin");

    // Generate
    println!();
    println!("Sample Output:");
    let prompt = "wildandev";
    let mut gen_tokens = tokenizer.encode(prompt);
    print!("  \"{}\" -> \"{}{}", prompt, prompt, "");

    let v = model.cfg.vocab_size;
    for _ in 0..24 {
        if gen_tokens.len() >= 24 { break; }
        let logits = model.forward(&gen_tokens);
        let last = gen_tokens.len() - 1;
        let row = &logits[last * v..(last + 1) * v];
        let (mut best_t, mut best_val) = (0usize, row[0]);
        for (t, &val) in row.iter().enumerate().skip(1) {
            if val > best_val { best_val = val; best_t = t; }
        }
        gen_tokens.push(best_t);
        print!("{}", tokenizer.decode(&[best_t]));
    }
    println!("\"");
}
