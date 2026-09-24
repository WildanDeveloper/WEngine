use std::time::Instant;
use wildandev_core::tokenizer::WildandevTokenizer;
use wildandev_core::{ElisTransformer, WildandevConfig};

fn main() {
    println!("WEngine: WildanDev Elis AI Model Trainer");

    // Corpus training: kalimat bahasa Indonesia & Inggris tentang WildanDev & Elis
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

    println!("[1/4] Training BPE Tokenizer dari corpus...");
    let tokenizer = WildandevTokenizer::train(corpus, 320);
    println!("      Vocab size: {} tokens", tokenizer.vocab_size());
    println!("      Merges:     {} pairs", tokenizer.merges_count());

    let token_ids = tokenizer.encode(corpus);
    println!("      Corpus encoded: {} tokens", token_ids.len());

    println!("\n[2/4] Initializing ElisTransformer...");
    let cfg = WildandevConfig {
        vocab_size: tokenizer.vocab_size(),
        d_model: 128,
        n_heads: 4,
        n_layers: 2,
        d_ff: 256,
        seq_len: 32,
    };
    let mut model = ElisTransformer::new(cfg, 42);
    println!("      Model parameters: {}", model.param_count());
    println!("      Memory RAM:       {:.2} KB", model.memory_bytes() as f64 / 1024.0);

    println!("\n[3/4] Mulai Training (Autoregressive Next-Token Prediction)...");
    let seq_len = 16;
    let epochs = 300;
    let t0 = Instant::now();

    for epoch in 1..=epochs {
        let mut total_loss = 0.0;
        let mut batches = 0;

        for start in (0..token_ids.len().saturating_sub(seq_len + 1)).step_by(seq_len) {
            let inputs = &token_ids[start..start + seq_len];
            let targets = &token_ids[start + 1..start + seq_len + 1];
            let loss = model.compute_loss_and_grads(inputs, targets);
            total_loss += loss;
            batches += 1;
        }

        if epoch % 50 == 0 || epoch == 1 {
            let avg_loss = if batches > 0 { total_loss / batches as f32 } else { 0.0 };
            println!("      Epoch {:3}/{} | Loss: {:.4}", epoch, epochs, avg_loss);
        }
    }
    println!("      Training selesai dalam {:?}", t0.elapsed());

    println!("\n[4/4] Menyimpan Checkpoint...");
    let ckpt_path = "/root/WEngine/elis_checkpoint.bin";
    model.save_checkpoint(ckpt_path).expect("Failed saving checkpoint");
    println!("      Checkpoint tersimpan di {}", ckpt_path);

    // Test load kembali
    let loaded = ElisTransformer::load_checkpoint(ckpt_path).expect("Failed loading checkpoint");
    println!("      Verifikasi checkpoint load: OK (params: {})", loaded.param_count());

    // Generate teks hasil training
    println!("\n[Sample Output Hasil Belajar]");
    let prompt = "wildandev membuat ";
    let mut gen_tokens = tokenizer.encode(prompt);
    print!("Prompt: \"{}\"\nHasil:  {}", prompt, prompt);

    let v = model.cfg.vocab_size;
    for _ in 0..15 {
        if gen_tokens.len() >= 32 { break; }
        let logits = model.forward(&gen_tokens);
        let last_idx = gen_tokens.len() - 1;
        let row = &logits[last_idx * v..(last_idx + 1) * v];

        let mut best_t = 0;
        let mut best_val = row[0];
        for (t, &val) in row.iter().enumerate().skip(1) {
            if val > best_val {
                best_val = val;
                best_t = t;
            }
        }
        gen_tokens.push(best_t);
        print!("{}", tokenizer.decode(&[best_t]));
    }
    println!();
}
