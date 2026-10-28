//! Single-thread engine microbench: native vs ORT at the workload's real
//! shapes. Not a test — run manually:
//!   cargo run --release --example bench_engines -- [native|ort|both] [reps]
//!
//! Uses names from names_5k.json (falling back to a built-in battery) so token
//! lengths match production (~3.4 mean, ≤8).

use std::path::PathBuf;
use std::time::Instant;

use custom_names_embeddings_server::engine::Engine;

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "both".into());
    let reps: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(100);

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let md = std::env::var("CT4_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest.join("model"));

    let names: Vec<String> = std::fs::read_to_string(manifest.join("names_5k.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| {
            vec![
                "Vladimir Putin".into(),
                "Владимир Путин".into(),
                "Иванов Иван".into(),
                "毛泽东".into(),
                "Müller".into(),
            ]
        });

    let run = |label: &str, engine: &Engine| {
        println!("\n=== {label} (dim {}) ===", engine.dim());
        for &b in &[1usize, 8, 32, 64, 256] {
            let batch: Vec<String> = (0..b).map(|i| names[i % names.len()].clone()).collect();
            // warmup
            for _ in 0..3 {
                let ids = engine.tokenizer().encode_ids(&batch).unwrap();
                let _ = engine.infer_ids(&ids).unwrap();
            }
            let ids = engine.tokenizer().encode_ids(&batch).unwrap();
            let tokens: usize = ids.iter().map(|v| v.len()).sum();
            let t0 = Instant::now();
            let mut last = 0.0f64;
            for _ in 0..reps {
                let (_v, t) = engine.infer_ids(&ids).unwrap();
                last = t.inference_s;
            }
            let per = t0.elapsed().as_secs_f64() / reps as f64;
            println!(
                "  B={b:3} tokens={tokens:4}  {:9.3} ms  → {:8.0} names/s",
                per * 1e3,
                b as f64 / per
            );
        }
    };

    if which == "native" || which == "both" {
        let e = Engine::load_native(&md, &md.join("tokenizer.json"), 64).expect("native");
        run("native", &e);
    }
    if which == "ort" || which == "both" {
        let e = Engine::load_ort(
            &md.join("model.onnx"),
            &md.join("tokenizer.json"),
            std::env::var("CT4_ORT_INTRA_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(1),
            1,
            64,
        )
        .expect("ort");
        run("ort (1 thread)", &e);
    }
}
