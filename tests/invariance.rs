//! Native-engine properties beyond Gate A (NATIVE_RUST_PLAN §4):
//!  * **bit-exact batch-invariance** — a row's vector is IDENTICAL whether
//!    embedded alone or inside any ragged batch (the packed layout must make
//!    the training repo's `x * keep` discipline structural);
//!  * **determinism** — same inputs → identical bits across runs;
//!  * **cross-engine agreement** — native vs ORT ≤ 1e-5 on a stress battery;
//!  * **edge rows** — empty string (pad fallback), single char, 64-token
//!    truncation: finite, unit-norm.
//!
//! Skips with a notice when the model dir is absent.

use std::path::PathBuf;

use custom_names_embeddings_server::engine::Engine;

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn model_dir() -> PathBuf {
    std::env::var("CT4_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest().join("model"))
}

fn native() -> Option<Engine> {
    let md = model_dir();
    if md.join("weights.bin").exists() {
        Some(Engine::load_native(&md, &md.join("tokenizer.json"), 64).expect("native engine load"))
    } else {
        eprintln!("SKIP: no model dir with weights.bin at {}", md.display());
        None
    }
}

fn embed(engine: &Engine, texts: &[String]) -> Vec<Vec<f32>> {
    let ids = engine.tokenizer().encode_ids(texts).expect("encode");
    engine.infer_ids(&ids).expect("infer").0
}

const BATTERY: &[&str] = &[
    "Иванов Иван",
    "Q",
    "",
    "Vladimir Putin",
    "Владимир Путин",
    "محمد بن سلمان",
    "毛泽东",
    "Müller-Žarković",
];

#[test]
fn native_bit_exact_batch_invariance() {
    let Some(engine) = native() else { return };
    // Long outlier forces 64-token truncation; rows of every length mix.
    let ragged: Vec<String> = BATTERY
        .iter()
        .map(|s| s.to_string())
        .chain([format!("А{}", "б".repeat(400)), "Gazprom 000".to_string()])
        .collect();

    let batch_vecs = embed(&engine, &ragged);
    for (i, text) in ragged.iter().enumerate() {
        let alone = embed(&engine, &[text.clone()]).remove(0);
        let in_batch = batch_vecs[i].clone();
        assert_eq!(
            alone, in_batch,
            "row {i} ({text:?}) differs between alone and ragged-batch embedding"
        );
    }
}

#[test]
fn native_determinism() {
    let Some(engine) = native() else { return };
    let texts: Vec<String> = BATTERY.iter().map(|s| s.to_string()).collect();
    let a = embed(&engine, &texts);
    let b = embed(&engine, &texts);
    // Also through a differently-shaped call (chunked in twos) — rows must be
    // bit-identical regardless of batch composition.
    let mut c = Vec::new();
    for pair in texts.chunks(2) {
        c.extend(embed(&engine, &pair.to_vec()));
    }
    assert_eq!(a, b, "same call twice differs");
    assert_eq!(a, c, "different chunking differs");
}

#[test]
fn native_edge_rows_finite_unit_norm() {
    let Some(engine) = native() else { return };
    let texts: Vec<String> = BATTERY.iter().map(|s| s.to_string()).collect();
    let long = format!("А{}", "б".repeat(400)); // >64 tokens → truncated
    let mut all = texts.clone();
    all.push(long);
    for (i, v) in embed(&engine, &all).iter().enumerate() {
        assert_eq!(v.len(), 256, "dim");
        assert!(v.iter().all(|x| x.is_finite()), "non-finite at row {i} ({:?})", all[i]);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "row {i} norm {norm}");
    }
}

#[test]
fn native_vs_ort_cross_engine() {
    let md = model_dir();
    if !md.join("model.onnx").exists() || !md.join("weights.bin").exists() {
        eprintln!("SKIP: need both model.onnx and weights.bin in {}", md.display());
        return;
    }
    let nat = Engine::load_native(&md, &md.join("tokenizer.json"), 64).expect("native");
    let ort = Engine::load_ort(
        &md.join("model.onnx"),
        &md.join("tokenizer.json"),
        std::env::var("CT4_ORT_INTRA_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(8),
        1,
        64,
    )
    .expect("ort");

    // Stress battery: scripts/gen_names sample + edge cases, chunked in 37s to
    // vary batch shapes against the ORT fixed-32 path.
    let mut texts: Vec<String> = BATTERY.iter().map(|s| s.to_string()).collect();
    if let Ok(names) = std::fs::read_to_string(manifest().join("names_5k.json")) {
        if let Ok(list) = serde_json::from_str::<Vec<String>>(&names) {
            texts.extend(list.into_iter().take(1000));
        }
    }
    texts.push(format!("А{}", "б".repeat(400)));

    let mut max_abs = 0.0f32;
    let mut min_cos = 1.0f32;
    let mut checked = 0usize;
    for chunk in texts.chunks(37) {
        let a = embed(&nat, &chunk.to_vec());
        let b = embed(&ort, &chunk.to_vec());
        for (va, vb) in a.iter().zip(b.iter()) {
            let mut dot = 0.0f32;
            for (x, y) in va.iter().zip(vb.iter()) {
                dot += x * y;
                max_abs = max_abs.max((x - y).abs());
            }
            min_cos = min_cos.min(dot);
            checked += 1;
        }
    }
    println!("cross-engine: N={checked} max|Δ|={max_abs:.3e} min_cos={min_cos:.8}", );
    assert!(checked > 0);
    assert!(max_abs <= 1e-5, "native vs ort max|Δ| {max_abs:.2e} > 1e-5");
    assert!(min_cos >= 0.99999, "native vs ort min cos {min_cos}");
}
