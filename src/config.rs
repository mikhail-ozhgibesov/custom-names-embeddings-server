//! Configuration: env vars, with a thread-budget warning.
//!
//! Every results-affecting knob lives here and is logged at startup so a run is
//! fully described by its environment. Prefix `CT4_` — the served model is the
//! CT4 names-encoder, checkpoint h016 (step 84 000).

use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineKind {
    /// Hand-written f32 CT4 forward over repacked weights (default).
    Native,
    /// ONNX Runtime reference implementation (A/B baseline).
    Ort,
}

impl EngineKind {
    pub fn from_str(s: &str) -> anyhow::Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "native" => Ok(EngineKind::Native),
            "ort" => Ok(EngineKind::Ort),
            other => anyhow::bail!("unknown engine {other:?} (expected `native` or `ort`)"),
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            EngineKind::Native => "native",
            EngineKind::Ort => "ort",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    /// Dir holding `model.onnx` + `weights.bin` + `weights_manifest.json` +
    /// `tokenizer.json`, built by `scripts/build_model_dir.py`.
    pub model_dir: PathBuf,
    /// Inference backend: native CT4 forward or ONNX Runtime.
    pub engine: EngineKind,
    /// Inference workers. Native: one per physical core by default (they share
    /// ONE `Arc<Weights>` copy and run data-parallel). ORT: always 1
    /// (`Session::run` needs `&mut`; a second session would duplicate weights).
    pub workers: usize,
    /// Name reported by the OpenAI-compatible `model` response field.
    pub model_name: String,
    pub port: u16,
    pub max_batch: usize,
    pub batch_timeout_ms: u64,
    /// Bounded request-queue size; the backpressure threshold.
    pub queue_capacity: usize,
    pub ort_intra_threads: usize,
    pub ort_inter_threads: usize,
    /// Truncation length in tokens. CT4's positional table is learned up to
    /// 64 (`CT4Config.max_len`) — do not raise without re-exporting.
    pub max_seq_len: usize,
    /// tokio HTTP runtime worker threads (kept small; mostly I/O-bound).
    pub http_threads: usize,
}

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        let model_dir: PathBuf = std::env::var("CT4_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("./model"));
        let physical = num_cpus::get_physical().max(1);
        let engine = EngineKind::from_str(
            &std::env::var("CT4_ENGINE").unwrap_or_else(|_| "native".to_string()),
        )?;
        let workers_default = match engine {
            EngineKind::Native => physical,
            EngineKind::Ort => 1,
        };
        let mut cfg = Config {
            model_dir,
            engine,
            workers: env_or("CT4_WORKERS", workers_default),
            model_name: std::env::var("CT4_MODEL_NAME").unwrap_or_else(|_| "ct4-h016".to_string()),
            port: env_or("CT4_PORT", 8080),
            max_batch: env_or("CT4_MAX_BATCH", 32),
            batch_timeout_ms: env_or("CT4_BATCH_TIMEOUT_MS", 0),
            queue_capacity: env_or("CT4_QUEUE_CAPACITY", 1024),
            ort_intra_threads: env_or("CT4_ORT_INTRA_THREADS", physical),
            ort_inter_threads: env_or("CT4_ORT_INTER_THREADS", 1),
            max_seq_len: env_or("CT4_MAX_SEQ_LEN", 64),
            http_threads: env_or("CT4_HTTP_THREADS", 4usize.min(physical)),
        };
        if cfg.max_seq_len > 64 {
            tracing::warn!(
                max_seq_len = cfg.max_seq_len,
                "CT4's learned positional table covers 64 positions; longer inputs would misindex — clamping to 64"
            );
            cfg.max_seq_len = 64;
        }
        if cfg.engine == EngineKind::Ort && cfg.workers > 1 {
            tracing::warn!(
                workers = cfg.workers,
                "CT4_WORKERS>1 ignored for the ORT engine (Session::run is &mut; extra sessions would duplicate weights). Native engine supports a worker pool."
            );
            cfg.workers = 1;
        }
        cfg.check_thread_budget(physical);
        Ok(cfg)
    }

    /// Total inference threads must stay <= physical cores; oversubscription
    /// destroys tail latency.
    fn check_thread_budget(&self, physical: usize) {
        let total = match self.engine {
            // native: 1 thread per worker; ort: intra-op pool per (single) session
            EngineKind::Native => self.workers,
            EngineKind::Ort => self.workers.saturating_mul(self.ort_intra_threads),
        };
        if total > physical {
            tracing::warn!(
                engine = self.engine.as_str(),
                workers = self.workers,
                ort_intra_threads = self.ort_intra_threads,
                total_inference_threads = total,
                physical_cores = physical,
                "thread budget exceeds physical cores — expect tail-latency degradation"
            );
        }
    }

    pub fn model_onnx(&self) -> PathBuf { self.model_dir.join("model.onnx") }
    pub fn weights_bin(&self) -> PathBuf { self.model_dir.join("weights.bin") }
    pub fn weights_manifest(&self) -> PathBuf { self.model_dir.join("weights_manifest.json") }
    pub fn tokenizer_json(&self) -> PathBuf { self.model_dir.join("tokenizer.json") }
}
