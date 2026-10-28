//! Entry point: load config → load engine (one weights copy) → warmup → start
//! the queue/scheduler/pool → serve HTTP → graceful shutdown that drains
//! in-flight.
//!
//! Native engine: `weights.bin` + tokenizer load ONCE; every worker gets an
//! `Engine` sharing the same `Arc<Weights>` (data-parallel, one weights copy).
//! ORT engine: a single session behind the scheduler (reference mode).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Context;

use custom_names_embeddings_server::batcher;
use custom_names_embeddings_server::config::{Config, EngineKind};
use custom_names_embeddings_server::engine::Engine;
use custom_names_embeddings_server::metrics::Metrics;
use custom_names_embeddings_server::server::{self, AppState};
use custom_names_embeddings_server::tokenize::NamesTokenizer;
use custom_names_embeddings_server::weights::Weights;

fn main() -> anyhow::Result<()> {
    init_tracing();
    let cfg = Config::from_env()?;

    let physical = num_cpus::get_physical();
    let logical = num_cpus::get();

    tracing::info!(
        engine = cfg.engine.as_str(),
        workers = cfg.workers,
        model_dir = %cfg.model_dir.display(),
        model_name = %cfg.model_name,
        max_batch = cfg.max_batch,
        batch_timeout_ms = cfg.batch_timeout_ms,
        queue_capacity = cfg.queue_capacity,
        max_seq_len = cfg.max_seq_len,
        ort_intra_threads = if cfg.engine == EngineKind::Ort { Some(cfg.ort_intra_threads) } else { None },
        ort_inter_threads = if cfg.engine == EngineKind::Ort { Some(cfg.ort_inter_threads) } else { None },
        physical_cores = physical,
        logical_cores = logical,
        "starting custom-names-embeddings-server (ct4-h016)"
    );

    let load_t0 = std::time::Instant::now();
    let engines: Vec<Engine> = match cfg.engine {
        EngineKind::Native => {
            // One weights copy; one Engine per worker (cheap, Arc-shared).
            let weights = Arc::new(Weights::load(&cfg.weights_bin(), &cfg.weights_manifest())?);
            let tok = NamesTokenizer::load(&cfg.tokenizer_json(), cfg.max_seq_len)?;
            (0..cfg.workers)
                .map(|_| Engine::native_from_shared(weights.clone(), tok.clone()))
                .collect()
        }
        EngineKind::Ort => {
            vec![Engine::load_ort(
                &cfg.model_onnx(),
                &cfg.tokenizer_json(),
                cfg.ort_intra_threads,
                cfg.ort_inter_threads,
                cfg.max_seq_len,
            )
            .context("engine load")?]
        }
    };
    let dim = engines[0].dim();
    tracing::info!(
        backend = engines[0].backend_name(),
        dim,
        workers = engines.len(),
        load_ms = load_t0.elapsed().as_millis() as u64,
        "engine ready"
    );

    // tokio runtime for the HTTP layer (kept small; inference is off-runtime).
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(cfg.http_threads.max(1))
        .enable_all()
        .build()?;

    rt.block_on(async move { serve(cfg, engines).await })
}

async fn serve(cfg: Config, engines: Vec<Engine>) -> anyhow::Result<()> {
    let metrics = Arc::new(Metrics::new()?);

    let bcfg = batcher::BatchConfig {
        max_batch: cfg.max_batch,
        batch_timeout_ms: cfg.batch_timeout_ms,
        queue_capacity: cfg.queue_capacity,
        // size × max_len budget: a full batch of short (≤32 tok) names.
        pad_cell_budget: cfg.max_batch * 32,
    };
    let submitter = batcher::start(engines, bcfg, metrics.clone());

    let state = Arc::new(AppState {
        submitter,
        metrics: metrics.clone(),
        ready: AtomicBool::new(false),
        model_name: cfg.model_name.clone(),
    });

    // A real warmup request through the full path before flipping /health.
    {
        if let Ok(rx) = state.submitter.submit("warmup".to_string()) {
            let _ = rx.await;
        }
        state.ready.store(true, Ordering::Relaxed);
        tracing::info!("warmup complete; /health is now 200");
    }

    let app = server::router(state.clone());
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], cfg.port));
    let listener = tokio::net::TcpListener::bind(addr).await.context("bind")?;
    tracing::info!(%addr, "listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("serve")?;
    tracing::info!("shutdown complete");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async { tokio::signal::ctrl_c().await.ok(); };
    #[cfg(unix)]
    let term = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = term => {},
    }
    tracing::info!("shutdown signal received; draining");
}

fn init_tracing() {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};
    // ORT bridges its very chatty allocator logs into tracing at INFO; quiet it.
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,ort::logging=warn"));
    tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().json())
        .init();
}
