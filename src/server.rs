//! HTTP layer (axum): OpenAI-compatible `POST /embeddings`, `GET /health`,
//! `GET /metrics`. The async handlers never block on inference — they submit to
//! the bounded queue and await a oneshot, so the executor stays responsive and
//! a full queue is shed as 503.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

use crate::batcher::Submitter;
use crate::metrics::Metrics;

pub struct AppState {
    pub submitter: Submitter,
    pub metrics: Arc<Metrics>,
    pub ready: AtomicBool,
    pub model_name: String,
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/embeddings", post(embeddings))
        .route("/v1/embeddings", post(embeddings)) // OpenAI-style alias
        .route("/health", get(health))
        .route("/metrics", get(metrics_handler))
        .with_state(state)
}

// ---- request / response shapes (OpenAI /embeddings) ----

#[derive(Deserialize)]
struct EmbeddingsRequest {
    #[serde(default)]
    model: Option<String>,
    input: Input,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Input {
    Single(String),
    Many(Vec<String>),
}

impl Input {
    fn into_vec(self) -> Vec<String> {
        match self {
            Input::Single(s) => vec![s],
            Input::Many(v) => v,
        }
    }
}

#[derive(Serialize)]
struct EmbeddingObject {
    object: &'static str,
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct EmbeddingsResponse {
    object: &'static str,
    data: Vec<EmbeddingObject>,
    model: String,
    usage: Usage,
}

#[derive(Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}
#[derive(Serialize)]
struct ErrorDetail {
    message: String,
    r#type: &'static str,
}

fn err(status: StatusCode, etype: &'static str, msg: impl Into<String>) -> Response {
    (status, Json(ErrorBody { error: ErrorDetail { message: msg.into(), r#type: etype } })).into_response()
}

async fn embeddings(State(st): State<Arc<AppState>>, body: Result<Json<EmbeddingsRequest>, axum::extract::rejection::JsonRejection>) -> Response {
    let Json(req) = match body {
        Ok(j) => j,
        Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", format!("malformed JSON: {e}")),
    };
    let inputs = req.input.into_vec();
    if inputs.is_empty() {
        return err(StatusCode::BAD_REQUEST, "invalid_request_error", "`input` must be a non-empty string or array");
    }

    st.metrics.requests_total.inc();
    st.metrics.items_total.inc_by(inputs.len() as u64);
    st.metrics.in_flight.inc();
    let t0 = Instant::now();

    // Submit all items; a full queue → 503 (shed, don't buffer).
    let mut rxs = Vec::with_capacity(inputs.len());
    for text in inputs {
        match st.submitter.submit(text) {
            Ok(rx) => rxs.push(rx),
            Err(_) => {
                st.metrics.in_flight.dec();
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    [(header::RETRY_AFTER, "1")],
                    Json(ErrorBody {
                        error: ErrorDetail { message: "server overloaded; retry".into(), r#type: "overloaded_error" },
                    }),
                )
                    .into_response();
            }
        }
    }

    // Await results in order.
    let mut data = Vec::with_capacity(rxs.len());
    let mut total_tokens = 0usize;
    for (index, rx) in rxs.into_iter().enumerate() {
        match rx.await {
            Ok(Ok((embedding, tokens))) => {
                total_tokens += tokens;
                data.push(EmbeddingObject { object: "embedding", index, embedding });
            }
            Ok(Err(msg)) => {
                st.metrics.in_flight.dec();
                st.metrics.errors_total.inc();
                return err(StatusCode::INTERNAL_SERVER_ERROR, "inference_error", msg);
            }
            Err(_) => {
                st.metrics.in_flight.dec();
                st.metrics.errors_total.inc();
                return err(StatusCode::INTERNAL_SERVER_ERROR, "inference_error", "worker dropped");
            }
        }
    }

    st.metrics.lat_total_s.observe(t0.elapsed().as_secs_f64());
    st.metrics.in_flight.dec();
    let model = req.model.unwrap_or_else(|| st.model_name.clone());
    Json(EmbeddingsResponse {
        object: "list",
        data,
        model,
        usage: Usage { prompt_tokens: total_tokens, total_tokens },
    })
    .into_response()
}

async fn health(State(st): State<Arc<AppState>>) -> Response {
    if st.ready.load(Ordering::Relaxed) {
        (StatusCode::OK, "ok").into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "warming up").into_response()
    }
}

async fn metrics_handler(State(st): State<Arc<AppState>>) -> Response {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        st.metrics.render(),
    )
        .into_response()
}
