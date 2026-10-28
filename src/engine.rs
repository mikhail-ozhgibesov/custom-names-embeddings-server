//! Engine front-end: owns the tokenizer and one inference `Backend`
//! (native CT4 forward or ONNX Runtime), exposing a single `infer_ids`.
//!
//! The server, batcher and tests are backend-agnostic. Selection:
//!  * `NativeBackend` (default) — hand-written f32 forward over repacked
//!    `weights.bin` (see `native.rs`); weights behind `Arc`, `&self` inference
//!    → many workers may share ONE weights copy.
//!  * `OrtBackend` — the original ONNX Runtime path (kept as the reference
//!    implementation and A/B baseline; `Session::run` needs `&mut`, so it
//!    runs behind a `Mutex` and always single-worker).
//!
//! ORT graph probing (input names/dtypes, Form A/B detection) lives in
//! `ort_backend()`; the native path needs no graph — it IS the model.

use std::path::Path;
use std::sync::{Arc, Mutex};

use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionOutputs};
use ort::value::{Tensor, TensorElementType, ValueType};

use crate::head;
use crate::native::NativeBackend;
use crate::tokenize::NamesTokenizer;
use crate::weights::Weights;

pub struct Timing {
    pub tokenize_s: f64,
    pub inference_s: f64,
    pub batch: usize,
    pub seq: usize,
    pub total_tokens: usize,
}

pub trait Backend: Send {
    fn name(&self) -> &'static str;
    fn dim(&self) -> usize;
    fn infer_ids(&self, ids: &[Vec<i64>]) -> anyhow::Result<(Vec<Vec<f32>>, Timing)>;
}

pub struct Engine {
    tokenizer: NamesTokenizer,
    backend: Box<dyn Backend>,
}

impl Engine {
    /// ONNX Runtime backend (reference implementation).
    pub fn load_ort(
        model_onnx: &Path,
        tokenizer_json: &Path,
        intra: usize,
        inter: usize,
        max_seq_len: usize,
    ) -> anyhow::Result<Self> {
        let tokenizer = NamesTokenizer::load(tokenizer_json, max_seq_len)?;
        let backend = OrtBackend::load(model_onnx, tokenizer.pad_id(), intra, inter)?;
        Ok(Self { tokenizer, backend: Box::new(backend) })
    }

    /// Native backend; loads `weights.bin` + manifest from the model dir.
    pub fn load_native(model_dir: &Path, tokenizer_json: &Path, max_seq_len: usize) -> anyhow::Result<Self> {
        let tokenizer = NamesTokenizer::load(tokenizer_json, max_seq_len)?;
        let weights = Arc::new(Weights::load(
            &model_dir.join("weights.bin"),
            &model_dir.join("weights_manifest.json"),
        )?);
        Ok(Self { tokenizer, backend: Box::new(NativeBackend::new(weights)) })
    }

    /// Native backend from already-loaded shared weights + tokenizer
    /// (cheap — used to build one Engine per worker with ONE weights copy).
    pub fn native_from_shared(weights: Arc<Weights>, tokenizer: NamesTokenizer) -> Self {
        Self { tokenizer, backend: Box::new(NativeBackend::new(weights)) }
    }

    pub fn tokenizer(&self) -> &NamesTokenizer { &self.tokenizer }
    pub fn dim(&self) -> usize { self.backend.dim() }
    pub fn backend_name(&self) -> &'static str { self.backend.name() }

    pub fn infer_ids(&self, ids: &[Vec<i64>]) -> anyhow::Result<(Vec<Vec<f32>>, Timing)> {
        self.backend.infer_ids(ids)
    }
}

// ---------------------------------------------------------------------------
// ONNX Runtime backend (reference; ported from the previous Engine)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Form {
    /// Graph outputs final `[batch, dim]` vectors (CT4's export form).
    A,
    /// Graph outputs `last_hidden_state [batch, seq, dim]` (not CT4's export).
    B,
}

struct OrtBackend {
    session: Mutex<Session>,
    out_name: String,
    form: Form,
    dim: usize,
    int32_inputs: bool,
    has_token_types: bool,
    pad_id: i64,
}

impl Backend for OrtBackend {
    fn name(&self) -> &'static str { "ort" }
    fn dim(&self) -> usize { self.dim }

    fn infer_ids(&self, ids: &[Vec<i64>]) -> anyhow::Result<(Vec<Vec<f32>>, Timing)> {
        anyhow::ensure!(!ids.is_empty(), "empty sub-batch");
        let total_tokens: usize = ids.iter().map(|v| v.len()).sum();
        let (input_ids, mask, seq) = pad_to_batch(ids, self.pad_id);
        let batch = ids.len();

        let t0 = std::time::Instant::now();
        let vecs = {
            // Outputs borrow the session — extract within the guard's scope.
            let mut session = self.session.lock().expect("ort session poisoned");
            let outputs = run_session(
                &mut session,
                &input_ids,
                &mask,
                batch,
                seq,
                self.int32_inputs,
                self.has_token_types,
            )?;
            let (_shape, data) = outputs[self.out_name.as_str()].try_extract_tensor::<f32>()?;
            match self.form {
                Form::A => head::form_a(data, batch, self.dim),
                Form::B => anyhow::bail!("Form B head not configured for this model"),
            }
        };
        let inference_s = t0.elapsed().as_secs_f64();
        Ok((vecs, Timing { tokenize_s: 0.0, inference_s, batch, seq, total_tokens }))
    }
}

impl OrtBackend {
    fn load(model_onnx: &Path, pad_id: i64, intra: usize, inter: usize) -> anyhow::Result<Self> {
        // ort's builder-config errors carry the (non-Send) builder for retry, so
        // they don't convert into anyhow via `?`; map them to messages.
        let mut b = Session::builder().map_err(|e| anyhow::anyhow!("session builder: {e}"))?;
        b = b.with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("set optimization level: {e}"))?;
        b = b.with_intra_threads(intra).map_err(|e| anyhow::anyhow!("set intra threads: {e}"))?;
        b = b.with_inter_threads(inter).map_err(|e| anyhow::anyhow!("set inter threads: {e}"))?;
        let session = b
            .commit_from_file(model_onnx)
            .map_err(|e| anyhow::anyhow!("load onnx {}: {e}", model_onnx.display()))?;

        let input_names: Vec<String> = session.inputs().iter().map(|i| i.name().to_string()).collect();
        anyhow::ensure!(
            input_names.iter().any(|n| n == "input_ids"),
            "graph has no `input_ids` input (found {input_names:?})"
        );
        anyhow::ensure!(
            input_names.iter().any(|n| n == "attention_mask"),
            "graph has no `attention_mask` input (found {input_names:?})"
        );
        let has_token_types = input_names.iter().any(|n| n == "token_type_ids");
        let int32_inputs = tensor_element(&session, "input_ids")? == TensorElementType::Int32;

        let out_name = session
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .ok_or_else(|| anyhow::anyhow!("model has no outputs"))?;

        // Form detection from a warmup run (any single-token input works).
        let mut eng = OrtBackend {
            session: Mutex::new(session),
            out_name,
            form: Form::A,
            dim: 0,
            int32_inputs,
            has_token_types,
            pad_id,
        };
        (eng.form, eng.dim) = eng.detect_form()?;
        Ok(eng)
    }

    fn detect_form(&self) -> anyhow::Result<(Form, usize)> {
        let ids = [vec![self.pad_id]];
        let (input_ids, mask, seq) = pad_to_batch(&ids, self.pad_id);
        let mut session = self.session.lock().expect("ort session poisoned");
        let outputs =
            run_session(&mut session, &input_ids, &mask, 1, seq, self.int32_inputs, self.has_token_types)?;
        let (shape, _) = outputs[self.out_name.as_str()].try_extract_tensor::<f32>()?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        match dims.len() {
            2 => Ok((Form::A, *dims.last().unwrap() as usize)),
            3 => anyhow::bail!(
                "graph outputs `last_hidden_state` {dims:?} (Form B): the CT4 export is Form A \
                 (pooling inside the graph). Re-export with scripts/build_model_dir.py."
            ),
            _ => anyhow::bail!("unrecognized output shape {dims:?}; expected [b,dim] or [b,seq,dim]"),
        }
    }
}

fn tensor_element(session: &Session, name: &str) -> anyhow::Result<TensorElementType> {
    let input = session
        .inputs()
        .iter()
        .find(|i| i.name() == name)
        .ok_or_else(|| anyhow::anyhow!("no input {name}"))?;
    match input.dtype() {
        ValueType::Tensor { ty, .. } => Ok(*ty),
        other => anyhow::bail!("input {name} is not a tensor: {other:?}"),
    }
}

/// Pad per-item ids to the batch max length; build flat ids + attention mask.
/// Padded cells carry `pad_id` — exact mirror of the training path.
fn pad_to_batch(ids: &[Vec<i64>], pad_id: i64) -> (Vec<i64>, Vec<i64>, usize) {
    let seq = ids.iter().map(|v| v.len()).max().unwrap_or(1).max(1);
    let batch = ids.len();
    let mut input_ids = vec![pad_id; batch * seq];
    let mut mask = vec![0i64; batch * seq];
    for (b, item) in ids.iter().enumerate() {
        let base = b * seq;
        for (j, &id) in item.iter().enumerate() {
            input_ids[base + j] = id;
            mask[base + j] = 1;
        }
    }
    (input_ids, mask, seq)
}

fn run_session<'s>(
    session: &'s mut Session,
    input_ids: &[i64],
    attention_mask: &[i64],
    batch: usize,
    seq: usize,
    int32: bool,
    has_token_types: bool,
) -> anyhow::Result<SessionOutputs<'s>> {
    let shape = [batch, seq];
    if int32 {
        let ids: Vec<i32> = input_ids.iter().map(|&v| v as i32).collect();
        let mask: Vec<i32> = attention_mask.iter().map(|&v| v as i32).collect();
        let ids = Tensor::from_array((shape, ids))?;
        let mask = Tensor::from_array((shape, mask))?;
        Ok(if has_token_types {
            let tt = Tensor::from_array((shape, vec![0i32; batch * seq]))?;
            session.run(ort::inputs! {
                "input_ids" => ids,
                "attention_mask" => mask,
                "token_type_ids" => tt,
            })?
        } else {
            session.run(ort::inputs! {
                "input_ids" => ids,
                "attention_mask" => mask,
            })?
        })
    } else {
        let ids = Tensor::from_array((shape, input_ids.to_vec()))?;
        let mask = Tensor::from_array((shape, attention_mask.to_vec()))?;
        Ok(if has_token_types {
            let tt = Tensor::from_array((shape, vec![0i64; batch * seq]))?;
            session.run(ort::inputs! {
                "input_ids" => ids,
                "attention_mask" => mask,
                "token_type_ids" => tt,
            })?
        } else {
            session.run(ort::inputs! {
                "input_ids" => ids,
                "attention_mask" => mask,
            })?
        })
    }
}
