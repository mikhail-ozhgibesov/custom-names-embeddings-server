//! Native weights (M0, NATIVE_RUST_PLAN §3.1): loads `weights.bin` +
//! `weights_manifest.json` produced by `scripts/build_model_dir.py`.
//!
//! The manifest lists f32 tensors in EXECUTION layout (Linear weights already
//! transposed to `[in, out]` sgemm B-operands; depthwise k-major `[5, C]`), so
//! loading is a plain read + slice — no ONNX parser, no repacking at runtime.
//! A load-time census test re-derives the parameter count and must equal the
//! manifest's `param_total` (10 536 768 for CT4 h016).

use std::path::Path;

use anyhow::Context;
use serde::Deserialize;

#[derive(Deserialize)]
struct Manifest {
    #[serde(rename = "format")]
    _format: usize,
    dims: Dims,
    pattern: Vec<String>,
    blocks: Vec<BlockMeta>,
    tensors: Vec<TensorEntry>,
    param_total: usize,
}

#[derive(Deserialize, Debug)]
struct Dims {
    vocab: usize,
    d_token: usize,
    h: usize,
    heads: usize,
    head_dim: usize,
    max_len: usize,
    dim: usize,
}

#[derive(Deserialize)]
struct BlockMeta {
    index: usize,
    kind: String, // "conv" | "att"
}

#[derive(Deserialize)]
struct TensorEntry {
    name: String,
    offset: usize, // f32 elements from the start of weights.bin
    numel: usize,
    #[serde(default)]
    _shape: Vec<usize>,
}

pub struct ConvBlock {
    /// k-major `[5 * C]`: `dw_w[k * C + c]` — contiguous over channels.
    pub dw_w: Vec<f32>,
    pub dw_b: Vec<f32>,
    /// `[C(in) * C(out)]` row-major `[in, out]` (sgemm B operand).
    pub pw_w: Vec<f32>,
    pub pw_b: Vec<f32>,
    pub norm_w: Vec<f32>,
}

pub struct AttBlock {
    pub norm_w: Vec<f32>,
    /// `[256 * 768]` = `[in, out]`.
    pub qkv_w: Vec<f32>,
    pub qkv_b: Vec<f32>,
    /// `[256 * 256]` = `[in, out]`.
    pub out_w: Vec<f32>,
    pub out_b: Vec<f32>,
}

pub enum Block {
    Conv(ConvBlock),
    Att(AttBlock),
}

pub struct Weights {
    /// `[vocab * d_token]` row-major.
    pub tok_emb: Vec<f32>,
    /// `[d_token * h]` = `[64, 256]`.
    pub proj_w: Vec<f32>,
    pub proj_b: Vec<f32>,
    /// `[max_len * h]` rows by position.
    pub pos_emb: Vec<f32>,
    pub blocks: Vec<Block>,

    pub vocab: usize,
    pub d_token: usize,
    pub h: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub max_len: usize,
    pub dim: usize,

    /// Census guard: total f32 parameters as loaded.
    pub param_total: usize,
}

impl Weights {
    pub fn load(weights_bin: &Path, manifest_json: &Path) -> anyhow::Result<Self> {
        let raw = std::fs::read(weights_bin)
            .with_context(|| format!("read {}", weights_bin.display()))?;
        let mf: Manifest = serde_json::from_str(
            &std::fs::read_to_string(manifest_json)
                .with_context(|| format!("read {}", manifest_json.display()))?,
        )
        .context("parse weights_manifest.json")?;
        anyhow::ensure!(mf.dims.h == 256 && mf.dims.dim == 256, "unexpected dims {:?}", mf.dims);

        // Slice f32 tensors out of the blob (safe copy; 42 MB once at startup).
        let bytes = raw.len();
        let get = |name: &str| -> anyhow::Result<Vec<f32>> {
            let e = mf
                .tensors
                .iter()
                .find(|t| t.name == name)
                .with_context(|| format!("tensor {name} missing from manifest"))?;
            let start = e.offset * 4;
            let end = start + e.numel * 4;
            anyhow::ensure!(end <= bytes, "tensor {name} out of bounds");
            let src = &raw[start..end];
            Ok(src
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };

        let mut census = 0usize;
        let mut count = |v: &[f32]| census += v.len();

        let tok_emb = get("tok_emb")?;
        let proj_w = get("proj_w")?;
        let proj_b = get("proj_b")?;
        let pos_emb = get("pos_emb")?;

        let mut blocks = Vec::with_capacity(mf.blocks.len());
        for bm in &mf.blocks {
            let p = format!("blk{}.", bm.index);
            let norm_w = get(&(p.clone() + "norm_w"))?;
            let blk = match bm.kind.as_str() {
                "conv" => {
                    let dw_w = get(&(p.clone() + "dw_w"))?;
                    let dw_b = get(&(p.clone() + "dw_b"))?;
                    let pw_w = get(&(p.clone() + "pw_w"))?;
                    let pw_b = get(&(p.clone() + "pw_b"))?;
                    count(&dw_w); count(&dw_b); count(&pw_w); count(&pw_b); count(&norm_w);
                    anyhow::ensure!(dw_w.len() == 5 * mf.dims.h, "dw_w len {}", dw_w.len());
                    anyhow::ensure!(pw_w.len() == mf.dims.h * mf.dims.h, "pw_w len {}", pw_w.len());
                    Block::Conv(ConvBlock { dw_w, dw_b, pw_w, pw_b, norm_w })
                }
                "att" => {
                    let qkv_w = get(&(p.clone() + "qkv_w"))?;
                    let qkv_b = get(&(p.clone() + "qkv_b"))?;
                    let out_w = get(&(p.clone() + "out_w"))?;
                    let out_b = get(&(p.clone() + "out_b"))?;
                    count(&qkv_w); count(&qkv_b); count(&out_w); count(&out_b); count(&norm_w);
                    anyhow::ensure!(qkv_w.len() == mf.dims.h * 3 * mf.dims.h, "qkv_w len {}", qkv_w.len());
                    anyhow::ensure!(out_w.len() == mf.dims.h * mf.dims.h, "out_w len {}", out_w.len());
                    Block::Att(AttBlock { norm_w, qkv_w, qkv_b, out_w, out_b })
                }
                other => anyhow::bail!("unknown block kind {other:?}"),
            };
            blocks.push(blk);
        }
        count(&tok_emb); count(&proj_w); count(&proj_b); count(&pos_emb);

        anyhow::ensure!(
            census == mf.param_total,
            "weights census {census} != manifest param_total {}",
            mf.param_total
        );
        anyhow::ensure!(
            mf.param_total == 10_536_768,
            "param_total {} != CT4 h016 expected 10 536 768",
            mf.param_total
        );
        anyhow::ensure!(tok_emb.len() == mf.dims.vocab * mf.dims.d_token, "tok_emb size");
        anyhow::ensure!(pos_emb.len() == mf.dims.max_len * mf.dims.h, "pos_emb size");

        // Pattern sanity: C C A C C A.
        let kinds: Vec<&str> = blocks
            .iter()
            .map(|b| match b {
                Block::Conv(_) => "C",
                Block::Att(_) => "A",
            })
            .collect();
        anyhow::ensure!(kinds == mf.pattern, "block pattern {kinds:?} != manifest {:?}", mf.pattern);

        Ok(Weights {
            tok_emb,
            proj_w,
            proj_b,
            pos_emb,
            blocks,
            vocab: mf.dims.vocab,
            d_token: mf.dims.d_token,
            h: mf.dims.h,
            heads: mf.dims.heads,
            head_dim: mf.dims.head_dim,
            max_len: mf.dims.max_len,
            dim: mf.dims.dim,
            param_total: census,
        })
    }
}
