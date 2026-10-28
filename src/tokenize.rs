//! Tokenization via the HF `tokenizers` crate, using the CT4 `tokenizer.json`
//! (Qwen2.5-0.5B BPE) verbatim — byte-level BPE, no UNK, no added specials.
//!
//! Mirrors `retrieval.tokenizers.qwen_bpe.QwenBpeTokenizer.encode_batch`:
//!  * `add_special_tokens = false` (the Python wrapper's exact flag);
//!  * truncation to `max_seq_len` (CT4: 64 — learned positional table);
//!  * no padding here — padding to a sub-batch's own max happens in the engine
//!    after length-bucketing, so a long outlier never pads the short names.
//!  * zero-token rows (empty string) get a single pad token — CT4's
//!    NaN-guard for masked softmax/mean-pooling (see CT4Encoder.encode). We do
//!    NOT add our own normalization — the tokenizer is self-contained.
//!
//! `Clone` shares the underlying tokenizer (`Arc`), so one parse feeds every
//! inference worker.

use std::sync::Arc;

use tokenizers::Tokenizer;

#[derive(Clone)]
pub struct NamesTokenizer {
    tok: Arc<Tokenizer>,
    pad_id: i64,
}

impl NamesTokenizer {
    pub fn load(path: &std::path::Path, max_seq_len: usize) -> anyhow::Result<Self> {
        let mut tok = Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", path.display()))?;
        tok.with_truncation(Some(tokenizers::TruncationParams {
            max_length: max_seq_len,
            ..Default::default()
        }))
        .map_err(|e| anyhow::anyhow!("set truncation: {e}"))?;
        tok.with_padding(None);
        // Qwen's pad token (`<|endoftext|>` = 151643) — used both for the
        // empty-string fallback and for padded cells (mirrors the training path).
        let pad_id = tok
            .token_to_id("<|endoftext|>")
            .ok_or_else(|| anyhow::anyhow!("tokenizer has no `<|endoftext|>` pad token"))?
            as i64;
        Ok(Self { tok: Arc::new(tok), pad_id })
    }

    /// Qwen pad id (used for padded cells and the empty-row fallback).
    pub fn pad_id(&self) -> i64 { self.pad_id }

    /// Per-item token ids (truncated, unpadded). Empty strings yield the
    /// single-pad-token row `[pad_id]` — CT4's empty-input contract.
    pub fn encode_ids(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<i64>>> {
        let encs = self
            .tok
            .encode_batch(texts.to_vec(), false)
            .map_err(|e| anyhow::anyhow!("encode_batch: {e}"))?;
        Ok(encs
            .iter()
            .map(|e| {
                let ids: Vec<i64> = e.get_ids().iter().map(|&x| x as i64).collect();
                if ids.is_empty() {
                    vec![self.pad_id]
                } else {
                    ids
                }
            })
            .collect())
    }
}
