//! Pooling + projection head producing the final L2-normalized embedding.
//!
//! CT4 (h016) exports the ENTIRE post-embedding path — blocks, masked mean
//! pooling, and L2 normalization — inside the ONNX graph (that's what makes the
//! padding discipline exactly the trained one). So this module only handles
//! Form A: L2-normalize rows of the `[batch, 256]` graph output defensively
//! (idempotent — the graph already normalizes).
//!
//! If a future export stops at `last_hidden_state` (Form B), implement the
//! model's pooling here; keep the rules: row-independent (batch-invariance is
//! a tested property of CT4) and always finish with L2.

use ndarray::Array2;

/// L2-normalize each row in place.
fn l2_normalize_rows(m: &mut Array2<f32>) {
    for mut row in m.rows_mut() {
        let norm = row.dot(&row).sqrt();
        if norm > 0.0 {
            row.mapv_inplace(|v| v / norm);
        }
    }
}

/// Form A: graph output is already `[batch, dim]`; L2-normalize defensively.
/// `dim` is read from the graph at load time (not hardcoded).
pub fn form_a(out_flat: &[f32], batch: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut out = Array2::from_shape_vec((batch, dim), out_flat.to_vec())
        .expect("form A output not [batch,dim]");
    l2_normalize_rows(&mut out);
    (0..batch).map(|b| out.row(b).to_vec()).collect()
}
