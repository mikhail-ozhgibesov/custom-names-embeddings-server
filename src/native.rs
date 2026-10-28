//! Native CT4 forward (M1/M2, NATIVE_RUST_PLAN §2–3): a hand-written f32
//! implementation of the exact exported graph, replacing ONNX Runtime on the
//! hot path.
//!
//! Layout: activations are **packed real tokens only** — `[T, 256]` row-major
//! with a per-row `(start, len)` table. Padded tokens don't exist, so the
//! training repo's `x * keep` mask discipline is enforced by construction:
//!  * depthwise conv reads zeros beyond a row's ends (F.pad semantics);
//!  * attention is computed per row over its real keys only (no −inf needed);
//!  * pooling sums exactly the row's tokens.
//! Consequently the implementation is **bit-exactly batch-invariant** (each
//! row's arithmetic is independent of every other row) and deterministic.
//!
//! GEMMs go through `matrixmultiply::sgemm` (runtime AVX2/FMA dispatch); all
//! other passes (bias/position add, RMSNorm, softmax, pooling, L2) are simple
//! contiguous-SIMD loops fused into few passes. Buffers live in an `Arena`
//! reused across calls — steady state performs no allocation (the row table
//! included).

use std::sync::{Arc, Mutex};

use crate::engine::{Backend, Timing};
use crate::weights::{Block, Weights};

const C: usize = 256; // hidden dim
const K_DW: usize = 5; // depthwise kernel size
const EPS: f32 = 1e-6; // RMSNorm eps (matches _RMSNorm)

/// Per-worker scratch buffers, reused across calls.
pub struct Arena {
    emb: Vec<f32>,    // [T, 64]  gathered token embeddings (GEMM A operand)
    x: Vec<f32>,      // [T, 256] residual stream
    y: Vec<f32>,      // [T, 256] scratch: dw-conv out / pre-norm h / attention ctx
    h: Vec<f32>,      // [T, 256] GEMM out scratch
    qkv: Vec<f32>,    // [T, 768]
    scores: Vec<f32>, // [L, L] attention scores (≤ 64×64)
    starts: Vec<usize>,
    lens: Vec<usize>,
}

impl Arena {
    fn new() -> Self {
        Arena {
            emb: Vec::new(),
            x: Vec::new(),
            y: Vec::new(),
            h: Vec::new(),
            qkv: Vec::new(),
            scores: Vec::new(),
            starts: Vec::new(),
            lens: Vec::new(),
        }
    }

    fn reserve(&mut self, t: usize, rows: usize) {
        self.emb.resize(t * 64, 0.0);
        self.x.resize(t * C, 0.0);
        self.y.resize(t * C, 0.0);
        self.h.resize(t * C, 0.0);
        self.qkv.resize(t * 3 * C, 0.0);
        self.starts.resize(rows, 0);
        self.lens.resize(rows, 0);
        if self.scores.len() < 64 * 64 {
            self.scores.resize(64 * 64, 0.0);
        }
    }
}

/// C[m,n] = A[m,k] · B[k,n], all row-major. Thin safe wrapper over
/// matrixmultiply's pointer API (runtime AVX2/FMA kernels inside).
fn gemm(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    debug_assert!(m > 0 && k > 0 && n > 0);
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    unsafe {
        matrixmultiply::sgemm(
            m, k, n, 1.0,
            a.as_ptr(), k as isize, 1,
            b.as_ptr(), n as isize, 1,
            0.0,
            c.as_mut_ptr(), n as isize, 1,
        );
    }
}

/// RMSNorm over one 256-channel row, in place: `v * rsqrt(mean(v²)+eps) * w`.
fn rmsnorm(v: &mut [f32], w: &[f32]) {
    debug_assert_eq!(v.len(), C);
    let mut ss = 0.0f32;
    for &x in v.iter() {
        ss += x * x;
    }
    let inv = 1.0 / ((ss / C as f32) + EPS).sqrt();
    for (x, &wv) in v.iter_mut().zip(w.iter()) {
        *x = *x * inv * wv;
    }
}

pub struct NativeBackend {
    w: Arc<Weights>,
    arena: Mutex<Arena>, // per-backend; uncontended (one backend per worker)
}

impl Backend for NativeBackend {
    fn name(&self) -> &'static str { "native" }
    fn dim(&self) -> usize { self.w.dim }

    fn infer_ids(&self, ids: &[Vec<i64>]) -> anyhow::Result<(Vec<Vec<f32>>, Timing)> {
        NativeBackend::infer_ids(self, ids)
    }
}

impl NativeBackend {
    pub fn new(w: Arc<Weights>) -> Self {
        NativeBackend { w, arena: Mutex::new(Arena::new()) }
    }

    pub fn weights(&self) -> &Arc<Weights> { &self.w }

    /// Embed a length-similar sub-batch (per-row ids, each ≥1 token, ≤64).
    /// Returns per-row unit vectors in input order.
    pub fn infer_ids(&self, ids: &[Vec<i64>]) -> anyhow::Result<(Vec<Vec<f32>>, Timing)> {
        anyhow::ensure!(!ids.is_empty(), "empty sub-batch");
        let t0 = std::time::Instant::now();
        let w = &*self.w;
        let rows = ids.len();
        let total_tokens: usize = ids.iter().map(|v| v.len()).sum();
        anyhow::ensure!(total_tokens > 0, "no tokens");
        for row in ids {
            anyhow::ensure!(!row.is_empty() && row.len() <= w.max_len, "row len out of 1..=max_len");
        }
        let seq = ids.iter().map(|v| v.len()).max().unwrap_or(1);

        let mut arena = self.arena.lock().expect("arena poisoned");
        arena.reserve(total_tokens, rows);

        // Row table (packed layout: row r occupies [starts[r], starts[r]+lens[r])).
        {
            let mut off = 0usize;
            for (r, row) in ids.iter().enumerate() {
                arena.starts[r] = off;
                arena.lens[r] = row.len();
                off += row.len();
            }
        }

        // ---- 1. gather token embeddings → proj GEMM → +pos → +bias ----
        {
            let Arena { emb, x, starts, lens, .. } = &mut *arena;
            for (r, row) in ids.iter().enumerate() {
                let base = starts[r] * w.d_token;
                for (t, &id) in row.iter().enumerate() {
                    let idu = id as usize;
                    anyhow::ensure!(idu < w.vocab, "token id {id} out of vocab {}", w.vocab);
                    let src = idu * w.d_token;
                    emb[base + t * w.d_token..base + (t + 1) * w.d_token]
                        .copy_from_slice(&w.tok_emb[src..src + w.d_token]);
                }
            }
            gemm(total_tokens, w.d_token, C, emb, &w.proj_w, x);
            for r in 0..rows {
                let (s, l) = (starts[r], lens[r]);
                for t in 0..l {
                    let xr = &mut x[(s + t) * C..(s + t + 1) * C];
                    let pos = &w.pos_emb[t * C..(t + 1) * C];
                    for ((xc, &pv), &bv) in xr.iter_mut().zip(pos.iter()).zip(w.proj_b.iter()) {
                        *xc += pv + bv;
                    }
                }
            }
        }

        // ---- 2. blocks C C A C C A (pad re-zeroing is implicit: no pad rows) ----
        for blk in &w.blocks {
            match blk {
                Block::Conv(cb) => {
                    let Arena { x, y, h, starts, lens, .. } = &mut *arena;
                    // y = depthwise conv k=5 (same, zero edges, row-aware) + bias
                    for r in 0..rows {
                        let (s, l) = (starts[r], lens[r]);
                        for t in 0..l {
                            let dst = &mut y[(s + t) * C..(s + t + 1) * C];
                            dst.copy_from_slice(&cb.dw_b);
                            for k in 0..K_DW {
                                let src_t = t as isize + k as isize - 2;
                                if src_t >= 0 && (src_t as usize) < l {
                                    let wk = &cb.dw_w[k * C..(k + 1) * C];
                                    let xs = &x[(s + src_t as usize) * C..(s + src_t as usize + 1) * C];
                                    for (d, (&xv, &wv)) in dst.iter_mut().zip(xs.iter().zip(wk.iter())) {
                                        *d += xv * wv;
                                    }
                                }
                            }
                        }
                    }
                    // h = y @ pw_w ; then x = rmsnorm(x + h + pw_b)
                    gemm(total_tokens, C, C, y, &cb.pw_w, h);
                    for idx in 0..total_tokens {
                        let xr = &mut x[idx * C..(idx + 1) * C];
                        let hr = &h[idx * C..(idx + 1) * C];
                        for (xc, (&hv, &bv)) in xr.iter_mut().zip(hr.iter().zip(cb.pw_b.iter())) {
                            *xc += hv + bv;
                        }
                        rmsnorm(xr, &cb.norm_w);
                    }
                }
                Block::Att(ab) => {
                    let Arena { x, y, h, qkv, scores, starts, lens, .. } = &mut *arena;
                    // y = rmsnorm(x)
                    for idx in 0..total_tokens {
                        let yr = &mut y[idx * C..(idx + 1) * C];
                        let xr = &x[idx * C..(idx + 1) * C];
                        yr.copy_from_slice(xr);
                        rmsnorm(yr, &ab.norm_w);
                    }
                    // qkv = y @ qkv_w + qkv_b
                    gemm(total_tokens, C, 3 * C, y, &ab.qkv_w, qkv);
                    for idx in 0..total_tokens {
                        let qr = &mut qkv[idx * 3 * C..(idx + 1) * 3 * C];
                        for (qv, &bv) in qr.iter_mut().zip(ab.qkv_b.iter()) {
                            *qv += bv;
                        }
                    }
                    // per-row, per-head attention → ctx accumulated into y
                    y[..total_tokens * C].fill(0.0);
                    let scale = 1.0 / (w.head_dim as f32).sqrt();
                    for r in 0..rows {
                        let (s, l) = (starts[r], lens[r]);
                        for head in 0..w.heads {
                            let qo = head * w.head_dim;
                            let ko = C + head * w.head_dim;
                            let vo = 2 * C + head * w.head_dim;
                            // scores + masked-free softmax (all keys real) + ctx
                            for i in 0..l {
                                let qi = &qkv[(s + i) * 3 * C + qo..(s + i) * 3 * C + qo + w.head_dim];
                                let srow = &mut scores[i * l..i * l + l];
                                let mut mx = f32::NEG_INFINITY;
                                for (j, sc) in srow.iter_mut().enumerate() {
                                    let kj = &qkv[(s + j) * 3 * C + ko..(s + j) * 3 * C + ko + w.head_dim];
                                    let mut dot = 0.0f32;
                                    for (&a, &b) in qi.iter().zip(kj.iter()) {
                                        dot += a * b;
                                    }
                                    *sc = dot * scale;
                                    if *sc > mx {
                                        mx = *sc;
                                    }
                                }
                                let mut sum = 0.0f32;
                                for sc in srow.iter_mut() {
                                    *sc = (*sc - mx).exp();
                                    sum += *sc;
                                }
                                let inv_sum = 1.0 / sum;
                                let cr = &mut y[(s + i) * C + head * w.head_dim
                                    ..(s + i) * C + head * w.head_dim + w.head_dim];
                                for (j, &a) in srow[..l].iter().enumerate() {
                                    let aj = a * inv_sum;
                                    let vj = &qkv[(s + j) * 3 * C + vo..(s + j) * 3 * C + vo + w.head_dim];
                                    for (cc, &vv) in cr.iter_mut().zip(vj.iter()) {
                                        *cc += aj * vv;
                                    }
                                }
                            }
                        }
                    }
                    // x += ctx @ out_w + out_b
                    gemm(total_tokens, C, C, y, &ab.out_w, h);
                    for idx in 0..total_tokens {
                        let xr = &mut x[idx * C..(idx + 1) * C];
                        let hr = &h[idx * C..(idx + 1) * C];
                        for (xc, (&hv, &bv)) in xr.iter_mut().zip(hr.iter().zip(ab.out_b.iter())) {
                            *xc += hv + bv;
                        }
                    }
                }
            }
        }

        // ---- 3. masked mean pool (row's tokens) + L2 → results ----
        let Arena { x, starts, lens, .. } = &mut *arena;
        let mut out = Vec::with_capacity(rows);
        for r in 0..rows {
            let (s, l) = (starts[r], lens[r]);
            let mut acc = [0.0f32; C];
            for t in 0..l {
                let xr = &x[(s + t) * C..(s + t + 1) * C];
                for (a, &v) in acc.iter_mut().zip(xr.iter()) {
                    *a += v;
                }
            }
            let inv_l = 1.0 / l as f32;
            let mut ss = 0.0f32;
            for a in acc.iter_mut() {
                *a *= inv_l;
                ss += *a * *a;
            }
            let inv_norm = 1.0 / ss.sqrt();
            for a in acc.iter_mut() {
                *a *= inv_norm;
            }
            out.push(acc.to_vec());
        }

        let inference_s = t0.elapsed().as_secs_f64();
        Ok((
            out,
            Timing { tokenize_s: 0.0, inference_s, batch: rows, seq, total_tokens },
        ))
    }
}
