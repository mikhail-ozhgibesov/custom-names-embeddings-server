# NATIVE_RUST_PLAN — replacing ONNX Runtime with a hand-written CT4 forward pass in Rust

> **STATUS: EXECUTED (M0–M3).** Results in
> [`NATIVE_RUST_REPORT.md`](NATIVE_RUST_REPORT.md): Gate A max\|Δ\| 1.94e-7
> (10k names), bit-exact batch-invariance, 3.2× single-thread batched
> throughput vs ORT, 0.8 ms single-request p50. M4 evaluated and deliberately
> skipped (see report §5). The document below is the original plan, kept for
> the measured motivation and design rationale.

Plan for rewriting the CT4 (h016) model forward natively in Rust inside this
server, dropping ONNX Runtime from the hot path. The ONNX engine stays as a
reference/fallback (feature-gated) and all existing gates keep running against
both.

---

## 1. Why: what the measurements say

### 1.1 The workload is micro-sequences

`names_5k` token lengths (Qwen BPE): **mean 3.4, p50 3, p90 5, p99 6, max 8**.
The model's `max_len=64` is a distant upper bound — in practice every name fits
in a handful of tokens. Hot shapes are `[B, 3..8]`; attention is tiny
(`4 heads × ≤8×8`); everything is GEMM + gather bound.

### 1.2 ORT is compute-bound but scales badly

ORT 1.31, this box (4 physical cores), fp32, per-`Run()` wall time:

| Shape | 1 thread | 4 threads | 4t/1t |
|---|--:|--:|--:|
| B=1, L=5 | 285 µs | — | — |
| B=32, L=5 | 6.39 ms | 2.93 ms | 2.2× |
| B=64, L=8 | 13.6 ms | 7.02 ms | 1.9× |
| B=256, L=8 | 54.7 ms | 26.2 ms | 2.1× |

Two observations:

1. **Per-name cost is ~flat in B** (0.21 ms/name at every batch size, 1
   thread ≈ 4.7–5.0k names/s regardless of B). The model is genuinely
   GEMM-bound at ~30–40 GFLOP/s effective — good kernels, little per-Run
   overhead left to win at large B on one thread.
2. **Multi-core scaling stops at ~2× on 4 cores.** ORT's intra-op pool
   parallelizes *inside* individual small GEMMs with per-node barriers; for
   skinny matrices the barriers dominate. This is the single biggest lever a
   native rewrite owns: **data-parallel over sub-batches** (each core runs the
   whole tiny network on a slice of rows) scales ~linearly because
   batch-invariance guarantees identical results for any row partitioning.

### 1.3 The graph is 19 compute ops drowning in 274 plumbing nodes

Op census of `model.onnx` (293 nodes total): **9 MatMul, 8 Conv, 7 Gather,
2 Softmax, 6 ReduceMean** — the rest is 90 `Constant` + ~180 shape/mask
plumbing (`Cast/Transpose/Reshape/Unsqueeze/Shape/Slice/Squeeze/Pad/Concat/
Where/Expand/Range/Mul/Add/Div/Pow/Sqrt/Not/…`). Every Run re-interprets the
mask bookkeeping that, natively, becomes compile-time control flow (a row
range). Also visible in the live server: fixed per-Run overhead sets the
~285 µs single-name latency floor.

### 1.4 ORT's `&mut Session` forces ONE inference stream

The server keeps a single worker because `Session::run` borrows `&mut self`
(a second session would duplicate weights). A native forward over
**immutable, `Arc`-shared weights** runs `fn forward(&self, …)` from any
thread: N workers, one weights copy, true parallel inference. Combined with
data-parallel batching this is where the concurrency-model win comes from.

### 1.5 Secondary wins

- **RSS**: drop the ORT arena → target ~70–90 MB steady (weights 42 MB +
   arenas) vs 145 MB today.
- **Startup**: mmap a `weights.bin` instead of parsing a 40 MB ONNX proto.
- **Image/binary**: drop the statically linked ORT (~30–40 MB) from the
  distroless image.
- **Latency floor**: single-name forward target ~100–150 µs (vs 285 µs).

**Non-goals**: quantization/bf16 compute (the T-032 parity bar is max|Δ| ≤
1e-3 vs torch fp32 — keep fp32+FMA), GPU, changing the served API or the
queue/batcher architecture (they stay as-is).

---

## 2. Exact model contract (what must be reproduced)

Source of truth: `retrieval/models/conv_hybrid.py` (already exported 1:1 into
`model.onnx`; native must match it within the same gates).

Notation: `T` = padded token count of the sub-batch (`T = B × L`), `C = 256`,
`V = 151 665`, `d_tok = 64`, `H = 4` heads, `D = 64` head dim, `L ≤ 64`.

```
ids [B,L] i32, mask [B,L] i32 (1 = real token; every row has ≥ 1)

1  x[t] = proj(tok_emb[id_t]) + pos_emb[pos_t]      # gather + [T,64]×[64,256] GEMM
   x[t] = x[t] * mask_t                             # zero pad rows
2  blocks in order C C A C C A, after EVERY block: x *= mask
   C: y = pointwise(depthwise_conv_k5_same(x))      # per channel k=5, then
      #  [T,256]×[256,256] GEMM, both with bias
      x = rmsnorm(x + y)                            # post-norm, residual
   A: h = rmsnorm(x)                                # pre-norm
      qkv = h @ W_qkv + b                           # [T,256]×[256,768]
      per head: s = (q kᵀ)/√D; s[pad_key] = -inf; a = softmax(s); ctx = a v
      x = x + (reshape(ctx) @ W_out + b)
3  pooled[b] = Σ_t mask·x[t] / Σ_t mask[t]          # masked mean over row range
4  out[b] = pooled[b] / ‖pooled[b]‖₂                # float32, unit rows
```

- `rmsnorm(v) = v * rsqrt(mean(v², axis=-1) + 1e-6) * w` — float32 internal,
  eps = 1e-6, no mean subtraction, weight per channel.
- **Batch-invariance contract** (tested in the training repo, preserved by the
  export, mandatory for the native impl): a row's output depends only on its
  own tokens. Pad rows are zeroed after every block; pad keys are −inf in
  attention; pooling sums only over the row's real prefix.
- **Empty-string rule**: a zero-token row becomes `[pad_id]` with mask 1
  (server already does this in `tokenize.rs`).
- Parameter census (must sum to 10 536 768): tok_emb 9 706 560 · proj 16 640 ·
  pos_emb 16 384 · conv blocks ×4 (dw 1 536 + pw 65 792 + norm 256) ·
  attention ×2 (qkv 197 376 + out 65 792 + norm 256).
- FLOP model (per real token, L≈5): conv 267 K MAC (32 %) · attention 550 K
  MAC (66 %) · proj 16 K (2 %) → ~0.84 M MAC/token ≈ 1.7 MFLOP/token.
  GEMM-bound; the embedding gather is negligible compute but touches the
  38.8 MB table (random cache lines).

---

## 3. Design

### 3.1 Weights: load-time repacking, `weights.bin` + manifest

Extend `scripts/build_model_dir.py` to ALSO emit a `weights.bin` (+ small
`weights_manifest.json`) with tensors in **execution layout**, so Rust just
mmaps and computes (no ONNX parser dependency):

| Tensor | Stored layout | Why |
|---|---|---|
| `tok_emb` | `[V, 64]` f32 (row-major, 16-float padded rows) | gather + SIMD dot into proj without transpose |
| `proj` | `[64, 256]` transposed for A·B (row = in-channel) | direct GEMM operand |
| `pos_emb` | `[64, 256]` | row add |
| pointwise / qkv / out | K-major packed blocks (see 3.3) | microkernel operand order |
| depthwise | `[C, 5]` + bias `[C]` | scalar-lane SIMD |
| norms | `w [C]` | epilogue |

Manifest records shapes, dtypes, offsets, sha256 of the source checkpoint +
ONNX (provenance chain: native weights and the ONNX reference come from the
same `best.pt`). A Rust unit test re-derives the parameter census from the
manifest (sums to 10 536 768) as a load-time guard.

Optional later (flagged, only if Gate B still passes): store `tok_emb` as f16
(19.4 MB vs 38.8 MB) and upconvert on gather — halves table cache pressure.

### 3.2 Activation layout: token-major `[T, C]`, arenas, zero alloc

- All activations live in a per-worker `Arena` (`Vec<f32>` chunks) sized for
  `max_batch × 64` tokens once at startup; steady-state forward performs **no
  allocation** (buffer indices, not pushes).
- `[T, C]` row-major: channel dim is contiguous → RMSNorm, mask-zeroing,
  residual, pooling all become simple SIMD loops over 256-float rows; GEMMs
  consume contiguous A-rows.
- **Pad rows are never computed.** Because right-padding + the mask contract
  make pad rows dead data, every kernel takes a per-row `[len]` slice; pad
  rows stay zero in the arena (memset once per batch) and are simply skipped.
  Length-bucketing (already in `batcher.rs`) keeps waste near zero anyway —
  at mean L≈3.4 with bucketed sub-batches, computed tokens ≈ real tokens.

### 3.3 Kernel inventory (the entire network is 7 kernels)

1. **embed**: for each row, for each real token: gather 64 floats → dot with
   `proj` (K=64, unrolled 16-wide) → add `pos_emb[l]` → mask-zero. Writes
   `[T, C]`. (T·16 K MAC — memory-bound on the table gather; irrelevant cost.)
2. **dwconv_k5**: `out[l,c] = Σ_{k=0..4} w[c,k]·x[l+k-2,c] + b[c]` — SIMD over
   c-lanes, k unrolled; reads zeros beyond row ends (same-padding semantics
   preserved exactly by construction).
3. **gemm_bias_act** — the workhorse: `[T,K]×[K,N]` for K∈{64,256},
   N∈{256,768} with a fused epilogue chain selected at compile time
   (`const EPILOGUE`): `+bias` → `+residual` → `*mask` → `rmsnorm`
   (two-pass within the epilogue: GEMM writes the row, then a 256-wide
   norm pass while the row is L1-hot). Covers: proj (epilogue
   pos+mask), pointwise (bias+residual+rmsnorm), qkv (bias, split
   q/k/v pointers), out-proj (bias+residual; followed by mask pass).
4. **attention**, per head (H=4, D=64, L≤row_len≤64): scores GEMM
   `[L,64]×[64,L]` with fused `×1/√D`; **masked softmax** with the row-range
   (pad keys −inf; ≥1 real key guaranteed by the empty-string rule; max-subtract
   for stability exactly like `F.softmax`); ctx GEMM `[L,L]×[L,64]`. All
   buffers ≤ 64×64 — L1-resident. Per-row (not per-batch) since rows in a
   bucket share L.
5. **pool_l2**: per row: masked column-sum over the row's real tokens
   (accumulate [C] in registers across L) → L2 normalize 256-d (rsqrt).
   Emits `Vec<f32>` results straight into the job slots — no intermediate
   `[B,256]` copy.

Kernel policy: **first correct version on safe Rust + `matrixmultiply`**
(already in the dependency tree via ndarray; mature AVX2 kernels) with manual
pre/post passes for the "epilogues". Then, only for the 3 hot GEMM shapes,
hand-rolled microkernels (`6×16` FMA tiles) behind runtime CPU dispatch
(SSE4.2 / AVX2 / AVX-512), each gated by a criterion microbench + Gate A/B.
Do not vendor a BLAS; do not adopt a framework (candle/burn) — their generic
op dispatch re-introduces the per-node overhead we are removing (decision
matrix kept in §7 of this doc's history if needed).

### 3.4 Numeric policy

fp32 with FMA everywhere; SIMD reduction orders may differ from torch — that
is exactly what the gates measure (ONNX already deviates 1.3–1.9e-7 from
torch; native is expected in the same 1e-6–1e-7 class, bar is 1e-3).
Softmax: max-subtract, fp32 accumulation. rsqrt: fine-grained `1/sqrt` in
fp32 (avoid rsqrt-approx intrinsics — cheap enough at these sizes, keeps
parity). Determinism: same kernel + same shapes → same bits run-to-run
(required by the soak test).

### 3.5 Threading & server integration (the structural change)

```
                    ┌─ worker 0 (arena 0) ─ forward(&weights, slice 0) ─┐
batcher sub-batch → ├─ worker 1 (arena 1) ─ forward(&weights, slice 1) ─┤ → oneshots
                    ├─ worker 2 (arena 2) ─ …                            ─┤
                    └─ worker 3 (arena 3) ─ …                            ─┘
                        weights: Arc<Weights>, read-only, ONE copy
```

- `Weights: Sync + Send`, immutable after load; `forward(&self, rows: &[Row],
  out: &mut …)`. No `&mut` session → **N workers share one weights copy**.
  N defaults to physical cores; the old `CT4_WORKERS × INTRA ≤ cores` warning
  becomes a plain `workers = cores` default.
- Splitting rule (two levels, both exact thanks to batch-invariance):
  sub-batches from the existing length-bucketer are split round-robin into
  per-worker slices when `B ≥ 2·workers` (amortize weights streaming); below
  that a single worker takes it (latency path — no cross-thread sync at all).
- Work distribution: the existing single inference thread stays as the
  *scheduler*; workers are fed via a tiny bounded crossbeam-style channel or a
  preallocated `Chunk` array + spin/park — no async, no locks on the hot path.
- **In-batch + LRU dedup**: identical strings are common in name workloads
  (1 127 duplicated strings in a 10k sample). The batcher already groups;
  add: hash-dedup rows within a sub-batch (embed once, replicate) + a small
  `(text → vec)` LRU (e.g. 4 k entries, ~4 MB) across requests. Both are
  exact (same input ⇒ same vector by determinism).

### 3.6 What stays untouched

`server.rs` (HTTP/OpenAI API), `batcher.rs` queue/linger/length-bucketing,
`metrics.rs` (stage histograms get *lower* numbers, same names),
`tokenize.rs` (already mirrors QwenBpeTokenizer exactly), Docker/compose
(minus the ORT note), all gate scripts. `engine.rs` becomes a trait:

```rust
trait EmbeddingEngine {
    fn dim(&self) -> usize;
    fn infer_ids(&mut self, ids: &[Vec<i64>]) -> Result<(Vec<Vec<f32>>, Timing)>;
}
```

with `OrtEngine` (feature `engine-ort`) and `NativeEngine` (feature
`engine-native`, default). Compile-time default, runtime switchable by
feature build; both linkable simultaneously for A/B testing in one process.

---

## 4. Validation (all gates re-run per engine, build-failing)

1. **Gate A** (`cargo test --release --test parity`): native vs `golden_ort/`
   — mean cos ≥ 0.9999, max|Δ| < 1e-3 (expect ~1e-6).
2. **Gate B** (`scripts/quality_vs_reference.py`): native (via a tiny dump
   binary or the ONNX→native equivalence test) vs `reference_pt/` torch —
   max|Δ| ≤ 1e-3, Re@10 overlap ≥ 0.999. Prefer: a `--engine native|ort`
   flag on `make_golden.py` that spins the Rust engine through a small CLI —
   then Gate B is measured on the *served* path, not a simulation of it.
3. **Batch-invariance test** (new, cargo): random ragged batches — each row's
   vector must equal the alone-embedded vector bit-for-bit (native can and
   should be *exactly* invariant, not just ≤1e-6).
4. **Edge cases** (reuse the golden battery): empty string, single char,
   >64-token truncation, mixed scripts, diacritics, all-in-one-batch +
   each-alone.
5. **Determinism/soak**: same input twice → identical bits; 30-min soak, flat
   RSS, zero errors; backpressure beyond queue capacity unchanged.
6. **Cross-engine test** (new): native vs ort engine on the same random
   batches, max|Δ| < 1e-5 — catches drift between the two implementations.

## 5. Performance targets & measurement protocol

| Metric | Today (ORT) | Target (native) | Why believable |
|---|--:|--:|---|
| Batched B=64, 4 cores | 7.6–8.3k names/s | **18–25k names/s** | data-parallel ~linear (ORT stuck at 2.1×), fusion removes plumbing passes, dedup ~11 % free |
| Single-name forward | 285 µs | **100–150 µs** | no Run()/node overhead, single worker path, no arena churn |
| End-to-end p95, C=1 (linger 0) | 4.4 ms (linger 3 ms) | **< 1 ms** | forward + HTTP only |
| RSS steady | 145 MB | **70–90 MB** | no ORT arena |
| Startup (load → /health 200) | ~0.5 s | **< 100 ms** | mmap weights vs ONNX parse |

Protocol: criterion microbenches per kernel (checked into `benches/`); the
existing `ct4_inference_seconds` / `ct4_tokenize_seconds` histograms for the
integration view; `scripts/bench.py` sweep (sequential / C∈{1,10,50,100} /
B∈{64,512}) on this box + the 16-vCPU box; results into `BENCH.md` with the
ORT numbers kept as the baseline column.

## 6. Milestones

- **M0 — weights.bin** (build script + manifest + Rust loader + census test).
  Acceptance: byte-identical tensors to ONNX initializers (test compares
  against `model.onnx` via a debug feature or a Python check).
- **M1 — correct single-threaded native forward** (safe Rust + matrixmultiply,
  no fusion yet), `engine-native` behind feature. Acceptance: Gates A/B pass,
  batch-invariance exact, cross-engine ≤ 1e-5.
- **M2 — fusion + arenas + pad-skip** (7-kernel form, zero-alloc steady
  state). Acceptance: single-thread ≥ ORT single-thread (5k names/s) at
  B∈{32,64}; single-name forward ≤ 200 µs.
- **M3 — multi-worker data-parallel + dedup (batch + LRU)**. Acceptance:
  B=64 ≥ 18k names/s on 4 cores; C=100 p95 ≤ 40 ms; gates green.
- **M4 — microkernels + ISA dispatch** (only the 3 hot GEMMs). Acceptance:
  ≥ 20 % over M2 numbers per kernel in criterion, gates green.
- **M5 — ops hardening**: default `engine-native`, ORT demoted to reference
  feature; Docker/image slimming; soak + backpressure on the 16-vCPU box;
  BENCH.md/REPORT.md updated.

Rough effort: M0–M1 ≈ 1–2 days, M2 ≈ 2–3 days, M3 ≈ 1–2 days, M4 ≈ 2–3 days,
M5 ≈ 1 day — gated on machine time for the sweeps, not on code.

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Parity drift from SIMD reduction order | bar is 1e-3; ONNX already at 1.9e-7 with different kernels; Gate A/B run per PR/engine |
| Hand-rolled GEMM slower than MLAS | M2 ships with matrixmultiply first; microkernels land only if they beat it in criterion (M4 is optional upside, not a dependency) |
| Multi-worker complexity / tail regressions | single-worker fast path below `2·workers` rows; cross-engine + soak tests; ORT engine remains a release-flag fallback |
| Weights format drift vs checkpoint | manifest sha256 chain back to `best.pt`; census test; regenerate = one command |
| Softmax −inf NaN on degenerate rows | ≥1-real-token guarantee already enforced in `tokenize.rs`; explicit test rows (empty, 1-char, 64-truncated) |
| Overfitting to short names | bucketing keeps long rows correct by construction; golden battery includes 400-char inputs; kernels are generic in L ≤ 64 |
