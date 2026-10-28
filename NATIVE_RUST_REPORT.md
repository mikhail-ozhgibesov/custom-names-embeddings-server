# NATIVE_RUST_REPORT — CT4 forward rewritten natively in Rust

Report on executing [`NATIVE_RUST_PLAN.md`](NATIVE_RUST_PLAN.md): the ONNX
Runtime hot path was replaced with a hand-written f32 CT4 forward, all gates
re-verified, and the server re-benchmarked. Milestones **M0–M3 are complete and
validated**; M4 (hand-rolled GEMM microkernels) was evaluated and deliberately
skipped — rationale in §5. Numbers: [BENCH.md](BENCH.md).

---

## 1. TL;DR

The native engine is **numerically indistinguishable from torch and ONNX
Runtime** (Gate A max\|Δ\| = **1.94e-7** over 10 000 names, bar 1e-3; Gate B
unchanged and passing) and **bit-exactly batch-invariant and deterministic** —
strictly stronger than the ONNX path (which is only ≤1e-6 invariant).
Performance on the 4-core dev box, vs the ORT engine measured identically:

| Metric | ORT | native | |
|---|--:|--:|---|
| Single-thread batched (B=64, engine level) | 5.7k n/s | **18.4k n/s** | 3.2× |
| HTTP batched B=64 / B=512 | 7.6k / 8.3k n/s | **14.2k / 16.1k n/s** | ~1.9× |
| HTTP batched, 4 parallel clients | — | **20.3k n/s** | |
| Single-request p50/p95 (linger 0) | 4.2 / 4.4 ms | **0.8 / 0.9 ms** | ~5× |
| 80 s soak @ 3.3k req/s, 8 threads | — | **0 errors, RSS flat** | |
| RSS steady / engine load | 145 MB / ~0.5 s | **128 MB / 222 ms** | |

The default engine is now `native`; `CT4_ENGINE=ort` selects the reference
path (both compiled in, same gates run against both).

## 2. What was built

### M0 — execution-layout weights (`src/weights.rs` + export)

`scripts/build_model_dir.py` now also emits `weights.bin` (40.2 MiB) +
`weights_manifest.json`: every Linear/pointwise weight pre-transposed to
`[in, out]` (sgemm B-operand), depthwise stored k-major `[5, C]` (SIMD over
channels), provenance chained (`checkpoint_step`, `model_onnx_sha256`,
`weights_bin_sha256`). The Rust loader slices tensors by manifest offset and
**enforces the parameter census at load** (must equal 10 536 768) plus pattern
`C-C-A-C-C-A` and all shape invariants. One-time read of 42 MB; no ONNX
parser in the native path.

### M1/M2 — the native forward (`src/native.rs`)

The entire network is ~400 lines: 7 fused stages over **packed real tokens**
(`[T, 256]` row-major + per-row `(start, len)` table — padded tokens simply
don't exist, which makes the training repo's `x * keep` mask discipline
*structural* instead of executed):

1. gather embeddings → one `[T,64]×[64,256]` GEMM → fused `+pos +bias` epilogue;
2. conv blocks: row-aware depthwise k=5 (zero edges = F.pad semantics) →
   `[T,256]×[256,256]` GEMM → fused `+bias +residual → RMSNorm` epilogue;
3. attention blocks: RMSNorm pass → `[T,256]×[256,768]` QKV GEMM + bias →
   per-row per-head scores (no −inf needed — all keys are real), max-subtracted
   softmax exactly like `F.softmax`, ctx accumulation → out-proj GEMM →
   fused `+bias +residual`;
4. masked mean pool over the row's tokens + L2, writing results directly.

GEMMs are `matrixmultiply::sgemm` (runtime AVX2/FMA dispatch — no
`target-cpu` flags needed). Buffers live in a per-worker `Arena`, reused;
steady-state inference performs **zero allocations**. `&self` inference over
`Arc<Weights>` — many workers, one weights copy.

### M3 — data-parallel worker pool + dedup (`src/batcher.rs`)

The proven scheduler (block → sweep backlog → optional linger → length-bucket)
now fronts a **worker pool**: N engines sharing one `Arc<Weights>` copy; when a
batch has ≥ 4 rows/worker, its length-sorted unique rows are split into
contiguous (length-homogeneous) groups dispatched round-robin; the scheduler
runs group 0 and waits for acks; a saturated worker queue falls back to local
execution (never blocks). Below the threshold the scheduler runs alone — the
low-latency path does zero cross-thread synchronization. **In-batch dedup**:
identical texts (11 % of a generated 10k workload) embed once and the vector
is replicated to the whole job group.

Server changes: `CT4_ENGINE=native|ort` (default native), `CT4_WORKERS`
default = physical cores (native) / 1 (ORT), `CT4_BATCH_TIMEOUT_MS` default
**0** (linger no longer pays at 0.23 ms forwards; it cost 3 ms of latency per
request). ORT backend kept verbatim behind a `Mutex<Session>`.

## 3. Verification (all build-failing, all green)

| Gate | Scope | Result |
|---|---|---|
| Gate A vs Python-ORT golden, **10 000 names** | native | mean cos **1.000000**, max\|Δ\| **1.94e-7** |
| Gate A (same golden) | ORT | mean cos **1.000000**, max\|Δ\| **1.79e-7** |
| Bit-exact batch-invariance (alone vs ragged batch, incl. empty/1-char/64-token rows) | native | **bitwise identical** ✅ |
| Determinism (repeat runs + different chunkings) | native | **bitwise identical** ✅ |
| Edge rows (empty → pad fallback, single char, 400-char → truncated) | native | finite, unit norm ✅ |
| Cross-engine native↔ORT (1 009 stress rows, mixed batch shapes) | both | max\|Δ\| **1.19e-7** (bar 1e-5) ✅ |
| Gate B vs torch checkpoint (T-032) | via golden chain | max\|Δ\| **1.94e-7**, Re@10 agreement **1.00000** ✅ |
| Live HTTP vs golden over 10k names (previous session, ORT) | server | max\|Δ\| 1.8e-7 |
| 80 s soak, 8 threads, ~261k requests | server | **0 errors, 0 rejects, RSS Δ 0.5 MB** |

One bug was caught by the new tests and the live bench working together: the
first dedup implementation dropped oneshot replies for duplicate texts
("worker dropped" 500s) — fixed by carrying job *groups* in the dispatch task
and replicating each row's vector across its group.

## 4. Performance detail

- **Engine level** (`examples/bench_engines.rs`, single thread): native
  17.9–18.4k n/s for B=32…256 vs ORT's 5.3–5.7k — **3.2×**, from (a) packed
  real tokens (no padded-cell compute; names are 3–8 tokens), (b) no per-Run
  shape/mask plumbing (19 compute ops instead of 293 nodes), (c) fused
  epilogues, (d) zero-alloc arenas. B=1: 231 µs vs 279 µs — per-call GEMM
  B-packing dominates both at that size.
- **HTTP**: batched 1.9× (14.2k @B=64); single-request p50 **0.8 ms** with
  linger 0 (ORT needed linger 3 ms, p50 4.2 ms). The ~2k n/s plateau at high
  concurrency is the single Python client (proven: 4 parallel clients → 5.2k
  aggregate single-name, 20.3k aggregate batched).
- **Efficiency context**: single-thread native B=64 ≈ **19× the torch CT4
  reference** (18.4k vs 930 n/s @batch256, 1 thread, from the h016 eval).

## 5. M4 (hand-rolled GEMM microkernels) — evaluated, skipped

`matrixmultiply`'s runtime-dispatched AVX2/FMA kernels are already near the
practical roofline for this model's shapes (M=100–2000, K=256 skinny GEMMs):
measured 3.2× over ORT's MLAS at batch, and the remaining B=1 cost (~231 µs)
is per-call B-operand packing that pre-packing would address. That is a
bounded, optional refinement (~1 day, fused-epilogue microkernel for the 3 hot
shapes); it is not needed for any current target and adds the most
maintenance-sensitive code in the system. Decision: defer until a target
requires it (documented here and in the plan's milestone list).

## 6. Files changed

- New: `src/weights.rs`, `src/native.rs`, `tests/invariance.rs`,
  `examples/bench_engines.rs`; `weights.bin` + `weights_manifest.json` in the
  model dir (via `scripts/build_model_dir.py`).
- Reworked: `src/engine.rs` (`Backend` trait; ORT behind `Mutex`),
  `src/batcher.rs` (worker pool, dispatch, dedup), `src/config.rs` (engine
  selection, worker defaults, linger 0), `src/tokenize.rs` (`Clone`, shared),
  `src/main.rs`, `tests/parity.rs` (both engines), `Cargo.toml`
  (+`matrixmultiply`), compose/docs.
- Untouched by design: HTTP layer, metrics, gate scripts, tokenizer
  semantics, Docker structure.

## 7. Reproduction

```bash
python scripts/build_model_dir.py --out ./model        # ONNX + weights.bin (self-verifying)
cargo test --release                                   # Gates A ×2, invariance ×4, cross-engine
cargo run --release --example bench_engines -- both    # single-thread engine A/B
CT4_MODEL_DIR=./model cargo run --release              # serve (native, workers=cores, linger 0)
CT4_ENGINE=ort CT4_BATCH_TIMEOUT_MS=3 cargo run --release   # ORT reference mode
python scripts/bench.py --url http://127.0.0.1:8080/embeddings --names names_5k.json
```

## 8. What remains

1. Official 16-vCPU sweep + 30-min soak + backpressure overload run
   (BENCH.md stubs these).
2. Optional M4 pre-packed-B microkernel if sub-150 µs B=1 forwards matter.
3. Optional: native-only build (drop `ort`/`ndarray` deps) for a smaller
   image; today both engines ship in one binary for A/B.
4. Optional f16 embedding-table storage (halves the 39 MB table's cache
   footprint) — must re-pass Gate B; low priority at current numbers.

## 9. Docker verification (added post-report)

The digest-pinned image was built and verified end-to-end: **77.1 MB**
(rust:1.96-trixie builder → distroless/cc-debian13, same pins the labse study
used). In-container checks, both engines: OpenAI interface battery
(`/health`, `/v1/embeddings`, 400 bodies, metrics), golden parity — native
max\|Δ\| **9.7e-8**, ORT **1.2e-7** — and B=64 batched **13.0k names/s** under
`--cpus=4` (bare metal 14.2k). The native engine runs from a lean mount
(`weights.bin` + manifest + tokenizer only, no `model.onnx`). Compose unchanged
from the plan: `CT4_MODEL_HOST=… docker compose up --build`.
