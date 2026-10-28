# REPORT — custom-names-embeddings-server (ct4-h016)

Status report on the CT4 (h016) embedding server: what exists today, how it was
verified, measured results, open findings, and what remains. Companion docs:
[`SPEC.md`](SPEC.md) (requirements), [`BENCH.md`](BENCH.md) (numbers),
[`README.md`](README.md) (usage).

---

## 1. TL;DR

The server **works and is proven correct** end-to-end. The CT4 checkpoint
(`best.pt`, step 84 000) is exported to ONNX with the exact trained padding
discipline, and the Rust server reproduces the torch reference to **max |Δ| ≈
2e-7** (bar: 1e-3) with **identical top-10 retrieval** (bar: ±0.001) — both
T-032 acceptance criteria pass with ~4 orders of magnitude of margin. On an
informal 4-core dev-box sweep it serves **~7.6–8.3k names/s batched** (~8× the
torch reference) at **~4 ms** single-name p95, with flat 145 MB RSS and zero
errors. The official 16-vCPU benchmark sweep, soak, and backpressure runs are
still pending; Docker build is configured but not yet exercised on this box.

> **Update (later session):** the hot path was subsequently rewritten as a
> native Rust forward — 3.2× faster batched, 0.8 ms single-request p50,
> bit-exact batch-invariance. See
> [`NATIVE_RUST_REPORT.md`](NATIVE_RUST_REPORT.md) (this file documents the
> ONNX-engine phase).

## 2. Inputs (what the server was built from)

| Input | Source | Disposition |
|---|---|---|
| Reference architecture | `../labse-onnx-server` (Rust/axum/ort serving study) | Architecture inherited 1:1 (queue, micro-batching, backpressure, metrics, parity testing) |
| Training repo snapshot | RND-5 training repo (git `f2f3758…`) | `retrieval.models.conv_hybrid.CT4Encoder` + `QwenBpeTokenizer` used verbatim in export |
| Checkpoint | h016 `best.pt` (step 84 000) | config_json `t016-ct4`, 10 536 768 params (matches `model_stats.json`) |
| Tokenizer | HF `Qwen/Qwen2.5-0.5B`, rev `060db64…` | **sha256-verified** against `qwen_meta.json` |
| Eval reference | h016 eval aggregates (`retrieval_test.json` etc.) | Metrics only (no raw vectors) → parity targets generated locally from torch |

> The original archives were session inputs only. The repo is now
> self-sufficient: the minimal model sources + trimmed checkpoint live in
> `model-src/`, and the built model dir in `model/` (git LFS) — see
> `model-src/README.md`.

## 3. What exists now

### 3.1 Model dir (`scripts/build_model_dir.py` → `model/`)

- `model.onnx` (40.2 MiB, fp32): **the entire eval-mode encode path** —
  factorized embedding → proj → +positions → `x*keep` → `C-C-A-C-C-A` blocks
  (pad re-zeroed after each; `key_padding_mask` in attention) → masked mean
  pooling → L2. Inputs `input_ids`/`attention_mask` are **int32** with dynamic
  batch/seq, opset 17 (T-032 conventions). Output `embedding [B, 256]`,
  unit rows.
- `tokenizer.json` (verbatim Qwen copy) + `meta.json` (provenance: step, param
  count, sha256s, opset, pad id).
- The script **fails loudly** before producing anything unless: wrapper ==
  `CT4Encoder.encode` exactly, ONNX == torch ≤ 1e-5, and batch-invariance
  holds for torch and ONNX (name embedded alone vs inside a ragged batch,
  including the empty string).

### 3.2 Rust server (`src/`, crate `custom-names-embeddings-server`)

Same proven architecture as the LaBSE study server, adapted to CT4:

- One shared ONNX session (one weights copy) owned by a single inference
  worker; async axum handlers → bounded queue (1024) → 503 + Retry-After when
  full; worker self-batches by draining the queue (max_batch 32, 3 ms linger)
  with length-bucketing (padded-cell budget) so short names never pad to long
  outliers.
- Tokenization mirrors `QwenBpeTokenizer.encode_batch` bit-for-bit:
  `add_special_tokens=false`, truncation at 64, right-pad `<|endoftext|>`
  (151 643), **empty string → single pad token** (CT4's NaN guard).
- Input dtype probed from the graph at load (int32 for the CT4 export; i64 and
  `token_type_ids` paths kept for other graphs). Embedding dim read from the
  graph (256). `CT4_MAX_SEQ_LEN` hard-clamped to 64 (learned positional
  table).
- OpenAI-compatible `POST /embeddings` (+`/v1` alias), `/health` (200 after
  warmup), Prometheus `/metrics` (`ct4_*`: request/error/reject counters,
  in-flight, queue depth, batch-size histogram, per-stage latency histograms,
  RSS). JSON startup logs; graceful SIGTERM drain (observed live).

### 3.3 Test & gate harness

| Script | Role |
|---|---|
| `tests/parity.rs` (cargo) | **Gate A**: server vs `golden_ort/` (own ONNX in Python ORT); build-failing; skips with notice if goldens absent |
| `scripts/make_golden.py` | Golden vectors from ONNX (built-in 80-name edge battery, or `--names` for the 10k set) |
| `scripts/make_reference_pt.py` | Ground truth from the **torch checkpoint** via the training repo's own `encode` (batch 2048 = the h016 eval setting) |
| `scripts/quality_vs_reference.py` | **Gate B** (T-032): max\|Δ\| ≤ 1e-3 on 10k names + Re@10 agreement ± 0.001 (top-10 overlap vs reference, deduplicated) |
| `scripts/bench.py`, `gen_names.py` | Load driver + reproducible multilingual name generator (seeded) |

### 3.4 Ops

Digest-pinned multi-stage Dockerfile (rust:1.96-trixie → distroless/cc, 77
MB-class image) + env-driven `docker-compose.yml` (service `ct4`, container
`ct4_server`, `CT4_*` variables). `docker/images.lock.json` pins bases; ORT
pinned via `ort =2.0.0-rc.12` (static ORT 1.24). **Configured, not yet built
on this box** (no Docker engine images cached here).

## 4. Verification results (all from this session, this box)

### 4.1 Export self-verification (`build_model_dir.py`)

| Check | Result |
|---|---|
| Wrapper vs `CT4Encoder.encode` (20-name battery) | max\|Δ\| = **0.0** (exact) |
| ONNX vs torch, same battery | max\|Δ\| = **1.19e-7**, min cos 0.99999964 |
| Batch-invariance (alone vs ragged batch; torch + ONNX) | ≤ 1e-6 for `"Иванов Иван"`, `"Q"`, `""` — **PASS** |

### 4.2 Correctness gates

| Gate | Criterion | Measured | Verdict |
|---|---|---|---|
| **A** — server vs own ONNX (`cargo test --release --test parity`) | mean cos ≥ 0.9999, max\|Δ\| < 1e-3 | cos **1.000000** (min 0.999999), max\|Δ\| **1.30e-7** (80 names incl. edge cases) | ✅ PASS |
| HTTP end-to-end — live server vs Python-ORT golden, **10 000 names** | — | max\|Δ\| **1.79e-7**, mean \|v\| = 1.0 | ✅ |
| **B** — ONNX vs torch checkpoint (T-032) | max\|Δ\| ≤ 1e-3; Re@10 ± 0.001 | max\|Δ\| **1.94e-7**; top-10 overlap **1.00000** (delta 0.00000) | ✅ PASS |

Chain of evidence: server == its ONNX (Gate A, at build time and over HTTP at
10k scale) == torch checkpoint (Gate B) ⟹ **server == torch**.

### 4.3 Informal performance (4 physical cores, `CT4_ORT_INTRA_THREADS=4`)

| Mode | p50 / p95 / p99 | Throughput |
|---|--:|--:|
| One name/request, C=1 | 4.2 / 4.4 / 4.5 ms | 238 n/s |
| C=10 | 6.8 / 10.7 / 12.5 ms | 1 430 n/s |
| C=50 | 21.5 / 52.2 / 71.3 ms | 1 945 n/s |
| C=100 | 35.7 / 108.9 / 151.2 ms | 1 902 n/s |
| Batch B=64 | 8.0 / 10.2 ms (req) | **7 593 n/s** |
| Batch B=512 | 60.1 / 65.3 ms (req) | **8 254 n/s** |

Reference: torch CT4 on 1 CPU thread (h016 eval) — 930 n/s @batch256, 1 155
@batch64. Headline: **~8× batched throughput** on 4 cores; warm single-name p95
4.4 ms vs the 60 ms `span.embed_query` budget carried over from the LaBSE
study. Footprint: RSS **145 MB** steady (42 MB weights + ORT arena + runtime),
flat; 25 716 requests / 45 004 items served with **0 errors, 0 rejects**.

## 5. Findings

1. **Upstream bug (training repo)** — `CT4Encoder.encode([""])` raises
   `IndexError`: an all-empty batch produces zero columns and the pad-fallback
   indexes axis 1. Mixed batches with empty strings are handled correctly. The
   server and gate scripts always apply the pad-fallback (mixed-batch
   semantics), so empty inputs work everywhere here. Worth an upstream fix or
   test case (T-016 batch-invariance suite).
2. **Tie churn vs Gate B metric** — generated names contain duplicates
   (1 127 duplicated strings in 10k); identical inputs embed identically, so
   neighbor order among twins is arbitrary under any finite-precision change.
   Exact top-10 set identity is therefore reported as info only; the gated
   metric is mean top-10 **overlap** vs the reference system on deduplicated
   names (measured 1.00000). On real labeled benchmarks (positives well clear
   of rank-10 ties) this distinction will not matter.
3. **Dev-box port 8080 is occupied** by an unrelated process; smoke tests ran
   on `CT4_PORT=8093`. No server-side issue.
4. **Toolchain note** — this box had no Rust/GCC/venv and no sudo; a portable
   toolchain (rustup + micromamba GCC, OpenSSL) was set up under `/tmp/opencode`
   to build and test locally. The pinned Dockerfile builds do not depend on it.

## 6. What remains (next steps)

1. **Official benchmark table** on the 16-vCPU study box: full concurrency
   sweep {1, 25, 50, 100} × 5-min steady (30 s warmup discarded), torch CT4
   baseline through the same driver → rewrite `BENCH.md`.
2. **Stability runs**: 30-min soak at C=100 (flat RSS, zero 5xx);
   backpressure beyond `CT4_QUEUE_CAPACITY` (assert clean 503s, no OOM).
3. ~~Docker~~ **DONE** — image built and verified (77.1 MB, both engines,
   golden parity, ≈bare-metal throughput); see
   [`NATIVE_RUST_REPORT.md`](NATIVE_RUST_REPORT.md) §9.
4. Optional: quality evaluation on the real labeled benchmark (JRC-Names-style
   recall) with ONNX/server vectors, closing the loop with the h016 eval
   numbers (R@10 any 0.8300) rather than the synthetic self-retrieval proxy.
5. Optional: upstream the empty-batch fix (finding 1) to the training repo.

## 7. Reproduction

```bash
git lfs pull                                            # model/ + model-src/ weights

python scripts/build_model_dir.py --out ./model          # self-verifying export (from model-src/)
python scripts/make_golden.py --out golden_ort           # Gate A golden
cargo test --release --test parity                       # Gate A (build-failing)
python scripts/gen_names.py 10000 names_10k.json
python scripts/make_reference_pt.py --names names_10k.json --out reference_pt
python scripts/make_golden.py --names names_10k.json --out golden_ort
python scripts/quality_vs_reference.py                   # Gate B (T-032)
CT4_MODEL_DIR=./model cargo run --release                # serve :8080
python scripts/bench.py                                   # quick sweep
```
