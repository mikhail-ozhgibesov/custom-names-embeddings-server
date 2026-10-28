# custom-names-embeddings-server (ct4-h016)

A single-purpose, production-grade **Rust** microservice serving the **CT4
names-encoder** (checkpoint `h016`, step 84 000) over an OpenAI-compatible
HTTP API on CPU — built for **low p95/p99 latency under concurrency** and a
**flat, small memory footprint**.

Two interchangeable inference engines, both passing identical correctness
gates:

- **`native` (default)** — a hand-written f32 forward over repacked weights
  (`matrixmultiply` GEMM with runtime AVX2/FMA dispatch): **18.4k names/s
  single-thread batched**, **0.8 ms** single-request p50, bit-exactly
  batch-invariant, one `Arc<Weights>` copy shared by all workers.
- **`ort`** — the same model served through ONNX Runtime (reference
  implementation and A/B baseline, `CT4_ENGINE=ort`).

## The model

CT4 (T-016 / H-006): Qwen2.5-0.5B BPE tokens → factorized token embedding
(151 665×64 → 256) + learned absolute positions (≤64) → blocks `C-C-A-C-C-A`
(4× depthwise Conv1d k=5 + pointwise + residual + RMSNorm; 2× pre-RMSNorm
bidirectional attention, 4 heads; pad re-zeroed after every block) → masked
mean pooling → L2. **Output: `[B, 256]` float32, unit rows — cosine == dot.**
10 536 768 params (~42 MB fp32); batch-invariant by construction (a tested
property of the training repo, preserved by the export and made *structural*
in the native engine).

Reference quality (h016 `best.pt`, test): R@10 any **0.8300**, cross-script
0.8538, similarity AUC 0.9054; ~1.1k names/s on 1 CPU thread (torch).

## What the server does

- Loads `weights.bin` + `weights_manifest.json` + `tokenizer.json` (native) or
  `model.onnx` (ort) from one model dir; **exactly one** in-memory copy of the
  weights, shared across all requests and workers.
- Tokenization mirrors the training repo's `QwenBpeTokenizer.encode_batch`
  exactly: `add_special_tokens=false`, truncation at 64, right-padding with
  `<|endoftext|>` (151643), empty strings → one pad token (CT4's NaN-guard).
- **Dynamic micro-batching**: a scheduler thread groups requests from a
  bounded queue, length-buckets them (a long outlier never pads short names),
  and **deduplicates identical texts** (embed once, replicate).
- **Data-parallel workers** (native engine): one engine per physical core,
  sharing `Arc<Weights>`; batches split into length-homogeneous groups. Small
  batches run entirely on the scheduler — the low-latency path has zero
  cross-thread synchronization. Batch-invariance makes any split exact.
- **Backpressure**: the queue is bounded; overflow is shed with HTTP 503 +
  `Retry-After`, never buffered.
- Observable: `/health` (200 after warmup), Prometheus `/metrics` (per-stage
  latency, batch size, queue depth, in-flight, RSS), JSON startup logs,
  graceful SIGTERM drain.

## Architecture

```
axum (async) --try_send--> [bounded queue] --> scheduler thread
  POST /embeddings            (503 if full)    dedup → tokenize → length-bucket
  validate, await oneshot                      ├─ small batch: run locally
                                               └─ large batch: split into groups
                                                  → worker pool (Arc<Weights>,
                                                     one arena per worker)
                                                  → oneshots answered directly
```

Inference never touches the async executor. The native engine runs
`forward(&self, …)` over immutable weights — hence many workers, one copy. The
`ort` engine keeps a single `Mutex<Session>` on the scheduler (ONNX Runtime's
`Session::run` requires `&mut`, and a second session would duplicate weights).

## Quick start

```bash
# 0. One-time inputs (gitignored) — training-repo snapshot, checkpoint, tokenizer:
mkdir -p third_party && cd third_party
unzip /path/to/src.zip                       # repo with src/
unzip /path/to/h016.zip 'h016/best.pt'
PYTHONPATH=src python -m src.retrieval.tokenizers.download_qwen   # tokenizer.json, sha256-checked
cd ..

# 1. Python env for model prep (the server itself needs none):
python -m venv .venv && . .venv/bin/activate
pip install -r scripts/requirements-model.txt

# 2. Build the model dir (self-verifying: wrapper==encode, ONNX==torch,
#    batch-invariance; emits model.onnx + weights.bin + manifest + tokenizer):
python scripts/build_model_dir.py --out ./model

# 3. Run the server (native engine, workers = physical cores):
CT4_MODEL_DIR=./model cargo run --release

# 4. Query:
curl localhost:8080/embeddings -H 'content-type: application/json' \
  -d '{"model":"ct4-h016","input":["Vladimir Putin","Владимир Путин","Иванов Иван"]}'
curl localhost:8080/health
curl localhost:8080/metrics
```

## Configuration (env)

| Var | Meaning | Default |
|---|---|---|
| `CT4_MODEL_DIR` | dir with `weights.bin`, `weights_manifest.json`, `tokenizer.json` (+ `model.onnx` for the ort engine) | `./model` |
| `CT4_ENGINE` | `native` or `ort` | `native` |
| `CT4_MODEL_NAME` | name echoed in the OpenAI `model` field | `ct4-h016` |
| `CT4_PORT` | HTTP port | 8080 |
| `CT4_MAX_BATCH` | max items per inference batch | 32 |
| `CT4_BATCH_TIMEOUT_MS` | linger to coalesce arrivals (0 = off; native forwards are ~0.23 ms, so off by default) | 0 |
| `CT4_QUEUE_CAPACITY` | bounded queue size (503 threshold) | 1024 |
| `CT4_WORKERS` | inference workers: native = physical cores; ort = 1 | engine-dependent |
| `CT4_ORT_INTRA_THREADS` | ORT intra-op threads (ort engine only) | physical cores |
| `CT4_ORT_INTER_THREADS` | ORT inter-op threads (ort engine only) | 1 |
| `CT4_MAX_SEQ_LEN` | truncation length (hard-clamped to 64 — learned positions) | 64 |
| `CT4_HTTP_THREADS` | tokio HTTP worker threads | min(4, cores) |

Total inference threads stay ≤ physical cores (the server warns otherwise).

## Correctness gates (build-failing)

**Gate A — each engine vs the same graph in Python onnxruntime** (proves the
wrapper math):

```bash
python scripts/make_golden.py --out golden_ort        # built-in edge battery, or --names names_10k.json
cargo test --release                                  # Gate A × both engines + invariance suite
```

Measured (10 000 names): mean cos **1.000000**, max\|Δ\| native **1.94e-7** /
ort **1.79e-7** (bar: cos ≥ 0.9999, Δ < 1e-3). The native engine is
additionally **bit-exactly batch-invariant and deterministic** (cargo tests).

**Gate B — export vs the torch checkpoint (T-032)**: `max|Δ| ≤ 1e-3` on 10k
names and Re@10 agreement ± 0.001:

```bash
python scripts/gen_names.py 10000 names_10k.json
python scripts/make_reference_pt.py --names names_10k.json --out reference_pt
python scripts/make_golden.py --names names_10k.json --out golden_ort
python scripts/quality_vs_reference.py
```

Measured: max\|Δ\| **1.94e-7**, top-10 overlap **1.00000** → PASS (server ==
its graph by Gate A, so server == torch transitively).

## Benchmarks

```bash
python scripts/gen_names.py 5000 names_5k.json
python scripts/bench.py --url http://127.0.0.1:8080/embeddings --names names_5k.json
cargo run --release --example bench_engines -- both   # single-thread engine A/B
```

Headlines (4-core dev box, [`BENCH.md`](BENCH.md) for the full tables):
native **3.2×** the ort engine single-thread batched (18.4k vs 5.7k names/s at
B=64), HTTP batched **14–16k names/s** (20.3k with parallel clients),
single-request p50/p95 **0.8/0.9 ms**, 80 s soak at 3.3k req/s with **0
errors** and flat 128 MB RSS; ~19× the torch reference throughput.

## Docker

**Verified end-to-end** (built from this Dockerfile with `--pull --no-cache`,
both engines run in-container, parity vs golden confirmed, throughput ≈ bare
metal):

```bash
python scripts/build_model_dir.py --out ./model        # if not already built
CT4_MODEL_HOST=$PWD/model docker compose up --build    # serves on :8000, model at /model
```

- Image: **77.1 MB**, digest-pinned multi-stage build (rust:1.96-trixie →
  `distroless/cc-debian13`), ONNX Runtime pinned by the `ort` crate
  (`=2.0.0-rc.12` → ORT 1.24, statically linked).
- Default engine `native` needs only `weights.bin` + `weights_manifest.json` +
  `tokenizer.json` from the mount; `model.onnx` is only needed for
  `CT4_ENGINE=ort`.
- In-container checks: OpenAI interface battery, golden parity native
  max\|Δ\| **9.7e-8** / ort **1.2e-7**; B=64 batched **13.0k names/s** under
  `--cpus=4` (bare metal: 14.2k).

## CPU portability

No vendor- or ISA-specific code: the repo contains **zero** SIMD intrinsics,
`target_feature` annotations, `target-cpu`/RUSTFLAGS builds — the binary is
compiled at the **x86-64 baseline (SSE2)** and runs on any modern x86 CPU
(Intel or AMD; all benchmarks here were measured on an AMD Ryzen 9 5900X).
Faster paths are chosen **at runtime, by feature bit**:
`matrixmultiply` dispatches FMA/AVX2 vs SSE2 via `is_x86_feature_detected!`,
and ONNX Runtime's MLAS does the same internally — old CPUs without AVX2 fall
back to SSE2 automatically (correct, slower). The code also cross-compiles
for `aarch64-unknown-linux-gnu` (NEON kernels via cfg), so an ARM64 image can
be built on ARM hosts — x86_64 is the tested configuration.

## API

`POST /embeddings` (+`/v1/embeddings` alias), OpenAI-compatible: `input`
accepts a string or array; responses carry `object/data[{index,embedding}]`,
`model` (echoed if provided), `usage` token counts; 400 on malformed input,
503 + `Retry-After` when saturated. `GET /health`, `GET /metrics`
(Prometheus, `ct4_*`). Fully drop-in for existing OpenAI-style clients of the
previous names-embedding deployment (this server emits 256-d vectors; the
client-facing `model` name and env/metric prefixes changed from the old
deployment's).

## Documentation

- [`SPEC.md`](SPEC.md) — requirements & exact model contract
- [`NATIVE_RUST_REPORT.md`](NATIVE_RUST_REPORT.md) — the native-engine rewrite:
  design, verification, benchmarks
- [`NATIVE_RUST_PLAN.md`](NATIVE_RUST_PLAN.md) — the measured motivation
  (why ONNX Runtime was replaced) and design rationale
- [`REPORT.md`](REPORT.md) — the ONNX-engine phase report
- [`BENCH.md`](BENCH.md) — all benchmark tables
