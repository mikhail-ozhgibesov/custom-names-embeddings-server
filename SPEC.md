# SPEC — custom-names-embeddings-server (ct4-h016)

Single-purpose microservice serving the CT4 names-encoder (checkpoint h016,
step 84 000) over an OpenAI-compatible `POST /embeddings` API. Model specifics
per T-016/H-006; export conventions per T-032.

## 1. Mission

Load the CT4 ONNX graph and serve L2-normalized 256-d embeddings, optimized in
priority order:
1. Low p95/p99 latency under concurrent load (binding metric).
2. Low, flat memory footprint under sustained concurrency (one copy of weights).
3. High throughput.

The model is fixed. No model-management/multi-model/fine-tuning.

## 2. The model — CT4 (h016)

- Source: `retrieval.models.conv_hybrid.CT4Encoder` (+`CT4Config`), registry
  name `ct4`; checkpoint `results/h016/best.pt` (state_dict + config_json,
  step 84 000, dev Re@10 0.8474).
- Params: 10 536 768 (~42 MB fp32). Pattern `C-C-A-C-C-A`.
- Encode path: QwenBpeTokenizer.encode_batch(max_len=64,
  add_special_tokens=False, pad `<|endoftext|>`=151643) → factorized embedding
  V×64 → proj → +absolute positions (≤64) → `x*keep` → 6 blocks (pad re-zeroed
  after each; attention uses key_padding_mask) → masked mean pooling →
  L2 normalize in float32. No extra head; output `[B, 256]`.
- Batch-invariance is a tested property: the export must preserve the
  pad-zeroing + mask-aware pooling exactly (it does — verified at export time).
- Known upstream quirk: `encode([""])` (all-empty batch) IndexErrors in the
  training repo; mixed batches apply a pad-fallback for empty rows. The server
  always applies the pad-fallback (matches the mixed-batch semantics).
- Reference quality (test): R@10 any 0.8300, cross-script 0.8538,
  similarity AUC 0.9054. Throughput ref: ~1.1k names/s, 1 CPU thread (torch).

## 3. Graph form

Form A: the export wraps the ENTIRE post-embedding path (blocks, masked mean
pooling, L2) in the graph. Inputs `input_ids`/`attention_mask` are **int32**
(T-032), dynamic batch+seq; output `embedding [B, 256]`. The server L2-normalizes
defensively (idempotent). Rank-3 outputs (Form B) are refused at load.

## 4. Hard requirements (MUST)

- Load ONNX graph + matching HF tokenizer from a config dir.
- Exactly one in-memory copy of weights, shared across requests.
- Correct, L2-normalized float32 256-d embeddings within tolerance (§7).
- ≥100 concurrent clients without errors/unbounded memory; backpressure
  (503/Retry-After) when the queue is full.
- Decouple async HTTP from blocking inference.
- Dynamic micro-batching; length-bucketing so short names never pad to a long
  outlier.
- Configurable via env (§9), containerized (§10), observable
  (`/health`, `/metrics`).

NON-GOALS: runtime conversion/quantization/export; auth/multi-tenancy/
persistence/GPU; request-level model selection.

## 5. Architecture

`tokio/axum` async accept → validate → enqueue on a bounded MPSC (503 if full)
→ inference worker owning the session → drain → length-bucket → `Session::run`
→ L2 → fulfill oneshots. One weights copy. Async never blocks on `Run()`.
Thread budget: total ≤ physical cores.

## 6. API

`POST /embeddings` (OpenAI): `{ "model": "...", "input": ["...", ...] }` →
`{ "object":"list", "data":[{"object":"embedding","index":i,"embedding":[...]}],
"model":..., "usage":{"prompt_tokens":N,"total_tokens":N} }`. `input` accepts a
string or array. 503 + Retry-After when saturated; 400 on malformed input.
`GET /health` → 200 when loaded + warmup passed. `GET /metrics` → Prometheus.

## 7. Correctness (BLOCKING gates)

- **Gate A — parity (server vs own ONNX):** mean cosine ≥ 0.9999, max abs elem
  diff < 1e-3 over a fixed multilingual sample incl. edge cases (empty, single
  token, >64 tokens truncated, mixed scripts, combining diacritics). Fails the
  build. *Measured: mean cos 1.000000, max|Δ| 1.3e-7.*
- **Gate B — vs torch checkpoint (T-032):** max|Δ| ≤ 1e-3 over 10k names and
  Re@10 equal within ±0.001. *Measured: max|Δ| 1.9e-7, top-10 overlap (Re@10
  agreement) 1.00000 → PASS.* The binding numeric criterion is the max|Δ| bar;
  the recall agreement is computed as top-10 neighbor overlap vs the reference
  system on deduplicated names.

## 8. Performance targets

Baseline = torch CT4 (~1.1k names/s single CPU thread, from the h016 eval).
Report p50/p95/p99/p999 at concurrency {1,25,50,100}, steady RSS (flat —
weights ≈ 42 MB + ORT arena), texts/s, 30-min soak, and 503 backpressure
beyond `CT4_QUEUE_CAPACITY`. Results → `BENCH.md`.

## 9. Configuration (env)

`CT4_MODEL_DIR` (./model), `CT4_MODEL_NAME` (ct4-h016), `CT4_PORT` (8080),
`CT4_MAX_BATCH` (32), `CT4_BATCH_TIMEOUT_MS` (3), `CT4_QUEUE_CAPACITY` (1024),
`CT4_WORKERS` (1), `CT4_ORT_INTRA_THREADS` (physical cores),
`CT4_ORT_INTER_THREADS` (1), `CT4_MAX_SEQ_LEN` (64, hard-clamped — learned
positional table), `CT4_HTTP_THREADS` (min(4,cores)). Total threads ≤ physical
cores; warn if exceeded.

## 10. Dockerization

Multi-stage; minimal final image; base pinned by digest (`docker/images.lock.json`);
ORT version pinned (`ort =2.0.0-rc.12` → ORT 1.24, static). Config from env,
respects cgroup CPU limits. `docker-compose.yml` env-driven (service `ct4`,
container `ct4_server`).

## 11. Testing & benchmarking

1. Parity gates A/B (above).
2. Load test — `scripts/bench.py` (sequential / concurrent sweep / batched).
3. Soak — 30-min at 100 conc; flat RSS, zero 5xx within capacity.
4. Backpressure — beyond `CT4_QUEUE_CAPACITY`; assert 503s, not OOM.

## 12. Pitfalls (do NOT)

One session per request/worker; blocking the executor with `Run()`; adding
special tokens (Qwen path is `add_special_tokens=False`); forgetting the
empty-string pad fallback (NaN from masked softmax/mean pooling); feeding int64
to the int32 graph (or vice versa without probing); reimplementing pooling
outside the graph differently from training (breaks batch-invariance); raising
max_seq_len past 64 (learned positions); unbounded queue; `WORKERS × INTRA >`
physical cores; unpinned base/ORT; padding every batch to a fixed long length.
