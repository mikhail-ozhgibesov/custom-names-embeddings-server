# BENCH.md — custom-names-embeddings-server (ct4-h016)

All numbers from this 4-physical-core dev box (not the 16-vCPU study box) ·
`scripts/bench.py` · 5 000 multilingual names (mean 3.4 tokens, max 8) ·
native engine = hand-written Rust forward; ORT engine = ONNX Runtime 1.24
reference. Informal but same-box, same-driver A/B. Rerun on the target box for
the official table.

## Engine-level (single thread, `cargo run --release --example bench_engines`)

| Batch | native ms | native names/s | ORT ms | ORT names/s | speedup |
|--:|--:|--:|--:|--:|--:|
| B=1 | 0.231 | 4 241 | 0.279 | 3 641 | 1.2× |
| B=32 | 1.78 | 17 908 | 5.69 | 5 611 | **3.2×** |
| B=64 | 3.47 | 18 362 | 11.15 | 5 734 | **3.2×** |
| B=256 | 15.02 | 16 992 | 48.20 | 5 266 | **3.2×** |

## HTTP end-to-end (single client process; 4 workers, linger 0 ms)

| Mode | native p50/p95/p99 | native n/s | ORT (baseline) p50/p95 | ORT n/s |
|---|---|--:|--:|--:|
| one name/req, C=1 | **0.8 / 0.9 / 1.0 ms** | 1 217 | 4.2 / 4.4 ms | 238 |
| C=10 | 4.5 / 10.3 / 13.8 ms | 1 980 | 6.8 / 10.7 ms | 1 430 |
| C=50 | 19.5 / 55.2 / 76.5 ms | 2 032 | 21.5 / 52.2 ms | 1 945 |
| C=100 | 33.1 / 104.7 / 153.5 ms | 1 928 | 35.7 / 108.9 ms | 1 902 |
| batch B=64 | 4.4 / 5.4 ms (req) | **14 199** | 8.0 / 10.2 ms | 7 593 |
| batch B=512 | 31.6 / 32.7 ms (req) | **16 086** | 60.1 / 65.3 ms | 8 254 |

The ~2 000 n/s plateau at C≥10 is the **single Python client** (GIL — same
artifact documented in the LaBSE study). Parallel clients on the same box:

| Test | Aggregate |
|---|--:|
| 4 × C=25 single-name clients | **5 176 n/s** |
| 4 × B=128 batch clients | **20 332 n/s** |
| 8-thread × 80 s soak (urllib, C=100-equivalent) | **3 263 req/s**, p50 2.3 / p95 4.0 / p99 4.9 ms, **0 errors** |

## Footprint

| | native | ORT |
|---|--:|--:|
| RSS steady (4 workers) | **128 MB** | 145 MB |
| RSS across 80 s soak @3.3k req/s | flat (Δ 0.5 MB), 294 k requests | — |
| engine load → /health 200 | **222 ms** | ~500 ms |
| errors / 503s in all runs | 0 / 0 | 0 / 0 |

Weights: one 42 MB copy shared by all workers (`Arc<Weights>`).

## Correctness (this session, per-engine)

- Gate A vs Python-ORT golden (10 000 names): native mean cos **1.000000**,
  max\|Δ\| **1.94e-7** · ORT max\|Δ\| **1.79e-7** (bar: cos ≥ 0.9999, Δ < 1e-3).
- Cross-engine native↔ORT (1 009 stress rows): max\|Δ\| **1.19e-7** (bar 1e-5).
- Native is **bit-exactly batch-invariant** and deterministic (cargo tests).
- Gate B vs torch checkpoint (ONNX golden path): max\|Δ\| **1.94e-7**,
  Re@10 agreement **1.00000** (bar 1e-3 / ±0.001) — unchanged.

## Notes

- `CT4_BATCH_TIMEOUT_MS` default is **0** for the native engine (batching
  linger no longer pays: single-name forward is ~0.23 ms; ORT needed it).
- torch CT4 reference (h016 eval, 1 CPU thread): 930 n/s @batch256.
- Single-thread native B=64 is **~19× the torch reference** on comparable
  terms (18.4k vs 930 n/s; 4 cores vs 1 thread for the HTTP numbers above).
- Remaining headroom (future): B=1 latency 231 µs is dominated by per-call GEMM
  B-operand packing in `matrixmultiply`; pre-packed B would cut it further
  (NATIVE_RUST_PLAN M4, not needed for current targets).
