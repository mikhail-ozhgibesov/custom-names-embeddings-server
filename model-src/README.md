# model-src — vendored model sources (self-sufficiency)

The minimal, versioned inputs needed to **regenerate** the serving model dir
(`model/`) without the training repo or the original archives.

Extracted verbatim from the RND-5 training repo (git `f2f375856ee50439ddef754e2debb9dd3b4c3191`,
see `environment.json` provenance in the h016 run) and the h016 checkpoint:

| File | Origin | What it is |
|---|---|---|
| `src/retrieval/models/conv_hybrid.py` | verbatim | `CT4Encoder` + `CT4Config` — the exact trained encode path |
| `src/retrieval/models/base.py` | verbatim | `TextEncoder` contract + `maybe_checkpoint` (no-op in eval) |
| `src/retrieval/tokenizers/qwen_bpe.py` | verbatim | `QwenBpeTokenizer` wrapper (the tokenization contract) |
| `src/retrieval/tokenizers/qwen_meta.json` | verbatim | pinned Qwen tokenizer revision + sha256 |
| `tokenizer.json` | downloaded per pin | the Qwen2.5-0.5B tokenizer itself (sha256-checked against `qwen_meta.json`); copied into `model/` by the export script |
| `src/retrieval/__init__.py` | verbatim | package docstring |
| `src/retrieval/{models,tokenizers}/__init__.py` | **trimmed** | originals import every model family; here docstrings only |
| `ct4_h016_state.pt` | **extracted** | `best.pt` (step 84 000) reduced to `{step, config_json, model}` — the 10 536 768 state_dict values the model needs; optimizer/scheduler/rng state dropped (126.5 → 42.2 MB) |

`model/` itself (weights.bin, model.onnx, tokenizer.json) is also committed
via **git LFS**, so a clone is runnable without any export step; `model-src/`
is what regenerates it.

Everything else in the training repo (training loop, other model families,
data pipeline) is intentionally NOT vendored — serving and export don't use it.

## Usage

```bash
python scripts/build_model_dir.py --out ./model   # defaults already point here
```

`scripts/make_reference_pt.py` (torch ground truth for Gate B) uses the same
paths. The full training repo remains the place to retrain or evolve the
model; when a new checkpoint is trained, refresh this directory (verbatim
files if the architecture changed, plus a new trimmed state file) and re-run
the export + gates.
