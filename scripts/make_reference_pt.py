#!/usr/bin/env python3
"""Gate B reference: run the ORIGINAL torch CT4 (checkpoint h016) over a name
list, producing the ground-truth vectors the ONNX export must reproduce
(T-032: max |Δ| <= 1e-3 on 10k names, Re@10 equal within ±0.001).

Uses the training repo's `CT4Encoder.encode` — the exact production path,
tokenization included (so this is independent of our export tokenizer wiring).

Writes <out>/queries.json + <out>/vectors.npy (float32 [N,256]).
  python scripts/gen_names.py 10000 names_10k.json   # if not present
  python scripts/make_reference_pt.py --names names_10k.json --out reference_pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="model-src/src")
    ap.add_argument("--checkpoint", default="model-src/ct4_h016_state.pt")
    ap.add_argument("--tokenizer", default="model-src/tokenizer.json")
    ap.add_argument("--names", default="names_10k.json")
    ap.add_argument("--out", default="reference_pt")
    ap.add_argument("--batch", type=int, default=2048, help="eval_batch_size of the h016 run")
    args = ap.parse_args()

    import torch

    torch.set_grad_enabled(False)
    src_dir = os.path.abspath(args.src)
    repo_root = os.path.dirname(src_dir)
    for p in (repo_root, src_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    from retrieval.models.conv_hybrid import CT4Encoder
    from retrieval.tokenizers.qwen_bpe import QwenBpeTokenizer

    enc = CT4Encoder(tokenizer=QwenBpeTokenizer(args.tokenizer))
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    enc.load_state_dict(payload["model"])
    enc.eval()
    cfg = json.loads(payload["config_json"])
    assert cfg["params"]["model"] == "ct4"
    print(f"torch CT4 loaded: step={payload['step']}, params={enc.num_parameters():,}")

    names = json.load(open(args.names, encoding="utf-8"))
    print(f"embedding {len(names)} names (batch {args.batch}) …")
    parts = []
    for i in range(0, len(names), args.batch):
        parts.append(enc.encode(names[i:i + args.batch]).cpu().numpy())
        if (i // args.batch) % 5 == 0:
            print(f"  {i + args.batch}/{len(names)}")
    vecs = np.concatenate(parts, axis=0).astype(np.float32)

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    json.dump(names, open(os.path.join(out, "queries.json"), "w"), ensure_ascii=False)
    np.save(os.path.join(out, "vectors.npy"), vecs)
    print(f"wrote {len(names)} names, vectors {vecs.shape} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
