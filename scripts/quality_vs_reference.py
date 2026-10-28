#!/usr/bin/env python3
"""Gate B — ONNX export vs the torch reference (T-032 acceptance):

  * max |Δ| <= 1e-3 over the shared query set (10k names), AND
  * retrieval Re@10 equal within ±0.001 (self-retrieval on the same set,
    excluding self-hits — the metric the h016 eval reports).

Compares `reference_pt/` (torch) with `golden_ort/` (ONNX) when their query
lists match. The Rust server is chained to the same ONNX by Gate A, so
server == torch transitively.

  python scripts/gen_names.py 10000 names_10k.json
  python scripts/make_reference_pt.py --names names_10k.json --out reference_pt
  python scripts/make_golden.py --names names_10k.json --out golden_ort
  python scripts/quality_vs_reference.py
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ct4_common import maxdiff, read_names

MAX_DIFF_BAR = 1e-3
RE10_BAR = 1e-3  # ±0.001


def top10_sets(vectors: np.ndarray, chunk: int = 512) -> np.ndarray:
    """[N,10] indices of the 10 nearest neighbors (self excluded) per row."""
    n = vectors.shape[0]
    out = np.empty((n, 10), dtype=np.int64)
    for i in range(0, n, chunk):
        sims = vectors[i:i + chunk] @ vectors.T          # [c, N]
        rows = np.arange(i, min(i + chunk, n))
        sims[np.arange(len(rows)), rows] = -np.inf       # exclude self
        out[i:i + chunk] = np.argpartition(-sims, 10, axis=1)[:, :10]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="golden_ort")
    ap.add_argument("--reference", default="reference_pt")
    args = ap.parse_args()

    g_names = read_names(os.path.join(args.golden, "queries.json"))
    r_names = read_names(os.path.join(args.reference, "queries.json"))
    if g_names != r_names:
        shared = len(set(g_names) & set(r_names))
        raise SystemExit(
            f"query sets differ ({len(g_names)} vs {len(r_names)} names, {shared} shared) — "
            f"regenerate both from the same names file (see this script's docstring)."
        )
    names = g_names

    g = np.load(os.path.join(args.golden, "vectors.npy"))
    r = np.load(os.path.join(args.reference, "vectors.npy"))
    assert g.shape == r.shape, f"shape mismatch {g.shape} vs {r.shape}"

    # Deduplicate (keep first occurrence): identical input strings embed to
    # identical vectors, and their arbitrary mutual order among themselves
    # would churn top-10 sets without any retrieval-quality meaning.
    seen: set[str] = set()
    keep = [i for i, n in enumerate(names) if not (n in seen or seen.add(n))]
    names = [names[i] for i in keep]
    g, r = g[keep], r[keep]

    d = maxdiff(g, r)
    cos = np.sum(g * r, axis=1)
    print(f"N={len(names)} (deduped) dim={g.shape[1]}")
    print(f"max|Δ| = {d:.3e}   (bar ≤ {MAX_DIFF_BAR:.0e})")
    print(f"cos    : mean={cos.mean():.8f} min={cos.min():.8f}")
    ok_diff = d <= MAX_DIFF_BAR

    # Re@10 agreement (T-032: equal within ±0.001): recall of the torch
    # reference's top-10 neighbors inside the ONNX top-10, averaged over
    # queries. 1 - mean overlap bounds the Re@10 delta for any retrieval
    # metric whose positives are the reference neighbors. (fp-level |Δ| ~1e-7
    # can still legitimately reorder near-ties, so exact set identity is only
    # reported, not gated.)
    tg = top10_sets(g.astype(np.float32))
    tr = top10_sets(r.astype(np.float32))
    overlap = np.mean([
        len(set(a) & set(b)) / 10.0 for a, b in zip(tg.tolist(), tr.tolist())
    ])
    identical = np.mean([set(a) == set(b) for a, b in zip(tg.tolist(), tr.tolist())])
    re10_delta = 1.0 - overlap
    print(f"top-10 overlap (Re@10 agreement) = {overlap:.5f}  → delta {re10_delta:.5f} (bar ≤ {RE10_BAR})")
    print(f"top-10 sets identical (info only, near-ties reorder) = {identical*100:.2f}%")
    ok_re = re10_delta <= RE10_BAR

    print("GATE B:", "PASS" if (ok_diff and ok_re) else "FAIL")
    return 0 if (ok_diff and ok_re) else 1


if __name__ == "__main__":
    raise SystemExit(main())
