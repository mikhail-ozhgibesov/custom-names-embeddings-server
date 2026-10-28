#!/usr/bin/env python3
"""Benchmark the custom-names-embeddings-server over three submission modes.

Modes:
  sequential  : one HTTP request per name, one at a time.
  concurrent C: one HTTP request per name, C worker threads in parallel.
  batch     B : one HTTP request carrying B names at once, sequential requests.

For each mode we measure wall-clock total time to embed ALL 5000 names and
report throughput (names/s). For per-request modes we also collect per-request
latencies for p50/p95/p99.

Usage:
    python scripts/bench.py [--url URL] [--names PATH] [--warmup-sec N]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8080/embeddings")
    p.add_argument("--names", default="names_5k.json")
    p.add_argument("--warmup-sec", type=int, default=10)
    p.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50, 100, 200, 500],
    )
    p.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[5, 10, 50, 100, 500, 1000, 5000],
    )
    p.add_argument("--out", default="bench_results.json")
    return p.parse_args()


def wait_for_server(url: str, warmup_sec: int) -> None:
    deadline = time.time() + warmup_sec
    health = url.rsplit("/", 1)[0] + "/health"
    last = None
    while time.time() < deadline:
        try:
            r = requests.get(health, timeout=2)
            if r.status_code == 200:
                # Warmup the session with one real request.
                requests.post(url, json={"input": "warmup"}).raise_for_status()
                return
            last = f"health {r.status_code}"
        except Exception as e:
            last = str(e)
        time.sleep(0.5)
    raise SystemExit(f"server not ready at {health}: {last}")


def post_one(session: requests.Session, url: str, name: str) -> tuple[float, int]:
    """POST a single-name request; return (elapsed_s, status_code)."""
    t0 = time.perf_counter()
    r = session.post(url, json={"input": name})
    return time.perf_counter() - t0, r.status_code


def post_batch(session: requests.Session, url: str, batch: list[str]) -> tuple[float, int]:
    """POST a batch request; return (elapsed_s, status_code)."""
    t0 = time.perf_counter()
    r = session.post(url, json={"input": batch})
    return time.perf_counter() - t0, r.status_code


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def bench_sequential(url: str, names: list[str]) -> dict:
    session = requests.Session()
    t0 = time.perf_counter()
    lat_status = [post_one(session, url, n) for n in names]
    total = time.perf_counter() - t0
    session.close()
    lats = [lat for lat, sc in lat_status if sc == 200]
    n_err = sum(1 for _, sc in lat_status if sc != 200)
    return {
        "mode": "sequential",
        "total_s": total,
        "n": len(names),
        "n_errors": n_err,
        "throughput_names_per_s": len(names) / total,
        "p50_ms": pct(lats, 50) * 1000,
        "p95_ms": pct(lats, 95) * 1000,
        "p99_ms": pct(lats, 99) * 1000,
    }


def bench_concurrent(url: str, names: list[str], c: int) -> dict:
    session = requests.Session()
    lat_status: list[tuple[float, int]] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=c) as ex:
        futs = [ex.submit(post_one, session, url, n) for n in names]
        for f in as_completed(futs):
            lat_status.append(f.result())
    total = time.perf_counter() - t0
    session.close()
    lats = [lat for lat, sc in lat_status if sc == 200]
    n_err = sum(1 for _, sc in lat_status if sc != 200)
    return {
        "mode": "concurrent",
        "concurrency": c,
        "total_s": total,
        "n": len(names),
        "n_errors": n_err,
        "throughput_names_per_s": len(names) / total,
        "p50_ms": pct(lats, 50) * 1000,
        "p95_ms": pct(lats, 95) * 1000,
        "p99_ms": pct(lats, 99) * 1000,
    }


def bench_batch(url: str, names: list[str], batch_size: int) -> dict:
    session = requests.Session()
    batches = [names[i : i + batch_size] for i in range(0, len(names), batch_size)]
    lat_status: list[tuple[float, int]] = []
    t0 = time.perf_counter()
    for b in batches:
        lat_status.append(post_batch(session, url, b))
    total = time.perf_counter() - t0
    session.close()
    lats = [lat for lat, sc in lat_status if sc == 200]
    n_err = sum(1 for _, sc in lat_status if sc != 200)
    return {
        "mode": "batch",
        "batch_size": batch_size,
        "n_batches": len(batches),
        "total_s": total,
        "n": len(names),
        "n_errors": n_err,
        "throughput_names_per_s": len(names) / total if total else 0,
        "p50_req_ms": pct(lats, 50) * 1000,
        "p95_req_ms": pct(lats, 95) * 1000,
        "p99_req_ms": pct(lats, 99) * 1000,
    }


def main() -> None:
    args = parse_args()
    names = json.loads(Path(args.names).read_text())
    print(f"loaded {len(names)} names from {args.names}")
    wait_for_server(args.url, args.warmup_sec)

    results: list[dict] = []

    print("\n[1/3] sequential (one request per name, one at a time)")
    r = bench_sequential(args.url, names)
    results.append(r)
    print(
        f"  total={r['total_s']:.2f}s  thr={r['throughput_names_per_s']:.1f} n/s  "
        f"p50={r['p50_ms']:.1f}ms p95={r['p95_ms']:.1f}ms p99={r['p99_ms']:.1f}ms"
    )

    print("\n[2/3] concurrent sweep (one request per name, C workers)")
    for c in args.concurrency:
        r = bench_concurrent(args.url, names, c)
        results.append(r)
        err = f"  ERR={r['n_errors']}" if r["n_errors"] else ""
        print(
            f"  C={c:>4d}  total={r['total_s']:6.2f}s  thr={r['throughput_names_per_s']:7.1f} n/s  "
            f"p50={r['p50_ms']:6.1f}ms p95={r['p95_ms']:6.1f}ms p99={r['p99_ms']:6.1f}ms{err}"
        )

    print("\n[3/3] batch sweep (one request per B-name batch, sequential)")
    for b in args.batch_sizes:
        r = bench_batch(args.url, names, b)
        results.append(r)
        err = f"  ERR={r['n_errors']}" if r["n_errors"] else ""
        print(
            f"  B={b:>5d}  ({r['n_batches']:>4d} reqs)  total={r['total_s']:6.2f}s  "
            f"thr={r['throughput_names_per_s']:7.1f} n/s  "
            f"p50_req={r['p50_req_ms']:7.1f}ms p95_req={r['p95_req_ms']:7.1f}ms{err}"
        )

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
