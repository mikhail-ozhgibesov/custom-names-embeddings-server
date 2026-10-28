#!/usr/bin/env python3
"""Gate A golden: run the model dir's `model.onnx` in Python onnxruntime over a
fixed multilingual name sample (edge cases included), producing the reference
the Rust parity test (`tests/parity.rs`) must match: mean cos >= 0.9999,
max |Δ| < 1e-3 (in practice ~1e-7).

Writes <out>/queries.json (list[str]) + <out>/vectors.npy (float32 [N,256]).
  python scripts/make_golden.py --model-dir ./model --out golden_ort
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ct4_common import load_tokenizer, read_names, tokenize

# Edge cases first: empty (pad fallback), single char, >64 tokens (truncation),
# mixed scripts / combining diacritics.
EDGE = ["", "Q", "А" * 400, "Müller", "José Müller-Žarković", "Владимир Путин",
        "محمد بن سلمان", "毛泽东", "김정은", "Ñoño Çağ"]
NAMES_LATIN = ["Vladimir Putin", "Angela Merkel", "Barack Obama", "Olof Palme",
               "Sven-Göran Eriksson", "François Hollande", "Łukasz Kowalski",
               "Đorđe Petrović", "Mỹ Tâm", "Nguyễn Phú Trọng", "Recep Tayyip Erdoğan"]
NAMES_OTHER = ["Владимир Путин", "Дмитрий Медведев", "Українська мова",
               "محمد بن سلمان", "عبد الله", "الجزيرة", "毛泽东", "习近平", "上海",
               "김정은", "서울특별시", "東京", "日本語", "नरेन्द्र मोदी", "भारत",
               "Αθήνα", "Γιώργος", "ישראל", "תל אביב", "ราชอาณาจักรไทย"]
ORGS = ["United Nations", "Banca d'Italia", "Société Générale", "Газпром",
        "中国石油天然气集团", "삼성전자", "Deutsche Bank AG", "نفط الكويت"]


def build_sample() -> list[str]:
    names = EDGE + NAMES_LATIN + NAMES_OTHER + ORGS
    return names + NAMES_LATIN + NAMES_OTHER  # grow N without new data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="./model")
    ap.add_argument("--out", default="golden_ort")
    ap.add_argument("--names", default=None,
                    help="names JSON for Gate B (default: built-in edge-case sample for Gate A)")
    args = ap.parse_args()
    md = os.path.abspath(args.model_dir)

    import onnxruntime as ort

    tok, pad_id = load_tokenizer(md)
    sess = ort.InferenceSession(os.path.join(md, "model.onnx"),
                                providers=["CPUExecutionProvider"])

    names = read_names(args.names) if args.names else build_sample()

    def embed(batch: list[str]) -> np.ndarray:
        ids, mask = tokenize(tok, pad_id, batch)
        return sess.run(["embedding"], {
            "input_ids": ids.astype(np.int32),
            "attention_mask": mask.astype(np.int32),
        })[0].astype(np.float32)

    vecs = np.concatenate([embed(names[i:i + 64]) for i in range(0, len(names), 64)], axis=0)

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    json.dump(names, open(os.path.join(out, "queries.json"), "w"), ensure_ascii=False)
    np.save(os.path.join(out, "vectors.npy"), vecs)
    print(f"wrote {len(names)} names, vectors {vecs.shape} -> {out}")
    print(f"next: cargo test --release --test parity  (Gate A)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
