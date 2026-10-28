#!/usr/bin/env python3
"""Shared helpers for the CT4-h016 gate scripts (golden / reference / quality).

Tokenization mirrors `retrieval.tokenizers.qwen_bpe.QwenBpeTokenizer.encode_batch`
and `CT4Encoder.encode`'s empty-row guard exactly:
  add_special_tokens=False, truncate to max_len, right-pad with pad_id,
  bool mask (True = real token), zero-token rows get one pad token.
"""
from __future__ import annotations

import json
import os

import numpy as np

PAD_TOKEN = "<|endoftext|>"  # qwen_meta.json
MAX_LEN = 64                 # CT4Config.max_len


def load_tokenizer(model_dir: str):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    pad_id = tok.token_to_id(PAD_TOKEN)
    if pad_id is None:
        raise SystemExit(f"tokenizer has no {PAD_TOKEN!r}")
    return tok, int(pad_id)


def tokenize(tok, pad_id: int, texts: list[str], max_len: int = MAX_LEN):
    """-> (int64 ids [B,L], bool mask [B,L]) with ≥1 column per row."""
    encs = tok.encode_batch(texts, add_special_tokens=False)
    ids_list = [e.ids[:max_len] for e in encs]
    length = max(1, max(len(s) for s in ids_list))
    ids = np.full((len(ids_list), length), pad_id, dtype=np.int64)
    mask = np.zeros((len(ids_list), length), dtype=bool)
    for i, seq in enumerate(ids_list):
        ids[i, : len(seq)] = seq
        mask[i, : len(seq)] = True
    empty = ~mask.any(axis=1)
    if empty.any():  # CT4Encoder.encode's NaN-guard
        ids[empty, 0] = pad_id
        mask[empty, 0] = True
    return ids, mask


def read_names(path: str) -> list[str]:
    return json.load(open(path, encoding="utf-8"))


def read_npy(path: str) -> np.ndarray:
    return np.load(path)


def maxdiff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
