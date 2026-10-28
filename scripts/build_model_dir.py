#!/usr/bin/env python3
"""Build the CT4 (h016) model dir for the server: checkpoint -> ONNX.

Exports the EXACT eval-mode encode path of `retrieval.models.conv_hybrid.
CT4Encoder` (minus tokenization) as one ONNX graph:

    (input_ids int32 [B,L], attention_mask int32 [B,L])
      -> tok_emb(Vx64) -> proj(64->256) -> +pos_emb
      -> x*keep -> [C C A C C A blocks, x*keep after each]
      -> masked mean pool -> L2 normalize (float32)
      -> embedding [B, 256]

Conventions fixed by docs/tasks.md T-032: opset >= 17, int32 ids + mask.
Batch-invariance (pad-zeroing + mask-aware pooling) lives INSIDE the graph —
the server only has to feed the mask it derives from token lengths.

In-script verification (fails loudly):
  1. wrapper(torch) == CT4Encoder.encode(texts)            (wrapper fidelity)
  2. onnxruntime     == CT4Encoder.encode(texts)           (export fidelity)
  3. batch-invariance: embedding of a name alone == within a ragged batch,
     for both torch and ONNX (T-016 padding discipline).

Inputs (defaults below; all are vendored in this repo — no external archives
needed):
  --src          minimal training-repo subset `model-src/src` (CT4Encoder verbatim)
  --checkpoint   `model-src/ct4_h016_state.pt` (state_dict + config_json + step)
  --tokenizer    `model/tokenizer.json` (the pinned, sha256-checked Qwen copy)
  --out          output dir (default ./model)

Produces: model.onnx, tokenizer.json, meta.json.

Requirements: torch, tokenizers, onnx, onnxruntime (scripts/requirements-model.txt).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MAX_LEN = 64
OPSET = 17

# Set by load_encoder() once the training repo is importable (lazy because
# sys.path gets --src prepended first).
_AttentionBlock = None


def dump_native_weights(enc, out_dir: str, onnx_sha: str) -> dict:
    """M0 (NATIVE_RUST_PLAN §3.1): emit weights.bin + weights_manifest.json in
    EXECUTION layout so the Rust native backend mmaps and computes, no ONNX
    parser. Linear weights stored transposed [in, out] (sgemm B operand);
    depthwise stored k-major [5, C]; provenance chained to the checkpoint.
    """
    _Att = _AttentionBlock
    tensors: list[tuple[str, "np.ndarray", str]] = []

    def add(name, arr, layout):
        tensors.append((name, np.ascontiguousarray(arr, dtype=np.float32), layout))

    add("tok_emb", enc.tok_emb.weight.detach().numpy(), "[V,64] row-major (gather rows)")
    add("proj_w", enc.proj.weight.detach().numpy().T.copy(), "[64,256] = W^T (sgemm B)")
    add("proj_b", enc.proj.bias.detach().numpy(), "[256]")
    add("pos_emb", enc.pos_emb.weight.detach().numpy(), "[64,256] rows by position")

    blocks_meta = []
    for i, blk in enumerate(enc.blocks):
        if isinstance(blk, _Att):
            add(f"blk{i}.norm_w", blk.norm.weight.detach().numpy(), "[256]")
            add(f"blk{i}.qkv_w", blk.qkv.weight.detach().numpy().T.copy(), "[256,768] = W^T")
            add(f"blk{i}.qkv_b", blk.qkv.bias.detach().numpy(), "[768]")
            add(f"blk{i}.out_w", blk.out.weight.detach().numpy().T.copy(), "[256,256] = W^T")
            add(f"blk{i}.out_b", blk.out.bias.detach().numpy(), "[256]")
            blocks_meta.append({"index": i, "kind": "att", "heads": enc.config.heads})
        else:
            add(f"blk{i}.norm_w", blk.norm.weight.detach().numpy(), "[256]")
            dw = blk.depthwise.weight.detach().numpy().reshape(enc.config.h, -1)  # [C,k]
            add(f"blk{i}.dw_w", dw.T.copy(), "[k=5,C] k-major (SIMD over channels)")
            add(f"blk{i}.dw_b", blk.depthwise.bias.detach().numpy(), "[256]")
            pw = blk.pointwise.weight.detach().numpy().reshape(enc.config.h, enc.config.h)  # [out,in]
            add(f"blk{i}.pw_w", pw.T.copy(), "[256,256] = W^T (sgemm B)")
            add(f"blk{i}.pw_b", blk.pointwise.bias.detach().numpy(), "[256]")
            blocks_meta.append({"index": i, "kind": "conv", "kernel": 5})

    params = sum(t[1].size for t in tensors)
    assert params == enc.num_parameters(), (params, enc.num_parameters())

    blob = bytearray()
    entries = []
    for name, arr, layout in tensors:
        off = len(blob) // 4
        blob += arr.tobytes()
        entries.append({"name": name, "offset": off, "numel": int(arr.size),
                        "shape": list(arr.shape), "layout": layout})

    import hashlib as _h

    bin_path = os.path.join(out_dir, "weights.bin")
    with open(bin_path, "wb") as fh:
        fh.write(blob)
    manifest = {
        "format": 1,
        "provenance": {
            "checkpoint_step": None,  # filled by caller
            "model_onnx_sha256": onnx_sha,
            "weights_bin_sha256": _h.sha256(blob).hexdigest(),
        },
        "dims": {"vocab": int(enc.config.vocab_size), "d_token": int(enc.config.d_token),
                 "h": int(enc.config.h), "heads": int(enc.config.heads),
                 "head_dim": int(enc.config.h // enc.config.heads),
                 "max_len": int(enc.config.max_len), "dim": int(enc.dim)},
        "pattern": list(enc.pattern),
        "blocks": blocks_meta,
        "tensors": entries,
        "param_total": int(params),
        "dtype": "f32-le",
    }
    with open(os.path.join(out_dir, "weights_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest

# Multilingual battery incl. the edge cases that matter for CT4: empty string
# (zero-token row -> pad fallback), single char, >64 tokens (truncation),
# mixed scripts.
BATTERY = [
    "", "Q", "А" * 400,
    "Vladimir Putin", "Владимир Путин", "Володимир Путін",
    "محمد بن سلمان", "毛泽东", "김정은", "東京", "Müller", "Ñoño Çağ",
    "Sven-Göran Eriksson", "Nguyễn Phú Trọng", "Газпром", "삼성전자",
    "Иванов Иван Иванович", "Ana María Fernández-García y López",
]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CT4Export(nn.Module):
    """CT4 encode path with tokenization lifted out: (int32 ids, int32 mask)
    -> [B, 256] L2-normalized float32. Mirrors CT4Encoder.encode EXACTLY
    (same modules, same x*keep discipline, same pooling); in eval mode
    `maybe_checkpoint` is a plain call, so the block loop is the plain path.
    """

    def __init__(self, enc) -> None:
        super().__init__()
        self.enc = enc

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        enc = self.enc
        ids = input_ids.long()
        mask = attention_mask.bool()
        b, l = ids.shape

        x = enc.proj(enc.tok_emb(ids))
        x = x + enc.pos_emb(torch.arange(l))
        keep = mask.unsqueeze(-1).to(x.dtype)  # [B, L, 1]; 1 = real token
        x = x * keep  # padding -> zeros (basis of batch-invariance)
        for block in enc.blocks:
            x = block(x, mask) if isinstance(block, _AttentionBlock) else block(x)
            x = x * keep  # after every block padding is strictly zero again

        pooled = (x * keep).sum(dim=1) / keep.sum(dim=1)
        return F.normalize(pooled.float(), p=2, dim=-1)


def load_encoder(src_dir: str, ckpt_path: str, tok_path: str):
    global _AttentionBlock
    # Vendored minimal package (model-src/src) — `retrieval.*` imports only;
    # both the package root and its parent are importable for compatibility
    # with a full training-repo layout.
    src_dir = os.path.abspath(src_dir)
    repo_root = os.path.dirname(src_dir)
    for p in (repo_root, src_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    from retrieval.models.conv_hybrid import CT4Encoder, _AttentionBlock as AttBlock
    from retrieval.tokenizers.qwen_bpe import QwenBpeTokenizer

    _AttentionBlock = AttBlock
    tok = QwenBpeTokenizer(tok_path)
    enc = CT4Encoder(tokenizer=tok)
    payload_step = load_state(enc, ckpt_path)
    enc.eval()
    return enc, payload_step


def load_state(enc, ckpt_path: str) -> int:
    # Inline of retrieval.train.load_checkpoint (which imports polars/pyarrow
    # for the training loop — unnecessary here). Same semantics for the parts
    # we use: CPU load, strict state_dict, config_json metadata.
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc.load_state_dict(payload["model"])
    cfg = json.loads(payload["config_json"])
    assert cfg["params"]["model"] == "ct4", f"checkpoint is {cfg['params']['model']!r}, expected 'ct4'"
    return int(payload["step"])


def tokenize(tok, texts: list[str]):
    """Mirror CT4Encoder.encode's tokenization: no specials, truncate 64,
    right-pad to batch max, bool mask; zero-token rows get one pad token."""
    encs = tok._tok.encode_batch(texts, add_special_tokens=False)
    ids_list = [e.ids[:MAX_LEN] for e in encs]
    length = max(1, max(len(s) for s in ids_list))  # ≥1 col: all-empty batch needs the pad-fallback slot
    pad = tok.pad_id
    ids = np.full((len(ids_list), length), pad, dtype=np.int64)
    mask = np.zeros((len(ids_list), length), dtype=bool)
    for i, seq in enumerate(ids_list):
        ids[i, : len(seq)] = seq
        mask[i, : len(seq)] = True
    empty = ~mask.any(axis=1)
    if empty.any():
        ids[empty, 0] = pad
        mask[empty, 0] = True
    return ids, mask


def maxdiff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="model-src/src")
    ap.add_argument("--checkpoint", default="model-src/ct4_h016_state.pt")
    ap.add_argument("--tokenizer", default="model-src/tokenizer.json")
    ap.add_argument("--out", default="./model")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    print(f"[1/4] loading CT4 from {args.checkpoint} …")
    enc, step = load_encoder(os.path.abspath(args.src), args.checkpoint, args.tokenizer)
    n_params = enc.num_parameters()
    print(f"      step={step}, params={n_params:,} (expect 10,536,768), pattern={'-'.join(enc.pattern)}")
    assert n_params == 10_536_768, "param count mismatch — wrong checkpoint/config?"

    wrapper = CT4Export(enc).eval()

    def torch_embed(texts: list[str]) -> np.ndarray:
        return enc.encode(texts).cpu().numpy()

    def wrapper_embed(texts: list[str]) -> np.ndarray:
        ids, mask = tokenize(enc.tokenizer, texts)
        t = wrapper(torch.from_numpy(ids.astype(np.int32)), torch.from_numpy(mask.astype(np.int32)))
        return t.cpu().numpy()

    print("[2/4] wrapper fidelity (wrapper == CT4Encoder.encode) …")
    ref = torch_embed(BATTERY)
    wrp = wrapper_embed(BATTERY)
    d = maxdiff(ref, wrp)
    print(f"      max|Δ| = {d:.3e}")
    assert d == 0.0, "export wrapper diverges from CT4Encoder.encode"

    print("[3/4] exporting ONNX (opset %d, int32 ids+mask) …" % OPSET)
    onnx_path = os.path.join(out, "model.onnx")
    for f in ("model.onnx", "model.onnx.data"):
        p = os.path.join(out, f)
        if os.path.exists(p):
            os.remove(p)
    ids0, mask0 = tokenize(enc.tokenizer, BATTERY)
    torch.onnx.export(
        wrapper,
        (torch.from_numpy(ids0.astype(np.int32)), torch.from_numpy(mask0.astype(np.int32))),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"],
        opset_version=OPSET,
        dynamo=False,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "embedding": {0: "batch"},
        },
    )

    print("[4/4] ONNX fidelity vs torch (incl. batch-invariance) …")
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["input_ids", "attention_mask"]
    assert [i.type for i in sess.get_inputs()] == ["tensor(int32)", "tensor(int32)"]

    def onnx_embed(texts: list[str]) -> np.ndarray:
        ids, mask = tokenize(enc.tokenizer, texts)
        return sess.run(["embedding"], {
            "input_ids": ids.astype(np.int32),
            "attention_mask": mask.astype(np.int32),
        })[0]

    oe = onnx_embed(BATTERY)
    d = maxdiff(ref, oe)
    cos = np.sum(ref * oe, axis=1).min()
    print(f"      vs torch: max|Δ| = {d:.3e}, min cos = {cos:.8f}")
    assert d <= 1e-5 and cos >= 0.999999, f"ONNX diverges from torch: max|Δ|={d:.3e}"

    # Batch-invariance: a name embedded alone must equal its row inside a
    # ragged batch (short name + 400-char outlier), torch AND onnx.
    # NOTE: enc.encode([""]) alone trips an upstream IndexError in the training
    # repo (ids has zero columns, pad-fallback indexes axis 1) — the wrapper's
    # tokenizer applies the same pad-fallback and its math is verified identical
    # to encode() above, so it stands in for the torch side on the empty case.
    for name in ("Иванов Иван", "Q", ""):
        alone_t = wrapper_embed([name])[0]
        alone_o = onnx_embed([name])[0]
        batch_t = torch_embed([name, "А" * 400, "Gazprom 000"])[0]
        batch_o = onnx_embed([name, "А" * 400, "Gazprom 000"])[0]
        assert maxdiff(alone_t, batch_t) <= 1e-6, f"torch not batch-invariant for {name!r}"
        assert maxdiff(alone_o, batch_o) <= 1e-6, f"onnx not batch-invariant for {name!r}"
    print("      batch-invariance OK (torch + onnx)")

    # Assemble the model dir.
    import shutil

    shutil.copyfile(args.tokenizer, os.path.join(out, "tokenizer.json"))

    print("[extra] dumping native weights.bin + manifest …")
    manifest = dump_native_weights(enc, out, sha256(onnx_path))
    manifest["provenance"]["checkpoint_step"] = step
    with open(os.path.join(out, "weights_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    meta = {
        "model": "ct4-h016",
        "arch": "ct4",
        "source_repo_task": "T-016 / H-006 (h016)",
        "checkpoint": os.path.basename(args.checkpoint),
        "checkpoint_step": step,
        "params": n_params,
        "pattern": "-".join(enc.pattern),
        "dim": 256,
        "max_len": MAX_LEN,
        "pad_id": enc.tokenizer.pad_id,
        "vocab_size": enc.tokenizer.vocab_size,
        "opset": OPSET,
        "tokenizer_sha256": sha256(args.tokenizer),
        "model_onnx_sha256": sha256(onnx_path),
        "weights_bin_sha256": manifest["provenance"]["weights_bin_sha256"],
    }
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nmodel dir ready: {out}")
    for f in sorted(os.listdir(out)):
        sz = os.path.getsize(os.path.join(out, f))
        print(f"  {f:16s} {sz/1024/1024:.1f} MiB" if sz > 1e6 else f"  {f:16s} {sz} B")
    print("\nrun:  CT4_MODEL_DIR=%s cargo run --release" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
