"""Обёртка над Qwen BPE (tokenizer.json) для моделей T4-Q / CT4.

Зачем обёртка: единый источник токен-идов и паддинга для всех моделей,
работающих на Qwen-словаре (docs/tasks.md T-013/T-016). Qwen — byte-level
BPE: UNK-токен отсутствует по построению (любой байт покрывается fallback),
что проверяется тестом T-013.

Файл tokenizer.json не хранится в git (правило: артефакты моделей — вне git,
docs/rules.md §3). Его скачивает ``download_qwen.py`` в
``models/tokenizers/<name>/``; контрольная сумма фиксируется в
``qwen_meta.json`` рядом с этим модулем — так установка воспроизводима.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

#: Метаданные зафиксированного токенизатора (репозиторий HF + sha256 файла).
_META_PATH = Path(__file__).with_name("qwen_meta.json")

#: Дефолтный корень для артефактов моделей (gitignored).
DEFAULT_MODELS_ROOT = Path("models") / "tokenizers"


def _load_meta() -> dict:
    with open(_META_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def qwen_tokenizer_path(models_root: str | Path | None = None) -> Path:
    """Путь к tokenizer.json локальной копии Qwen (без проверки существования)."""
    meta = _load_meta()
    root = Path(models_root) if models_root else DEFAULT_MODELS_ROOT
    return root / meta["local_name"] / "tokenizer.json"


class QwenBpeTokenizer:
    """Обёртка ``tokenizers.Tokenizer``: ид-ы, маска, паддинг, усечение.

    :param path: путь к tokenizer.json (по умолчанию — стандартная локальная
        копия через :func:`qwen_tokenizer_path`).
    """

    def __init__(self, path: str | Path | None = None):
        resolved = Path(path) if path else qwen_tokenizer_path()
        if not resolved.exists():
            raise FileNotFoundError(
                f"{resolved} не найден; выполните `python -m "
                "src.retrieval.tokenizers.download_qwen` для скачивания "
                f"(repo {_load_meta()['repo_id']})"
            )
        self._tok = Tokenizer.from_file(str(resolved))
        self._pad_id = self._tok.token_to_id(_load_meta()["pad_token"])
        if self._pad_id is None:
            raise ValueError("pad-токен не найден в словаре Qwen")

    @property
    def vocab_size(self) -> int:
        """Размер словаря (с учётом всех спец-токенов)."""
        return self._tok.get_vocab_size(with_added_tokens=True)

    @property
    def pad_id(self) -> int:
        """Ид паддинга (``<|endoftext|>`` у Qwen)."""
        return int(self._pad_id)

    def encode(self, text: str, max_len: int = 512) -> list[int]:
        """Текст → ид-ы без спец-токенов, усечение до max_len."""
        enc = self._tok.encode(text, add_special_tokens=False)
        return enc.ids[:max_len]

    def encode_unknown_count(self, text: str) -> int:
        """Число UNK-токенов в разметке (у byte-level BPE всегда 0)."""
        unk_id = self._tok.token_to_id("<unk>")
        if unk_id is None:
            return 0
        return sum(1 for tid in self.encode(text) if tid == unk_id)

    def encode_batch(
        self,
        texts: list[str],
        max_len: int = 512,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Батч → (``np.int64 [B, L]`` ид-ов, ``np.bool_ [B, L]`` маска).

        Паддинг — справа значением :attr:`pad_id`; маска True = настоящий
        токен. Токенизация батчем идёт внутри Rust-ядра (быстро даже на 1 ядре).
        """
        if not texts:
            raise ValueError("пустой батч недопустим")
        encs = self._tok.encode_batch(texts, add_special_tokens=False)
        ids_list = [enc.ids[:max_len] for enc in encs]
        batch = len(ids_list)
        length = max(len(seq) for seq in ids_list)
        ids = np.full((batch, length), self._pad_id, dtype=np.int64)
        mask = np.zeros((batch, length), dtype=bool)
        for i, seq in enumerate(ids_list):
            ids[i, : len(seq)] = seq
            mask[i, : len(seq)] = True
        return ids, mask
