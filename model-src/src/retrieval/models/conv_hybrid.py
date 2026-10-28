"""Энкодер CT4 — гибрид «свёртка + внимание» на Qwen-токенах (T-016, H-006).

Гипотеза H-006: локальные свёрточные признаки (соседние символы имени)
в сочетании с глобальным вниманием дадут качество T4-Q при меньшей латентности.
Реализация удовлетворяет контракту :class:`retrieval.models.base.TextEncoder`
(«тексты → [B, 256] float32, L2-нормированные строки»).

Архитектура
-----------
1. Факторизованная токен-эмбеддинг: словарь Qwen 151 665 × 64 → линейная
   проекция 64 → 256 (экономия параметров — зеркало подхода T4-Q;
   полная матрица 151 665 × 256 стоила бы ~38.8M параметров).
2. Блоки в порядке **C-C-A-C-C-A** (выбор из допустимых по T-016: 4 conv-блока
   и 2 attention-блока). Пара conv-блоков извлекает локальный контекст
   (окно ±2 символа), затем attention-блок перемешивает информацию по всей
   длине имени; этап «2×C → A» повторён дважды, поэтому глобальное смешение
   происходит и в середине, и в конце стека. Порядок фиксируется функцией
   :func:`_build_pattern` (равномерное разбиение conv-блоков на группы
   по одному attention-блоку).
3. Conv-блок ``C``: depthwise ``Conv1d`` (k=5, группы=h, ручной same-паддинг
   по 2 позиции с обеих сторон) → pointwise ``Conv1d`` 1×1 → residual →
   RMSNorm (по одному RMSNorm на блок, post-norm).
4. Attention-блок ``A``: pre-RMSNorm → собственная многоголовая self-attention
   (4 головы, head_dim=64, двунаправленное внимание, ``key_padding_mask`)
   → residual. Позиционная информация — обучаемые абсолютные эмбеддинги
   позиций (max_len=64); RoPE НЕ используется: реализация независима от
   параллельной T4-модели (на момент написания ``tiny_bert`` в репозитории
   отсутствовал), а для имён длиной ≤ 64 фиксированные обучаемые позиции —
   простейшее и достаточное решение.
5. Маскированное mean-пуллинг по настоящим токенам → L2-нормировка в float32.

Батч-инвариантность
-------------------
После эмбеддинга и после КАЖДОГО блока позиции паддинга принудительно
обнуляются (``x = x * mask``). Внимание дополнительно исключает pad-ключи
через ``key_padding_mask`` (−inf до softmax). Благодаря этому вектор имени
не зависит от того, закодирован он один или в батче с более длинными именами
(проверяется тестом T-016), т.к. все операции видят на pad-позициях нули
независимо от длины батча. RMSNorm от нулевого вектора корректно даёт ноль
(деление на ``sqrt(0 + eps)``).

Параметры: ≈ 10.5M (цель T-016 < 15M); dtype по умолчанию — float32,
под ``torch.autocast(bfloat16)`` выход приводится к float32 без NaN.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..tokenizers.qwen_bpe import QwenBpeTokenizer
from .base import TextEncoder, maybe_checkpoint

__all__ = ["CT4Config", "CT4Encoder"]


@dataclass
class CT4Config:
    """Гиперпараметры CT4 (значения по умолчанию — конфиг T-016).

    :param n_conv: число conv-блоков ``C`` (T-016: 4).
    :param n_att: число attention-блоков ``A`` (T-016: 2).
    :param h: скрытая размерность (T-016: 256).
    :param heads: число голов внимания (делит ``h`` нацело).
    :param d_token: размерность факторизованной токен-эмбеддинга (64, как T4-Q).
    :param max_len: максимум токенов на имя (усечение в ``encode_batch``).
    :param vocab_size: размер словаря эмбеддинга (словарь Qwen — 151 665).
    """

    n_conv: int = 4
    n_att: int = 2
    h: int = 256
    heads: int = 4
    d_token: int = 64
    max_len: int = 64
    vocab_size: int = 151_665

    def __post_init__(self) -> None:
        if self.n_conv < 1:
            raise ValueError("требуется n_conv ≥ 1")
        if self.n_att < 0:
            raise ValueError("требуется n_att ≥ 0")
        if self.h < 1 or self.heads < 1 or self.h % self.heads != 0:
            raise ValueError("требуется h ≥ 1, heads ≥ 1 и h % heads == 0")
        if self.d_token < 1 or self.max_len < 1 or self.vocab_size < 1:
            raise ValueError("размерности d_token/max_len/vocab_size должны быть ≥ 1")


def _build_pattern(n_conv: int, n_att: int) -> tuple[str, ...]:
    """Порядок блоков: conv-блоки делятся на ``n_att`` равных групп.

    Для конфига T-016 (4, 2) → группы [2, 2] → ``('C', 'C', 'A', 'C', 'C', 'A')``.
    Если ``n_conv`` не делится на ``n_att`` нацело, первые группы получают
    по одному дополнительному conv-блоку.
    """
    if n_att == 0:
        return ("C",) * n_conv
    base, rem = divmod(n_conv, n_att)
    pattern: list[str] = []
    for i in range(n_att):
        pattern.extend(["C"] * (base + (1 if i < rem else 0)))
        pattern.append("A")
    return tuple(pattern)


class _RMSNorm(nn.Module):
    """RMSNorm с внутренним счётом в float32 (стабильность под autocast)."""

    def __init__(self, h: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(h))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(x.dtype)


class _ConvBlock(nn.Module):
    """Conv-блок ``C``: depthwise k=5 (ручной same-паддинг) → pointwise 1×1
    → residual → RMSNorm.

    Вход/выход: ``[B, L, h]``. Свёртка работает в раскладке ``[B, h, L]``,
    паддинг выполняется явным ``F.pad`` (по 2 позиции с обеих сторон для
    k=5), а не аргументом ``padding`` — ровно как «ручной same-паддинг»
    сформулирован в T-016.
    """

    def __init__(self, h: int, kernel_size: int = 5) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ожидается нечётное ядро (same-паддинг)")
        self.pad = kernel_size // 2
        self.depthwise = nn.Conv1d(h, h, kernel_size, groups=h)
        self.pointwise = nn.Conv1d(h, h, 1)
        self.norm = _RMSNorm(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)  # [B, h, L]
        y = F.pad(y, (self.pad, self.pad))
        y = self.depthwise(y)
        y = self.pointwise(y)
        return self.norm(x + y.transpose(1, 2))  # residual + RMSNorm


class _AttentionBlock(nn.Module):
    """Attention-блок ``A``: pre-RMSNorm → многоголовое двунаправленное
    self-attention (``key_padding_mask``) → residual.

    Вход/выход: ``[B, L, h]``; маска ``[B, L]`` (True = настоящий токен).
    Скалярное произведение и softmax считаются в float32 независимо от
    autocast — NaN от −inf-маски исключены (в каждой строке есть ≥ 1
    незамаскированный ключ). Pad-строки-запросы дают произвольный выход,
    но обнуляются маской в цикле энкодера и не попадают в пуллинг.
    """

    def __init__(self, h: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.d_head = h // heads
        self.norm = _RMSNorm(h)
        self.qkv = nn.Linear(h, 3 * h)
        self.out = nn.Linear(h, h)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        b, l, _ = h.shape
        qkv = (
            self.qkv(h)
            .reshape(b, l, 3, self.heads, self.d_head)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # каждый: [B, heads, L, d_head]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        pad = ~mask  # True = позиция паддинга (запрещённый ключ)
        scores = scores.float().masked_fill(pad[:, None, None, :], float("-inf"))
        attn = F.softmax(scores, dim=-1).to(v.dtype)
        ctx = (attn @ v).transpose(1, 2).reshape(b, l, -1)
        return x + self.out(ctx)


class CT4Encoder(TextEncoder):
    """Энкодер CT4: Qwen-BPE токены → гибрид conv+attention → [B, 256] fp32.

    :param config: гиперпараметры :class:`CT4Config` (по умолчанию — T-016).
    :param tokenizer: обёртка Qwen BPE; по умолчанию — стандартная локальная
        копия (см. :mod:`retrieval.tokenizers.qwen_bpe`).
    """

    #: Имя модели в реестре (``--model ct4``).
    MODEL_NAME = "ct4"

    def __init__(
        self,
        config: CT4Config | None = None,
        tokenizer: QwenBpeTokenizer | None = None,
    ) -> None:
        super().__init__()
        self.config = config or CT4Config()
        self.tokenizer = tokenizer or QwenBpeTokenizer()
        if self.tokenizer.vocab_size != self.config.vocab_size:
            raise ValueError(
                "vocab_size конфига "
                f"({self.config.vocab_size}) не совпадает со словарём "
                f"токенизатора ({self.tokenizer.vocab_size})"
            )
        self.dim = 256

        cfg = self.config
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_token)
        self.proj = nn.Linear(cfg.d_token, cfg.h)
        # Обучаемые абсолютные позиции (RoPE-free дизайн, см. докстринг модуля).
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.h)
        # Стандартная инициализация эмбеддингов (BERT/GPT-конвенция): N(0, 0.02).
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        self.pattern = _build_pattern(cfg.n_conv, cfg.n_att)
        self.blocks = nn.ModuleList(
            _ConvBlock(cfg.h) if kind == "C" else _AttentionBlock(cfg.h, cfg.heads)
            for kind in self.pattern
        )

    # -- контракт TextEncoder -------------------------------------------------

    def encode(
        self,
        texts: Sequence[str],
        device: str | torch.device | None = None,
    ) -> torch.FloatTensor:
        """Тексты → ``[B, 256]`` float32, строки L2-нормированы (см. base.py)."""
        if device is None:
            device = next(self.parameters()).device
        device = torch.device(device)

        # Усечение до max_len происходит внутри encode_batch (T-016).
        ids, mask = self.tokenizer.encode_batch(list(texts), max_len=self.config.max_len)
        # Защита от пустых строк: ряд без настоящего токена получает один
        # pad-токен — иначе masked softmax/пуллинг дали бы NaN.
        empty = ~mask.any(axis=1)
        if empty.any():
            ids[empty, 0] = self.tokenizer.pad_id
            mask[empty, 0] = True

        ids_t = torch.from_numpy(ids).to(device)
        mask_t = torch.from_numpy(mask).to(device)

        b, l = ids_t.shape
        x = self.proj(self.tok_emb(ids_t))
        x = x + self.pos_emb(torch.arange(l, device=device))
        keep = mask_t.unsqueeze(-1).to(x.dtype)  # [B, L, 1]; 1 = настоящий токен
        x = x * keep  # паддинг → нули (основа батч-инвариантности)
        for block in self.blocks:
            if isinstance(block, _AttentionBlock):
                x = maybe_checkpoint(self, block, x, mask_t)
            else:
                x = maybe_checkpoint(self, block, x)
            x = x * keep  # после каждого блока паддинг снова строго нулевой

        # Маскированное mean-пуллинг по настоящим токенам.
        pooled = (x * keep).sum(dim=1) / keep.sum(dim=1)
        # Выход контракта — float32 с L2-нормой 1 (и под autocast).
        return F.normalize(pooled.float(), p=2, dim=-1)
