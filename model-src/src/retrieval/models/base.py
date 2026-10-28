"""Контракт модели «текст → эмбеддинг» для всех архитектур RND-5 (T-011).

Единый цикл обучения (:mod:`retrieval.train`) и оценщики ничего не знают об
архитектурах: они работают только с этим контрактом. Все модели Phase 1–2
(NGRAM, T4-Q, BT4, CT4, CNN6, …) реализуют ровно его.

Контракт ``encode``
-------------------
``encode(texts, device=None) -> torch.FloatTensor``:

- вход: непустая последовательность строк (``list[str]``) произвольной длины
  и письменности (unicode); токенизация/пэддинг/пуллинг — ВНУТРИ модели,
  цикл обучения лишь передаёт сырые тексты;
- выход: тензор ``torch.float32`` формы ``[B, self.dim]`` (dim = 256),
  строки L2-нормированы (нормировка выполняется в float32);
- ``device``: ``str | torch.device`` устройства входа/выхода; ``None`` —
  устройство параметров модели. Модель должна быть заранее перенесена на
  это устройство (``.to(device)``) вызывающим кодом;
- вызывается и с градиентами (тренировка), и под ``torch.autocast``
  (модель обязана привести выход к float32), и внутри
  ``torch.inference_mode()`` (оценка);
- детерминизм: в режиме ``eval()`` на одном устройстве повторный вызов с
  теми же текстами даёт те же векторы (нет стохастики вроде dropout).

Дополнительно каждая модель обязана:

- иметь атрибут ``dim: int`` — размерность эмбеддинга;
- наследовать :meth:`num_parameters` — счётчик параметров для отчётов.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

__all__ = ["TextEncoder", "maybe_checkpoint"]


def maybe_checkpoint(module: torch.nn.Module, fn: Callable, *args, **kwargs):
    """Прогон блока модели под activation checkpointing при обучении.

    В режиме ``training`` при активных градиентах блок выполняется через
    ``torch.utils.checkpoint`` (use_reentrant=False): внутренние активации
    не сохраняются, а пересчитываются в backward — это вдвое-втрое снижает
    пик VRAM при неизменной математике (тот же батч, тот же лосс; dropout
    у моделей Phase 1 отключён, поэтому пересчёт детерминирован).
    В eval/inference_mode/без градиентов блок вызывается напрямую —
    на память и скорость оценки это не влияет.
    """
    if module.training and torch.is_grad_enabled():
        return _torch_checkpoint(fn, *args, use_reentrant=False, **kwargs)
    return fn(*args, **kwargs)


class TextEncoder(torch.nn.Module):
    """Базовый класс контракта «текст → 256-мерный L2-нормированный вектор».

    См. докстринг модуля — точная спецификация :meth:`encode`.
    """

    #: Размерность эмбеддинга (переопределяется в подклассах при желании).
    dim: int = 256

    def encode(
        self,
        texts: Sequence[str],
        device: str | torch.device | None = None,
    ) -> torch.FloatTensor:
        """Закодировать тексты → ``[B, dim]`` float32, L2-нормированные строки.

        Точная спецификация — в докстринге модуля. Подклассы обязаны
        реализовать метод без обращения к ``super()``.
        """
        raise NotImplementedError

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Число параметров модели (по умолчанию — только обучаемых)."""
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad or not trainable_only
        )
