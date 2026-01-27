from __future__ import annotations

from typing import Tuple, cast

import torch
from torch import nn


class MultilabelBinaryAccuracy(nn.Module):
    """
    Element-wise binary accuracy for multi-label classification tasks.

    This metric computes accuracy over all label elements rather than
    per sample. Each label position is treated as an independent
    binary classification problem.

    Assumptions:
        - `outputs` are raw logits with arbitrary shape (N, ...).
        - `targets` are binary tensors (0 or 1) with the same shape
          as `outputs`.
        - A sigmoid activation is applied internally to convert logits
          to probabilities.

    Args:
        threshold (float, optional):
            Threshold used to convert probabilities into binary
            predictions. Default is 0.5.

    Returns:
        Tuple[int, int]:
            - num_correct: Number of correctly predicted label elements.
            - num_total: Total number of label elements.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[int, int]:
        probs = torch.sigmoid(outputs)
        pred = (probs > self.threshold).to(dtype=targets.dtype)

        num_correct = (pred == targets).sum().item()
        num_total = targets.numel()

        return cast(int, num_correct), cast(int, num_total)
