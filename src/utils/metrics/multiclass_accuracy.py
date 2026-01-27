from __future__ import annotations

from typing import Tuple, cast

import torch
from torch import nn


class MulticlassAccuracy(nn.Module):
    """
    Accuracy metric for multi-class classification tasks.

    This metric computes the number of correctly classified samples
    and the total number of samples in a batch.

    Assumptions:
        - `outputs` are raw logits with shape (N, C), where
          N is the batch size and C is the number of classes.
        - `targets` are class indices with shape (N,), where
          each value is in the range [0, C-1].

    Returns:
        Tuple[int, int]:
            - num_correct: Number of correctly predicted samples.
            - num_total: Total number of samples in the batch.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[int, int]:
        # Predicted class indices
        pred = outputs.argmax(dim=1)

        # Count correct predictions
        num_correct = (pred == targets).sum().item()
        num_total = targets.size(0)

        return cast(int, num_correct), cast(int, num_total)
