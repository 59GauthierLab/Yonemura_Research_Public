from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .epoch_runner import TargetTransform, _run_epoch


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    metric: nn.Module,
    target_transform: TargetTransform = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[float, float]:
    """Evaluate the model on test data."""
    return _run_epoch(
        model=model,
        data_loader=test_loader,
        optimizer=None,
        criterion=criterion,
        metric=metric,
        device=device,
        target_transform=target_transform,
        verbose=verbose,
        desc="Evaluating",
    )
