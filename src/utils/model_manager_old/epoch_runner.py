from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

TargetTransform = Optional[Callable[[Tensor], Any]]


def _run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: Optional[optim.Optimizer],
    criterion: nn.Module,
    metric: nn.Module,
    target_transform: TargetTransform = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    desc: str = "Running",
) -> Tuple[float, float]:
    """
    Execute one epoch of training or evaluation.

    This function provides a task-agnostic epoch runner that is shared
    by both training and evaluation pipelines. It performs forward
    passes, loss computation, optional backpropagation, and metric
    aggregation while delegating task-specific logic to injected
    components.

    Design principles:
        - The model is responsible only for producing outputs.
        - The criterion computes a mathematically well-defined loss
          given model outputs and targets.
        - The metric computes a mathematically well-defined evaluation
          statistic given model outputs and targets.
        - Any adjustment of target shape or representation is handled
          exclusively by `target_transform`.

    Args:
        model (nn.Module):
            Model to be trained or evaluated.
        data_loader (torch.utils.data.DataLoader):
            DataLoader yielding batches of (inputs, targets).
        optimizer (optim.Optimizer, optional):
            Optimizer used for training. If None, the model is run
            in evaluation mode without gradient updates.
        criterion (nn.Module):
            Loss function that computes a scalar loss from
            (outputs, targets). It must not assume any specific
            target shape beyond mathematical validity.
        metric (nn.Module):
            Evaluation metric that returns batch-level statistics.
            It must return a tuple of (num_correct, num_total).
        target_transform (Callable[[Tensor], Any], optional):
            Optional function applied to targets before loss and
            metric computation. This is intended for target shape
            or representation adjustment.
        device (torch.device, optional):
            Device on which all computations are performed.
            If None, no device transfer is performed.
        verbose (bool, optional):
            Whether to display a progress bar and intermediate logs.
        desc (str, optional):
            Description shown in the progress bar.

    Returns:
        Tuple[float, float]:
            - avg_loss (float): Average loss per sample over the epoch.
            - metric_value (float): Aggregated metric value (e.g., accuracy in %).

    Notes:
        - Training or evaluation mode is determined solely by whether
          `optimizer` is provided.
        - This function does not encode any task-specific assumptions
          and can be reused across different problem settings
          (e.g., single-label, multi-label, or attribute-based tasks).
    """

    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss, total_samples = 0.0, 0
    correct, total = 0, 0

    loader = tqdm(data_loader, desc=desc, leave=False, disable=not verbose)

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs: Tensor
            targets: Tensor

            # Move data to the specified device if provided
            if device is not None:
                inputs, targets = inputs.to(device), targets.to(device)

            # Apply target transformation if provided
            if target_transform is not None:
                targets = target_transform(targets)

            # Zero gradients if in training mode
            if is_train:
                optimizer.zero_grad()

            # Forward pass
            outputs: Tensor = model(inputs)
            loss: Tensor = criterion(outputs, targets)

            # Backward pass and optimization step if in training mode
            if is_train:
                loss.backward()
                optimizer.step()

            # Accumulate loss
            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)

            # Compute metrics
            c, t = metric(outputs.detach(), targets)
            correct += c
            total += t

            # Update progress bar
            if verbose:
                loader.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Acc": f"{100.0 * correct / total:.2f}%",
                    }
                )

    # Compute average loss and accuracy
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy
