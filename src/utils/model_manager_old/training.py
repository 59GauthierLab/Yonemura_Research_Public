from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .epoch_runner import TargetTransform, _run_epoch
from .evaluation import evaluate_model


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    metric: nn.Module,
    target_transform: TargetTransform = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[float, float]:
    """Train the model for one epoch."""
    return _run_epoch(
        model=model,
        data_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        metric=metric,
        target_transform=target_transform,
        device=device,
        verbose=verbose,
        desc="Training",
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: Optional[DataLoader],
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    metric: nn.Module,
    epochs: int,
    target_transform: TargetTransform = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, list]:
    """
    Train the model for multiple epochs with optional validation.
    """
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(epochs):
        if verbose:
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 30)

        train_loss, train_acc = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            metric=metric,
            target_transform=target_transform,
            device=device,
            verbose=verbose,
        )
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)

        if test_loader is not None:
            val_loss, val_acc = evaluate_model(
                model=model,
                test_loader=test_loader,
                criterion=criterion,
                metric=metric,
                target_transform=target_transform,
                device=device,
                verbose=verbose,
            )
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            if verbose:
                print(
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"
                )
                print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        else:
            if verbose:
                print(
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"
                )

    return history
