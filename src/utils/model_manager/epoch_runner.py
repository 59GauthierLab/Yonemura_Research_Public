from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeAlias,
)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

TargetTransform: TypeAlias = Optional[Callable[[Tensor], Any]]


class _SchedulerProtocol(Protocol):
    def step(self) -> None: ...


SchedulerProtocol = _SchedulerProtocol | lr_scheduler.ReduceLROnPlateau


def all_same_length(*lists: Sequence[Any]) -> bool:
    """Check if all lists have the same length."""
    if not lists:
        return True
    first_length = len(lists[0])
    return all(len(lst) == first_length for lst in lists)


@dataclass(frozen=True)
class EpochRunResult:
    loss: float
    accuracy: float
    lr: Optional[float] = None


def train(
    data_loader: DataLoader,
    models: Sequence[nn.Module],
    optimizers: Sequence[optim.Optimizer],
    criterions: Sequence[nn.Module],
    metrics: Sequence[nn.Module],
    schedulers: Sequence[
        Sequence[
            Tuple[
                SchedulerProtocol,
                Literal["batch", "epoch", "validation"],
            ]
        ]
    ],
    target_transforms: Sequence[TargetTransform],
    device: Optional[torch.device] = None,
) -> List[EpochRunResult]:
    # Validate input list lengths
    if not all_same_length(
        models, optimizers, criterions, metrics, target_transforms
    ):
        raise ValueError(
            "All model-related argument lists must have the same length."
        )

    num_models = len(models)

    # Set train / eval mode per model
    for model in models:
        model.train()

    # Statistics per model
    total_loss = [0.0 for _ in range(num_models)]
    total_samples = [0 for _ in range(num_models)]
    correct = [0 for _ in range(num_models)]
    total = [0 for _ in range(num_models)]
    lr = [0.0 for _ in range(num_models)]

    loader = tqdm(data_loader, desc="Training", leave=False)

    with torch.enable_grad():
        for inputs, targets in loader:
            inputs: Tensor
            targets: Tensor

            # Move data to the specified device if provided
            if device is not None:
                inputs = inputs.to(device)
                targets = targets.to(device)

            for i in range(num_models):
                model = models[i]
                optimizer = optimizers[i]
                criterion = criterions[i]
                metric = metrics[i]
                target_transform = target_transforms[i]
                scheduler = schedulers[i]

                # Apply target transformation if provided
                targ = targets
                if target_transform is not None:
                    targ = target_transform(targ)

                # Zero gradients
                optimizer.zero_grad(set_to_none=True)

                # Forward pass
                outputs: Tensor = model(inputs)
                loss: Tensor = criterion(outputs, targ)

                # Backward pass and optimization step
                loss.backward()
                optimizer.step()

                # Step the scheduler that steps per batch
                for sch in scheduler:
                    if sch[1] == "batch":
                        sch[0].step()

                # Accumulate loss
                batch_size = targ.size(0)
                total_loss[i] += loss.item() * batch_size
                total_samples[i] += batch_size

                # Compute metrics
                c, t = metric(outputs.detach(), targ.detach())
                correct[i] += c
                total[i] += t

                # Update learning rate for logging if scheduler is lr_scheduler
                for sch in scheduler:
                    if isinstance(sch[0], lr_scheduler.LRScheduler):
                        lr[i] = sch[0].get_last_lr()[0]
                        break

            postfix = {
                f"M{i + 1}_Acc": f"{100.0 * correct[i] / max(total[i], 1):.2f}%"
                for i in range(num_models)
            }
            loader.set_postfix(postfix)

        # Step the schedulers that step per epoch
        for i in range(num_models):
            scheduler = schedulers[i]
            for sch in scheduler:
                if sch[1] == "epoch":
                    sch[0].step()

    # Aggregate results
    results: List[EpochRunResult] = [
        EpochRunResult(
            total_loss[i] / total_samples[i],
            100.0 * correct[i] / total[i],
            lr[i],
        )
        for i in range(num_models)
    ]
    return results


def evaluate(
    data_loader: DataLoader,
    models: Sequence[nn.Module],
    criterions: Sequence[nn.Module],
    metrics: Sequence[nn.Module],
    target_transforms: Sequence[TargetTransform],
    device: Optional[torch.device] = None,
) -> List[EpochRunResult]:
    # Validate input list lengths
    if not all_same_length(models, criterions, metrics, target_transforms):
        raise ValueError(
            "All model-related argument lists must have the same length."
        )

    num_models = len(models)

    # Set train / eval mode per model
    for model in models:
        model.eval()

    # Statistics per model
    total_loss = [0.0 for _ in range(num_models)]
    total_samples = [0 for _ in range(num_models)]
    correct = [0 for _ in range(num_models)]
    total = [0 for _ in range(num_models)]

    loader = tqdm(data_loader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for inputs, targets in loader:
            inputs: Tensor
            targets: Tensor

            # Move data to the specified device if provided
            if device is not None:
                inputs = inputs.to(device)
                targets = targets.to(device)

            for i in range(num_models):
                model = models[i]
                criterion = criterions[i]
                metric = metrics[i]
                target_transform = target_transforms[i]

                # Apply target transformation if provided
                targ = targets
                if target_transform is not None:
                    targ = target_transform(targ)

                # Forward pass
                outputs: Tensor = model(inputs)
                loss: Tensor = criterion(outputs, targ)

                # Accumulate loss
                batch_size = targ.size(0)
                total_loss[i] += loss.item() * batch_size
                total_samples[i] += batch_size

                # Compute metrics
                c, t = metric(outputs.detach(), targ.detach())
                correct[i] += c
                total[i] += t

            # Update progress bar
            postfix = {
                f"M{i + 1}_Acc": f"{100.0 * correct[i] / max(total[i], 1):.2f}%"
                for i in range(num_models)
            }
            loader.set_postfix(postfix)

    # Aggregate results
    results: List[EpochRunResult] = [
        EpochRunResult(
            total_loss[i] / total_samples[i], 100.0 * correct[i] / total[i]
        )
        for i in range(num_models)
    ]
    return results
