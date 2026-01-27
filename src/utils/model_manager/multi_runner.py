from __future__ import annotations

from typing import Callable, Optional, Sequence, cast

import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from .epoch_runner import evaluate, train
from .model_manager import ModelManager
from .recoder import TestData, TrainData, TrainHistory


def train_multi_model(
    model_managers: Sequence[ModelManager],
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int,
    device: torch.device,
    epoch_hook: Optional[
        Callable[[Sequence[ModelManager], int, int], None],
    ] = None,
) -> None:
    """
    Train multiple models using their respective ModelManager instances.
    """

    # Input validation
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if any(mm.device != device for mm in model_managers):
        raise ValueError("All models must be on the same device.")

    print(f"🚀 Starting training for {len(model_managers)} models")
    print(f"🔥 Device: {device}")

    # prepare training data recorder
    for i, mm in enumerate(model_managers):
        print(
            f"Starting training for Model {i + 1}/{len(model_managers)}: {str(mm.model)}"
        )
        mm.train_data = TrainData(mm.model_data, mm.dataset_provider)

    print("=" * 50 + "\n")

    models = [mm.model for mm in model_managers]
    optimizers = [mm.optimizer for mm in model_managers]
    criterions = [mm.criterion for mm in model_managers]
    metrics = [mm.metric for mm in model_managers]
    schedulers = [mm.scheduler for mm in model_managers]
    target_transforms = [mm.target_transform for mm in model_managers]

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Train all models
        train_result = train(
            models=models,
            optimizers=optimizers,
            data_loader=train_loader,
            criterions=criterions,
            metrics=metrics,
            target_transforms=target_transforms,
            schedulers=schedulers,
            device=device,
        )

        print("📈 Train:")
        for i, res in enumerate(train_result):
            print(
                f"Model {i + 1}: "
                f"Loss={res.loss:.4f}, "
                f"Accuracy={res.accuracy:.2f}%, "
                f"LR={res.lr:.6f}"
            )

        # Evaluate all models
        eval_result = evaluate(
            models=models,
            data_loader=valid_loader,
            criterions=criterions,
            metrics=metrics,
            target_transforms=target_transforms,
            device=device,
        )

        print("📉 Validation:")
        for i, res in enumerate(eval_result):
            print(
                f"Model {i + 1}: "
                f"Loss={res.loss:.4f}, "
                f"Accuracy={res.accuracy:.2f}%"
            )

        # Update history
        for mm, train_res, eval_res in zip(
            model_managers, train_result, eval_result
        ):
            cast(TrainData, mm.train_data).add_history(
                TrainHistory(
                    epochs=epoch + 1,
                    train_loss=train_res.loss,
                    train_accuracy=train_res.accuracy,
                    valid_loss=eval_res.loss,
                    valid_accuracy=eval_res.accuracy,
                    lr=train_res.lr,
                )
            )

        # Step the schedulers that step per validation
        for mm, eval_res in zip(model_managers, eval_result):
            for sch in mm.scheduler:
                if sch[1] == "validation":
                    if isinstance(sch[0], lr_scheduler.ReduceLROnPlateau):
                        sch[0].step(
                            eval_res.loss
                        )  # ReduceLROnPlateau needs validation loss
                    else:
                        sch[0].step()  # Other schedulers

        # Call epoch hook if provided
        if epoch_hook is not None:
            epoch_hook(model_managers, epoch + 1, epochs)

    print("\n" + "=" * 50)
    print("✅ Training completed!")


def test_multi_model(
    model_managers: Sequence[ModelManager],
    test_loader: DataLoader,
    device: torch.device,
) -> None:
    """
    Test multiple models using their respective ModelManager instances.
    """

    print(f"🚀 Starting testing for {len(model_managers)} models")
    print(f"🔥 Device: {device}")

    models = [mm.model for mm in model_managers]
    criterions = [mm.criterion for mm in model_managers]
    metrics = [mm.metric for mm in model_managers]
    target_transforms = [mm.target_transform for mm in model_managers]

    test_result = evaluate(
        models=models,
        data_loader=test_loader,
        criterions=criterions,
        metrics=metrics,
        target_transforms=target_transforms,
        device=device,
    )

    print("📉 Test Results:")
    for i, res in enumerate(test_result):
        print(
            f"Model {i + 1}: "
            f"Loss: {res.loss:.4f}, "
            f"Accuracy: {res.accuracy:.2f}%"
        )

    # Update test results
    for mm, res in zip(model_managers, test_result):
        mm.test_data = TestData(
            model=mm.model_data,
            dataset=mm.dataset_provider,
            test_loss=res.loss,
            test_accuracy=res.accuracy,
        )

    print("✅ Test completed!")
