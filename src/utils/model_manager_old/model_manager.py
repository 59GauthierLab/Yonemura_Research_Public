from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.figure import Figure
from torch.utils.data import DataLoader
from typing_extensions import deprecated

from ..dataset.dataset_provider import DatasetProvider
from .epoch_runner import TargetTransform
from .evaluation import evaluate_model
from .recoder import (
    ModelData,
    TestData,
    TestDataDict,
    TrainData,
    TrainDataDict,
    TrainHistory,
)
from .training import train_epoch


class ModelManager:
    """
    Model training, evaluation, saving, and logging management class.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        metric: nn.Module,
        device: torch.device,
        seed: int,
        dataset_provider: DatasetProvider,
        target_transform: TargetTransform = None,
        save_dir: Optional[Path] = None,
    ):
        """
        Initialize the ModelTrainer.

        Args:
            model: PyTorch model to train
            optimizer: Optimizer
            criterion: Loss function
            metric: Evaluation metric function
            device: Device to run training on
            target_transform: Optional function to transform targets
            seed: Random seed for reproducibility (used for recording only)
            dataset_provider: DatasetProvider instance (used for recording only)
            save_dir: Directory to save models and logs
        """

        # Model, criterion, optimizer, device, seed
        self._model = model.to(device)
        self._optimizer = optimizer
        self._criterion = criterion.to(device)
        self._metric = metric.to(device)
        self._target_transform = target_transform
        self._device = device
        self._seed = seed
        self._dataset_provider = dataset_provider

        # Root directory setup
        self._save_dir = (
            save_dir or Path(__import__("__main__").__file__).parent
        )
        self._root_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self._root_dir = self._save_dir / self._root_name

        # recorder
        self.model_data = ModelData(
            model=self._model,
            optimizer=self._optimizer,
            criterion=self._criterion,
            device=self._device,
            seed=self._seed,
        )
        self.train_data: Optional[TrainData] = None
        self.test_data: Optional[TestData] = None

    def set_save_dir(
        self, save_dir: Path, root_name: Optional[str] = None
    ) -> None:
        """Set the directory to save models and logs."""
        self._save_dir = save_dir
        if root_name is not None:
            self._root_name = root_name
        self._root_dir = self._save_dir / self._root_name

    @property
    @deprecated("Use root_dir property instead")
    def _get_experiment_dir(self) -> Path:
        """Get the directory where models and logs are saved."""
        return self._root_dir

    @property
    def root_dir(self) -> Path:
        """Get the directory where models and logs are saved."""
        return self._root_dir

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 30,
        epoch_hook: Optional[
            Callable[[int, int, float, float, float, float], None]
        ] = None,
    ) -> TrainDataDict:
        """
        Train the model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            save_every: Save model every N epochs
            epoch_hook: Optional function called at the end of each epoch with signature
                (current_epoch: int, total_epochs: int, train_loss: float, train_acc: float,
                 val_loss: float, val_acc: float)
        Returns:
            Training data
        """
        # Create experiment directory
        self._root_dir.mkdir(parents=True, exist_ok=True)

        # Input validation
        if epochs <= 0:
            raise ValueError("epochs must be positive")

        # prepare training data recorder
        self.train_data = TrainData(
            model=self.model_data,
            dataset=self._dataset_provider,
        )

        print(f"🚀 Starting training: {self._root_name}")
        print(f"📁 Saving to: {self._root_dir}")
        print(f"🔥 Device: {self._device}")
        print("=" * 50 + "\n")

        # start training
        self.train_data.set_start_time()

        for epoch in range(1, epochs + 1):
            print(f"📊 Epoch {epoch}/{epochs}")
            print("-" * 30)

            # Training
            train_loss, train_acc = train_epoch(
                self._model,
                train_loader,
                self._optimizer,
                self._criterion,
                self._metric,
                self._target_transform,
                self._device,
                verbose=True,
            )
            print(f"📈 Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")

            # Validation
            val_loss, val_acc = evaluate_model(
                self._model,
                test_loader,
                self._criterion,
                self._metric,
                self._target_transform,
                self._device,
                verbose=True,
            )
            print(f"📊 Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

            # Update history
            self.train_data.add_history(
                TrainHistory(
                    epochs=epoch,
                    train_loss=train_loss,
                    train_accuracy=train_acc,
                    val_loss=val_loss,
                    val_accuracy=val_acc,
                )
            )

            # Save best model
            is_best = self.train_data.is_latest_epoch_best()
            if is_best:
                print(f"💾 Best model updated at epoch {epoch}")

            # Epoch hook
            if epoch_hook is not None:
                epoch_hook(
                    epoch, epochs, train_loss, train_acc, val_loss, val_acc
                )

            print()

        # End of training
        self.train_data.set_end_time()

        print("\n" + "=" * 50)
        print("✅ Training completed!")

        best_epoch = self.train_data.get_best_epoch()
        print(
            f"📊 Best validation accuracy: {best_epoch.val_accuracy:.2f}% (epoch {best_epoch.epochs})"
        )
        print()

        return self.train_data.to_dict()

    def evaluate(
        self,
        test_loader: torch.utils.data.DataLoader,
    ) -> TestDataDict:
        """
        Evaluate the model on test data.

        Args:
            test_loader: Test data loader

        Returns:
            Tuple of (test_loss, test_accuracy)
        """
        # Create experiment directory
        self._root_dir.mkdir(parents=True, exist_ok=True)

        print(f"🚀  Starting evaluation: {self._root_name}")
        print(f"📁 Saving to: {self._root_dir}")
        print(f"🔥 Device: {self._device}")
        print("=" * 50 + "\n")

        val_loss, val_acc = evaluate_model(
            self._model,
            test_loader,
            self._criterion,
            self._metric,
            self._target_transform,
            self._device,
            verbose=True,
        )

        self.test_data = TestData(
            model=self.model_data,
            dataset=self._dataset_provider,
            test_loss=val_loss,
            test_accuracy=val_acc,
        )

        print("\n" + "=" * 50)
        print("✅ Evaluation completed!")

        print(
            f"📊 Test Results: Loss={self.test_data.test_loss:.4f}, Accuracy={self.test_data.test_accuracy:.2f}%"
        )
        print()

        return self.test_data.to_dict()

    def save_train_logs(self) -> None:
        if self.train_data is None:
            raise ValueError("No training data to save.")
        self._save_json(self.train_data.to_dict(), "log_training")

    def save_test_logs(self) -> None:
        if self.test_data is None:
            raise ValueError("No test data to save.")
        self._save_json(self.test_data.to_dict(), "log_evaluation")

    def save_train_graphs(self) -> None:
        if self.train_data is None:
            raise ValueError("No training data to save.")
        self._save_graph(
            self.train_data.loss_graph(),
            "plt_loss.png",
        )
        self._save_graph(
            self.train_data.accuracy_graph(),
            "plt_accuracy.png",
        )

    def save_model(self, name: Optional[str] = None) -> None:
        """Save a model."""
        # Create experiment directory
        self._root_dir.mkdir(parents=True, exist_ok=True)

        # Create model directory
        _model_dir = self._root_dir / "models"
        _model_dir.mkdir(parents=True, exist_ok=True)

        save_path = _model_dir / f"{name}.pth"

        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
            },
            save_path,
        )

    def _save_json(self, object: Any, name: str) -> None:
        """Save training logs."""

        save_path = self._root_dir / f"{name}.json"

        # Save as JSON
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(object, f, indent=2)

    def _save_graph(self, figure: Figure, name: str) -> None:
        """Save training graphs."""

        save_path = self._root_dir / name
        figure.savefig(save_path, dpi=300, bbox_inches="tight")

    def __repr__(self) -> str:
        return (
            f"ModelData(model={self._model.__class__.__name__}\n"
            f"optimizer={self._optimizer.__class__.__name__}\n"
            f"criterion={self._criterion.__class__.__name__}\n"
            f"device={self._device}\n"
            f"seed={self._seed})\n"
        )

    @classmethod
    def load_experiment(cls):
        pass
