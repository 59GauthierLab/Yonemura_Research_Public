from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.figure import Figure
from torch.utils.data import DataLoader
from typing_extensions import deprecated

from ..dataset.dataset_provider import DatasetProvider
from .epoch_runner import SchedulerProtocol, TargetTransform, evaluate, train
from .recoder import (
    ModelData,
    TestData,
    TestDataDict,
    TrainData,
    TrainDataDict,
    TrainHistory,
)


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
        scheduler: Optional[
            Sequence[
                Tuple[
                    SchedulerProtocol,
                    Literal["batch", "epoch", "validation"],
                ]
            ]
        ] = None,
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
            seed: Random seed for reproducibility (used for recording only)
            dataset_provider: DatasetProvider instance (used for recording only)
            scheduler: Learning rate scheduler
            target_transform: Optional function to transform targets
            save_dir: Directory to save models and logs
        """

        # Model, criterion, optimizer, device, seed
        self._model: nn.Module = model.to(device)
        self._optimizer: optim.Optimizer = optimizer
        self._criterion: nn.Module = criterion.to(device)
        self._metric: nn.Module = metric.cpu()  # Metrics are computed on CPU
        self._scheduler: Sequence[
            Tuple[
                SchedulerProtocol,
                Literal["batch", "epoch", "validation"],
            ]
        ] = scheduler if scheduler is not None else []
        self._target_transform: TargetTransform = target_transform
        self._device: torch.device = device
        self._seed: int = seed
        self._dataset_provider = dataset_provider

        # Root directory setup
        self._save_dir = (
            save_dir or Path(__import__("__main__").__file__).parent
        )
        self._root_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self._root_dir = self._save_dir / self._root_name

        # recorder
        self._model_data = ModelData(
            model=self._model,
            optimizer=self._optimizer,
            criterion=self._criterion,
            scheduler=self._scheduler,
            device=self._device,
            seed=self._seed,
        )
        self._train_data: Optional[TrainData] = None
        self._test_data: Optional[TestData] = None
        self.memory: Dict[str, Any] = {}

    @deprecated("Use set_save_dir method instead.")
    def set_save_dir(
        self, save_dir: Path, root_name: Optional[str] = None
    ) -> None:
        """Set the directory to save models and logs."""
        self._save_dir = save_dir
        if root_name is not None:
            self._root_name = root_name
        self._root_dir = self._save_dir / self._root_name

    @property
    def model(self) -> nn.Module:
        """Get the model."""
        return self._model

    @property
    def optimizer(self) -> optim.Optimizer:
        """Get the optimizer."""
        return self._optimizer

    @property
    def criterion(self) -> nn.Module:
        """Get the criterion."""
        return self._criterion

    @property
    def metric(self) -> nn.Module:
        """Get the metric."""
        return self._metric

    @property
    def scheduler(
        self,
    ) -> Sequence[
        Tuple[SchedulerProtocol, Literal["batch", "epoch", "validation"]]
    ]:
        """Get the scheduler."""
        return self._scheduler

    @property
    def target_transform(self) -> Optional[TargetTransform]:
        """Get the target transform."""
        return self._target_transform

    @property
    def device(self) -> torch.device:
        """Get the device."""
        return self._device

    @property
    def seed(self) -> int:
        """Get the random seed."""
        return self._seed

    @property
    def dataset_provider(self) -> DatasetProvider:
        """Get the dataset provider."""
        return self._dataset_provider

    @property
    def save_dir(self) -> Path:
        """Get the save directory."""
        return self._save_dir

    @save_dir.setter
    def save_dir(self, path: Path) -> None:
        """Set the save directory."""
        self._save_dir = path
        self._root_dir = self._save_dir / self._root_name

    @property
    def root_name(self) -> str:
        """Get the root name."""
        return self._root_name

    @property
    def root_dir(self) -> Path:
        """Get the root directory."""
        return self._root_dir

    @property
    def model_data(self) -> ModelData:
        """Get the model data."""
        return self._model_data

    @property
    def train_data(self) -> Optional[TrainData]:
        """Get the training data."""
        return self._train_data

    @train_data.setter
    def train_data(self, data: TrainData) -> None:
        """Set the training data."""
        self._train_data = data

    @property
    def test_data(self) -> Optional[TestData]:
        """Get the test data."""
        return self._test_data

    @test_data.setter
    def test_data(self, data: TestData) -> None:
        """Set the test data."""
        self._test_data = data

    def get_train_data_safety(self) -> TrainData:
        """Get the training data, raising an error if not available."""
        if self._train_data is None:
            raise ValueError("No training data available.")
        return self._train_data

    def get_test_data_safety(self) -> TestData:
        """Get the test data, raising an error if not available."""
        if self._test_data is None:
            raise ValueError("No test data available.")
        return self._test_data

    @property
    def memory(self) -> Dict[str, Any]:
        """Get the memory attribute."""
        return self._memory

    @memory.setter
    def memory(self, value: Dict[str, Any]) -> None:
        """Set the memory attribute."""
        self._memory = value

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        epochs: int = 30,
        epoch_hook: Optional[
            Callable[[int, int, float, float, float, float], None]
        ] = None,
    ) -> TrainDataDict:
        """
        Train the model for multiple epochs.

        Args:
            train_loader: Training data loader
            valid_loader: Validation data loader
            epochs: Number of epochs to train
            epoch_hook: Optional function called at the end of each epoch with signature
                (current_epoch: int, total_epochs: int, train_loss: float, train_acc: float,
                 valid_loss: float, valid_acc: float)
        Returns:
            Training data
        """

        # Input validation
        if epochs <= 0:
            raise ValueError("epochs must be positive")

        # prepare training data recorder
        self._train_data = TrainData(
            model=self._model_data,
            dataset=self._dataset_provider,
        )

        print(f"🚀 Starting training: {self._root_name}")
        print(f"🔥 Device: {self._device}")
        print("=" * 50 + "\n")

        # start training
        self._train_data.set_start_time()

        for epoch in range(1, epochs + 1):
            print(f"📊 Epoch {epoch}/{epochs}")
            print("-" * 30)

            # Training
            train_resut = train(
                data_loader=train_loader,
                models=[self._model],
                optimizers=[self._optimizer],
                criterions=[self._criterion],
                metrics=[self._metric],
                schedulers=[self._scheduler],
                target_transforms=[self._target_transform],
                device=self._device,
            )
            train_loss, train_acc, train_lr = (
                train_resut[0].loss,
                train_resut[0].accuracy,
                train_resut[0].lr,
            )

            print(
                f"📈 Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%, LR={train_lr:.6f}"
            )

            # Validation
            valid_result = evaluate(
                data_loader=valid_loader,
                models=[self._model],
                criterions=[self._criterion],
                metrics=[self._metric],
                target_transforms=[self._target_transform],
                device=self._device,
            )
            valid_loss, valid_acc = (
                valid_result[0].loss,
                valid_result[0].accuracy,
            )

            print(f"📊 Valid: Loss={valid_loss:.4f}, Acc={valid_acc:.2f}%")

            # Update history
            self._train_data.add_history(
                TrainHistory(
                    epochs=epoch,
                    train_loss=train_loss,
                    train_accuracy=train_acc,
                    valid_loss=valid_loss,
                    valid_accuracy=valid_acc,
                    lr=train_lr,
                )
            )

            # Best model
            is_best = self._train_data.is_latest_epoch_best()
            if is_best:
                print(f"💾 Best model updated at epoch {epoch}")

            # Epoch hook
            if epoch_hook is not None:
                epoch_hook(
                    epoch, epochs, train_loss, train_acc, valid_loss, valid_acc
                )

            print()

        # End of training
        self._train_data.set_end_time()

        print("\n" + "=" * 50)
        print("✅ Training completed!")

        best_epoch = self._train_data.get_best_epoch()
        print(
            f"📊 Best validation accuracy: {best_epoch.valid_accuracy:.2f}% (epoch {best_epoch.epochs})"
        )
        print()

        return self._train_data.to_dict()

    def test(
        self,
        test_loader: torch.utils.data.DataLoader,
    ) -> TestDataDict:
        """
        Test the model on test data.

        Args:
            test_loader: Test data loader

        Returns:
            Test data
        """

        print(f"🚀  Starting testing: {self._root_name}")
        print(f"🔥 Device: {self._device}")

        test_result = evaluate(
            models=[self._model],
            data_loader=test_loader,
            criterions=[self._criterion],
            metrics=[self._metric],
            target_transforms=[self._target_transform],
            device=self._device,
        )
        test_loss, test_accuracy = test_result[0].loss, test_result[0].accuracy

        self._test_data = TestData(
            model=self._model_data,
            dataset=self._dataset_provider,
            test_loss=test_loss,
            test_accuracy=test_accuracy,
        )

        print("✅ Testing completed!")

        print(
            f"📊 Test Results: Loss={test_loss:.4f}, Accuracy={test_accuracy:.2f}%"
        )
        print()

        return self._test_data.to_dict()

    def save_train_logs(self) -> None:
        """Save training logs."""
        if self._train_data is None:
            raise ValueError("No training data to save.")
        self._root_dir.mkdir(parents=True, exist_ok=True)
        self._save_json(self._train_data.to_dict(), "log_training")

    def save_test_logs(self) -> None:
        """Save test logs."""
        if self._test_data is None:
            raise ValueError("No test data to save.")
        self._root_dir.mkdir(parents=True, exist_ok=True)
        self._save_json(self._test_data.to_dict(), "log_evaluation")

    def save_train_graphs(self) -> None:
        """Save training graphs."""
        if self._train_data is None:
            raise ValueError("No training data to save.")
        self._root_dir.mkdir(parents=True, exist_ok=True)
        self._save_graph(
            self._train_data.loss_graph(),
            "plt_loss.png",
        )
        self._save_graph(
            self._train_data.accuracy_graph(),
            "plt_accuracy.png",
        )

    def save_model(self, name: Optional[str] = None) -> None:
        """Save a model."""
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

    def save_memory(self, name: str) -> None:
        """Save memory data."""
        self._root_dir.mkdir(parents=True, exist_ok=True)
        self._save_json(self.memory, f"memory_{name}")

    def _save_json(
        self, data: Any, name: str, float_precision: int = 4
    ) -> None:
        """Save training logs."""

        save_path = self._root_dir / f"{name}.json"
        formatted_data: Any = json.loads(
            json.dumps(data),
            parse_float=lambda x: round(float(x), float_precision),
        )

        # Save as JSON
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(formatted_data, f, indent=2)

    def _save_graph(self, figure: Figure, name: str) -> None:
        """Save training graphs."""

        save_path = self._root_dir / name
        figure.savefig(save_path, dpi=300, bbox_inches="tight")

    def __repr__(self) -> str:
        return (
            f"ModelData(model={self._model.__class__.__name__}\n"
            f"optimizer={self._optimizer.__class__.__name__}\n"
            f"criterion={self._criterion.__class__.__name__}\n"
            f"metric={self._metric.__class__.__name__}\n"
            f"scheduler={self._scheduler.__class__.__name__ if self._scheduler else None}\n"
            f"device={self._device}\n"
            f"seed={self._seed})\n"
        )

    @classmethod
    def load_experiment(cls):
        pass
