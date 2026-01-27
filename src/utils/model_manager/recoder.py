from __future__ import annotations

from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
)

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.figure import Figure

from ..dataset.dataset_provider import DatasetProvider
from .epoch_runner import SchedulerProtocol


class ModelDataDict(TypedDict):
    model: str
    criterion: str
    optimizer: str
    scheduler: str
    device: str
    seed: int


class ModelData:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        scheduler: Sequence[
            Tuple[
                SchedulerProtocol,
                Literal["batch", "epoch", "validation"],
            ]
        ],
        device: torch.device,
        seed: int,
    ) -> None:
        self.model: nn.Module = model
        self.criterion: nn.Module = criterion
        self.optimizer: optim.Optimizer = optimizer
        self.scheduler: Sequence[
            Tuple[
                SchedulerProtocol,
                Literal["batch", "epoch", "validation"],
            ]
        ] = scheduler
        self.device: torch.device = device
        self.seed: int = seed

    def to_dict(self) -> ModelDataDict:
        return {
            "model": str(self.model),
            "criterion": str(self.criterion),
            "optimizer": str(self.optimizer),
            "scheduler": str(self.scheduler),
            "device": str(self.device),
            "seed": self.seed,
        }


class TrainHistoryDict(TypedDict):
    epochs: int
    train_loss: float
    train_accuracy: float
    valid_loss: float
    valid_accuracy: float
    lr: float


class TrainHistory:
    def __init__(
        self,
        epochs: int,
        train_loss: float,
        train_accuracy: float,
        valid_loss: float,
        valid_accuracy: float,
        lr: float,
    ) -> None:
        self.epochs: int = epochs
        self.train_loss: float = train_loss
        self.train_accuracy: float = train_accuracy
        self.valid_loss: float = valid_loss
        self.valid_accuracy: float = valid_accuracy
        self.lr: float = lr

    def to_dict(self) -> TrainHistoryDict:
        return {
            "epochs": self.epochs,
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "valid_loss": self.valid_loss,
            "valid_accuracy": self.valid_accuracy,
            "lr": self.lr,
        }


class BestEpochDict(TypedDict):
    epochs: int
    valid_accuracy: float
    valid_loss: float


class TrainDataDict(TypedDict):
    model: ModelDataDict
    dataset: Dict[str, Any]
    epochs: int
    start_time: Optional[str]
    end_time: Optional[str]
    duration: Optional[float]
    best_epoch: Optional[BestEpochDict]
    history: List[TrainHistoryDict]


class TrainData:
    def __init__(
        self,
        model: ModelData,
        dataset: DatasetProvider,
    ) -> None:
        self.model: ModelData = model
        self.dataset: DatasetProvider = dataset
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.duration: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.history: List[TrainHistory] = []

    def set_start_time(self, start_time: Optional[datetime] = None) -> None:
        if start_time is None:
            start_time = datetime.now()
        self.start_time = start_time

    def set_end_time(self, end_time: Optional[datetime] = None) -> None:
        if end_time is None:
            end_time = datetime.now()
        self.end_time = end_time

    def add_history(self, train_history: TrainHistory) -> None:
        self.history.append(train_history)

        this_epoch = len(self.history)
        if self.best_epoch is None:
            self.best_epoch = this_epoch
        else:
            if (
                self.history[this_epoch - 1].valid_accuracy
                > self.history[self.best_epoch - 1].valid_accuracy
            ):
                self.best_epoch = this_epoch

    def is_latest_epoch_best(self) -> bool:
        if self.best_epoch is None:
            raise ValueError("No best epoch recorded.")
        return self.best_epoch == len(self.history)

    def _calc_duration(self):
        # duration
        if self.start_time is None or self.end_time is None:
            self.duration = None
        else:
            self.duration = (
                self.end_time.timestamp() - self.start_time.timestamp()
            )

    def to_dict(self) -> TrainDataDict:
        self._calc_duration()

        return {
            "model": self.model.to_dict(),
            "dataset": self.dataset.get_annotation(),
            "epochs": len(self.history),
            "start_time": self.start_time.isoformat()
            if self.start_time
            else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "best_epoch": {
                "epochs": self.best_epoch,
                "valid_accuracy": self.history[
                    self.best_epoch - 1
                ].valid_accuracy,
                "valid_loss": self.history[self.best_epoch - 1].valid_loss,
            }
            if self.best_epoch is not None
            else None,
            "history": [h.to_dict() for h in self.history],
        }

    def accuracy_graph(self) -> Figure:
        fig = Figure(frameon=False)

        ax = fig.add_axes((0, 0, 1, 1))
        epochs = [i + 1 for i in range(len(self.history))]
        train_accuracies = [h.train_accuracy for h in self.history]
        valid_accuracies = [h.valid_accuracy for h in self.history]
        ax.plot(epochs, train_accuracies, label="Train Accuracy", color="blue")
        ax.plot(
            epochs,
            valid_accuracies,
            label="Validation Accuracy",
            color="orange",
        )

        ax.set_title("Model Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True)
        ax.legend()

        if self.best_epoch is not None:
            ax.scatter(
                self.best_epoch,
                valid_accuracies[self.best_epoch - 1],
                color="red",
                s=100,
                zorder=5,
                label=f"Best (Epoch {self.best_epoch})",
            )
        return fig

    def loss_graph(self) -> Figure:
        fig = Figure(frameon=False)

        ax = fig.add_axes((0, 0, 1, 1))
        epochs = [i + 1 for i in range(len(self.history))]
        train_losses = [h.train_loss for h in self.history]
        valid_losses = [h.valid_loss for h in self.history]
        ax.plot(epochs, train_losses, label="Train Loss", color="blue")
        ax.plot(epochs, valid_losses, label="Validation Loss", color="orange")

        ax.set_title("Model Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True)
        ax.legend()

        if self.best_epoch is not None:
            ax.scatter(
                self.best_epoch,
                valid_losses[self.best_epoch - 1],
                color="red",
                s=100,
                zorder=5,
                label=f"Best (Epoch {self.best_epoch})",
            )
        return fig

    def get_best_epoch(self) -> TrainHistory:
        if self.best_epoch is None:
            raise ValueError("No best epoch recorded.")
        return self.history[self.best_epoch - 1]


class TestDataDict(TypedDict):
    model: ModelDataDict
    dataset: Dict[str, Any]
    test_loss: float
    test_accuracy: float


class TestData:
    def __init__(
        self,
        model: ModelData,
        dataset: DatasetProvider,
        test_loss: float,
        test_accuracy: float,
    ) -> None:
        self.model: ModelData = model
        self.dataset: DatasetProvider = dataset
        self.test_loss: float = test_loss
        self.test_accuracy: float = test_accuracy

    def to_dict(self) -> TestDataDict:
        return {
            "model": self.model.to_dict(),
            "dataset": self.dataset.get_annotation(),
            "test_loss": self.test_loss,
            "test_accuracy": self.test_accuracy,
        }
