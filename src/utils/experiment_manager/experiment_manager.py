from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import torch
from torch.utils.data import DataLoader

from ..model_manager import ModelManager
from ..model_manager.multi_runner import test_multi_model, train_multi_model

T = TypeVar("T", covariant=True)


class ExperimentManager:
    def __init__(
        self,
        save_dir: Optional[Path] = None,
    ) -> None:
        # Root directory setup
        self._save_dir = (
            save_dir or Path(__import__("__main__").__file__).parent
        )
        self._root_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self._root_dir = self._save_dir / self._root_name

        # Model managers
        self.model_managers: dict[str, ModelManager] = {}

    @property
    def save_dir(self) -> Path:
        """Get the directory where models and logs are saved."""
        return self._save_dir

    @property
    def root_dir(self) -> Path:
        """Get the directory where models and logs are saved."""
        return self._root_dir

    @property
    def managers(self) -> Dict[str, ModelManager]:
        """Get all ModelManagers."""
        return self.model_managers

    def set_save_dir(
        self, save_dir: Path, root_name: Optional[str] = None
    ) -> None:
        """Set the directory to save models and logs."""
        self._save_dir = save_dir
        if root_name is not None:
            self._root_name = root_name
        self._root_dir = self._save_dir / self._root_name

    def add_model_manager(
        self, name: str, model_manager: ModelManager
    ) -> None:
        """Add a ModelManager to the ExperimentManager."""
        if name in self.model_managers:
            raise ValueError(
                f"ModelManager with name '{name}' already exists."
            )
        model_manager.set_save_dir(self._root_dir, name)
        self.model_managers[name] = model_manager

    def remove_model_manager(self, name: str) -> None:
        """Remove a ModelManager from the ExperimentManager."""
        if name not in self.model_managers:
            raise ValueError(
                f"ModelManager with name '{name}' does not exist."
            )
        del self.model_managers[name]

    def __getitem__(self, name: str) -> ModelManager:
        return self.model_managers[name]

    def __len__(self) -> int:
        return len(self.model_managers)

    def __iter__(self) -> Iterator[Tuple[str, ModelManager]]:
        return iter(self.model_managers.items())

    def train_all(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        epochs: int,
        device: torch.device,
        epoch_hook: Optional[
            Callable[[Sequence[ModelManager], int, int], None],
        ] = None,
        model_managers: Optional[Iterable[str]] = None,
    ) -> None:
        """Train all models managed by the ExperimentManager."""
        run_model_managers = list(
            map(lambda pair: pair[1], self._load_all_models(model_managers))
        )

        train_multi_model(
            model_managers=run_model_managers,
            train_loader=train_loader,
            valid_loader=valid_loader,
            epochs=epochs,
            device=device,
            epoch_hook=epoch_hook,
        )

    def test_all(
        self,
        test_loader: DataLoader,
        device: torch.device,
        model_managers: Optional[Iterable[str]] = None,
    ) -> None:
        """Test all models managed by the ExperimentManager."""
        run_model_managers = list(
            map(lambda pair: pair[1], self._load_all_models(model_managers))
        )

        test_multi_model(
            model_managers=run_model_managers,
            test_loader=test_loader,
            device=device,
        )

    def for_each_all(
        self,
        func: Callable[[str, ModelManager], None],
        model_names: Optional[Iterable[str]] = None,
    ) -> None:
        """Apply a function to all ModelManagers."""
        for name, mm in self._load_all_models(model_names):
            func(name, mm)

    def map_all(
        self,
        func: Callable[[str, ModelManager], T],
        model_names: Optional[Iterable[str]] = None,
    ) -> Iterable[T]:
        """Map a function over all ModelManagers and yield results."""
        for name, mm in self._load_all_models(model_names):
            yield func(name, mm)

    def save_train_graphs_all(
        self, model_names: Optional[Iterable[str]] = None
    ) -> None:
        """Save computation graphs for all models."""

        def func(name: str, mm: ModelManager) -> None:
            mm.save_train_graphs()

        self.for_each_all(func, model_names)

    def save_train_logs_all(
        self, model_names: Optional[Iterable[str]] = None
    ) -> None:
        """Save training logs for all models."""

        def func(name: str, mm: ModelManager) -> None:
            mm.save_train_logs()

        self.for_each_all(func, model_names)

    def save_test_logs_all(
        self, model_names: Optional[Iterable[str]] = None
    ) -> None:
        """Save test logs for all models."""

        def func(name: str, mm: ModelManager) -> None:
            mm.save_test_logs()

        self.for_each_all(func, model_names)

    def save_model_all(
        self, model_names: Optional[Iterable[str]] = None
    ) -> None:
        """Save model weights for all models."""

        def func(name: str, mm: ModelManager) -> None:
            mm.save_model()

        self.for_each_all(func, model_names)

    def save_memory_all(
        self, model_names: Optional[Iterable[str]] = None
    ) -> None:
        """Save memory data for all models."""

        def func(name: str, mm: ModelManager) -> None:
            mm.save_memory(name)

        self.for_each_all(func, model_names)

    def _load_all_models(
        self, model_names: Optional[Iterable[str]]
    ) -> Iterable[Tuple[str, ModelManager]]:
        """Load all models managed by the ExperimentManager."""
        name_manager_pairs: Iterable[Tuple[str, ModelManager]] = []

        if model_names is None:
            name_manager_pairs = self.model_managers.items()
        else:
            for name in model_names:
                if name not in self.model_managers:
                    raise ValueError(
                        f"ModelManager with name '{name}' does not exist."
                    )
            name_manager_pairs = (
                (name, self.model_managers[name]) for name in model_names
            )
        return name_manager_pairs

    def save_model_summary(
        self,
        model_names: Optional[Iterable[str]] = None,
        include_memory: bool = False,
    ) -> None:
        """Save model summaries for all models."""

        def func(name: str, mm: ModelManager) -> Tuple[str, Dict[str, Any]]:
            out: Dict[str, Any] = {}

            train_result = mm.get_train_data_safety()
            test_result = mm.get_test_data_safety()

            out["train_loss"] = [h.train_loss for h in train_result.history]
            out["val_loss"] = [h.valid_loss for h in train_result.history]
            out["train_acc"] = [h.train_accuracy for h in train_result.history]
            out["val_acc"] = [h.valid_accuracy for h in train_result.history]
            out["test_loss"] = test_result.test_loss
            out["test_acc"] = test_result.test_accuracy
            if include_memory:
                out["memory"] = mm.memory
            return name, out

        summaries: Dict[str, Any] = dict(self.map_all(func, model_names))

        self.save_json(summaries, "model_summaries.json")

    def save_json(
        self, data: Any, filename: str, float_precision: int = 4
    ) -> None:
        """Save an object as a JSON file in the root directory."""

        file_path = self._root_dir / filename
        formatted_data: Any = json.loads(
            json.dumps(data),
            parse_float=lambda x: round(float(x), float_precision),
        )

        with open(file_path, "w") as f:
            json.dump(formatted_data, f, indent=4)
