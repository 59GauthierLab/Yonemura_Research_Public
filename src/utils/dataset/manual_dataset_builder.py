from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class ManualDatasetBuilder(ABC):
    """
    Abstract base class for Image dataset preparation and dataset construction.

    This class defines a standardized workflow for:
    1. Downloading raw dataset files from external sources.
    2. Processing and organizing the data into a format compatible with
       torchvision datasets (e.g., ImageFolder).
    3. Providing ready-to-use PyTorch Dataset objects for training and testing.

    The dataset directory structure is expected to be:

        <root>/
        ├─ raw/                 # Raw downloaded data (archives, original files)
        │   └─ ...
        ├─ train/               # Processed training data (ImageFolder format)
        │   ├─ class_0/
        │   │   ├─ img001.png
        │   │   └─ ...
        │   └─ class_1/
        │       └─ ...
        ├─ test/                # Processed test data (ImageFolder format)
        │   ├─ class_0/
        │   └─ class_1/
        ├─ val/                 # Processed validation data (ImageFolder format) (optional)
        │   ├─ class_0/
        │   └─ class_1/
        └─ info.txt              # Dataset metadata and preparation information

    Subclasses must implement:
    - download(): Download raw data into the raw/ directory.
    - process(): Convert raw data into train/ and test/ directories
                 following the ImageFolder format.
    - info(): Return a human-readable string describing the dataset,
              preprocessing steps, and version information.

    The `prepare()` method ensures that the dataset is downloaded and
    processed only once by using `info.txt` as a preparation flag.

    Training, validation, and test datasets can be obtained via:
        - get_train(transform)
        - get_val(transform)
        - get_test(transform)

    These methods return torchvision-compatible Dataset instances that can
    be directly passed to torch.utils.data.DataLoader.
    """

    def __init__(self, root: Path) -> None:
        self.root: Path = root
        self.raw_dir: Path = self.root / "raw"
        self.train_dir: Path = self.root / "train"
        self.test_dir: Path = self.root / "test"
        self.val_dir: Path = self.root / "val"
        self.info_file: Path = self.root / "info.txt"

        self.info_dict: Dict[str, Any] = {}

    def prepare(self) -> None:
        if not self.info_file.exists():
            self.download()
            self.process()
            # write info
            with open(self.info_file, "w", encoding="utf-8") as f:
                f.write(self.info())

    @abstractmethod
    def download(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def process(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def info(self) -> str:
        raise NotImplementedError

    def get_train(self, transform: Optional[Callable] = None) -> Dataset:
        """
        Return the training dataset as a torchvision Dataset.
        """
        return ImageFolder(
            root=str(self.train_dir),
            transform=transform,
        )

    def get_test(self, transform: Optional[Callable] = None) -> Dataset:
        """
        Return the test dataset as a torchvision Dataset.
        """
        return ImageFolder(
            root=str(self.test_dir),
            transform=transform,
        )

    def get_val(self, transform: Optional[Callable] = None) -> Dataset:
        """
        Return the validation dataset as a torchvision Dataset.
        """
        if not self.val_dir.exists():
            raise NotImplementedError(
                "This dataset does not provide a validation split."
            )
        return ImageFolder(
            root=str(self.val_dir),
            transform=transform,
        )
