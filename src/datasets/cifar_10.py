from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

from utils.dataset import DatasetProvider, LngthDict


class CIFAR10Wrapper(CIFAR10):
    def __getitem__(self, index: int) -> tuple[Any, Any]:
        data, label = super().__getitem__(index)
        data: torch.Tensor
        label: int
        return data, torch.tensor([label], dtype=torch.long)


class CIFAR10Provider(DatasetProvider):
    """
    DatasetProvider implementation for the CIFAR-10 dataset.
    """

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir / "CIFAR10")

    def prepare(self) -> None:
        # The CIFAR10 class handles downloading and preparing the dataset.
        tmp1 = CIFAR10Wrapper(
            root=str(self.root),
            train=True,
            download=True,
        )
        tmp2 = CIFAR10Wrapper(
            root=str(self.root),
            train=False,
            download=True,
        )

        self.label_names = copy.copy(tmp1.classes)
        self.len_info: LngthDict = {
            "train": len(tmp1),
            "test": len(tmp2),
            "val": None,
        }

    def get_train(self, transform: Optional[Callable] = None) -> Dataset:
        return CIFAR10Wrapper(
            root=str(self.root),
            train=True,
            transform=transform,
            download=False,
        )

    def get_valid(self, transform: Optional[Callable] = None) -> Dataset:
        raise NotImplementedError(
            "CIFAR-10 does not have a separate validation set."
        )

    def get_test(self, transform: Optional[Callable] = None) -> Dataset:
        return CIFAR10Wrapper(
            root=str(self.root),
            train=False,
            transform=transform,
            download=False,
        )

    def get_num_labels(self) -> int:
        return 1

    def get_label_names(self) -> List[str]:
        return self.label_names

    def get_length(self) -> LngthDict:
        return self.len_info

    def get_annotation(self) -> Optional[Mapping[str, Any]]:
        return {
            "name": "CIFAR-10",
            "description": "The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.",
            "source": "https://www.cs.toronto.edu/~kriz/cifar.html",
            "task": "image classification",
            "num_labels": self.get_num_labels(),
            "label_names": self.get_label_names(),
            "length": self.len_info,
        }
