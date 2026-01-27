from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import torch
from kagglehub import dataset_download
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from utils.dataset import DatasetProvider, LngthDict, ManualDatasetBuilder


class ImageFolderWrapper(ImageFolder):
    def __getitem__(self, index: int) -> tuple[Any, Any]:
        data, label = super().__getitem__(index)
        data: torch.Tensor
        label: int
        return data, torch.tensor([label], dtype=torch.long)


# Dataset structure after dataset_download():
# ~/.cache/kagglehub/datasets/masoudnickparvar/brain-tumor-mri-dataset/versions/1/
#     ├─ Testing/
#     │   ├─ glioma/
#     │   ├─ meningioma/
#     │   ├─ notumor/
#     │   └─ pituitary/
#     └─ training/
#         ├─ glioma/
#         ├─ meningioma/
#         ├─ notumor/
#         └─ pituitary/


class BrainTumorMRI(ManualDatasetBuilder):
    """
    DatasetProvider implementation for the Brain Tumor MRI dataset.
    """

    def __init__(self, root: Path) -> None:
        super().__init__(root)

    def download(self) -> None:
        # Create necessary directories
        self.root.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # Download the dataset from Kaggle (to cache)
        save_path = Path(
            dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
        )

        # Copy the dataset from cache to raw_dir
        self.dataset_root = self.raw_dir / save_path.name
        if not self.dataset_root.exists():
            shutil.copytree(save_path, self.dataset_root)

        # get dataset version
        self.version = save_path.name  # 1

    def process(self) -> None:
        src_train = self.dataset_root / "Training"
        src_test = self.dataset_root / "Testing"

        self.train_dir.symlink_to(
            src_train.relative_to(self.train_dir.parent),
            target_is_directory=True,
        )

        self.test_dir.symlink_to(
            src_test.relative_to(self.test_dir.parent),
            target_is_directory=True,
        )

    def info(self) -> str:
        return (
            f"Ref: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset\n"
            f"Version: {self.version}\n"
            f"Downloaded at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"if redownload is needed, delete this file.\n"
        )

    def get_train(self, transform: Optional[Callable] = None) -> Dataset:
        return ImageFolderWrapper(
            root=str(self.train_dir),
            transform=transform,
        )

    def get_test(self, transform: Optional[Callable] = None) -> Dataset:
        return ImageFolderWrapper(
            root=str(self.test_dir),
            transform=transform,
        )


class BrainTumorMRIProvider(DatasetProvider):
    """
    DatasetProvider implementation for the Brain Tumor MRI dataset.
    """

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir / "BrainTumorMRI")
        self.builder = BrainTumorMRI(self.root)

    def prepare(self) -> None:
        self.builder.prepare()
        self.len_info: LngthDict = {
            "train": len(self.get_train()),
            "test": len(self.get_test()),
            "val": None,
        }

    def get_train(self, transform: Optional[Callable] = None) -> Dataset:
        return self.builder.get_train(transform=transform)

    def get_valid(
        self, transform: Callable[..., Any] | None = None
    ) -> Dataset:
        raise NotImplementedError(
            "This dataset does not provide a validation split."
        )

    def get_test(self, transform: Optional[Callable] = None) -> Dataset:
        return self.builder.get_test(transform=transform)

    def get_num_labels(self) -> int:
        return 4

    def get_label_names(self) -> list[str]:
        return ["glioma", "meningioma", "notumor", "pituitary"]

    def get_length(self) -> LngthDict:
        return self.len_info

    def get_annotation(self) -> Optional[Mapping[str, Any]]:
        return {
            "name": "Brain Tumor MRI",
            "description": "MRI images of brain tumors categorized into four types.",
            "source": "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset",
            "task": "image classification",
            "num_labels": self.get_num_labels(),
            "label_names": self.get_label_names(),
            "length": self.len_info,
        }
