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
# ~/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/
#     └─ chest_xray/
#         ├─ __MACOSX/
#         │   ...(macOS hidden files)...
#         ├─ chest_xray/
#         │   ├─ test/
#         │   ├─ train/
#         │   └─ val/
#         ├─ test/
#         │   ├─ NORMAL/
#         │   └─ PNEUMONIA/
#         ├─ train/
#         │   ├─ NORMAL/
#         │   └─ PNEUMONIA/
#         └─ val/
#             ├─ NORMAL/
#             └─ PNEUMONIA/


class ChestXrayPneumonia(ManualDatasetBuilder):
    """
    DatasetProvider implementation for the Chest X-Ray Image (Pneumonia) dataset.
    """

    def __init__(self, root: Path) -> None:
        super().__init__(root)

    def download(self) -> None:
        # Create necessary directories
        self.root.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # Download the dataset from Kaggle (to cache)
        save_path = Path(
            dataset_download("paultimothymooney/chest-xray-pneumonia")
        )

        # Copy the dataset from cache to raw_dir
        self.dataset_root = self.raw_dir / save_path.name
        if not self.dataset_root.exists():
            shutil.copytree(save_path, self.dataset_root)

        # get dataset version
        self.version = save_path.name  # 2

    def process(self) -> None:
        src_train = self.dataset_root / "chest_xray" / "train"
        src_test = self.dataset_root / "chest_xray" / "test"
        src_val = self.dataset_root / "chest_xray" / "val"

        self.train_dir.symlink_to(
            src_train.relative_to(self.train_dir.parent),
            target_is_directory=True,
        )

        self.test_dir.symlink_to(
            src_test.relative_to(self.test_dir.parent),
            target_is_directory=True,
        )

        self.val_dir.symlink_to(
            src_val.relative_to(self.val_dir.parent),
            target_is_directory=True,
        )

    def info(self) -> str:
        return (
            f"Ref: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia\n"
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

    def get_val(self, transform: Optional[Callable] = None) -> Dataset:
        return ImageFolderWrapper(
            root=str(self.val_dir),
            transform=transform,
        )


class ChestXrayPneumoniaProvider(DatasetProvider):
    """
    DatasetProvider implementation for the Chest X-Ray Image (Pneumonia) dataset.
    """

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir / "ChestXrayPneumonia")
        self.builder = ChestXrayPneumonia(self.root)

    def prepare(self) -> None:
        self.builder.prepare()
        self.len_info: LngthDict = {
            "train": len(self.get_train()),
            "test": len(self.get_test()),
            "val": len(self.get_valid()),
        }

    def get_train(self, transform: Optional[Callable] = None) -> Dataset:
        return self.builder.get_train(transform=transform)

    def get_valid(
        self, transform: Callable[..., Any] | None = None
    ) -> Dataset:
        return self.builder.get_val(transform=transform)

    def get_test(self, transform: Optional[Callable] = None) -> Dataset:
        return self.builder.get_test(transform=transform)

    def get_num_labels(self) -> int:
        return 2

    def get_label_names(self) -> list[str]:
        return ["NORMAL", "PNEUMONIA"]

    def get_length(self) -> LngthDict:
        return self.len_info

    def get_annotation(self) -> Optional[Mapping[str, Any]]:
        return {
            "name": "Chest X-Ray Image (Pneumonia)",
            "description": "X-ray images categorized into normal and pneumonia classes.",
            "source": "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia",
            "task": "image classification",
            "num_labels": self.get_num_labels(),
            "label_names": self.get_label_names(),
            "length": self.len_info,
        }
