from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from torch.utils.data import Dataset
from torchvision.datasets import CelebA

from utils.dataset import DatasetProvider, LngthDict


class CelebAProvider(DatasetProvider):
    """
    DatasetProvider implementation for the CelebA dataset.
    """

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir / "CelebA")

    def prepare(self) -> None:
        # The CelebA class handles downloading and preparing the dataset.
        tmp1 = CelebA(
            root=str(self.root),
            split="train",
            download=True,
        )
        tmp2 = CelebA(
            root=str(self.root),
            split="test",
            download=True,
        )
        tmp3 = CelebA(
            root=str(self.root),
            split="valid",
            download=True,
        )

        raw_label_names = tmp1.attr_names
        self.label_names = [name for name in raw_label_names if name != ""]
        self.len_info: LngthDict = {
            "train": len(tmp1),
            "test": len(tmp2),
            "val": len(tmp3),
        }

    def get_train(self, transform: Optional[Callable] = None) -> Dataset:
        return CelebA(
            root=str(self.root),
            split="train",
            transform=transform,
            download=False,
        )

    def get_valid(self, transform: Optional[Callable] = None) -> Dataset:
        return CelebA(
            root=str(self.root),
            split="valid",
            transform=transform,
            download=False,
        )

    def get_test(self, transform: Optional[Callable] = None) -> Dataset:
        return CelebA(
            root=str(self.root),
            split="test",
            transform=transform,
            download=False,
        )

    def get_num_labels(self) -> int:
        # attribute names
        return len(self.label_names)

    def get_label_names(self) -> list[str]:
        return self.label_names

    def get_length(self) -> LngthDict:
        return self.len_info

    def get_annotation(self) -> Optional[Mapping[str, Any]]:
        return {
            "name": "CelebA",
            "description": "Large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations.",
            "source": "https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html",
            "task": "attribute classification",
            "num_labels": self.get_num_labels(),
            "label_names": self.get_label_names(),
            "length": self.len_info,
        }


# memo
"""\
@inproceedings{liu2015faceattributes,
  title = {Deep Learning Face Attributes in the Wild},
  author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
  month = {December},
  year = {2015}
}
"""
