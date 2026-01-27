from __future__ import annotations

from pathlib import Path
from typing import List, Type

from config import config
from utils.dataset import DatasetProvider

from .brain_tumor_mri import BrainTumorMRIProvider
from .celeb_a import CelebAProvider
from .chest_xray_pneumonia import ChestXrayPneumoniaProvider
from .cifar_10 import CIFAR10Provider

providers: List[Type[DatasetProvider]] = [
    BrainTumorMRIProvider,
    ChestXrayPneumoniaProvider,
    CIFAR10Provider,
    CelebAProvider,
]


def install_all(data_dir: Path):
    for provider_cls in providers:
        try:
            instance = provider_cls(data_dir)
            instance.prepare()
            print(f"[OK] Installed {provider_cls.__name__}")
        except Exception as e:
            print(f"[FAIL] {provider_cls.__name__}: {e}")


if __name__ == "__main__":
    install_all(config.path.data)
