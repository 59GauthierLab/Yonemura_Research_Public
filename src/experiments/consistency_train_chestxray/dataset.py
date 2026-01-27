from typing import NamedTuple

import torch
import torchvision.transforms.v2 as transforms
from torch import Tensor

from config import config
from datasets import ChestXrayPneumoniaProvider

# Chest X-ray Pneumonia
provider = ChestXrayPneumoniaProvider(config.path.data)
provider.prepare()  # 副作用あり: データダウンロード & 解凍

# transform定義: 全レベル共通
# 224x224, 1チャネル

class Size(NamedTuple):
    height: int
    width: int
    channels: int


input_size = Size(height=224, width=224, channels=1)

transform_train = transforms.Compose(
    [
        transforms.ToImage(),
        # Augmentations
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(input_size.height, scale=(0.9, 1.0)),
        transforms.Grayscale(num_output_channels=input_size.channels),
        transforms.Resize((input_size.height, input_size.width)),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

transform_test = transforms.Compose(
    [
        # ToTensor & Normalize
        transforms.ToImage(),
        transforms.Grayscale(num_output_channels=input_size.channels),
        transforms.Resize((input_size.height, input_size.width)),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

train_dataset = provider.get_train(transform=transform_train)
test_dataset = provider.get_test(transform=transform_test)
sample_data = provider.get_test(transform=transform_test)


# 逆正規化(transformの逆変換)定義
def denormalize(tensor: Tensor) -> Tensor:
    return (tensor * 0.5) + 0.5


# 教師データ変換関数定義
def target_transform(targets: torch.Tensor) -> torch.Tensor:
    return targets.squeeze(1)
