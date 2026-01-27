from typing import Sequence, Tuple, Type

import torch
import torch.nn as nn
from grad_cam import GradCAM
from torch import Tensor


# モデル予測関数定義
def predict(model: nn.Module, inputs: Tensor) -> Tuple[Tensor, Tensor]:
    """
    モデル予測(単一属性/多クラス分類)を取得
    Args:
        model (nn.Module): 予測モデル
        inputs (torch.Tensor): 入力データ (shape: [B,C,H,W])
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 予測確率分布 (shape: [B, K])、予測ラベル (shape: [B])
    """

    model.eval()
    with torch.no_grad():
        outputs: Tensor = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        labels = probs.argmax(dim=1)
    return probs, labels


# モデル定義: レベル1 〜 5 までで構成
# レベル    主な違い    比較観点
#   1	    最小	    CAM の最低限
#   2	    標準CNN	    現実的なおベースライン
#   3	    Residual	勾配安定性
#   4	    Attention	注視領域の集中
#   5	    高容量	    精度 vs 説明性


# Level 1: 最小ベースライン CNN
# - Grad-CAM が「どこを見るか」の最下限比較用
# - 計算量・精度ともに最低
class Model1(GradCAM):
    """
    Level 1: Minimal CNN baseline
    """

    def __init__(self, num_classes: int, num_channels: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),  # hook point
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

        self._register_target_layer(self.features[-2])

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# Level 2: 標準的CNN
# - 標準的なCNN構造
# - 精度・Grad-CAM品質のバランスを評価
class Model2(GradCAM):
    """
    Level 2: Standard CNN
    """

    def __init__(self, num_classes: int, num_channels: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),  # hook point
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self._register_target_layer(self.features[-2])

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# Level 3: Residual Block 導入（Grad-CAM 安全）
# - Residual Block を導入したCNN
# - 深いネットワークでの勾配消失対策とGrad-CAM品質の評価
# - Grad-CAM は最後の Convを見るので問題なし
class Model3(GradCAM):
    """
    Level 3: Residual CNN
    """

    def __init__(self, num_classes: int, num_channels: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2),
            ResidualBlock(128, 256),  # hook point
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self._register_target_layer(self.features[-1])

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
        )

        self.skip = (
            nn.Identity()
            if in_ch == out_ch and stride == 1
            else nn.Conv2d(in_ch, out_ch, 1, stride=stride)
        )

        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.conv(x) + self.skip(x))


# Level 4: Attention Mechanism 導入（注視領域の集中）
# - 「どの特徴チャネルを見るか」を明示的に学習
# - Grad-CAM と相性が良い？
class Model4(GradCAM):
    """
    Level 4: CNN + SE Attention
    """

    def __init__(self, num_classes: int, num_channels: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SEBlock(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SEBlock(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            SEBlock(256),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self._register_target_layer(self.features[-1])

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.classifier(x)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        w = self.fc(x).view(x.size(0), -1, 1, 1)
        return x * w


# Level 5: 高表現力CNN(深層 + 広層)
# - ネットワーク容量を大幅に増加
# - 精度向上とGrad-CAM品質(精度は出るが CAM が散る)のトレードオフ評価
class Model5(GradCAM):
    """
    Level 5: High-capacity CNN (deep + wide)
    """

    def __init__(self, num_classes: int, num_channels: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2),
            ResidualBlock(128, 256),
            nn.MaxPool2d(2),
            ResidualBlock(256, 512),  # hook point
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.6),
            nn.Linear(512, num_classes),
        )

        self._register_target_layer(self.features[-1])

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.classifier(x)


models_classes: Sequence[Type[nn.Module]] = [
    Model1,
    Model2,
    Model3,
    Model4,
    Model5,
]
