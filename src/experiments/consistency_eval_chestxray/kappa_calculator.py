from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class KappaCalculator:
    def __init__(
        self,
        num_classes: int,
        models: Sequence[nn.Module],
        predict: Callable[[nn.Module, Tensor], Tuple[Tensor, Tensor]],
        target_transform: Optional[Callable[[Tensor], Tensor]],
        device: torch.device,
        dataset: Optional[Dataset] = None,
    ) -> None:
        self.num_classes: int = num_classes
        self.models: Sequence[nn.Module] = models
        self.predict: Callable[[nn.Module, Tensor], Tuple[Tensor, Tensor]] = (
            predict
        )
        self.target_transform: Optional[Callable[[Tensor], Tensor]] = (
            target_transform
        )
        self.device: torch.device = device
        self.dataset: Optional[Dataset] = None

        if dataset is not None:
            self.set_dataset(dataset)

        # モデルごとの混同行列
        self.confusions: list[Tensor] = [
            torch.zeros(
                num_classes, num_classes, dtype=torch.long, device=device
            )
            for _ in range(len(models))
        ]

    def set_dataset(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def _calc_confusions(self) -> None:
        if self.dataset is None:
            raise ValueError(
                "Dataset is not set. Please set the dataset first."
            )

        # 初期化
        for m in range(len(self.confusions)):
            self.confusions[m].zero_()

        loader = DataLoader(
            self.dataset, batch_size=32, shuffle=False, num_workers=0
        )

        for inputs, targets in loader:
            inputs: Tensor
            targets: Tensor

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            targ = targets
            if self.target_transform is not None:
                targ = self.target_transform(targ)  # targ: (B,)

            for m_idx, model in enumerate(self.models):
                _, preds = self.predict(model, inputs)  # preds: (B,)

                # 混同行列更新
                idx = targ.long() * self.num_classes + preds.long()
                counts = torch.bincount(
                    idx, minlength=self.num_classes * self.num_classes
                ).view(self.num_classes, self.num_classes)

                self.confusions[m_idx] += counts

    def calc_kappa(self, dataset: Optional[Dataset] = None) -> list[float]:
        if dataset is not None:
            self.set_dataset(dataset)

        self._calc_confusions()

        one = torch.tensor(
            1.0, device=self.device
        )  # 1.0 tensor for comparison
        kappas: list[float] = []

        for confusion in self.confusions:
            n = confusion.sum().item()
            if n == 0:
                kappas.append(float("nan"))
                continue

            # 観測一致率 p_o（Accuracy）
            p_o = torch.trace(confusion).float() / n

            # 真・予測ラベルの周辺分布
            true_dist = confusion.sum(dim=1).float() / n
            pred_dist = confusion.sum(dim=0).float() / n

            # 偶然一致率 p_e
            p_e = torch.sum(true_dist * pred_dist)

            if torch.isclose(p_e, one):
                kappas.append(0.0)
            else:
                kappa = (p_o - p_e) / (1.0 - p_e)
                kappas.append(kappa.item())

        return kappas
