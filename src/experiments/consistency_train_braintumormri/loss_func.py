from typing import Optional, Protocol, cast

import torch
import torch.nn as nn
from grad_cam import GradCAM
from torch import Tensor


class CamConsistencyLoss(nn.Module):
    def __init__(
        self,
        model: GradCAM,
        class_num: int,
        cam_size: tuple[int, int],
        ema_alpha: float = 0.9,
    ) -> None:
        super().__init__()
        self.model: GradCAM = model
        self.class_num: int = class_num
        self.h: int = cam_size[0]
        self.w: int = cam_size[1]
        self.alpha: float = ema_alpha

        # クラスごとの EMA CAM
        # shape: [class_num, 1, 1, H', W']
        self.register_buffer(
            "_ema_cam",
            torch.zeros((class_num, 1, 1, self.h, self.w), dtype=torch.float),
        )
        self.register_buffer(
            "_ema_initialized",
            torch.zeros(class_num, dtype=torch.bool),
        )

    def _get_ema_cam(self) -> Tensor:
        """
        Get EMA CAM
        Returns:
            EMA CAM: Tensor of shape [class_num, 1, 1, H', W']
        """
        return cast(Tensor, self._ema_cam)

    def _get_ema_initialized(self) -> Tensor:
        """
        Get EMA initialized flags
        Returns:
            EMA initialized flags: Tensor of shape [class_num]
        """
        return cast(Tensor, self._ema_initialized)

    @torch.no_grad()
    def _update_ema(self, class_id: int, batch_rep_cam: Tensor) -> None:
        """
        Update EMA CAM for class_id
        Args:
            class_id: class ID
            batch_rep_cam: representative CAM of the current batch(shape: [1, 1, H', W'])
        """
        if self._get_ema_initialized()[class_id]:
            # EMA = α * EMA_pre + (1 - α) * new_value
            self._get_ema_cam()[class_id] = (
                self.alpha * self._get_ema_cam()[class_id]
                + (1 - self.alpha) * batch_rep_cam
            )
        else:
            # EMA = new_value
            self._get_ema_cam()[class_id] = batch_rep_cam.clone()
            self._get_ema_initialized()[class_id] = True

    def _rep(self, cams: Tensor) -> Tensor:
        """
        Compute representative CAM
        Args:
            cams: CAMs(shape: [N, 1, H, W])
        Returns:
            representative CAM(shape: [1, 1, H, W])
        """
        return cams.mean(dim=0, keepdim=True)

    def _distance(self, cams: Tensor, rep_cam: Tensor) -> Tensor:
        """
        Compute distance between CAMs and representative CAM.
        distance is computed as L2 norm.
        L2 norm between each CAM and the representative CAM,
        flattened over spatial dimensions.
        normalization by H*W is done in forward()
        Args:
            cams: Tensor of shape [N, 1, H', W']
            rep_cam: Tensor of shape [1, 1, H', W']

        Returns:
            distances: Tensor of shape [N]
        """
        assert cams.shape[1:] == rep_cam.shape[1:], "Shape mismatch"

        diff = cams - rep_cam
        l2 = torch.norm(
            diff.flatten(start_dim=1),
            p=2,
            dim=1,
        )
        return l2

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            cams: Grad-CAM of correct-class scores(shape: [B, 1, H', W'])
            outputs: model outputs(shape: [B, class_num])
            targets: ground-truth labels(shape: [B])
        Returns:
            consistency loss (scalar Tensor)
        """
        loss_sum: Tensor = torch.tensor(0.0, device=outputs.device)
        count: int = 0

        for c in cast(Tensor, targets.unique()):  # エディタ補完用にキャスト
            class_id = int(c.item())

            # 予測ラベル取得
            preds = outputs.argmax(dim=1)  # shape: [B]

            # クラスcに属する正解予測サンプルのインデックス取得
            # Nc: クラスcに属する正解予測サンプル数
            mask = (targets == c) & (preds == c)
            idx = mask.nonzero(as_tuple=False).squeeze(1)  # shape: [Nc]

            # サンプル数が2未満の場合はスキップ
            if idx.numel() < 2:
                continue

            # Grad-CAM計算
            cams_c = self.model.calc_cam(
                logits=outputs,
                targets=targets,
                indices=idx,
            )  # shape: [Nc, 1, H', W']

            rep_cam = self._rep(cams_c)  # shape: [1, 1, H', W']

            # EMA更新(勾配なし)
            self._update_ema(class_id, rep_cam.detach())

            # EMAを基準分布として使用
            ref_cam = self._get_ema_cam()[class_id]  # shape: [1, 1, H', W']
            l_xc = self._distance(cams_c, ref_cam)  # shape: [Nc]

            loss_sum += l_xc.sum()
            count += l_xc.numel()

        # 全ての予測を間違えた場合 -> 現実的には起こりえないが形式的に対処
        if count == 0:
            return outputs.sum() * 0.0
        return loss_sum / (count * self.h * self.w)


class CamConsistencyLossWithCrossEntropy(nn.Module):
    def __init__(
        self,
        model: GradCAM,
        class_num: int,
        cam_size: tuple[int, int],
        ema_alpha: float = 0.9,
        lambda_: float = 0,
    ) -> None:
        super().__init__()
        self.model = model
        self.cam_consistency_loss = CamConsistencyLoss(
            model=model,
            class_num=class_num,
            cam_size=cam_size,
            ema_alpha=ema_alpha,
        )
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.lambda_ = lambda_

    def get_lambda(self) -> float:
        return self.lambda_

    def set_lambda(self, lambda_: float) -> None:
        self.lambda_ = lambda_

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            outputs: model outputs(shape: [B, class_num])
            targets: ground-truth targets(shape: [B])
        Returns:
            total loss (scalar Tensor)
        """

        # lambda_ == 0 の場合は余計な計算をしないように
        if self.lambda_ == 0:
            return self.cross_entropy_loss(outputs, targets)

        # eval中は勾配を計算できないため、一貫性損失を計算しない(CE損失のみ)
        if not self.model.training:
            return self.cross_entropy_loss(outputs, targets)

        loss_ce = self.cross_entropy_loss(outputs, targets)
        loss_cam = self.cam_consistency_loss(outputs, targets)

        loss = loss_ce + self.lambda_ * loss_cam
        return loss


class LambdaModule(Protocol):
    def get_lambda(self) -> float: ...

    def set_lambda(self, lambda_: float) -> None: ...


class LambdaScheduler:
    """
    LmbdaScheduler gradually changes lambda value in a LambdaModule
    from start_lambda to final_lambda between epoch t0 and t1.
    Before t0, lambda is start_lambda.
    After t1, lambda is final_lambda.

    step() method updates the lambda value based on the current epoch.
    call step() every epoch.
    """

    def __init__(
        self,
        loss_module: LambdaModule,
        start_lambda: float,
        final_lambda: float,
        t0: int,
        t1: int,
    ) -> None:
        self.loss_module: LambdaModule = loss_module
        self.epoch_cnt: int = 0

        if t0 <= 0 or t1 <= 0:
            raise ValueError("t0 and t1 must be positive integers")

        if not t0 <= t1:
            raise ValueError("t1 must be greater than t0")

        self.start_lambda: float = start_lambda
        self.final_lambda: float = final_lambda
        self.t0: int = t0
        self.t1: int = t1

        self.step()  # 初期値設定

    def _lerp(
        self, x0: float, y0: float, x1: float, y1: float, x: float
    ) -> float:
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    def _calc_lambda(self, epoch: int) -> float:
        if epoch <= self.t0:
            return self.start_lambda
        elif self.t1 <= epoch:
            return self.final_lambda
        else:
            return self._lerp(
                x0=self.t0,
                y0=self.start_lambda,
                x1=self.t1,
                y1=self.final_lambda,
                x=epoch,
            )

    def step(self, epoch: Optional[int] = None) -> None:
        if epoch is None:
            self.epoch_cnt += 1
        else:
            self.epoch_cnt = epoch

        lambda_ = self._calc_lambda(self.epoch_cnt)
        self.loss_module.set_lambda(lambda_)
