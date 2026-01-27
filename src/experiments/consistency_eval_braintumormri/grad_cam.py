from __future__ import annotations

from typing import (
    Callable,
    Literal,
    Optional,
    Tuple,
    TypeAlias,
    TypeGuard,
    cast,
)

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

PredictFunction: TypeAlias = Callable[
    [nn.Module, Tensor], Tuple[Tensor, Tensor]
]


class GradCAM(nn.Module):
    """
    Grad-CAM対応モデルの基底クラス
    サブクラスは以下を実装すること:
    - ネットワークアーキテクチャの定義(`forward`メソッドの実装)
    - 初期化時にターゲットとなる畳み込み層に対して`_register_target_layer(layer)`を呼び出す
    """

    def __init__(self) -> None:
        super().__init__()
        self._target_layer: Optional[nn.Module] = None
        self._cam_enabled: bool = False

        self._fwd_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._bwd_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._activations: Optional[Tensor] = None
        self._gradients: Optional[Tensor] = None

    def _register_target_layer(self, layer: nn.Module) -> None:
        """
        Grad-CAMのターゲットとなる畳み込み層にフックを登録する

        Args:
            layer (nn.Module): ターゲットとなる畳み込み層
        """

        self._target_layer = layer

    def cam_enable(self) -> None:
        """
        Grad-CAMを有効にする
        """
        if self._target_layer is None:
            raise RuntimeError(
                "Target layer is not registered. Call _register_target_layer(layer) first."
            )

        if self._cam_enabled:
            return

        def forward_hook(
            module: nn.Module,
            input: Tuple[Tensor, ...] | Tensor,
            output: Tuple[Tensor, ...] | Tensor,
        ) -> None:
            # Save activations
            self._activations = cast(Tensor, output).detach()

        def backward_hook(
            module: nn.Module,
            input_grad: Tuple[Tensor, ...] | Tensor,
            output_grad: Tuple[Tensor, ...] | Tensor,
        ) -> None:
            # grad_output[0]: gradient w.r.t. the output of the layer
            self._gradients = cast(Tensor, output_grad[0].detach())

        self._fwd_handle = self._target_layer.register_forward_hook(
            forward_hook
        )
        self._bwd_handle = self._target_layer.register_full_backward_hook(
            backward_hook
        )

        self.clear_cam_cache()
        self._cam_enabled = True

    def cam_disable(self) -> None:
        """
        Grad-CAMを無効にする
        """
        if self._target_layer is None:
            raise RuntimeError(
                "Target layer is not registered. Call _register_target_layer(layer) first."
            )

        if not self._cam_enabled:
            return

        if self._fwd_handle is not None:  # 型チェッカー対策
            self._fwd_handle.remove()
            self._fwd_handle = None

        if self._bwd_handle is not None:  # 型チェッカー対策
            self._bwd_handle.remove()
            self._bwd_handle = None

        self.clear_cam_cache()
        self._cam_enabled = False

    def generate_heatmap(self) -> Tensor:
        """
        Grad-CAMヒートマップを生成する。

        順伝播と逆伝播の後に呼び出す必要がある。

        Returns:
            Tensor: Grad-CAMヒートマップ (shape: [B, 1, H, W])
        """
        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "Grad-CAM requires forward() and backward() before calling "
                "generate_heatmap()."
            )

        # _gradients: [B,C,H,W]
        # _activations: [B,C,H,W]

        # グローバル平均プーリングでチャネル重みを求める
        weights = self._gradients.mean(
            dim=(2, 3), keepdim=True
        )  # shape: [B,C,1,1]

        # 重み付き和でCAMを計算
        cam = (weights * self._activations).sum(
            dim=1, keepdim=True
        )  # shape: [B,1,H,W]

        cam = F.relu(cam)

        # 正規化（0〜1）
        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam  # shape: [B,1,H,W]

    def clear_cam_cache(self) -> None:
        """
        Grad-CAMのキャッシュをクリアする
        """
        self._activations = None
        self._gradients = None


def is_gradcam(model: nn.Module) -> TypeGuard[GradCAM]:
    """
    モデルがGradCAMのインスタンスかどうかを判定する

    Args:
        model (nn.Module): 判定対象のモデル

    Returns:
        bool: GradCAMのインスタンスであればTrue、そうでなければFalse
    """
    return isinstance(model, GradCAM)


def interpolate_cam(
    cam: Tensor,
    size: Tuple[int, ...],
) -> Tensor:
    """
    CAMを指定サイズに補間する

    Args:
        cam (Tensor): CAMヒートマップ (shape: [B,1,H,W])
        size (Tuple[int, int]): 出力サイズ (H, W)

    Returns:
        Tensor: 補間後のCAMヒートマップ (shape: [B,1,H,W])
    """

    return F.interpolate(
        cam,
        size=size,
        mode="bilinear",
        align_corners=False,
    )


def grad_cam(
    model: GradCAM,
    inputs: Tensor,
    target_class: Optional[Tensor | int] = None,
    predict: Optional[PredictFunction] = None,
    sign: Literal[1, -1] = 1,
    interpolate: bool = True,
) -> Tensor:
    """
    Grad-CAMを計算する

    Args:
        model (GradCAM): Grad-CAM対応モデル
        inputs (Tensor ): 入力データ (shape: [B,C,H,W])
        target_class (Optional[Tensor | int]): ターゲットクラスラベル (shape: [B])。Noneの場合、予測ラベルを使用。
        sign (int): 勾配の方向。1(正)または-1(負)。デフォルトは1。
        predict (Optional[Callable[[nn.Module, Tensor], Tuple[Tensor, Tensor]]]): 予測関数。predict(model, inputs) -> (probs, labels)
        interpolate (bool): ヒートマップを入力サイズに補間するかどうか。デフォルトはTrue。

    Returns:
        Tensor: Grad-CAMヒートマップ (shape: [B,1,H,W])
    """

    if target_class is None:
        if predict is None:
            raise ValueError(
                "Either target_class or predict function must be provided."
            )
        else:
            _, target_class = predict(model, inputs)
    elif isinstance(target_class, int):
        target_class = torch.full(
            (inputs.size(0),),
            target_class,
            device=inputs.device,
            dtype=torch.long,
        )
    elif isinstance(target_class, Tensor):
        if target_class.dim() != 1:
            raise ValueError(
                f"target_class must be 1D Tensor [B] but got {target_class.shape}"
            )
        if target_class.size(0) != inputs.size(0):
            raise ValueError("target_class size must match batch size")
        target_class = target_class.to(device=inputs.device, dtype=torch.long)
    else:
        raise TypeError("target_class must be int or Tensor")

    model.cam_enable()
    model.eval()
    model.zero_grad(set_to_none=True)

    try:
        # 順伝播
        outputs: Tensor = model(inputs)  # [B, K] (K: クラス数 or 属性数)
        scores = outputs.gather(
            dim=1,
            index=target_class.unsqueeze(1),  # [B, 1]
        )  # -> [B, 1]
        score = sign * scores.sum()

        # 逆伝搬
        score.backward()
        model.zero_grad(set_to_none=True)
        # 上補足: indexの出力を大きくする(または小さくする)方向に寄与した特徴マップの勾配を取得

        # Grad-CAMヒートマップ
        heatmaps = model.generate_heatmap()
        if interpolate:
            heatmaps = interpolate_cam(
                heatmaps,
                size=inputs.shape[2:],  # [H, W]
            )

        return heatmaps
    finally:
        model.zero_grad(set_to_none=True)
        model.cam_disable()
