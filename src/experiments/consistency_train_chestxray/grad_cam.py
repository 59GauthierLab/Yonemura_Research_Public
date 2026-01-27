from __future__ import annotations

from typing import Literal, Optional, Tuple, TypeGuard, cast

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


class GradCAM(nn.Module):
    """
    Base class for Grad-CAM compatible models.
    Subclasses should implement the following:
    - Define the network architecture (implement the `forward` method)
    - Call `_register_target_layer(layer)` on the target convolutional layer during initialization
    """

    def __init__(self) -> None:
        super().__init__()
        self._target_layer: Optional[nn.Module] = None
        self._cam_enabled: bool = False

        self._fwd_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._activations: Optional[Tensor] = None

    def _register_target_layer(self, layer: nn.Module) -> None:
        """
        register the target convolutional layer for Grad-CAM.

        Args:
            layer (nn.Module): Target convolutional layer for Grad-CAM
        """

        self._target_layer = layer

    def cam_enable(self) -> None:
        """
        Enable Grad-CAM.
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
            self._activations = cast(Tensor, output)

        self._fwd_handle = self._target_layer.register_forward_hook(
            forward_hook
        )

        self.clear_cam_cache()
        self._cam_enabled = True

    def cam_disable(self) -> None:
        """
        Disable Grad-CAM.
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

        self.clear_cam_cache()
        self._cam_enabled = False

    def get_activations(self, indices: Optional[Tensor] = None) -> Tensor:
        """
        Get the activations captured by the forward hook.
        Args:
            indices: Indices for which to get activations (shape: [N]) or None
        Returns:
            Tensor: Activations (shape: [N,C',H',W'])
        """
        if self._activations is None:
            raise RuntimeError(
                "Grad-CAM activations are not available. Ensure that forward() has been called."
            )

        if indices is None:
            return self._activations
        else:
            return self._activations[indices]

    def get_gradients(
        self,
        logits: Tensor,
        targets: Tensor,
        indices: Optional[Tensor] = None,
        sign: Literal[1, -1] = 1,
    ) -> Tensor:
        """
        Get the gradients captured.
        Args:
            logits: Model output tensor (shape: [B, K])
            targets: Target class indices (shape: [B])
            indices: Indices for which to get gradients (shape: [N]) or None to get all
            sign: 1 for gradients of scores, -1 for gradients of negative scores

        Returns:
            Tensor: Gradients (shape: [N,C',H',W'])
        """
        if self._activations is None:
            raise RuntimeError(
                "Grad-CAM activations are not available. Ensure that forward() has been called."
            )

        if self._activations.size(0) != logits.size(0):
            raise ValueError(
                "Batch size of activations and predictions must match."
            )

        if indices is None:
            score = (
                sign
                * logits.gather(
                    dim=1,
                    index=targets.unsqueeze(1),  # [B, 1]
                ).sum()
            )
        else:
            score = (
                sign
                * logits[indices]
                .gather(
                    dim=1,
                    index=targets[indices].unsqueeze(1),  # [N, 1]
                )
                .sum()
            )

        # Compute gradients
        grads = torch.autograd.grad(
            outputs=score,
            inputs=self._activations,
            retain_graph=True,
        )[0]

        # Select gradients for specified indices
        if indices is not None:
            grads = grads[indices]

        return grads  # shape: [N,C',H',W']

    def calc_cam(
        self,
        logits: Tensor,
        targets: Tensor,
        indices: Optional[Tensor] = None,
        sign: Literal[1, -1] = 1,
    ) -> Tensor:
        """
        Calculate Grad-CAM heatmaps.
        call after forward()

        Args:
            logits: Model output tensor (shape: [B, K])
            targets: Target class indices (shape: [B])
            indices: Indices for which to generate heatmaps (shape: [N])
            sign: 1 for gradients of scores, -1 for gradients of negative scores

        Returns:
            Tensor: Grad-CAM heatmaps (shape: [N,1,H',W'])
        """
        activations = self.get_activations(indices)  # shape: [N,C',H',W']
        gradients = self.get_gradients(
            logits, targets, indices, sign
        )  # shape: [N,C',H',W']

        # グローバル平均プーリングでチャネル重みを求める
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # shape: [N,C,1,1]

        # 重み付き和でCAMを計算
        cam = (weights * activations).sum(
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
        Clear Grad-CAM cache.
        """
        self._activations = None

    @torch.no_grad()
    def get_cam_size(self, h: int, w: int, c: int) -> tuple[int, int]:
        """
        Infer CAM spatial size (H', W') for a given input image size.

        This method performs a dummy forward pass while preserving:
        - cam_enabled state
        - cached activations
        - model train/eval mode

        Args:
            h: input image height
            w: input image width
            c: input image channels
        Returns:
            (H', W'): spatial size of Grad-CAM
        """
        if self._target_layer is None:
            raise RuntimeError(
                "Target layer is not registered. Call _register_target_layer(layer) first."
            )

        # --- 状態退避 ---
        prev_cam_enabled = self._cam_enabled
        prev_activations = self._activations
        prev_training = self.training

        try:
            # CAM を一時的に有効化
            if not self._cam_enabled:
                self.cam_enable()

            # モデルを eval モードに設定
            self.eval()

            # ダミー入力作成
            device = next(self.buffers(), None)
            device = torch.device("cpu") if device is None else device.device

            dummy = torch.zeros(1, c, h, w, device=device)

            # forward（勾配なし）
            _ = self(dummy)

            if self._activations is None:
                raise RuntimeError(
                    "Failed to capture activations for CAM size inference."
                )

            cam_h, cam_w = self.get_activations().shape[2:]

            return cam_h, cam_w

        finally:
            # --- 状態復元 ---
            self._activations = prev_activations

            if not prev_cam_enabled:
                self.cam_disable()

            self.train(prev_training)


def is_gradcam(model: nn.Module) -> TypeGuard[GradCAM]:
    """
    モデルがGradCAMのインスタンスかどうかを判定する

    Args:
        model (nn.Module): 判定対象のモデル

    Returns:
        bool: GradCAMのインスタンスであればTrue、そうでなければFalse
    """
    return isinstance(model, GradCAM)


def is_gradcam_cls(model_cls: type[nn.Module]) -> TypeGuard[type[GradCAM]]:
    """
    モデルクラスがGradCAMのサブクラスかどうかを判定する

    Args:
        model_cls (type[nn.Module]): 判定対象のモデルクラス

    Returns:
        bool: GradCAMのサブクラスであればTrue、そうでなければFalse
    """
    return issubclass(model_cls, GradCAM)
