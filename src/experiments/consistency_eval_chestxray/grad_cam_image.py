from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple

import numpy as np
import torch
from grad_cam import GradCAM, PredictFunction, grad_cam
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from torch import Tensor
from torch.utils.data import DataLoader

from utils import ModelManager


def save_figure(fig: Figure, path: Path) -> None:
    """
    Figureを画像として保存
    Args:
        fig (Figure): 画像のFigure
        path (Path): 保存先パス
    """
    # 保存先ディレクトリ作成
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path, bbox_inches="tight", pad_inches=0)


def original_figure(
    image: Tensor,
    denormalize: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tuple[Figure, Axes]:
    """
    元画像をFigureとして保存
    Args:
        image (torch.Tensor): 元画像テンソル (shape: [1,C,H,W], C=1 or 3)
        denormalize (Optional[Callable[[torch.Tensor], torch.Tensor]]): 逆正規化関数
    Returns:
        Tuple[Figure, Axes]: 元画像のFigureとAxes
    """

    # Tensor: [B, C, H, W] -> Tensor: [1, C, H, W])
    image = image[0:1]

    # 逆正規化
    if denormalize is not None:
        image = denormalize(image)

    # Tensor: [1, C, H, W] -> Tensor: [C, H, W]
    image = image.squeeze(0).clip(0.0, 1.0).cpu()

    c = image.shape[0]
    if c == 1:
        # Tensor: [1, H, W] -> ndarray: [H, W]
        image_numpy = image[0].numpy()
        cmap = "gray"
    else:
        # Tensor: [C, H, W] -> ndarray: [H, W, 3]
        image_numpy = image.permute(1, 2, 0).numpy()
        cmap = None

    # Figure 生成
    fig = Figure(frameon=False)
    FigureCanvasAgg(fig)  # backend 明示

    ax = fig.add_axes((0, 0, 1, 1))  # 余白なし
    ax.imshow(image_numpy, cmap=cmap)  # 画像表示
    ax.axis("off")  # 軸非表示

    return fig, ax


def gradcam_figure(
    model: GradCAM,
    image: Tensor,
    target: int,
    sign: Literal[1, -1] = 1,
    denormalize: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tuple[Figure, Axes]:
    """
    元画像と各クラスのGrad-CAMヒートマップを保存
    Args:
        model (GradCAM): Grad-CAM対応モデル
        data (torch.Tensor): 入力画像テンソル (shape: [1,C,H,W], C=1 or 3)
        target (int): ターゲット(出力層のインデックス)
        denormalize (Optional[Callable[[torch.Tensor], torch.Tensor]]): 逆正規化関数
    Returns:
        Tuple[Figure, Axes]: 元画像とヒートマップを重ねたFigureとAxes
    """

    # Tensor: [B, C, H, W] -> Tensor: [1, C, H, W])
    image = image[0:1]

    heatmap = grad_cam(
        model=model,
        inputs=image,
        target_class=target,
        sign=sign,
        interpolate=True,
    )

    # Tensor: [1, 1, H, W] -> ndarray: [H, W]
    heatmap_numpy = heatmap[0, 0].cpu().numpy()

    # 元画像Figure生成
    fig, ax = original_figure(image, denormalize=denormalize)

    # ヒートマップ重ね描き
    alpha = np.clip(heatmap_numpy, 0.0, 1.0)
    ax.imshow(heatmap_numpy, cmap="jet", alpha=alpha)
    ax.axis("off")

    # Grad-CAM無効化
    model.cam_disable()

    return fig, ax


def save_result_multiclass(
    manager: ModelManager,
    model: GradCAM,
    data_loader: DataLoader,
    classes: List[str],
    predict: PredictFunction,
    denormalize: Optional[Callable[[Tensor], Tensor]] = None,
    device: Optional[torch.device] = None,
) -> None:
    """
    元画像と各クラスのGrad-CAMヒートマップを保存 (多クラス分類対応版)
    Args:
        manager (ModelManager): モデルマネージャ
        model (GradCAM): Grad-CAM対応モデル
        data_loader (DataLoader): データローダ
        classes (List[str]): クラス名リスト
        denormalize (Optional[Callable[[torch.Tensor], torch.Tensor]]): 逆正規化関数
    """

    save_classes: List[int] = list(range(len(classes)))

    # 保存先ディレクトリ作成
    save_dir = manager.root_dir / "gradcam_images"
    save_dir.mkdir(parents=True, exist_ok=True)

    # モデル移動
    if device is not None:
        model = model.to(device)

    for i, (images, labels) in enumerate(data_loader):
        images: Tensor
        labels: Tensor

        # 各サンプルのディレクトリ作成
        sample_dir = save_dir / f"sample_{i + 1}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Tensor: [B, C, H, W] -> Tensor: [1, C, H, W])
        image = images[0:1]
        if device is not None:
            image = image.to(device)

        # 予測データ取得
        _, pre_labels = predict(model, image)  # Tensor: [1, K], Tensor: [1]

        # 元画像保存
        fig, _ = original_figure(
            image,
            denormalize=denormalize,
        )
        save_figure(fig, sample_dir / "original_image.png")

        for cls_idx in save_classes:
            fig, _ = gradcam_figure(
                model,
                image,
                target=cls_idx,
                denormalize=denormalize,
                sign=1,
            )
            save_figure(fig, sample_dir / f"class_{classes[cls_idx]}_pos.png")

            fig, _ = gradcam_figure(
                model,
                image,
                target=cls_idx,
                denormalize=denormalize,
                sign=-1,
            )
            save_figure(fig, sample_dir / f"class_{classes[cls_idx]}_neg.png")

        # 情報保存
        (sample_dir / "info.txt").write_text(
            f"Classes,Truth,Predicted,Match: {','.join(classes)}\n"
            f"Truth: {classes[labels[0][0]]}\n"
            f"Predicted: {classes[pre_labels[0]]}\n"
        )
