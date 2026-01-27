from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from grad_cam import GradCAM, PredictFunction, grad_cam
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from utils.dataset import extract_by_prediction_result, split_subset_by_class


def compute_representative_cam(cams: Tensor) -> Tensor:
    """
    Compute representative CAM by averaging.

    Args:
        cams: Tensor of shape [B, 1, H, W]

    Returns:
        rep_cam: Tensor of shape [1, 1, H, W]
    """
    assert cams.ndim == 4, f"Expected [B, 1, H, W], got {cams.shape}"
    return cams.mean(dim=0, keepdim=True)


def compute_cam_distance(
    cams: Tensor,
    rep_cam: Tensor,
) -> Tensor:
    """
    Compute distance between CAMs and representative CAM.
    distance is computed as L2 norm.
    L2 norm between each CAM and the representative CAM,
    flattened over spatial dimensions.

    Args:
        cams: Tensor of shape [B, 1, H, W]
        rep_cam: Tensor of shape [1, 1, H, W]

    Returns:
        distances: Tensor of shape [B]
    """
    assert cams.shape[1:] == rep_cam.shape[1:], "Shape mismatch"

    diff = cams - rep_cam
    distances = torch.norm(diff.reshape(diff.size(0), -1), p=2, dim=1)
    return distances


def compute_gradcam_for_dataset(
    dataset: Subset[Tuple[Tensor, int]],  # __len__() を持つことを保証
    model: GradCAM,
    device: torch.device,
    predict: PredictFunction,
    _batch_size: int = 16,
) -> Tensor:
    """
    Calculate Grad-CAM for all samples in the dataset.
    Args:
        dataset: Subset  containing (input, label) tuples
        model: Grad-CAM compatible model
        device: Device to run computations on
        predict: Prediction function
        _batch_size: Batch size for DataLoader
    Returns:
        cams: Tensor of shape [N, 1, H', W']
    """

    # CAMサイズ取得用のサンプル
    sample_input, sample_target = dataset[0]  # shape: [C, H, W], int
    sample_input = sample_input.unsqueeze(0).to(device)  # shape: [1, C, H, W]

    # CAMサイズ取得
    (
        h,
        w,
    ) = grad_cam(
        model=model,
        inputs=sample_input,
        target_class=sample_target,
        interpolate=False,
    ).shape[2:]

    # データローダー作成
    dataloader = DataLoader(
        dataset,
        batch_size=_batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 全CAM格納用テンソル
    cams = torch.empty((len(dataset), 1, h, w), device=device)

    # 各サンプルのCAM計算・格納
    try:
        model.cam_enable()

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs: Tensor  # shape: [B, C, H, W]
            labels: Tensor  # shape: [B, 1]
            inputs = inputs.to(device)
            labels = labels.to(device).reshape(-1)  # shape: [B]

            cam = grad_cam(
                model=model,
                inputs=inputs,
                target_class=labels,
                predict=predict,
                interpolate=False,
            )  # shape: [B, 1, H', W'] H', W'はCAMの空間サイズ

            start_idx = batch_idx * _batch_size
            end_idx = start_idx + inputs.size(0)
            cams[start_idx:end_idx] = cam
    finally:
        model.cam_disable()

    return cams


@dataclass
class ConsistencyResult:
    consistency: float  # nan if not computable
    representative_cam: Tensor  # shape: [1, 1, H, W]
    data_num: int


def calc_consistency(
    dataset: Dataset[Tuple[Tensor, int]],
    model: GradCAM,
    device: torch.device,
    predict: PredictFunction,
    _batch_size: int = 16,
) -> Tuple[Dict[int, ConsistencyResult], float]:
    """
    Calculate Grad-CAM consistency for each class and overall
    Args:
        dataset: Dataset containing (input, label) tuples
        model: Grad-CAM compatible model
        device: Device to run computations on
        predict: Prediction function
        _batch_size: Batch size for DataLoader
    Returns:
        Tuple containing:
            - Dict mapping class ID to ConsistencyResult
            - Overall consistency score (float)
    """

    result: Dict[int, ConsistencyResult] = {}

    model = model.to(device)

    # データセットを正解予測サンプルに絞る
    dataset_c = extract_by_prediction_result(
        dataset=dataset,
        model=model,
        predict=predict,
        device=device,
        selection="correct",
    )

    # 正解予測サンプルが存在しない場合は例外を投げる
    if len(dataset_c) == 0:
        raise RuntimeError("No correctly predicted samples")

    # クラスごとにデータセットを分割
    dataset_c_by_class = split_subset_by_class(dataset_c)

    for class_id, subset_c in dataset_c_by_class.items():
        # 正解予測サンプルが2個未満の場合はスキップ
        if len(subset_c) in (0, 1):
            result[class_id] = ConsistencyResult(
                consistency=float("nan"),
                representative_cam=torch.empty(0, device=device),
                data_num=len(subset_c),
            )
            continue

        consistency: float = 0.0
        representative_cam: Tensor = torch.empty(0, device=device)
        data_num: int = len(subset_c)

        # CAM計算
        cams = compute_gradcam_for_dataset(
            dataset=subset_c,
            model=model,
            device=device,
            predict=predict,
            _batch_size=_batch_size,
        )  # shape: [N, 1, H, W]

        # CAMサイズ取得
        d, _, h, w = cams.shape

        # 代表値CAM計算
        representative_cam = compute_representative_cam(
            cams
        )  # shape: [1, 1, H', W]'

        # 各CAMの代表値CAMからの距離合計計算
        distance_sum: float = (
            compute_cam_distance(cams, representative_cam).sum().item()
        )
        # CAMサイズで正規化 -> l(xi, c)を算出
        consistency = distance_sum / (h * w)

        result[class_id] = ConsistencyResult(
            consistency=consistency,
            representative_cam=representative_cam,
            data_num=data_num,
        )

    # モデル全体の一貫性スコア計算 -> L(D)を算出
    valid_consistency = [
        v.consistency for v in result.values() if not math.isnan(v.consistency)
    ]
    valid_data_num = [
        v.data_num for v in result.values() if not math.isnan(v.consistency)
    ]

    if len(valid_consistency) == 0:
        total_consistency: float = float("nan")
    else:
        total_consistency: float = sum(valid_consistency) / sum(valid_data_num)

    return result, total_consistency
