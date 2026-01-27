from __future__ import annotations

from collections import defaultdict
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Tuple,
    TypeVar,
    cast,
)

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

X = TypeVar("X", covariant=True)  # input / data
Y = TypeVar("Y", covariant=True)  # label


def split_subset_by_class(
    dataset: Dataset[Tuple[X, Y]],
) -> Dict[int, Subset[Tuple[X, Y]]]:
    """
    Split a dataset into class-wise Subsets.

    Note:
        A minimal cast is used only to satisfy torch.utils.data.Subset,
        which requires a nominal Dataset type.
    """
    class_to_indices: Dict[int, List[int]] = defaultdict(list)

    # DatasetをIterableとして扱うためにcastを使用
    iterable_dataset: Iterable[Tuple[X, Y]] = cast(
        Iterable[Tuple[X, Y]], dataset
    )

    for i, (_, label) in enumerate(iterable_dataset):
        label: Tensor
        class_to_indices[int(label.item())].append(i)

    torch_dataset = cast(Dataset[Tuple[X, Y]], dataset)

    return {
        c: Subset(torch_dataset, idxs)
        for c, idxs in sorted(class_to_indices.items(), key=lambda x: x[0])
    }


def extract_by_prediction_result(
    dataset: Dataset[Tuple[Tensor, int]],
    model: nn.Module,
    device: torch.device,
    predict: Callable[[nn.Module, Tensor], Tuple[Tensor, Tensor]],
    selection: Literal["correct", "incorrect"] = "correct",
    _batch_size: int = 32,
) -> Subset[Tuple[Tensor, int]]:
    """
    Extract subset of samples based on prediction correctness.

    Args:
        dataset: Dataset of (image, label) pairs
        model_manager: ModelManager containing the model
        predict: Prediction function
        selection: "correct" to extract correctly classified samples,
                   "incorrect" for misclassified samples
        _batch_size: Batch size for DataLoader
    Returns:
        Subset of (CAM, label) pairs for the selected samples
    """

    selected_indices: List[int] = []

    lorder = DataLoader(
        dataset,
        batch_size=_batch_size,
        shuffle=False,
        num_workers=0,
    )

    for batch_idx, (images, labels) in enumerate(lorder):
        images: Tensor
        labels: Tensor

        # Move to device
        images = images.to(device)
        labels = labels.to(device)

        # Predict
        _, preds = predict(
            model, images
        )  # probabilities: Tensor[B, K], predictions: Tensor[B]

        if selection == "incorrect":
            selected_mask = preds.view(-1) != labels.view(
                -1
            )  # Tensor[B] (bool)
        elif selection == "correct":
            selected_mask = preds.view(-1) == labels.view(
                -1
            )  # Tensor[B] (bool)
        else:
            raise ValueError(f"Invalid selection: {selection}")

        offset = batch_idx * _batch_size
        selected_indices.extend(
            (
                offset + i
                for i, is_selected in enumerate(selected_mask)
                if is_selected
            )
        )

    return Subset(dataset, selected_indices)
