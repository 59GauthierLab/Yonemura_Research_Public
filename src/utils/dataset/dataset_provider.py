from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypedDict

from torch.utils.data import Dataset


class LngthDict(TypedDict, total=False):
    train: int
    test: int
    val: int


class DatasetProvider(ABC):
    """
    Unified interface for providing PyTorch-compatible datasets.

    This class abstracts over the differences between:
    - Library-provided datasets (e.g. torchvision.datasets)
    - Manually managed datasets built via custom preparation logic

    Implementations are responsible for returning ready-to-use Dataset
    objects that can be passed directly to torch.utils.data.DataLoader instances.

    Each dataset sample is expected to be a tuple of:
    - torch.Tensor with shape [C, H, W] and floating-point type
    - torch.Tensor with shape [A] and integer or boolean type
    Here, C denotes the number of channels, H and W denote height and width,
    and A denotes the number of attributes or classes.
    """

    def __init__(self, root: Path) -> None:
        """
        Initialize the DatasetProvider.

        :param root: The root directory where the dataset is stored or will be downloaded to.
        :type root: Path
        """
        self.root: Path = root

    @abstractmethod
    def prepare(self) -> None:
        """
        Prepare the dataset for use.
        This may involve downloading, extracting, and processing data.
        """
        raise NotImplementedError

    @abstractmethod
    def get_train(self, transform: Optional[Callable] = None) -> Dataset:
        """
        Return the training split of the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def get_valid(self, transform: Optional[Callable] = None) -> Dataset:
        """
        Return the validation split of the dataset.
        """
        raise NotImplementedError(
            "This dataset does not provide a validation split."
        )

    @abstractmethod
    def get_test(self, transform: Optional[Callable] = None) -> Dataset:
        """
        Return the test split of the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def get_num_labels(self) -> int:
        """
        Return the number of classes or attributes if available.

        Raises NotImplementedError if the number of labels is not provided.
        """
        raise NotImplementedError("Number of labels is not provided.")

    @abstractmethod
    def get_label_names(self) -> list[str]:
        """
        Return the list of classes or attributes if available.

        Raises NotImplementedError if label names are not provided.
        """
        raise NotImplementedError("Label names are not provided.")

    @abstractmethod
    def get_length(self) -> LngthDict:
        """
        Return the lengths of different dataset splits if available.

        Returns a mapping with keys like 'train', 'valid', 'test' and their corresponding lengths.

        Raises NotImplementedError if lengths are not provided.
        """
        raise NotImplementedError("Dataset lengths are not provided.")

    @abstractmethod
    def get_annotation(self) -> Dict[str, Any]:
        """
        Return dataset annotations or metadata if available.

        Examples:
        - Class names
        - Label descriptions
        - Dataset version
        - Citation information

        Returns None if no annotation is provided.
        """
        raise NotImplementedError
