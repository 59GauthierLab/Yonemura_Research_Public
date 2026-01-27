from .brain_tumor_mri import BrainTumorMRIProvider
from .celeb_a import CelebAProvider
from .chest_xray_pneumonia import ChestXrayPneumoniaProvider
from .cifar_10 import CIFAR10Provider

__all__ = [
    "CelebAProvider",
    "CIFAR10Provider",
    "BrainTumorMRIProvider",
    "ChestXrayPneumoniaProvider",
]
