
from .sha import SHA
from .ucf_qnrf import UCF_QNRF

from .crowd import Crowd, available_datasets, standardize_dataset_name, NWPUTest
from .transforms import RandomCrop, Resize, RandomResizedCrop, RandomHorizontalFlip, Resize2Multiple, ZeroPad2Multiple
from .transforms import ColorJitter, RandomGrayscale, GaussianBlur, RandomApply, PepperSaltNoise
from .utils import collate_fn

def get_dataset(name):
    name = name.lower()
    if name == "sha": return sha
    if name == "shb": return sha    
    if name == "ucf_qnrf": return UCF_QNRF
    raise ValueError(f"Dataset {name} non riconosciuto.")


__all__ = [
    "Crowd", "available_datasets", "standardize_dataset_name", "NWPUTest",
    "RandomCrop", "Resize", "RandomResizedCrop", "RandomHorizontalFlip", "Resize2Multiple", "ZeroPad2Multiple",
    "ColorJitter", "RandomGrayscale", "GaussianBlur", "RandomApply", "PepperSaltNoise",
    "collate_fn",
]