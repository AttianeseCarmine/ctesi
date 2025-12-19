# datasets/__init__.py
from .crowd import Crowd
from .jhu import JHUDataset
from .shha import SHHA
from .utils import collate_fn

__all__ = ['Crowd', 'JHUDataset', 'SHHA', 'collate_fn']

def get_dataset(name):
    name = name.lower()
    if name == "jhu": return JHUDataset
    if name == "sha": return SHHA
    if name == "shb": return SHHA
    if name == "crowd": return Crowd
    raise ValueError(f"Dataset {name} non riconosciuto.")