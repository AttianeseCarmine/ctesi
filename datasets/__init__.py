from .jhu import JHUDataset
from .shha import SHHA

def get_dataset(name):
    name = name.lower()
    if name == "jhu": return JHUDataset
    if name == "sha": return SHHA
    if name == "shb": return SHHA
    raise ValueError(f"Dataset {name} non riconosciuto.")