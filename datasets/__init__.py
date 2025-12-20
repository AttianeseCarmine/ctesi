
from .shha import SHHA

def get_dataset(name):
    name = name.lower()
    if name == "shha": return SHHA
    raise ValueError(f"Dataset {name} non riconosciuto.")