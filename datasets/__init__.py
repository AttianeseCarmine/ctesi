from .jhu import JHU_Crowd
from .shha import SHHA

def get_dataset(name):
    name = name.lower()
    if name == "jhu": return JHU_Crowd
    if name == "sha": return SHHA
    if name == "shb": return SHHA
    raise ValueError(f"Dataset {name} non riconosciuto.")