
from .sha import SHA

def get_dataset(name):
    name = name.lower()
    if name == "sha": return sha
    if name == "shb": return sha    
    raise ValueError(f"Dataset {name} non riconosciuto.")