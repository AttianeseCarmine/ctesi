
from .sha import SHA
from .ucf_qnrf import UCF_QNRF
def get_dataset(name):
    name = name.lower()
    if name == "sha": return sha
    if name == "shb": return sha    
    if name == "ucf_qnrf": return UCF_QNRF
    raise ValueError(f"Dataset {name} non riconosciuto.")