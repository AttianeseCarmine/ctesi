# models/model.py
import torch.nn as nn

# Questo file è un placeholder. 
# Per CLIP-EBC, usiamo models/clip/model.py

class Regressor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("Per CLIP-EBC non usare questa classe. Usa models.clip.model.CLIP_EBC")

class Classifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("Per CLIP-EBC non usare questa classe. Usa models.clip.model.CLIP_EBC")

def _classifier(*args, **kwargs):
    raise NotImplementedError("Funzione deprecata per CLIP-EBC.")

def _regressor(*args, **kwargs):
    raise NotImplementedError("Funzione deprecata per CLIP-EBC.")