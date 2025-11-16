# models/__init__.py
import torch
from torch import nn
from typing import Dict

# Importa il *costruttore* del modello dal tuo file model.py
# Assicurati che il tuo file sia in: models/clip_ebc/model.py
# e che definisca _clip_ebc
from .model import CLIP_EBC, _clip_ebc


def build_model(cfg: Dict, verbose: bool = True) -> nn.Module:
    """
    Costruisce il modello in base alla configurazione.
    """
    # Legge la configurazione del modello dal file .yaml
    model_cfg = cfg['model']
    model_name = model_cfg['name']

    if verbose:
        print(f"Costruzione modello: {model_name}")

    if model_name == "_clip_ebc":
        # Passa *tutti* i parametri da model_cfg al costruttore _clip_ebc
        # Questo include 'model_name', 'weight_name', 'lora', 'pi_thresh', ecc.
        model = _clip_ebc(**model_cfg)
    else:
        # (Opzionale: aggiungi qui la logica per altri modelli come 'ebc' se ti servono)
        raise ValueError(f"Nome modello '{model_name}' in config non riconosciuto.")

    return model

__all__ = [
    "CLIP_EBC",
    "_clip_ebc",
]