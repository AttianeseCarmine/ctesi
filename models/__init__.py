# models/__init__.py
import torch
from torch import nn
from typing import Dict

# Importa la funzione costruttore dal *modulo* clip_ebc (la cartella)
from .clip_ebc import _clip_ebc

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
        
        # Copia il dizionario per evitare di modificare l'originale
        build_params = model_cfg.copy()
        
        # Rimuovi 'name' perché _clip_ebc non se lo aspetta
        build_params.pop('name', None) 
        
        model = _clip_ebc(**build_params)
    else:
        # (Opzionale: aggiungi qui la logica per altri modelli se ti servono)
        raise ValueError(f"Nome modello '{model_name}' in config non riconosciuto.")

    return model

__all__ = [
    "build_model",
]