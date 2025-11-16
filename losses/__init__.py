# losses/__init__.py
import torch
from torch import nn
from typing import Dict

# Importa la classe di loss principale
from .loss import QuadLoss
# Importa le altre loss che potresti usare
from .zero_inflated_poisson_nll import ZIPoissonNLL, ZICrossEntropy
# (Assicurati di avere anche gli altri file loss come dm_loss.py, poisson_nll.py, etc.)

def build_loss(cfg: Dict) -> nn.Module:
    """
    Costruisce la funzione di loss in base alla configurazione.
    """
    loss_cfg = cfg['loss']
    model_cfg = cfg['model']
    
    loss_name = loss_cfg.get('name', 'zip_nll') # Default a zip_nll

    # In questo progetto, usiamo sempre QuadLoss come "contenitore"
    # per le altre loss (ZIPNLL, ZICE, L1, etc.)
    
    # Prepara gli argomenti per QuadLoss
    build_params = {
        'input_size': model_cfg['input_size'],
        'block_size': model_cfg['block_size'],
        'bins': model_cfg['bins'],
        'reg_loss': loss_cfg.get('reg_loss', 'zipnll'),
        'aux_loss': loss_cfg.get('aux_loss', 'none'),
        'weight_cls': loss_cfg.get('weight_cls', 1.0),
        'weight_reg': loss_cfg.get('weight_reg', 1.0),
        'weight_aux': loss_cfg.get('weight_aux', 0.0),
        # (Aggiungi qui altri parametri da cfg['loss'] se QuadLoss li richiede)
    }

    return QuadLoss(**build_params)

__all__ = [
    "build_loss",
    "QuadLoss"
]