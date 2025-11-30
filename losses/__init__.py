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
    loss_cfg = cfg.get('loss', {})
    model_cfg = cfg.get('model', {})
    
    # ✅ Usa ebc_bins (non più 'bins' generico)
    # Per la loss serve l'intero range (0 + ebc_bins)
    # quindi ricostruiamo i bins completi
    ebc_bins = model_cfg['ebc_bins']
    
    # ✅ Aggiungi bin zero all'inizio (per π-head)
    full_bins = [[0, 0]] + ebc_bins
    
    loss_name = loss_cfg.get('name', 'quad_loss')
    
    if loss_name == 'quad_loss':
        return QuadLoss(
            input_size=model_cfg.get('input_size', 256),
            block_size=model_cfg.get('block_size', 16),
            bins=full_bins,  # ✅ Bins completi (con zero)
            weight_cls=loss_cfg.get('weight_cls', 1.0),
            weight_reg=loss_cfg.get('weight_reg', 1.0),
            weight_aux=loss_cfg.get('weight_aux', 0.0),
            pi_loss_weight_bce=loss_cfg.get('pi_loss_weight_bce', 1.0),
        )
    else:
        raise NotImplementedError(f"Loss '{loss_name}' non implementata")

__all__ = [
    "build_loss",
    "QuadLoss"
]