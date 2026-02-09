"""
Zero-Inflated Poisson Negative Log-Likelihood Loss.

Identica implementazione usata in P2R-ZIP (Christian1301/P2R_ZIP).

Formula:
    se y == 0:  -log( (1-π) + π * e^{-λ} )
    se y > 0:   -log( π * e^{-λ} * λ^y / y! )

Dove:
    - π: probabilità che il blocco sia OCCUPATO (contiene persone)
    - λ: rate Poisson (conteggio atteso se occupato)
    - y: conteggio ground truth per blocco
"""

import torch
import torch.nn.functional as F


def zip_nll(pi, lam, target_counts, eps=1e-8, reduction="mean"):
    """
    Zero-Inflated Poisson Negative Log-Likelihood per blocco.
    
    Args:
        pi: Probabilità blocco OCCUPATO [0,1], shape [B, 1, Hb, Wb]
        lam: Rate Poisson >= 0, shape [B, 1, Hb, Wb]
        target_counts: Conteggi GT >= 0 per blocco, shape [B, 1, Hb, Wb]
        eps: Epsilon per stabilità numerica
        reduction: 'mean', 'sum', o 'none'
        
    Returns:
        loss: Scalar (se reduction='mean' o 'sum') o tensor [B, 1, Hb, Wb]
    """
    # Gestione dimensioni diverse (interpolazione se necessario)
    target_h, target_w = target_counts.shape[-2:]
    if pi.shape[-2:] != (target_h, target_w):
        pi = F.interpolate(pi, size=(target_h, target_w), mode='bilinear', align_corners=False)
    if lam.shape[-2:] != (target_h, target_w):
        lam = F.interpolate(lam, size=(target_h, target_w), mode='bilinear', align_corners=False)

    # Clamp per stabilità numerica
    pi = torch.clamp(pi, 0.0 + eps, 1.0 - eps)
    lam = torch.clamp(lam, eps, 1e6)
    y = target_counts

    # Maschere per blocchi vuoti e non-vuoti
    is_zero = (y == 0).float()
    is_pos = 1.0 - is_zero


    log_p0 = torch.log((1.0 - pi) + pi * torch.exp(-lam) + eps)


    log_pi = torch.log(pi + eps)
    log_fact = torch.lgamma(y + 1.0)  
    log_py = log_pi - lam + y * torch.log(lam + eps) - log_fact

    nll = -(is_zero * log_p0 + is_pos * log_py)

    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    return nll


class ZIPNLLLoss(torch.nn.Module):
    """
    Wrapper nn.Module per zip_nll.
    """
    
    def __init__(self, eps: float = 1e-8, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, pi, lam, target_counts):
        """
        Args:
            pi: [B, 1, Hb, Wb] - P(blocco occupato)
            lam: [B, 1, Hb, Wb] - rate Poisson
            target_counts: [B, 1, Hb, Wb] - conteggi GT
            
        Returns:
            loss: Scalar
        """
        return zip_nll(pi, lam, target_counts, self.eps, self.reduction)
