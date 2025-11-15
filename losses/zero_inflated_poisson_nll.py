import torch
from torch import nn, Tensor
import torch.nn.functional as F

EPS = 1e-8

def _safe_mean(x: Tensor) -> Tensor:
    return x.mean() if x.numel() > 0 else x.new_tensor(0.0)

def _nan_guard(t: Tensor) -> Tensor:
    return torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)

class ZIPoissonNLL(nn.Module):
    """
    Zero-Inflated Poisson NLL, numericamente stabile.
    Ritorna (loss, info_dict) in modo coerente con la chiamata in losses/loss.py
    """
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        assert reduction in {"none", "mean", "sum"}
        self.reduction = reduction
        self.eps = 1e-8

    def forward(
        self,
        pred_logit_pi_map: Tensor,      # [B, 2, H, W] (logits: [vuoto, non-vuoto])
        pred_lambda_map: Tensor,        # [B, 1, H, W] (raw -> softplus)
        gt_den_map_blocks: Tensor       # [B, 1, H, W] (interi >= 0)
    ):
        B, C, H, W = pred_logit_pi_map.shape
        assert C == 2, f"ZIPoissonNLL si aspetta 2 canali (vuoto/non-vuoto), ma ne ha ricevuti {C}"
        assert pred_lambda_map.shape == (B, 1, H, W), \
            f"pred_lambda_map shape attesa {(B,1,H,W)}, ricevuta {pred_lambda_map.shape}"
        assert gt_den_map_blocks.shape == (B, 1, H, W), \
            f"gt_den_map_blocks shape attesa {(B,1,H,W)}, ricevuta {gt_den_map_blocks.shape}"

        # Probabilità π1 (non-vuoto) e π0 (vuoto) dai logits 2-canali
        # Usiamo log-softmax per stabilità numerica: log π_c = log_softmax(logits)[c]
        log_pi = F.log_softmax(pred_logit_pi_map, dim=1)     # [B,2,H,W]
        log_pi0 = log_pi[:, 0, :, :]                         # [B,H,W]
        log_pi1 = log_pi[:, 1, :, :]                         # [B,H,W]

        # λ > 0 con softplus
        lam = F.softplus(pred_lambda_map).squeeze(1) + self.eps   # [B,H,W]

        y = gt_den_map_blocks.squeeze(1)                      # [B,H,W]
        zero_mask = (y == 0)
        nonzero_mask = ~zero_mask

        # y == 0:  -log( π0 + π1 * e^{-λ} ) = -logaddexp( log π0, log π1 - λ )
        log_prob_zero = torch.logaddexp(log_pi0, log_pi1 - lam)   # [B,H,W]
        nll_zero_map = -(log_prob_zero) * zero_mask

        # y > 0:   -[ log π1 + y log λ - λ - log(y!) ]
        log_factorial = torch.lgamma(y[nonzero_mask] + 1.0)       # [N_pos]
        log_prob_pos  = log_pi1[nonzero_mask] \
                        + (y[nonzero_mask] * torch.log(lam[nonzero_mask] + self.eps)) \
                        - lam[nonzero_mask] - log_factorial
        nll_pos_map_full = y.new_zeros(y.shape)
        nll_pos_map_full[nonzero_mask] = -log_prob_pos

        nll_map = _nan_guard(nll_zero_map + nll_pos_map_full)     # [B,H,W]

        if self.reduction == "mean":
            loss = _safe_mean(nll_map)
        elif self.reduction == "sum":
            loss = nll_map.sum()
        else:
            loss = nll_map  # 'none'

        info = {
            "nll": loss if isinstance(loss, Tensor) and loss.dim() == 0 else _safe_mean(nll_map)
        }
        return loss, info


class ZICrossEntropy(nn.Module):
    """
    CE per la componente 'vuoto/non-vuoto' (2 canali).
    Ritorna (loss, info_dict) per coerenza con losses/loss.py
    """
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        assert reduction in {"none", "mean", "sum"}
        # Usiamo reduction='none', poi riduciamo a mano per poter fare masking se serve
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.reduction = reduction

    def forward(
        self,
        pred_logit_pi_map: Tensor,      # [B,2,H,W]
        gt_den_map_blocks: Tensor       # [B,1,H,W]
    ):
        B, C, H, W = pred_logit_pi_map.shape
        assert C == 2, f"ZICrossEntropy si aspetta 2 canali (vuoto/non-vuoto), ma ne ha ricevuti {C}"
        assert gt_den_map_blocks.shape == (B, 1, H, W), \
            f"gt_den_map_blocks shape attesa {(B,1,H,W)}, ricevuta {gt_den_map_blocks.shape}"

        # target binaria: 0=vuoto, 1=non-vuoto
        gt = (gt_den_map_blocks > 0).long().squeeze(1)  # [B,H,W]
        ce_map = self.ce(pred_logit_pi_map, gt)         # [B,H,W]
        ce_map = _nan_guard(ce_map)

        if self.reduction == "mean":
            loss = _safe_mean(ce_map)
        elif self.reduction == "sum":
            loss = ce_map.sum()
        else:
            loss = ce_map  # 'none'

        info = {
            "bce": loss if isinstance(loss, Tensor) and loss.dim() == 0 else _safe_mean(ce_map)
        }
        return loss, info
