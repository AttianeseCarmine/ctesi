import torch
from torch import nn, Tensor
from typing import List, Tuple
# Importa le funzioni helper dal file utils.py
import torch.nn.functional as F  
from .utils import _reshape_density, _bin_count

EPS = 1e-8


class ZIPoissonNLL(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        assert reduction in ["none", "mean", "sum"], f"Expected reduction to be one of ['none', 'mean', 'sum'], got {reduction}."
        self.reduction = reduction

    def forward(
        self,
        logit_pi_maps: Tensor,
        lambda_maps: Tensor,
        gt_den_maps: Tensor,
    ) -> Tensor:
        
        # --- FIX per NaN: Assicura che lambda sia positivo ---
        lambda_maps = F.softplus(lambda_maps)

        assert len(logit_pi_maps.shape) == len(lambda_maps.shape) == len(gt_den_maps.shape) == 4, f"Expected 4D (B, C, H, W) tensor, got {logit_pi_maps.shape}, {lambda_maps.shape}, and {gt_den_maps.shape}"
        B, _, H, W = lambda_maps.shape
        assert logit_pi_maps.shape == (B, 2, H, W), f"Expected logit_pi_maps to have shape (B, 2, H, W), got {logit_pi_maps.shape}"
        assert lambda_maps.shape == (B, 1, H, W), f"Expected lambda_maps to have shape (B, 1, H, W), got {lambda_maps.shape}"
        if gt_den_maps.shape[2:] != (H, W):
            gt_h, gt_w = gt_den_maps.shape[-2], gt_den_maps.shape[-1]
            assert gt_h % H == 0 and gt_w % W == 0 and gt_h // H == gt_w // W, f"Expected the spatial dimension of gt_den_maps to be a multiple of that of logit_maps, got {gt_den_maps.shape} and {logit_pi_maps.shape}"
            gt_den_maps = _reshape_density(gt_den_maps, block_size=gt_h // H)
        assert gt_den_maps.shape == (B, 1, H, W), f"Expected gt_den_maps to have shape (B, 1, H, W), got {gt_den_maps.shape}"
        
        gt_zero_mask = (gt_den_maps == 0).float()
        gt_nonzero_mask = 1.0 - gt_zero_mask

        log_pi_zero = torch.log_softmax(logit_pi_maps, dim=1)[:, 0:1, :, :]
        log_pi_nonzero = torch.log_softmax(logit_pi_maps, dim=1)[:, 1:2, :, :]

        # Evita log(0) in log_poisson
        log_poisson = torch.lgamma(gt_den_maps + 1.0)
        log_poisson = -lambda_maps + gt_den_maps * torch.log(lambda_maps + EPS) - log_poisson
        
        nll = -log_pi_zero * gt_zero_mask
        nll -= (log_pi_nonzero + log_poisson) * gt_nonzero_mask

        if self.reduction == "mean":
            nll = nll.mean()
        elif self.reduction == "sum":
            nll = nll.sum()
        
        return nll, {"nll": nll.detach()}


# --- CLASSE RINOMINATA E CORRETTA ---
# Definisce 'ZICrossEntropy' (come si aspetta loss.py)
# e corregge l'AssertionError (non prende 'bins')
class ZICrossEntropy(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        assert reduction in ["none", "mean", "sum"], f"Expected reduction to be one of ['none', 'mean', 'sum'], got {reduction}."
        self.cls_loss_fn = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, pred_logit_map: Tensor, gt_den_map_blocks: Tensor) -> Tensor:
        B, C, H, W = pred_logit_map.shape
        
        # L'asserzione ora controlla C == 2
        assert C == 2, f"ZICrossEntropy (PI-Head loss) si aspetta 2 canali, ma ne ha ricevuti {C}"
        assert gt_den_map_blocks.shape == (B, 1, H, W), f"Expected gt_den_map_blocks to have shape (B, 1, H, W), got {gt_den_map_blocks.shape}"
        
        # Crea la ground truth a 2 classi: 0 = vuoto, 1 = non-vuoto
        gt_class_map = (gt_den_map_blocks > 0).long().squeeze(1) # Forma: [B, H, W]
        
        # Calcola la loss
        bce = self.cls_loss_fn(pred_logit_map, gt_class_map)
            
        return bce, {"bce": bce.detach()}