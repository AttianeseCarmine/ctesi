import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union

# Importa le dipendenze
from .zero_inflated_poisson_nll import ZIPoissonNLL, ZICrossEntropy

EPS = 1e-8

# --- FUNZIONI HELPER ---

def _reshape_density(gt_den_map: Tensor, block_size: int) -> Tensor:
    """Raggruppa la density map in blocchi."""
    B, C, H, W = gt_den_map.shape
    
    # Controllo preventivo dimensioni
    if H % block_size != 0 or W % block_size != 0:
        # Se non ﾃｨ divisibile, interpoliamo alla dimensione corretta piﾃｹ vicina
        new_h = (H // block_size) * block_size
        new_w = (W // block_size) * block_size
        gt_den_map = F.interpolate(gt_den_map, size=(new_h, new_w), mode='bilinear', align_corners=False)
        # Aggiorniamo H, W
        B, C, H, W = gt_den_map.shape

    gt_den_map = gt_den_map.view(
        B, C, H // block_size, block_size, W // block_size, block_size
    )
    # Permute porta a: (B, C, H_grid, W_grid, H_block, W_block)
    gt_den_map = gt_den_map.permute(0, 1, 2, 4, 3, 5).contiguous()
    
    # Somma su H_block (-2) e W_block (-1) per ottenere il conteggio nel blocco
    gt_den_map = gt_den_map.sum(dim=(-1, -2)) 
    
    return gt_den_map

def _bin_count(gt_den_map: Tensor, bins: List[Tuple[float, float]]) -> Tensor:
    """Converte density map in classi bin."""
    float_bins = [(float(low), float(high)) for low, high in bins]
    B, C, H, W = gt_den_map.shape
    num_bins = len(bins)
    
    gt_class_map = torch.zeros((B, H, W), dtype=torch.long, device=gt_den_map.device)
    gt_den_map_flat = gt_den_map.squeeze(1)
    
    bin_edges = [b[0] for b in float_bins] + [float('inf')]

    for i in range(num_bins):
        low = bin_edges[i]
        high = bin_edges[i+1]
        mask = (gt_den_map_flat >= low) & (gt_den_map_flat < high)
        gt_class_map[mask] = i

    return gt_class_map

# --- CLASSE LOSS PRINCIPALE ---
# losses/loss.py

class QuadLoss(nn.Module):
    def __init__(
        self,
        input_size: int,
        block_size: int,
        bins: List[Tuple[float, float]],
        weight_cls: float = 1.0,
        weight_reg: float = 1.0,
        weight_aux: float = 0.0,
        pi_loss_weight_bce: float = 1.0,
        **kwargs
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.block_size = block_size
        self.bins = bins 
        self.bins_lambda = bins[1:]  # EBC bins (senza zero)
        self.num_blocks_h = input_size // block_size
        self.num_blocks_w = input_size // block_size

        self.weight_cls = weight_cls
        self.weight_reg = weight_reg
        self.weight_aux = weight_aux
        self.pi_loss_weight_bce = pi_loss_weight_bce

        # Loss components
        self.pi_loss_bce = ZICrossEntropy(reduction="mean")  # Solo BCE per π
        self.lambda_loss_ebc = nn.CrossEntropyLoss(reduction="mean")
        self.cnt_loss_fn = nn.L1Loss(reduction="mean")

    def forward(
        self,
        pred_logit_map: Tensor,          # EBC logits
        pred_den_map: Tensor,            # Density finale
        gt_den_map: Tensor,              # GT density (in blocchi)
        gt_points: List[Tensor],
        pred_logit_pi_map: Optional[Tensor] = None,  # π logits
        pred_lambda_map: Optional[Tensor] = None,     # ❌ NON PIÙ USATO
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        
        # Sanitizzazione GT
        gt_den_map = torch.clamp(gt_den_map, min=0.0)
        
        # Reshape GT in blocchi (se necessario)
        if gt_den_map.shape[-2:] != (self.num_blocks_h, self.num_blocks_w):
            gt_den_map_blocks = _reshape_density(gt_den_map, block_size=self.block_size)
        else:
            gt_den_map_blocks = gt_den_map

        loss_info = {}
        pi_loss = torch.tensor(0.0, device=pred_den_map.device)
        lambda_loss = torch.tensor(0.0, device=pred_den_map.device)
        cnt_loss = torch.tensor(0.0, device=pred_den_map.device)

        # --- 1. LOSS π (Stage 1 + 3) ---
        if self.weight_reg > 0 and pred_logit_pi_map is not None:
            # Sanitizzazione
            pi_logits = torch.clamp(pred_logit_pi_map.float(), -10, 10)
            
            # ✅ Solo Binary Cross Entropy (vuoto vs pieno)
            pi_loss, pi_info = self.pi_loss_bce(pi_logits, gt_den_map_blocks)
            
            loss_info["pi_bce"] = pi_info['bce'].detach()

        # --- 2. LOSS λ_EBC (Stage 2 + 3) ---
        if self.weight_cls > 0:
            # Sanitizzazione
            if torch.isnan(pred_logit_map).any():
                pred_logit_map = torch.nan_to_num(pred_logit_map, 0.0)
            
            # Converti GT in classi bin (EBC)
            gt_class_map = _bin_count(gt_den_map_blocks, bins=self.bins_lambda)
            
            # Maschera: considera solo blocchi NON vuoti
            gt_mask_nonzero = (gt_den_map_blocks >= self.bins_lambda[0][0]).squeeze(1)
            
            num_active = gt_mask_nonzero.sum()
            
            if num_active > 0:
                logits = pred_logit_map.permute(0, 2, 3, 1)[gt_mask_nonzero]
                targets = gt_class_map[gt_mask_nonzero]
                lambda_loss = self.lambda_loss_ebc(logits, targets)
            
            loss_info["ebc_loss"] = lambda_loss.detach()

        # --- 3. LOSS CONTEGGIO (AUX - opzionale) ---
        if self.weight_aux > 0:
            if torch.isnan(pred_den_map).any():
                pred_den_map = torch.nan_to_num(pred_den_map, 0.0)

            gt_cnt = torch.tensor([len(p) for p in gt_points], dtype=torch.float32, device=pred_den_map.device)
            pred_cnt = pred_den_map.float().sum(dim=(1, 2, 3))
            cnt_loss = self.cnt_loss_fn(pred_cnt, gt_cnt)

            cnt_loss = torch.clamp(cnt_loss, max=50.0)

            loss_info["cnt_loss"] = cnt_loss.detach()

        # --- TOTALE ---
        total_loss = (self.weight_reg * pi_loss) + \
                     (self.weight_cls * lambda_loss) + \
                     (self.weight_aux * cnt_loss)
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("💀 Loss NaN/Inf! Fallback a 100.0")
            total_loss = torch.tensor(100.0, device=total_loss.device, requires_grad=True)
            
        loss_info["total_loss"] = total_loss.detach()
        
        return total_loss, loss_info