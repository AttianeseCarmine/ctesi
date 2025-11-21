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
    assert H % block_size == 0 and W % block_size == 0
    
    gt_den_map = gt_den_map.view(
        B, C, H // block_size, block_size, W // block_size, block_size
    )
    gt_den_map = gt_den_map.permute(0, 1, 2, 4, 3, 5).contiguous()
    gt_den_map = gt_den_map.sum(dim=(-1, -3)) # Somma sui blocchi (H_block, W_block)
    
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
        pi_mask_threshold: float = 0.5,
        **kwargs
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.block_size = block_size
        self.bins = bins 
        self.bins_lambda = bins[1:] 
        self.num_blocks_h = input_size // block_size
        self.num_blocks_w = input_size // block_size

        self.weight_cls = weight_cls
        self.weight_reg = weight_reg
        self.weight_aux = weight_aux
        self.pi_loss_weight_bce = pi_loss_weight_bce

        # Componenti Loss
        self.pi_loss_nll = ZIPoissonNLL(reduction="mean")
        self.pi_loss_bce = ZICrossEntropy(reduction="mean")
        self.lambda_loss_ebc = nn.CrossEntropyLoss(reduction="mean")
        self.cnt_loss_fn = nn.L1Loss(reduction="mean")

    def forward(
        self,
        pred_logit_map: Tensor, # EBC Logits
        pred_den_map: Tensor,   # EBC Density
        gt_den_map: Tensor,     # GT Density (Full Res)
        gt_points: List[Tensor],
        pred_logit_pi_map: Optional[Tensor] = None, # ZIP Logits
        pred_lambda_map: Optional[Tensor] = None,   # ZIP Lambda
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        
        B = pred_den_map.shape[0]

        # Reshape GT in blocchi se necessario
        if gt_den_map.shape[-2:] != (self.num_blocks_h, self.num_blocks_w):
            gt_den_map_blocks = _reshape_density(gt_den_map, block_size=self.block_size)
        else:
            gt_den_map_blocks = gt_den_map

        loss_info = {}
        pi_loss = torch.tensor(0.0, device=pred_den_map.device)
        lambda_loss = torch.tensor(0.0, device=pred_den_map.device)
        cnt_loss = torch.tensor(0.0, device=pred_den_map.device)

        # --- 1. LOSS ZIP (PI-Head + Lambda-Head) ---
        if self.weight_reg > 0 and pred_logit_pi_map is not None and pred_lambda_map is not None:
            # Input Sanitization per evitare NaN prima che entrino nelle funzioni
            pi_logits_stable = torch.clamp(pred_logit_pi_map.float(), -10, 10)
            lambda_map_stable = torch.clamp(pred_lambda_map.float(), 0, 100) # Lambda è positivo

            # A. ZIP NLL Loss (Probabilistica)
            pi_loss_n, n_info = self.pi_loss_nll(pi_logits_stable, lambda_map_stable, gt_den_map_blocks)
            
            # B. ZIP BCE Loss (Classificazione Vuoto/Pieno)
            pi_loss_b, b_info = self.pi_loss_bce(pi_logits_stable, gt_den_map_blocks)
            
            # C. ZIP COUNT Loss (L1 Stabilizzante - FONDAMENTALE PER STAGE 1)
            # Calcoliamo il conteggio predetto dalla testa ZIP sommando i lambda
            zip_count_pred = lambda_map_stable.sum(dim=(1,2,3))
            # Calcoliamo il conteggio GT
            zip_count_gt = gt_den_map_blocks.sum(dim=(1,2,3))
            
            # Pesa questa loss ausiliaria (es. 0.1) per guidare il training senza dominarlo
            zip_l1_loss = F.l1_loss(zip_count_pred, zip_count_gt)
            
            # Combinazione Loss ZIP: NLL + BCE + L1
            # Aggiungiamo 0.5 * L1 per stabilizzare i gradienti iniziali
            pi_loss = pi_loss_n + (self.pi_loss_weight_bce * pi_loss_b) + (0.5 * zip_l1_loss)
            
            loss_info.update({
                "zip_nll": n_info['nll'].detach(),
                "zip_bce": b_info['bce'].detach(),
                "zip_l1": zip_l1_loss.detach()
            })

        # --- 2. LOSS EBC (Classification Head) ---
        if self.weight_cls > 0:
            gt_class_map = _bin_count(gt_den_map_blocks, bins=self.bins_lambda)
            gt_mask_nonzero = (gt_den_map_blocks >= self.bins_lambda[0][0]).squeeze(1)
            
            num_active = gt_mask_nonzero.sum()
            
            if num_active > 0:
                # Filtra e calcola loss solo sui blocchi attivi
                logits = pred_logit_map.permute(0, 2, 3, 1)[gt_mask_nonzero]
                targets = gt_class_map[gt_mask_nonzero]
                lambda_loss = self.lambda_loss_ebc(logits, targets)
            
            loss_info["ebc_loss"] = lambda_loss.detach()

        # --- 3. LOSS CONTEGGIO GENERALE (AUX - EBC) ---
        if self.weight_aux > 0:
            gt_cnt = torch.tensor([len(p) for p in gt_points], dtype=torch.float32, device=pred_den_map.device)
            pred_cnt = pred_den_map.float().sum(dim=(1, 2, 3))
            cnt_loss = self.cnt_loss_fn(pred_cnt, gt_cnt)
            loss_info["cnt_loss"] = cnt_loss.detach()

        # --- TOTALE ---
        total_loss = (self.weight_reg * pi_loss) + \
                     (self.weight_cls * lambda_loss) + \
                     (self.weight_aux * cnt_loss)
        
        # Controllo NaN Critico
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("💀 ERRORE: Loss NaN rilevata! Tentativo di salvataggio con clamp...")
            # Clampiamo la loss a un valore alto ma finito per non rompere l'optimizer
            total_loss = torch.tensor(100.0, device=total_loss.device, requires_grad=True)

        loss_info["total_loss"] = total_loss.detach()
        
        return total_loss, loss_info