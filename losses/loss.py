import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union

# Importa le uniche dipendenze necessarie e presenti nel tuo progetto
from .zero_inflated_poisson_nll import ZIPoissonNLL, ZICrossEntropy

EPS = 1e-8

# --- FUNZIONI HELPER (da .utils mancante) ---

def _reshape_density(gt_den_map: Tensor, block_size: int) -> Tensor:
    """Raggruppa la density map in blocchi."""
    B, C, H, W = gt_den_map.shape
    assert H % block_size == 0 and W % block_size == 0, \
        f"Le dimensioni H e W ({H}x{W}) non sono divisibili per block_size ({block_size})"
    
    gt_den_map = gt_den_map.view(
        B, C, H // block_size, block_size, W // block_size, block_size
    )
    gt_den_map = gt_den_map.permute(0, 1, 2, 4, 3, 5).contiguous()
    gt_den_map = gt_den_map.view(
        B, C, (H // block_size) * (W // block_size), block_size, block_size
    )
    gt_den_map = gt_den_map.sum(dim=(-1, -2))
    gt_den_map = gt_den_map.view(
        B, C, H // block_size, W // block_size
    )
    return gt_den_map

def _bin_count(gt_den_map: Tensor, bins: List[Tuple[float, float]]) -> Tensor:
    """
    Converte la density map in una mappa di classi (bin).
    bins: DEVE essere la lista dei bin NON-ZERO (es. 13 bin).
    """
    
    float_bins = [(float(low), float(high)) for low, high in bins]
    
    B, C, H, W = gt_den_map.shape
    num_bins = len(bins) # es. 13
    
    # Inizializza con 0 (o un valore valido), non -100
    gt_class_map = torch.full(
        (B, H, W), fill_value=0, dtype=torch.long, device=gt_den_map.device
    )
    
    gt_den_map_flat = gt_den_map.squeeze(1) # Rimuovi canale C (B, H, W)
    
    # Crea i bordi dei bin
    bin_edges = [float_bins[0][0]] # Es: 1.0 (da bin [1,1])
    for i in range(len(float_bins) - 1):
        bin_edges.append(float_bins[i+1][0]) # Es: 2.0, 3.0, ..., 15.0
    bin_edges.append(float('inf')) # Bordo finale

    # bin_edges ora è [1.0, 2.0, ..., 15.0, inf]

    for i in range(num_bins): # i va da 0 a 12
        low = bin_edges[i]
        high = bin_edges[i+1]
        
        mask = (gt_den_map_flat >= low) & (gt_den_map_flat < high)
        gt_class_map[mask] = i # Assegna la classe 0, 1, ..., 12

    return gt_class_map

# --- CLASSE LOSS PRINCIPALE (Modificata) ---

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
        assert input_size % block_size == 0
        
        self.input_size = input_size
        self.block_size = block_size
        self.bins = bins # Lista completa di 14 bin
        self.bins_lambda = bins[1:] # Lista di 13 bin per EBC
        self.num_bins = len(bins)
        self.num_blocks_h = input_size // block_size
        self.num_blocks_w = input_size // block_size

        self.weight_cls = weight_cls
        self.weight_reg = weight_reg
        self.weight_aux = weight_aux
        self.pi_loss_weight_bce = pi_loss_weight_bce

        self.pi_loss_nll = ZIPoissonNLL(reduction="mean")
        self.pi_loss_bce = ZICrossEntropy(reduction="mean")
        
        # === MODIFICA: Riporta CrossEntropyLoss alla normalità ===
        # Non abbiamo più bisogno di reduction='none' o ignore_index
        self.lambda_loss_ebc = nn.CrossEntropyLoss(reduction="mean")
        
        self.cnt_loss_fn = nn.L1Loss(reduction="mean")

    def forward(
        self,
        pred_logit_map: Tensor, # Output EBC/Lambda-Head (B, 13, H, W)
        pred_den_map: Tensor,   # Output Densità Finale (B, 1, H, W)
        gt_den_map: Tensor,     # GT Densità (B, 1, H_full, W_full)
        gt_points: List[Tensor],
        pred_logit_pi_map: Optional[Tensor] = None, # Output PI-Head (B, 2, H, W)
        pred_lambda_map: Optional[Tensor] = None, # Output PI-Head (B, 1, H, W)
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        
        B = pred_den_map.shape[0]

        if gt_den_map.shape[-2:] != (self.num_blocks_h, self.num_blocks_w):
            assert gt_den_map.shape[-2:] == (self.input_size, self.input_size)
            gt_den_map_blocks = _reshape_density(gt_den_map, block_size=self.block_size)
        else:
            gt_den_map_blocks = gt_den_map

        loss_info = {}
        pi_loss = torch.tensor(0.0, device=pred_den_map.device)
        lambda_loss = torch.tensor(0.0, device=pred_den_map.device)
        cnt_loss = torch.tensor(0.0, device=pred_den_map.device)

        # --- 1. Calcolo Loss PI-Head (ZIP) ---
        if self.weight_reg > 0:
            assert pred_logit_pi_map is not None and pred_lambda_map is not None
            
            pi_logits_stable = pred_logit_pi_map.float()
            lambda_map_stable = pred_lambda_map.float()

            pi_loss_n, n_info = self.pi_loss_nll(pi_logits_stable, lambda_map_stable, gt_den_map_blocks)
            pi_loss_b, b_info = self.pi_loss_bce(pi_logits_stable, gt_den_map_blocks)
            
            pi_loss = pi_loss_n + (pi_loss_b * self.pi_loss_weight_bce)
            loss_info.update({"pi_nll_loss": n_info['nll'].detach(), "pi_bce_loss": b_info['bce'].detach()})

        # --- 2. Calcolo Loss LAMBDA-Head (EBC) ---
        if self.weight_cls > 0:
            # === INIZIO MODIFICA: Filtra i dati PRIMA della loss ===
            
            # 1. Crea la GT per la classificazione EBC (etichette 0-12)
            #    Usa i 13 bin non-zero
            gt_class_map = _bin_count(gt_den_map_blocks, bins=self.bins_lambda)

            # 2. Crea una maschera GT per i blocchi NON-VUOTI
            #    (gt_den_map_blocks > 0) ma usiamo >= 1.0 per coerenza con i bin
            gt_mask_nonzero = (gt_den_map_blocks >= self.bins_lambda[0][0]).squeeze(1) # (B, H, W)
            
            num_active_blocks = gt_mask_nonzero.sum()
            loss_info["mask_active_blocks"] = num_active_blocks.detach() / (B * self.num_blocks_h * self.num_blocks_w)
            
            if num_active_blocks > 0:
                # 3. Seleziona solo i logit dei blocchi non-vuoti
                #    pred_logit_map -> (B, 13, H, W)
                #    Trasponi a (B, H, W, 13) e poi applica la maschera (B, H, W)
                lambda_logits_filtered = pred_logit_map.permute(0, 2, 3, 1)[gt_mask_nonzero] # (N_active, 13)
                
                # 4. Seleziona solo le etichette dei blocchi non-vuoti
                #    gt_class_map -> (B, H, W)
                lambda_targets_filtered = gt_class_map[gt_mask_nonzero] # (N_active)

                # 5. Stabilizza e calcola la loss
                lambda_logits_stable = lambda_logits_filtered.float().clamp(min=-30, max=30)
                
                lambda_loss = self.lambda_loss_ebc(lambda_logits_stable, lambda_targets_filtered)
            else:
                # Se non ci sono blocchi attivi nel batch, la loss EBC è 0
                lambda_loss = torch.tensor(0.0, device=pred_den_map.device)
            
            # === FINE MODIFICA ===
            
            loss_info["lambda_ebc_loss"] = lambda_loss.detach()

        # --- 3. Calcolo Loss Conteggio Totale (AUX) ---
        if self.weight_aux > 0:
            gt_cnt_per_image = torch.tensor([len(p) for p in gt_points], dtype=torch.float32, device=pred_den_map.device)
            pred_cnt_per_image = pred_den_map.float().sum(dim=(1, 2, 3))
            
            cnt_loss = self.cnt_loss_fn(pred_cnt_per_image, gt_cnt_per_image)
            loss_info["cnt_loss"] = cnt_loss.detach()

        # --- Loss Totale Ponderata ---
        total_loss = (self.weight_reg * pi_loss) + \
                     (self.weight_cls * lambda_loss) + \
                     (self.weight_aux * cnt_loss)
        
        if torch.isnan(total_loss):
            print("--- RILEVATO NaN IN LOSS.PY ---")
            print(f"pi_loss: {pi_loss.item()}, lambda_loss: {lambda_loss.item()}, cnt_loss: {cnt_loss.item()}")
            total_loss = torch.tensor(0.0, device=pred_den_map.device, requires_grad=True)

        loss_info["total_loss"] = total_loss.detach()
        
        return total_loss, loss_info