import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

# --- IMPORTS DAI FILE UFFICIALI (Assicurati che siano nella cartella losses/) ---
from .dace_loss import DACELoss
from .dm_loss import DMLoss

# ============================================================
# STAGE 1: π-Head Loss (Classificazione vuoto/pieno)
# ============================================================
class PiHeadLoss(nn.Module):
    """
    Loss per π-head (Stage 1).
    Mantiene la logica originale che hai implementato, ottima per creare la maschera.
    """
    def __init__(
        self,
        pos_weight: float = 3.0,
        count_weight: float = 0.0, # Disattivato di default per Stage 1
        pi_reg_weight: float = 1e-3,
        pi_reg_target: float = 0.25,
        block_size: int = 16,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.count_weight = count_weight
        self.pi_reg_weight = pi_reg_weight
        self.pi_reg_target = pi_reg_target
        self.block_size = block_size
        
        # BCE con pos_weight
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]),
            reduction='none'
        )
    
    def compute_gt_occupancy(self, gt_density: torch.Tensor) -> torch.Tensor:
        # Calcola se un blocco è vuoto o pieno basandosi sulla density map
        gt_counts_per_block = F.avg_pool2d(
            gt_density,
            kernel_size=self.block_size,
            stride=self.block_size
        ) * (self.block_size ** 2)
        return (gt_counts_per_block > 0.5).float()
    
    def forward(self, predictions: Dict[str, torch.Tensor], gt_density: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        logit_pi_maps = predictions["logit_pi_maps"]
        logit_pieno = logit_pi_maps[:, 1:2, :, :]
        
        gt_occupancy = self.compute_gt_occupancy(gt_density)
        
        # Allinea dimensioni se necessario
        if gt_occupancy.shape != logit_pieno.shape:
            gt_occupancy = F.interpolate(gt_occupancy, size=logit_pieno.shape[-2:], mode='nearest')
            
        self.bce.pos_weight = self.bce.pos_weight.to(logit_pieno.device)
        bce_loss = self.bce(logit_pieno, gt_occupancy).mean()
        
        # Regularizzazione (opzionale)
        reg_loss = torch.tensor(0.0, device=logit_pieno.device)
        if self.pi_reg_weight > 0:
            pi_prob = torch.sigmoid(logit_pieno)
            reg_loss = (pi_prob.mean() - self.pi_reg_target) ** 2
            
        total_loss = bce_loss + self.pi_reg_weight * reg_loss
        
        return total_loss, {
            "pi_bce": bce_loss.detach(),
            "pi_reg": reg_loss.detach(),
            "pi_total": total_loss.detach()
        }

# ============================================================
# FACTORY FUNCTIONS (Costruttori delle Loss)
# ============================================================

def build_stage1_loss(config: Dict) -> PiHeadLoss:
    """Costruisce la loss per lo Stage 1."""
    loss_cfg = config.get("LOSS_STAGE1", {})
    data_cfg = config.get("DATA", {})
    pi_reg = loss_cfg.get("PI_REG", {})
    
    return PiHeadLoss(
        pos_weight=loss_cfg.get("POS_WEIGHT", 3.0),
        count_weight=0.0, # Forziamo a 0 per purezza maschera
        pi_reg_weight=pi_reg.get("WEIGHT", 1e-3) if pi_reg.get("ENABLE", False) else 0.0,
        pi_reg_target=pi_reg.get("TARGET_MEAN", 0.25),
        block_size=data_cfg.get("ZIP_BLOCK_SIZE", 16),
    )

def build_stage2_loss(config: Dict, device) -> nn.Module:
    """
    Costruisce la loss per Stage 2 usando la DACELoss ufficiale.
    Integra CrossEntropy (Classificazione) + Optimal Transport/MAE (Conteggio).
    """
    loss_cfg = config.get("LOSS_STAGE2", {})
    data_cfg = config.get("DATA", {})
    dataset_name = config.get("DATASET", "sha")
    
    # Recupera i bins dal config
    bins_cfg = config.get("BINS_CONFIG", {}).get(dataset_name, {})
    bins = bins_cfg.get("bins")
    
    # Parametri
    reduction = config.get("MODEL", {}).get("REDUCTION", 16) # di solito ZIP_BLOCK_SIZE
    input_size = data_cfg.get("CROP_SIZE", 256)
    
    # Parametri Loss Ufficiale
    count_loss_type = loss_cfg.get("COUNT_LOSS_TYPE", "dmcount") # 'mae' o 'dmcount'
    count_weight = loss_cfg.get("COUNT_WEIGHT", 1.0)
    
    print(f"🔧 Building Stage 2 DACELoss: Type={count_loss_type}, Weight={count_weight}")

    # Istanzia la loss ufficiale (dace_loss.py)
    return DACELoss(
        bins=bins,
        reduction=reduction,
        weight_count_loss=count_weight,
        count_loss=count_loss_type,
        input_size=input_size,
        # Parametri per Optimal Transport (usati solo se type='dmcount')
        norm_cood=True, 
        num_of_iter_in_ot=100,
        reg=10.0
    ).to(device)

def build_stage3_loss(config: Dict, device) -> nn.Module:
    """
    Wrapper per Stage 3: combina PiHeadLoss (nostra) + DACELoss (ufficiale).
    """
    # 1. Loss Classificazione (Pi)
    pi_loss_fn = build_stage1_loss(config).to(device)
    
    # 2. Loss Conteggio (EBC - DACELoss)
    # Per stabilità nello stage 3, a volte 'mae' è preferibile a 'dmcount', ma 'dmcount' è più preciso.
    # Usiamo la config dello stage 3 per decidere
    loss3_cfg = config.get("LOSS_STAGE3", {})
    
    # Creiamo una config temporanea per buildare la DACELoss con i parametri dello stage 3
    config_ebc = config.copy()
    config_ebc["LOSS_STAGE2"] = {
        "COUNT_LOSS_TYPE": loss3_cfg.get("COUNT_LOSS_TYPE", "mae"), # Default mae per fine-tuning
        "COUNT_WEIGHT": 1.0 # La DACELoss gestisce il peso interno, noi pesiamo la loss totale dopo
    }
    ebc_loss_fn = build_stage2_loss(config_ebc, device)
    
    class JointLossWrapper(nn.Module):
        def __init__(self, pi_loss, ebc_loss, alpha_pi, alpha_ebc):
            super().__init__()
            self.pi_loss = pi_loss
            self.ebc_loss = ebc_loss
            self.alpha_pi = alpha_pi
            self.alpha_ebc = alpha_ebc
            
        def forward(self, predictions, gt_density, target_points=None):
            # A. Calcola Pi Loss
            loss_pi, dict_pi = self.pi_loss(predictions, gt_density)
            
            # B. Calcola EBC Loss (DACELoss)
            # DACELoss richiede: (pred_class, pred_density, target_density, target_points)
            
            pred_class_logits = predictions['logit_bin_maps'] # [B, N_bins, H, W]
            
            # Per il conteggio, usiamo la densità finale pesata dalla probabilità π
            # Questo insegna a EBC a contare, ma solo dove π dice che c'è gente
            if 'pi_prob' in predictions:
                pred_density = predictions['lambda_maps'] * predictions['pi_prob']
            else:
                pred_density = predictions['lambda_maps']

            # Se target_points è None (es. validation senza punti), DACELoss potrebbe fallire se usa OT.
            # Gestiamo il caso passando lista vuota se necessario, ma train loop DEVE passare punti.
            points_arg = target_points if target_points is not None else []

            loss_ebc, dict_ebc = self.ebc_loss(
                pred_class_logits, 
                pred_density, 
                gt_density, 
                points_arg
            )
            
            # Somma pesata
            total_loss = self.alpha_pi * loss_pi + self.alpha_ebc * loss_ebc
            
            # Unisci i dizionari per il logging
            full_dict = {**dict_pi, **dict_ebc, "total_loss": total_loss.detach()}
            return total_loss, full_dict

    return JointLossWrapper(
        pi_loss_fn, 
        ebc_loss_fn, 
        loss3_cfg.get("ALPHA_PI", 0.05), 
        loss3_cfg.get("ALPHA_EBC", 1.0)
    ).to(device)