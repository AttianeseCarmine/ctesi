# ============================================================
# ZIP-CLIP-EBC: Loss Functions
# ============================================================
# Loss separate per ogni stage:
#   - Stage 1: BCE per π-head (vuoto/pieno)
#   - Stage 2: Cross-Entropy per EBC-head (bins di conteggio)
#   - Stage 3: Loss combinata
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math


# ============================================================
# STAGE 1: π-Head Loss (Classificazione vuoto/pieno)
# ============================================================

class PiHeadLoss(nn.Module):
    """
    Loss per π-head (Stage 1).
    
    Componenti:
    1. BCE Loss per classificazione binaria (vuoto vs pieno)
    2. Loss ausiliaria sul conteggio totale (opzionale)
    3. Regolarizzazione sulla distribuzione di π (opzionale)
    
    Args:
        pos_weight: Peso per la classe positiva (pieno)
        count_weight: Peso per la loss sul conteggio totale
        pi_reg_weight: Peso per la regolarizzazione su π
        pi_reg_target: Target per la media di π (es. 0.25 = 25% blocchi pieni)
        block_size: Dimensione del blocco per calcolare GT
    """
    
    def __init__(
        self,
        pos_weight: float = 3.0,
        count_weight: float = 0.1,
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
    
    def compute_gt_occupancy(
        self,
        gt_density: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcola la ground truth per l'occupancy (vuoto/pieno) dei blocchi.
        
        Args:
            gt_density: [B, 1, H, W] - mappa di densità GT
        
        Returns:
            gt_occupancy: [B, 1, H_block, W_block] - 1 se blocco contiene persone, 0 altrimenti
        """
        # Somma la densità per ogni blocco
        gt_counts_per_block = F.avg_pool2d(
            gt_density,
            kernel_size=self.block_size,
            stride=self.block_size
        ) * (self.block_size ** 2)
        
        # Soglia: se count > 0.5, blocco è "pieno"
        gt_occupancy = (gt_counts_per_block > 0.5).float()
        
        return gt_occupancy
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        gt_density: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calcola la loss per Stage 1.
        
        Args:
            predictions: dict con 'logit_pi_maps' [B, 2, H, W]
            gt_density: [B, 1, H, W] mappa di densità GT
        
        Returns:
            total_loss: Loss totale
            loss_dict: Dizionario con loss componenti per logging
        """
        logit_pi_maps = predictions["logit_pi_maps"]  # [B, 2, H_block, W_block]
        
        # Estrai logit per "pieno" (canale 1)
        logit_pieno = logit_pi_maps[:, 1:2, :, :]  # [B, 1, H_block, W_block]
        
        # GT occupancy
        gt_occupancy = self.compute_gt_occupancy(gt_density)  # [B, 1, H_block, W_block]
        
        # Allinea dimensioni se necessario
        if gt_occupancy.shape != logit_pieno.shape:
            gt_occupancy = F.interpolate(
                gt_occupancy, size=logit_pieno.shape[-2:],
                mode='nearest'
            )
        
        # BCE Loss
        self.bce.pos_weight = self.bce.pos_weight.to(logit_pieno.device)
        bce_loss = self.bce(logit_pieno, gt_occupancy).mean()
        
        # Count Loss (opzionale)
        count_loss = torch.tensor(0.0, device=logit_pieno.device)
        if self.count_weight > 0:
            pi_prob = torch.sigmoid(logit_pieno)  # P(pieno)
            # Stima grezza del conteggio: somma delle probabilità
            # (ogni blocco pieno contribuisce ~1 al conteggio medio)
            pred_count_proxy = pi_prob.sum(dim=[1, 2, 3])
            
            # GT: numero di blocchi pieni
            gt_count_blocks = gt_occupancy.sum(dim=[1, 2, 3])
            
            count_loss = F.l1_loss(pred_count_proxy, gt_count_blocks)
        
        # Regularizzazione su π (opzionale)
        reg_loss = torch.tensor(0.0, device=logit_pieno.device)
        if self.pi_reg_weight > 0:
            pi_prob = torch.sigmoid(logit_pieno)
            pi_mean = pi_prob.mean()
            reg_loss = (pi_mean - self.pi_reg_target) ** 2
        
        # Loss totale
        total_loss = bce_loss + self.count_weight * count_loss + self.pi_reg_weight * reg_loss
        
        # Loss dict per logging
        loss_dict = {
            "pi_bce_loss": bce_loss.detach(),
            "pi_count_loss": count_loss.detach(),
            "pi_reg_loss": reg_loss.detach(),
            "pi_total_loss": total_loss.detach(),
        }
        
        return total_loss, loss_dict


# ============================================================
# STAGE 2: EBC-Head Loss (Classificazione nei bins)
# ============================================================

class EBCHeadLoss(nn.Module):
    """
    Loss per EBC-head (Stage 2).
    
    Componenti:
    1. Cross-Entropy per classificazione nei bins di conteggio
    2. Loss ausiliaria sul conteggio totale
    
    Args:
        bins: Lista di tuple (min, max) per ogni bin
        label_smoothing: Label smoothing per CE
        count_weight: Peso per la loss sul conteggio
        block_size: Dimensione del blocco
    """
    
    def __init__(
        self,
        bins: List[Tuple[int, int]],
        label_smoothing: float = 0.1,
        count_weight: float = 0.5,
        block_size: int = 16,
    ):
        super().__init__()
        
        self.bins = bins
        self.label_smoothing = label_smoothing
        self.count_weight = count_weight
        self.block_size = block_size
        
        # Il primo bin [0,0] è gestito da π, quindi lo escludiamo
        if bins[0] == [0, 0] or bins[0] == (0, 0):
            self.ebc_bins = bins[1:]
        else:
            self.ebc_bins = bins
        
        self.num_ebc_bins = len(self.ebc_bins)
        
        # Cross-entropy con label smoothing
        self.ce = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction='none'
        )
    
    def get_bin_index(self, count: float) -> int:
        """Trova l'indice del bin per un dato conteggio."""
        for i, (lo, hi) in enumerate(self.ebc_bins):
            if lo <= count <= hi:
                return i
        # Se count > ultimo bin, assegna all'ultimo
        return self.num_ebc_bins - 1
    
    def compute_gt_bin_labels(
        self,
        gt_density: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcola i bin labels GT per ogni blocco.
        
        Args:
            gt_density: [B, 1, H, W]
        
        Returns:
            gt_bin_labels: [B, H_block, W_block] - indice del bin
            gt_mask: [B, H_block, W_block] - maschera per blocchi non-vuoti
        """
        # Conteggio per blocco
        gt_counts = F.avg_pool2d(
            gt_density,
            kernel_size=self.block_size,
            stride=self.block_size
        ) * (self.block_size ** 2)  # [B, 1, H_block, W_block]
        
        gt_counts = gt_counts.squeeze(1)  # [B, H_block, W_block]
        
        B, H, W = gt_counts.shape
        device = gt_counts.device
        
        # Crea tensor per bin labels
        gt_bin_labels = torch.zeros(B, H, W, dtype=torch.long, device=device)
        
        # Maschera per blocchi non-vuoti (count > 0.5)
        gt_mask = (gt_counts > 0.5).float()
        
        # Assegna bin index usando vectorized operations
        for i, (lo, hi) in enumerate(self.ebc_bins):
            in_bin = (gt_counts >= lo) & (gt_counts <= hi)
            gt_bin_labels[in_bin] = i
        
        # Per conteggi > ultimo bin, assegna ultimo indice
        last_hi = self.ebc_bins[-1][1]
        gt_bin_labels[gt_counts > last_hi] = self.num_ebc_bins - 1
        
        return gt_bin_labels, gt_mask
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        gt_density: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calcola la loss per Stage 2.
        
        Args:
            predictions: dict con 'logit_bin_maps', 'lambda_maps', 'pi_prob'
            gt_density: [B, 1, H, W]
        
        Returns:
            total_loss: Loss totale
            loss_dict: Componenti per logging
        """
        logit_bin_maps = predictions["logit_bin_maps"]  # [B, num_bins, H, W]
        lambda_maps = predictions["lambda_maps"]  # [B, 1, H, W]
        pi_prob = predictions.get("pi_prob")  # [B, 1, H, W] (opzionale)
        
        # GT
        gt_bin_labels, gt_mask = self.compute_gt_bin_labels(gt_density)
        
        # Allinea dimensioni
        B, C, H_pred, W_pred = logit_bin_maps.shape
        if gt_bin_labels.shape[-2:] != (H_pred, W_pred):
            # Ridimensiona GT
            gt_counts = F.avg_pool2d(
                gt_density,
                kernel_size=self.block_size,
                stride=self.block_size
            ) * (self.block_size ** 2)
            if gt_counts.shape[-2:] != (H_pred, W_pred):
                gt_counts = F.interpolate(gt_counts, size=(H_pred, W_pred), mode='nearest')
            gt_counts = gt_counts.squeeze(1)
            gt_mask = (gt_counts > 0.5).float()
            
            # Ricalcola bin labels
            gt_bin_labels = torch.zeros(B, H_pred, W_pred, dtype=torch.long, device=logit_bin_maps.device)
            for i, (lo, hi) in enumerate(self.ebc_bins):
                in_bin = (gt_counts >= lo) & (gt_counts <= hi)
                gt_bin_labels[in_bin] = i
            gt_bin_labels[gt_counts > self.ebc_bins[-1][1]] = self.num_ebc_bins - 1
        
        # Flatten per CE
        logits_flat = logit_bin_maps.permute(0, 2, 3, 1).reshape(-1, self.num_ebc_bins)
        labels_flat = gt_bin_labels.reshape(-1)
        mask_flat = gt_mask.reshape(-1)
        
        # CE Loss solo sui blocchi non-vuoti
        ce_all = self.ce(logits_flat, labels_flat)
        ce_masked = ce_all * mask_flat
        
        num_valid = mask_flat.sum().clamp(min=1)
        ce_loss = ce_masked.sum() / num_valid
        
        # Count Loss
        count_loss = torch.tensor(0.0, device=logit_bin_maps.device)
        if self.count_weight > 0:
            # Conteggio predetto = pi_prob * lambda
            if pi_prob is not None:
                pred_density = pi_prob * lambda_maps
            else:
                pred_density = lambda_maps
            
            pred_count = pred_density.sum(dim=[1, 2, 3])
            gt_count = gt_density.sum(dim=[1, 2, 3])
            
            count_loss = F.l1_loss(pred_count, gt_count)
        
        # Loss totale
        total_loss = ce_loss + self.count_weight * count_loss
        
        loss_dict = {
            "ebc_ce_loss": ce_loss.detach(),
            "ebc_count_loss": count_loss.detach(),
            "ebc_total_loss": total_loss.detach(),
        }
        
        return total_loss, loss_dict


# ============================================================
# STAGE 3: Joint Loss (Combinata)
# ============================================================

class JointLoss(nn.Module):
    """
    Loss combinata per Stage 3 (fine-tuning congiunto).
    
    Combina:
    - Loss π-head (BCE)
    - Loss EBC-head (CE)
    - Loss sul conteggio totale
    
    Args:
        bins: Lista di bins
        alpha_pi: Peso per loss π
        alpha_ebc: Peso per loss EBC
        count_weight: Peso per loss conteggio
        block_size: Dimensione blocco
    """
    
    def __init__(
        self,
        bins: List[Tuple[int, int]],
        alpha_pi: float = 0.3,
        alpha_ebc: float = 1.0,
        count_weight: float = 0.2,
        block_size: int = 16,
        pos_weight_pi: float = 3.0,
        label_smoothing_ebc: float = 0.1,
    ):
        super().__init__()
        
        self.alpha_pi = alpha_pi
        self.alpha_ebc = alpha_ebc
        self.count_weight = count_weight
        self.block_size = block_size
        
        # Loss components
        self.pi_loss = PiHeadLoss(
            pos_weight=pos_weight_pi,
            count_weight=0.0,  # Gestiamo il count separatamente
            pi_reg_weight=0.0,
            block_size=block_size,
        )
        
        self.ebc_loss = EBCHeadLoss(
            bins=bins,
            label_smoothing=label_smoothing_ebc,
            count_weight=0.0,  # Gestiamo il count separatamente
            block_size=block_size,
        )
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        gt_density: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calcola la loss combinata per Stage 3.
        """
        # π-head loss
        pi_total, pi_dict = self.pi_loss(predictions, gt_density)
        
        # EBC-head loss
        ebc_total, ebc_dict = self.ebc_loss(predictions, gt_density)
        
        # Count loss
        count_loss = torch.tensor(0.0, device=gt_density.device)
        if self.count_weight > 0:
            pred_count = predictions.get("pred_count")
            if pred_count is None:
                pi_prob = predictions["pi_prob"]
                lambda_maps = predictions["lambda_maps"]
                pred_density = pi_prob * lambda_maps
                pred_count = pred_density.sum(dim=[1, 2, 3])
            
            gt_count = gt_density.sum(dim=[1, 2, 3])
            count_loss = F.l1_loss(pred_count, gt_count)
        
        # Loss totale pesata
        total_loss = (
            self.alpha_pi * pi_dict["pi_bce_loss"] +
            self.alpha_ebc * ebc_dict["ebc_ce_loss"] +
            self.count_weight * count_loss
        )
        
        loss_dict = {
            **{f"joint_{k}": v for k, v in pi_dict.items()},
            **{f"joint_{k}": v for k, v in ebc_dict.items()},
            "joint_count_loss": count_loss.detach(),
            "joint_total_loss": total_loss.detach(),
        }
        
        return total_loss, loss_dict


# ============================================================
# Factory Functions
# ============================================================

def build_stage1_loss(config: Dict) -> PiHeadLoss:
    """Costruisce la loss per Stage 1."""
    loss_cfg = config.get("LOSS_STAGE1", {})
    data_cfg = config.get("DATA", {})
    pi_reg = loss_cfg.get("PI_REG", {})
    
    return PiHeadLoss(
        pos_weight=loss_cfg.get("POS_WEIGHT", 3.0),
        count_weight=loss_cfg.get("COUNT_WEIGHT", 0.1),
        pi_reg_weight=pi_reg.get("WEIGHT", 1e-3) if pi_reg.get("ENABLE", False) else 0.0,
        pi_reg_target=pi_reg.get("TARGET_MEAN", 0.25),
        block_size=data_cfg.get("ZIP_BLOCK_SIZE", 16),
    )


def build_stage2_loss(config: Dict) -> EBCHeadLoss:
    """Costruisce la loss per Stage 2."""
    loss_cfg = config.get("LOSS_STAGE2", {})
    data_cfg = config.get("DATA", {})
    dataset_name = config.get("DATASET", "sha")
    bins_cfg = config.get("BINS_CONFIG", {}).get(dataset_name, {})
    
    return EBCHeadLoss(
        bins=bins_cfg.get("bins", [[0, 0], [1, 1]]),
        label_smoothing=loss_cfg.get("LABEL_SMOOTHING", 0.1),
        count_weight=loss_cfg.get("COUNT_WEIGHT", 0.5),
        block_size=data_cfg.get("ZIP_BLOCK_SIZE", 16),
    )


def build_stage3_loss(config: Dict) -> JointLoss:
    """Costruisce la loss per Stage 3."""
    loss_cfg = config.get("LOSS_STAGE3", {})
    data_cfg = config.get("DATA", {})
    dataset_name = config.get("DATASET", "sha")
    bins_cfg = config.get("BINS_CONFIG", {}).get(dataset_name, {})
    
    return JointLoss(
        bins=bins_cfg.get("bins", [[0, 0], [1, 1]]),
        alpha_pi=loss_cfg.get("ALPHA_PI", 0.3),
        alpha_ebc=loss_cfg.get("ALPHA_EBC", 1.0),
        count_weight=loss_cfg.get("COUNT_WEIGHT", 0.2),
        block_size=data_cfg.get("ZIP_BLOCK_SIZE", 16),
    )


if __name__ == "__main__":
    # Test
    print("Testing Loss Functions...")
    
    device = "cpu"
    B, C, H, W = 2, 1, 256, 256
    H_block, W_block = H // 16, W // 16
    
    # Simula GT density
    gt_density = torch.rand(B, 1, H, W) * 0.1  # Sparse
    
    # Simula predictions
    predictions = {
        "logit_pi_maps": torch.randn(B, 2, H_block, W_block),
        "pi_prob": torch.sigmoid(torch.randn(B, 1, H_block, W_block)),
        "logit_bin_maps": torch.randn(B, 13, H_block, W_block),
        "lambda_maps": torch.abs(torch.randn(B, 1, H_block, W_block)) * 5,
    }
    
    # Bins
    bins = [[0, 0]] + [[i, i] for i in range(1, 11)] + [[11, 12], [13, 14], [15, 9999]]
    
    # Test Stage 1 Loss
    print("\nTesting PiHeadLoss (Stage 1)...")
    pi_loss = PiHeadLoss(block_size=16)
    loss, loss_dict = pi_loss(predictions, gt_density)
    print(f"  Total loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")
    
    # Test Stage 2 Loss
    print("\nTesting EBCHeadLoss (Stage 2)...")
    ebc_loss = EBCHeadLoss(bins=bins, block_size=16)
    loss, loss_dict = ebc_loss(predictions, gt_density)
    print(f"  Total loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")
    
    # Test Stage 3 Loss
    print("\nTesting JointLoss (Stage 3)...")
    joint_loss = JointLoss(bins=bins, block_size=16)
    loss, loss_dict = joint_loss(predictions, gt_density)
    print(f"  Total loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")
    
    print("\n✅ Test completato!")
