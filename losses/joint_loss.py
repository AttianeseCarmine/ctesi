"""
Loss Composita per ZIP-CLIP-EBC.

Combina:
1. ZIP Loss: Zero-Inflated Poisson NLL per classificazione vuoto/pieno + rate
2. CLIP-EBC Loss: Cross-Entropy sui bins + Count Loss

Formula (come in P2R-ZIP):
    L_total = L_ZIP + α * L_CLIP_EBC

Dove α (JOINT_ALPHA) bilancia i due obiettivi:
- α < 1: priorità a ZIP (struttura globale)
- α = 1: bilanciamento standard
- α > 1: priorità a CLIP-EBC (precisione locale)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

from .zip_nll import zip_nll
from .clip_ebc_loss import CLIPEBCLoss


class PiHeadLoss(nn.Module):
    """
    Loss per Stage 1 - Pre-training della testa π (classificazione binaria).
    
    Identica alla versione di Christian, usata per addestrare solo
    la classificazione vuoto/pieno prima del joint training.
    
    Args:
        pos_weight: Peso per blocchi pieni (che sono pochi)
        block_size: Dimensione del blocco (stride del backbone)
    """
    
    def __init__(
        self,
        pos_weight: float = 3.0,
        block_size: int = 16,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.block_size = block_size
        
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]),
            reduction='mean'
        )
    
    def compute_gt_occupancy(self, gt_density: torch.Tensor) -> torch.Tensor:
        """
        Genera maschera binaria GT dalla density map.
        
        Args:
            gt_density: [B, 1, H, W] - density map (o conteggi per pixel)
            
        Returns:
            gt_occupancy: [B, 1, Hb, Wb] - 1 se blocco contiene persone, 0 altrimenti
        """
        gt_counts_per_block = F.avg_pool2d(
            gt_density,
            kernel_size=self.block_size,
            stride=self.block_size
        ) * (self.block_size ** 2)
        
        gt_occupancy = (gt_counts_per_block > 0.5).float()
        return gt_occupancy
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        gt_density: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            predictions: Dict con 'logit_pi' [B, 1, Hb, Wb] o [B, 2, Hb, Wb]
            gt_density: [B, 1, H, W] - density map GT
            
        Returns:
            loss, loss_dict
        """
        logit_pi = predictions["logit_pi"]
        
        # Se è [B, 2, H, W], prendi il canale 1 (probabilità "pieno")
        if logit_pi.shape[1] == 2:
            logit_occupied = logit_pi[:, 1:2, :, :]
        else:
            # Se è [B, 1, H, W], è già la prob di occupato
            logit_occupied = logit_pi
        
        gt_occupancy = self.compute_gt_occupancy(gt_density)
        
        # Allinea dimensioni
        if gt_occupancy.shape[-2:] != logit_occupied.shape[-2:]:
            gt_occupancy = F.interpolate(
                gt_occupancy,
                size=logit_occupied.shape[-2:],
                mode='nearest'
            )
        
        # Sposta pos_weight sul device corretto
        if self.bce.pos_weight.device != logit_occupied.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logit_occupied.device)
        
        loss = self.bce(logit_occupied, gt_occupancy)
        
        # Calcola accuracy
        with torch.no_grad():
            pred_occupied = (torch.sigmoid(logit_occupied) > 0.5).float()
            accuracy = (pred_occupied == gt_occupancy).float().mean()
        
        return loss, {
            "pi_bce_loss": loss.detach(),
            "pi_accuracy": accuracy,
        }


class ZIPCLIPEBCLoss(nn.Module):
    """
    Loss Combinata per Joint Training (Stage 3).
    
    Formula: L_total = L_ZIP + α * L_CLIP_EBC
    
    Equivalente alla loss combinata di P2R-ZIP, dove P2R è sostituito da CLIP-EBC.
    
    Args:
        bins: Configurazione bins
        bin_centers: Centri dei bins
        alpha: Peso per L_CLIP_EBC (α nel paper)
        label_smoothing: Label smoothing per CE
        count_weight: Peso per count loss dentro L_CLIP_EBC (β)
        block_size: Dimensione blocco per calcolo conteggi GT
    """
    
    def __init__(
        self,
        bins: List[Tuple[int, int]],
        bin_centers: List[float],
        alpha: float = 1.0,
        label_smoothing: float = 0.1,
        count_weight: float = 0.1,
        block_size: int = 16,
    ):
        super().__init__()
        
        self.alpha = alpha
        self.block_size = block_size
        
        # CLIP-EBC Loss
        self.clip_ebc_loss = CLIPEBCLoss(
            bins=bins,
            bin_centers=bin_centers,
            label_smoothing=label_smoothing,
            count_weight=count_weight,
        )
    
    def compute_block_counts(self, gt_density: torch.Tensor) -> torch.Tensor:
        """
        Calcola i conteggi GT per blocco dalla density map.
        
        Args:
            gt_density: [B, 1, H, W] - density map
            
        Returns:
            block_counts: [B, 1, Hb, Wb] - conteggi per blocco
        """
        block_counts = F.avg_pool2d(
            gt_density,
            kernel_size=self.block_size,
            stride=self.block_size
        ) * (self.block_size ** 2)
        
        return block_counts
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        gt_density: torch.Tensor,
        target_counts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calcola la loss combinata.
        
        Args:
            predictions: Dict con output del modello:
                - pi: [B, 1, Hb, Wb] - P(blocco occupato)
                - lambda_zip: [B, 1, Hb, Wb] - rate Poisson
                - logits: [B, num_bins, Hb, Wb] - logits CLIP-EBC
                - bin_probs: [B, num_bins, Hb, Wb] - probabilità bins
            gt_density: [B, 1, H, W] - density map GT
            target_counts: [B, 1, Hb, Wb] - conteggi per blocco (opzionale)
            
        Returns:
            total_loss: Scalar
            loss_dict: Dict per logging
        """
        # Calcola conteggi per blocco se non forniti
        if target_counts is None:
            target_counts = self.compute_block_counts(gt_density)
        
        # === 1. ZIP Loss ===
        pi = predictions["pi"]
        lambda_zip = predictions["lambda_zip"]
        
        L_zip = zip_nll(pi, lambda_zip, target_counts, reduction="mean")
        
        # === 2. CLIP-EBC Loss ===
        logits = predictions["logits"]
        bin_probs = predictions.get("bin_probs", None)
        
        L_ebc, ebc_dict = self.clip_ebc_loss(logits, target_counts, bin_probs)
        
        # === 3. Loss Totale ===
        # L_total = L_ZIP + α * L_CLIP_EBC
        total_loss = L_zip + self.alpha * L_ebc
        
        # === Loss Dict ===
        loss_dict = {
            "loss_total": total_loss.detach(),
            "loss_zip": L_zip.detach(),
            "loss_ebc": L_ebc.detach(),
            "alpha": torch.tensor(self.alpha),
        }
        
        # Aggiungi metriche EBC
        for k, v in ebc_dict.items():
            loss_dict[k] = v
        
        # Metriche aggiuntive ZIP
        with torch.no_grad():
            pi_mean = pi.mean()
            lambda_mean = lambda_zip.mean()
            loss_dict["zip_pi_mean"] = pi_mean
            loss_dict["zip_lambda_mean"] = lambda_mean
        
        return total_loss, loss_dict


class Stage1ZIPLoss(nn.Module):
    """
    Loss per Stage 1: Pre-training ZIP Head.
    
    Solo ZIP NLL, senza CLIP-EBC.
    """
    
    def __init__(self, block_size: int = 16):
        super().__init__()
        self.block_size = block_size
    
    def compute_block_counts(self, gt_density: torch.Tensor) -> torch.Tensor:
        block_counts = F.avg_pool2d(
            gt_density,
            kernel_size=self.block_size,
            stride=self.block_size
        ) * (self.block_size ** 2)
        return block_counts
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        gt_density: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        target_counts = self.compute_block_counts(gt_density)
        
        pi = predictions["pi"]
        lambda_zip = predictions["lambda_zip"]
        
        loss = zip_nll(pi, lambda_zip, target_counts, reduction="mean")
        
        with torch.no_grad():
            pi_mean = pi.mean()
            lambda_mean = lambda_zip.mean()
            
            # Accuracy occupancy
            gt_occupied = (target_counts > 0.5).float()
            pred_occupied = (pi > 0.5).float()
            occupancy_acc = (pred_occupied == gt_occupied).float().mean()
        
        loss_dict = {
            "zip_nll_loss": loss.detach(),
            "zip_pi_mean": pi_mean,
            "zip_lambda_mean": lambda_mean,
            "zip_occupancy_acc": occupancy_acc,
        }
        
        return loss, loss_dict
class Stage2EBCLoss(nn.Module):
    def __init__(self, bins, bin_centers, label_smoothing=0.1, count_weight=0.1, block_size=16):
        super().__init__()
        self.block_size = block_size
        # Loss pura di CLIP-EBC
        self.clip_ebc_loss = CLIPEBCLoss(
            bins=bins,
            bin_centers=bin_centers,
            label_smoothing=label_smoothing,
            count_weight=count_weight,
        )
    
    def compute_block_counts(self, gt_density):
        # Converte la density map in conteggi per blocco 16x16
        return F.avg_pool2d(gt_density, self.block_size, stride=self.block_size) * (self.block_size**2)
    
    def forward(self, predictions, gt_density):
        target_counts = self.compute_block_counts(gt_density)
        
        # CALCOLA LOSS SU TUTTO (mask=None)
        # Il modello deve imparare che Muro = "Zero persone"
        loss, loss_dict = self.clip_ebc_loss(
            logits=predictions['ebc_logits'],
            target_counts=target_counts,
            mask=None  # <--- Nessuna maschera!
        )
        
        return loss, loss_dict