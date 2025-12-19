# losses/clip_ebc_loss.py
# -*- coding: utf-8 -*-
"""
CLIP-EBC Loss - Equivalente della P2R Loss per il modulo CLIP-EBC.

Mentre P2R usa supervisione diretta sulla density map pixel-wise,
CLIP-EBC classifica ogni blocco in un bin di conteggio discreto.

Componenti:
1. Cross-Entropy Loss: classifica ogni blocco nel bin corretto
2. Count Loss (L1): supervisione diretta sul conteggio totale

Formula:
    L_CLIP_EBC = L_CE + β * L_count
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

ENABLE_DEBUG_LOG = False


class CLIPEBCLoss(nn.Module):
    """
    Loss per il modulo CLIP-EBC.
    
    Classifica ogni blocco nel bin di conteggio corretto usando Cross-Entropy,
    più una loss L1 sul conteggio totale per supervisione diretta.
    
    Args:
        bins: Lista di tuple (min, max) per ogni bin
              Es: [(0,0), (1,1), (2,2), ..., (10,10), (11,15), (16,9999)]
        bin_centers: Centro di ogni bin per calcolare expected count
              Es: [0.0, 1.0, 2.0, ..., 10.0, 13.0, 20.0]
        label_smoothing: Label smoothing per robustezza
        count_weight: Peso per la count loss (β)
        reduction: 'mean' o 'sum'
    """
    
    def __init__(
        self,
        bins: List[Tuple[int, int]],
        bin_centers: List[float],
        label_smoothing: float = 0.1,
        count_weight: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        
        self.bins = bins
        self.num_bins = len(bins)
        self.label_smoothing = label_smoothing
        self.count_weight = count_weight
        self.reduction = reduction
        
        # Registra bin_centers come buffer
        self.register_buffer(
            "bin_centers",
            torch.tensor(bin_centers, dtype=torch.float32)
        )
        
        # Cross-Entropy con label smoothing
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction=reduction,
        )
    
    def _counts_to_bin_labels(self, counts: torch.Tensor) -> torch.Tensor:
        """
        Converte conteggi float in indici di bin.
        
        Args:
            counts: [B, 1, H, W] o [B, H, W] - conteggi per blocco
            
        Returns:
            labels: [B, H, W] - indici dei bins (long tensor)
        """
        if counts.dim() == 4:
            counts = counts.squeeze(1)
        
        B, H, W = counts.shape
        device = counts.device
        
        labels = torch.zeros(B, H, W, dtype=torch.long, device=device)
        
        # Assegna ogni count al bin corretto
        for i, (lo, hi) in enumerate(self.bins):
            mask = (counts >= lo) & (counts <= hi)
            labels[mask] = i
        
        # Valori oltre l'ultimo bin vanno nell'ultimo bin
        last_hi = self.bins[-1][1]
        labels[counts > last_hi] = self.num_bins - 1
        
        return labels
    
    def forward(
        self,
        logits: torch.Tensor,
        target_counts: torch.Tensor,
        bin_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calcola la loss CLIP-EBC.
        
        Args:
            logits: [B, num_bins, H, W] - logits (cosine similarity * temperature)
            target_counts: [B, 1, H, W] - conteggi GT per blocco
            bin_probs: [B, num_bins, H, W] - probabilità post-softmax (opzionale)
            
        Returns:
            total_loss: Scalar
            loss_dict: Dict per logging
        """
        B, C, H, W = logits.shape
        device = logits.device
        
        # Allinea dimensioni se necessario
        if target_counts.shape[-2:] != (H, W):
            target_counts = F.interpolate(
                target_counts,
                size=(H, W),
                mode='nearest'
            )
        
        # === 1. Cross-Entropy Loss ===
        target_labels = self._counts_to_bin_labels(target_counts)  # [B, H, W]
        
        # Reshape per CE: [B*H*W, C] e [B*H*W]
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
        labels_flat = target_labels.reshape(-1)
        
        ce_loss = self.ce_loss(logits_flat, labels_flat)
        
        # === 2. Count Loss (L1) ===
        # Calcola expected count dalla distribuzione sui bins
        if bin_probs is None:
            bin_probs = F.softmax(logits, dim=1)
        
        # Expected count = Σ (prob_bin * center_bin)
        centers = self.bin_centers.view(1, -1, 1, 1)  # [1, C, 1, 1]
        pred_density = (bin_probs * centers).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        
        pred_count = pred_density.sum(dim=[1, 2, 3])  # [B]
        gt_count = target_counts.sum(dim=[1, 2, 3])   # [B]
        
        count_loss = F.l1_loss(pred_count, gt_count, reduction=self.reduction)
        
        # === Total Loss ===
        total_loss = ce_loss + self.count_weight * count_loss
        
        # === Metriche per logging ===
        with torch.no_grad():
            pred_labels = logits_flat.argmax(dim=1)
            accuracy = (pred_labels == labels_flat).float().mean()
            
            mae = torch.abs(pred_count - gt_count).mean()
        
        loss_dict = {
            "ebc_ce_loss": ce_loss.detach(),
            "ebc_count_loss": count_loss.detach(),
            "ebc_total_loss": total_loss.detach(),
            "ebc_accuracy": accuracy,
            "ebc_mae": mae,
        }
        
        if ENABLE_DEBUG_LOG:
            print(f"[EBC DEBUG] CE={ce_loss.item():.4f}, "
                  f"Count={count_loss.item():.2f}, "
                  f"Acc={accuracy.item():.3f}, "
                  f"pred={pred_count.mean().item():.1f}, "
                  f"gt={gt_count.mean().item():.1f}")
        
        return total_loss, loss_dict
 