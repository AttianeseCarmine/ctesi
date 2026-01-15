# losses/clip_ebc_loss.py
# -*- coding: utf-8 -*-
"""
CLIP-EBC Loss Ufficiale (Fix Compatibilità Completa).
Restituisce 3 valori in OTLoss per matchare la logica originale.
Include fix per GPU e UnboundLocalError.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Union
from torch import Tensor

# Costanti per stabilità numerica
EPS = 1e-8
M_EPS = 1e-16

# ==============================================================================
# 1. ALGORITMO SINKHORN
# ==============================================================================
def sinkhorn(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    reg: float = 1e-1,
    maxIter: int = 1000,
    stopThr: float = 1e-9,
    log: bool = True,
    eval_freq: int = 10,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """Risolve il problema del trasporto ottimale (Sinkhorn-Knopp)."""
    device = a.device
    u = (torch.ones_like(a) / a.size(0)).to(device)
    v = (torch.ones_like(b) / b.size(0)).to(device)
    K = torch.exp(-C / reg)
    
    dict_log = {"err": []} if log else None

    for i in range(maxIter):
        u0 = u
        KTu = torch.matmul(u.view(1, -1), K).view(-1)
        v = torch.div(b, KTu + M_EPS)
        Kv = torch.matmul(K, v.view(-1, 1)).view(-1)
        u = torch.div(a, Kv + M_EPS)
        
        if log and i % eval_freq == 0:
            b_hat = (torch.matmul(u.view(1, -1), K) * v.view(1, -1)).view(-1)
            err = (b - b_hat).abs().sum().item()
            dict_log["err"].append(err)
            if err < stopThr:
                break

    if log:
        dict_log["alpha"] = u
        dict_log["beta"] = v
        return K, dict_log
    else:
        return K

# ==============================================================================
# 2. UTILS
# ==============================================================================
def _reshape_density(density: torch.Tensor, reduction: int) -> torch.Tensor:
    """Ridimensiona la GT density per matchare l'output del modello."""
    if isinstance(reduction, (list, tuple)): reduction = reduction[0]
    B, C, H, W = density.shape
    assert H % reduction == 0 and W % reduction == 0, f"Image size ({H}x{W}) not divisible by reduction {reduction}"
    return density.reshape(B, C, H // reduction, reduction, W // reduction, reduction).sum(dim=(-1, -3))

# ==============================================================================
# 3. OTLoss (Transport Loss)
# ==============================================================================
class OTLoss(nn.Module):
    def __init__(self, input_size: int, reduction: int, norm_cood: bool, num_of_iter_in_ot: int = 100, reg: float = 10.0):
        super().__init__()
        self.input_size = input_size
        self.reduction = reduction
        self.norm_cood = norm_cood
        self.num_of_iter_in_ot = num_of_iter_in_ot
        self.reg = reg
        
        # Griglia coordinate
        self.cood = torch.arange(0, input_size, step=reduction, dtype=torch.float32) + reduction / 2
        self.cood.unsqueeze_(0) 
        self.cood = self.cood / input_size * 2 - 1 if self.norm_cood else self.cood
        self.output_size = self.cood.size(1)

    def forward(self, pred_density: torch.Tensor, normed_pred_density: torch.Tensor, target_points: List[torch.Tensor]):
        device = pred_density.device
        loss = torch.zeros([1]).to(device)
        wd = 0
        ot_obj_values = torch.zeros([1]).to(device) # Terzo valore di ritorno
        
        cood = self.cood.to(device)
        
        for idx, points in enumerate(target_points):
            if len(points) > 0:
                # FIX 1: Sposta i punti su GPU
                points = points.to(device) 
                
                # Normalizzazione coordinate
                points = points / self.input_size * 2 - 1 if self.norm_cood else points
                x, y = points[:, 0].unsqueeze(1), points[:, 1].unsqueeze(1)
                
                # Calcolo distanze (FIX 2: UnboundLocalError risolto separando i passaggi)
                x_dist = -2 * torch.matmul(x, cood) + x * x + cood * cood
                y_dist = -2 * torch.matmul(y, cood) + y * y + cood * cood
                dist_raw = y_dist.unsqueeze(2) + x_dist.unsqueeze(1)
                dist = dist_raw.view(dist_raw.size(0), -1)

                source_prob = normed_pred_density[idx][0].view([-1]).detach()
                target_prob = (torch.ones([len(points)]) / len(points)).to(device)
                
                # Sinkhorn
                P, log = sinkhorn(target_prob, source_prob, dist, self.reg, maxIter=self.num_of_iter_in_ot, log=True)
                beta = log["beta"]
                
                # Accumula OT Objective
                ot_obj_values += torch.sum(normed_pred_density[idx] * beta.view([1, self.output_size, self.output_size]))

                # Gradienti
                source_density = pred_density[idx][0].view([-1]).detach()
                source_count = source_density.sum()
                gradient = ((source_count / (source_count**2 + EPS)) * beta) - \
                           ((source_density * beta).sum() / (source_count**2 + EPS))
                
                loss += torch.sum(pred_density[idx] * gradient.detach().view(1, self.output_size, self.output_size))
                wd += torch.sum(dist * P).item()

        return loss, wd, ot_obj_values # <--- RITORNA 3 VALORI (Official Style)

# ==============================================================================
# 4. DMLoss e DACELoss
# ==============================================================================
class DMLoss(nn.Module):
    def __init__(self, input_size=448, reduction=16, weight_ot=0.1, weight_tv=0.01, **kwargs):
        super().__init__()
        self.ot_loss = OTLoss(input_size, reduction, norm_cood=True, **kwargs)
        self.tv_loss = nn.L1Loss(reduction="none")
        self.count_loss = nn.L1Loss(reduction="mean")
        self.weight_ot = weight_ot
        self.weight_tv = weight_tv
        self.reduction = reduction

    def forward(self, pred_density, target_density, target_points):
        # Resize GT
        if target_density.shape[-2:] != pred_density.shape[-2:]:
            target_density = _reshape_density(target_density, self.reduction)
        
        # Normalizzazione
        pred_count = pred_density.view(pred_density.shape[0], -1).sum(dim=1)
        normed_pred = pred_density / (pred_count.view(-1, 1, 1, 1) + EPS)
        
        target_count = torch.tensor([len(p) for p in target_points], device=target_density.device, dtype=torch.float32)
        normed_target = target_density / (target_count.view(-1, 1, 1, 1) + EPS)

        # FIX 3: Unpack 3 valori (Matcha OTLoss)
        ot_loss_val, _, _ = self.ot_loss(pred_density, normed_pred, target_points)
        
        tv_loss_val = (self.tv_loss(normed_pred, normed_target).sum(dim=(1, 2, 3)) * target_count).mean()
        count_loss_val = self.count_loss(pred_count, target_count)

        return ot_loss_val * self.weight_ot + tv_loss_val * self.weight_tv + count_loss_val, {}

class DACELoss(nn.Module):
    def __init__(
        self,
        bins: List[Tuple[float, float]],
        reduction: int,
        weight_count: float = 1.0, 
        count_loss: str = "dmcount",
        weight_ot: float = 0.1,
        weight_tv: float = 0.01,
        label_smoothing: float = 0.0,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.bins = bins
        self.reduction = reduction
        self.cross_entropy_fn = nn.CrossEntropyLoss(reduction="none", label_smoothing=label_smoothing)

        count_loss = count_loss.lower()
        self.count_loss = count_loss
        self.weight_count_loss = weight_count
        
        if self.count_loss == "mae":
            self.use_dm_loss = False
            self.count_loss_fn = nn.L1Loss(reduction="none")
        elif self.count_loss == "mse":
            self.use_dm_loss = False
            self.count_loss_fn = nn.MSELoss(reduction="none")
        else: 
            self.use_dm_loss = True
            input_size = kwargs.get('input_size', 448) 
            self.count_loss_fn = DMLoss(
                input_size=input_size, 
                reduction=reduction, 
                weight_ot=weight_ot, 
                weight_tv=weight_tv, 
                **kwargs
            )

    def _bin_count(self, density_map):
        """
        Mappa ogni valore di densità al bin INTEGER corretto.
        Es: density=2.7 → bin [2,2] (indice 2)
            density=15.3 → bin [14,inf] (indice 14)
        """
        density_map = density_map.squeeze(1)  # [B, H, W]
        class_map = torch.zeros_like(density_map, dtype=torch.long)
        
        for idx, (low, high) in enumerate(self.bins):
            if high > 1000:  # Overflow bin
                mask = (density_map >= low)
            else:
                # Integer matching: floor(density) deve essere in [low, high]
                mask = (torch.floor(density_map) >= low) & (torch.floor(density_map) <= high)
            class_map[mask] = idx
        
        return class_map

    def forward(self, pred_class: Tensor, pred_density: Tensor, target_density: Tensor, target_points: List[Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        if target_density.shape[-2:] != pred_density.shape[-2:]:
            target_density = _reshape_density(target_density, reduction=self.reduction)
        
        target_class = self._bin_count(target_density)
        cross_entropy_loss = self.cross_entropy_fn(pred_class, target_class).sum(dim=(-1, -2)).mean()

        loss_info = {}
        if self.use_dm_loss:
            count_loss_val, _ = self.count_loss_fn(pred_density, target_density, target_points)
            loss_info["ce_loss"] = cross_entropy_loss.detach()
            count_loss_final = count_loss_val
        else:
            count_loss_final = self.count_loss_fn(pred_density, target_density).sum(dim=(-1, -2, -3)).mean()
            loss_info[f"{self.count_loss}_loss"] = count_loss_final.detach()

        loss = cross_entropy_loss + self.weight_count_loss * count_loss_final
        loss_info["loss"] = loss.detach()

        return loss, loss_info
# ==============================================================================
# 4. MAIN LOSS CLASS (CLIP-EBC Ufficiale Unificata)
# ==============================================================================
class CLIPEBCLoss(nn.Module):
    def __init__(self, bins, input_size, reduction, weight_ot=0.1, weight_tv=0.01, weight_count=1.0, label_smoothing=0.0, **kwargs):
        super().__init__()
        self.bins = bins
        self.reduction = reduction
        self.weight_ot = weight_ot
        self.weight_tv = weight_tv
        self.weight_count = weight_count
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='none')
        self.ot_loss = OTLoss(input_size, reduction, norm_cood=True)
        self.tv_loss = nn.L1Loss(reduction='none')
        self.count_l1 = nn.L1Loss(reduction='mean')

    def _bin_count(self, density_map):
        """
        Discretizza la densità nei bin per la CrossEntropy.
        FIX: Assicura che la density_map sia 3D [B, H, W] per matchare la class_map 3D.
        """
        # Se è 4D [B, 1, H, W], la rendiamo 3D [B, H, W]
        if density_map.dim() == 4:
            density_map = density_map.squeeze(1)
            
        B, H, W = density_map.shape
        class_map = torch.zeros((B, H, W), device=density_map.device, dtype=torch.long)
        
        for idx, (low, high) in enumerate(self.bins):
            high = float('inf') if high > 1000 else high
            # Ora mask è 3D, class_map è 3D -> Nessun errore
            mask = (density_map >= low) & (density_map <= high)
            class_map[mask] = idx
            
        return class_map

    def forward(self, pred_class, pred_density, target_density, target_points):
        if target_density.shape[-2:] != pred_density.shape[-2:]:
            target_density_reduced = _reshape_density(target_density, self.reduction)
        else:
            target_density_reduced = target_density

        target_class = self._bin_count(target_density_reduced)
        ce_loss_val = self.ce_loss(pred_class, target_class).mean()
        
        pred_count = pred_density.view(pred_density.shape[0], -1).sum(dim=1)
        target_count = torch.tensor([len(p) for p in target_points], device=pred_density.device, dtype=torch.float32)
        
        normed_pred = pred_density / (pred_count.view(-1, 1, 1, 1) + EPS)
        normed_target = target_density_reduced / (target_count.view(-1, 1, 1, 1) + EPS)
        
        ot_loss_val, _, _ = self.ot_loss(pred_density, normed_pred, target_points)
        tv_loss_val = (self.tv_loss(normed_pred, normed_target).sum(dim=(1, 2, 3)) * target_count).mean()
        count_mae = self.count_l1(pred_count, target_count)

        total_loss = ce_loss_val + (self.weight_ot * ot_loss_val) + (self.weight_tv * tv_loss_val) + (self.weight_count * count_mae)

        loss_dict = {
            'loss': total_loss.item(),
            'ce_loss': ce_loss_val.item(),
            'ot_loss': ot_loss_val.item(),
            'count_loss': count_mae.item(),
        }
        
        return total_loss, loss_dict