# losses/clip_ebc_loss.py
# -*- coding: utf-8 -*-
"""
CLIP-EBC Loss All-in-One.

Include:
1. Sinkhorn Algorithm (per Trasporto Ottimale)
2. OTLoss (Calcolo costo trasporto)
3. CLIPEBCLoss (Wrapper che unisce CrossEntropy + OT + TV)

Logica estratta e unificata dal repo ufficiale CLIP-EBC.
"""
from torch import nn, Tensor
from typing import List, Any, Tuple, Dict
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union

# Costanti per stabilità numerica
EPS = 1e-8
M_EPS = 1e-16

# ==============================================================================
# 1. ALGORITMO SINKHORN (da bregman_pytorch.py)
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
    """
    Risolve il problema del trasporto ottimale con regolarizzazione entropica
    usando l'algoritmo di Sinkhorn-Knopp.
    """
    device = a.device
    # Inizializzazione vettori u e v
    u = (torch.ones_like(a) / a.size(0)).to(device)
    v = (torch.ones_like(b) / b.size(0)).to(device)
    
    # Kernel K = exp(-C/reg)
    K = torch.exp(-C / reg)
    
    # Tensore per moltiplicazioni intermedie
    KTu = torch.zeros_like(v)
    Kv = torch.zeros_like(u)

    dict_log = {"err": []} if log else None

    for i in range(maxIter):
        # check relative error
        u0 = u
        
        # v = b / (K^T * u)
        KTu = torch.matmul(u.view(1, -1), K).view(-1)
        v = torch.div(b, KTu + M_EPS)
        
        # u = a / (K * v)
        Kv = torch.matmul(K, v.view(-1, 1)).view(-1)
        u = torch.div(a, Kv + M_EPS)
        
        if log and i % eval_freq == 0:
            # Calcolo errore marginale per convergenza
            # b_hat = (K^T * u) * v
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
# 2. UTILS (da utils.py)
# ==============================================================================
def _reshape_density(density: torch.Tensor, reduction: int) -> torch.Tensor:
    """
    Ridimensiona la mappa di densità sommando i valori nei blocchi (pooling).
    Serve per allineare la Ground Truth alla risoluzione di output del modello.
    """
    assert len(density.shape) == 4
    B, C, H, W = density.shape
    
    assert H % reduction == 0 and W % reduction == 0, \
        f"Image size ({H}x{W}) non divisibile per reduction {reduction}"
    
    # Reshape e Sum per ottenere la conta nel blocco
    return density.reshape(
        B, C, 
        H // reduction, reduction, 
        W // reduction, reduction
    ).sum(dim=(-1, -3))


# --- DACELoss (Invariata) ---
class DACELoss(nn.Module):
    def __init__(self, bins, bin_centers, weight_count=1.0, label_smoothing=0.0, block_size=16):
        super().__init__()
        self.bins = [tuple(b) for b in bins]
        self.num_bins = len(bins)
        self.weight_count = weight_count
        self.block_size = block_size
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction="mean")
        self.register_buffer("bin_centers", torch.tensor(bin_centers, dtype=torch.float32))
    
    def _get_bin_labels(self, block_counts):
        if block_counts.dim() == 4: block_counts = block_counts.squeeze(1)
        labels = torch.zeros_like(block_counts, dtype=torch.long)
        for idx, (low, high) in enumerate(self.bins):
            high_val = float('inf') if high > 9000 else high
            mask = (block_counts >= low) & (block_counts <= high_val)
            labels[mask] = idx
        return labels
    
    def _density_to_blocks(self, density, target_size):
        B, C, H, W = density.shape
        tH, tW = target_size
        if H == tH and W == tW: return density
        scale_h, scale_w = H // tH, W // tW
        if scale_h > 0 and scale_w > 0 and H % tH == 0 and W % tW == 0:
            return F.avg_pool2d(density, kernel_size=(scale_h, scale_w)) * (scale_h * scale_w)
        else:
            return F.adaptive_avg_pool2d(density, (tH, tW)) * (H * W) / (tH * tW)
    
    def forward(self, outputs, gt_density, points=None):
        logits = outputs['ebc_logits']
        B, C, H, W = logits.shape
        gt_blocks = self._density_to_blocks(gt_density, (H, W))
        target_labels = self._get_bin_labels(gt_blocks)
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
        labels_flat = target_labels.reshape(-1)
        ce_loss = self.ce_loss(logits_flat, labels_flat)
        bin_probs = outputs.get('bin_probs', F.softmax(logits, dim=1))
        centers = self.bin_centers.view(1, -1, 1, 1).to(logits.device)
        pred_density = (bin_probs * centers).sum(dim=1, keepdim=True)
        pred_count = pred_density.sum(dim=(1, 2, 3))
        if points is not None:
            gt_count = torch.tensor([len(p) for p in points], dtype=torch.float32, device=logits.device)
        else:
            gt_count = gt_blocks.sum(dim=(1, 2, 3))
        count_loss = F.l1_loss(pred_count, gt_count)
        total_loss = ce_loss + self.weight_count * count_loss
        with torch.no_grad():
            mae = torch.abs(pred_count - gt_count).mean()
        return total_loss, {'total_loss': total_loss.item(), 'mae': mae.item()}


# ==============================================================================
# 3. OT LOSS MODULE (da dm_loss.py)
# ==============================================================================

class OTLoss(nn.Module):
    def __init__(
        self,
        input_size: int,
        reduction: int,
        norm_cood: bool,
        num_of_iter_in_ot: int = 100,
        reg: float = 10.0
    ) -> None:
        super().__init__()
        assert input_size % reduction == 0

        self.input_size = input_size
        self.reduction = reduction
        self.norm_cood = norm_cood
        self.num_of_iter_in_ot = num_of_iter_in_ot
        self.reg = reg

        # coordinate is same to image space, set to constant since crop size is same
        self.cood = torch.arange(0, input_size, step=reduction, dtype=torch.float32) + reduction / 2
        self.density_size = self.cood.size(0)
        self.cood.unsqueeze_(0) # [1, #cood]
        self.cood = self.cood / input_size * 2 - 1 if self.norm_cood else self.cood
        self.output_size = self.cood.size(1)

    def forward(self, pred_density: Tensor, normed_pred_density: Tensor, target_points: List[Tensor]) -> Tuple[Tensor, float, Tensor]:
        batch_size = normed_pred_density.size(0)
        assert len(target_points) == batch_size, f"Expected target_points to have length {batch_size}, but got {len(target_points)}"
        assert self.output_size == normed_pred_density.size(2)
        device = pred_density.device

        loss = torch.zeros([1]).to(device)
        ot_obj_values = torch.zeros([1]).to(device)
        wd = 0 # Wasserstein distance
        cood = self.cood.to(device)
        for idx, points in enumerate(target_points):
            if len(points) > 0:
                # compute l2 square distance, it should be source target distance. [#gt, #cood * #cood]
                points = points / self.input_size * 2 - 1 if self.norm_cood else points
                x = points[:, 0].unsqueeze_(1)  # [#gt, 1]
                y = points[:, 1].unsqueeze_(1)
                x_dist = -2 * torch.matmul(x, cood) + x * x + cood * cood # [#gt, #cood]
                y_dist = -2 * torch.matmul(y, cood) + y * y + cood * cood
                y_dist.unsqueeze_(2)
                x_dist.unsqueeze_(1)
                dist = y_dist + x_dist
                dist = dist.view((dist.size(0), -1)) # size of [#gt, #cood * #cood]

                source_prob = normed_pred_density[idx][0].view([-1]).detach()
                target_prob = (torch.ones([len(points)]) / len(points)).to(device)
                # use sinkhorn to solve OT, compute optimal beta.
                P, log = sinkhorn(target_prob, source_prob, dist, self.reg, maxIter=self.num_of_iter_in_ot, log=True)
                beta = log["beta"] # size is the same as source_prob: [#cood * #cood]
                ot_obj_values += torch.sum(normed_pred_density[idx] * beta.view([1, self.output_size, self.output_size]))
                # compute the gradient of OT loss to predicted density (pred_density).
                # im_grad = beta / source_count - < beta, source_density> / (source_count)^2
                source_density = pred_density[idx][0].view([-1]).detach()
                source_count = source_density.sum()
                gradient_1 = (source_count) / (source_count * source_count+ EPS) * beta # size of [#cood * #cood]
                gradient_2 = (source_density * beta).sum() / (source_count * source_count + EPS) # size of 1
                gradient = gradient_1 - gradient_2
                gradient = gradient.detach().view([1, self.output_size, self.output_size])
                # Define loss = <im_grad, predicted density>. The gradient of loss w.r.t predicted density is im_grad.
                loss += torch.sum(pred_density[idx] * gradient)
                wd += torch.sum(dist * P).item()

        return loss, wd, ot_obj_values


class DMLoss(nn.Module):
    def __init__(
        self,
        input_size: int,
        reduction: int,
        norm_cood: bool = False,
        weight_ot: float = 0.1,
        weight_tv: float = 0.01,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.ot_loss = OTLoss(input_size, reduction, norm_cood, **kwargs)
        self.tv_loss = nn.L1Loss(reduction="none")
        self.count_loss = nn.L1Loss(reduction="mean")
        self.weight_ot = weight_ot
        self.weight_tv = weight_tv

    def forward(self, pred_density: Tensor, target_density: Tensor, target_points: List[Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        target_density = _reshape_density(target_density, reduction=self.ot_loss.reduction) if target_density.shape[-2:] != pred_density.shape[-2:] else target_density
        assert pred_density.shape == target_density.shape, f"Expected pred_density and target_density to have the same shape, got {pred_density.shape} and {target_density.shape}"

        pred_count = pred_density.view(pred_density.shape[0], -1).sum(dim=1)
        normed_pred_density = pred_density / (pred_count.view(-1, 1, 1, 1) + EPS)
        target_count = torch.tensor([len(p) for p in target_points], dtype=torch.float32).to(target_density.device)
        normed_target_density = target_density / (target_count.view(-1, 1, 1, 1) + EPS)

        ot_loss, _, _ = self.ot_loss(pred_density, normed_pred_density, target_points)

        tv_loss = (self.tv_loss(normed_pred_density, normed_target_density).sum(dim=(1, 2, 3)) * target_count).mean()

        count_loss = self.count_loss(pred_count, target_count)

        loss = ot_loss * self.weight_ot + tv_loss * self.weight_tv + count_loss

        loss_info = {
            "loss": loss.detach(),
            "ot_loss": ot_loss.detach(),
            "tv_loss": tv_loss.detach(),
            "count_loss": count_loss.detach(),
        }

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