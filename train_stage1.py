#!/usr/bin/env python3
"""
ZIP-CLIP-EBC: Stage 1 Training con ZIP NLL Loss

Implementazione fedele al paper "ZIP: Scalable Crowd Counting via Zero-Inflated Poisson Modeling".

Approccio:
- π (pi): P(blocco NON strutturalmente vuoto) = P(può contenere persone)
- λ (lambda): Rate Poisson (conteggio atteso se non vuoto)
- Joint training di π e λ con ZIP NLL loss

La ZIP distribution modella:
- Structural zeros: blocchi deterministicamente vuoti (background, corpo, etc.)
- Sampling zeros: blocchi che potrebbero avere persone ma non ne hanno in questa immagine

Loss totale: L = ω·L_CE + L_NLL + L_count

Per Stage 1, focalizziamo su π (classificazione strutturale) ma consideriamo anche λ.
"""

import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

from models.zip_clip_ebc_model import ZIPCLIPEBCModel
from datasets.sha import SHA
from datasets.transforms import build_transforms
from utils.train_utils import AverageMeter, seed_everything


# =============================================================================
# ZIP NLL LOSS - Implementazione fedele al paper
# =============================================================================
class ZIPNLLLoss(nn.Module):
    """
    Zero-Inflated Poisson Negative Log-Likelihood Loss.
    
    Formula dal paper ZIP:
        P(y=0|π,λ) = π + (1-π)·e^{-λ}
        P(y>0|π,λ) = (1-π)·(e^{-λ}·λ^y)/y!
        
    Dove:
        - π: P(structural zero) = P(blocco strutturalmente vuoto)
        - (1-π): P(blocco può contenere persone)  
        - λ: Rate Poisson (conteggio atteso quando non vuoto)
    
    NOTA: Nella nostra implementazione, usiamo la convenzione INVERSA:
        - π = P(blocco OCCUPATO), non P(vuoto)
        - Quindi (1-π) = P(structural zero)
    """
    def __init__(self, eps=1e-8, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, pi, lam, target_counts):
        """
        Args:
            pi: [B, 1, H, W] - P(blocco occupato), range [0, 1]
            lam: [B, 1, H, W] - Poisson rate, range [0, inf)
            target_counts: [B, 1, H, W] - Ground truth counts per block
        
        Returns:
            loss: Scalar
        """
        # Align dimensions
        if pi.shape[-2:] != target_counts.shape[-2:]:
            pi = F.interpolate(pi, size=target_counts.shape[-2:], 
                              mode='bilinear', align_corners=False)
        if lam.shape[-2:] != target_counts.shape[-2:]:
            lam = F.interpolate(lam, size=target_counts.shape[-2:], 
                               mode='bilinear', align_corners=False)
        
        # Numerical stability clamps
        pi = torch.clamp(pi, self.eps, 1.0 - self.eps)
        lam = torch.clamp(lam, self.eps, 1e6)
        y = target_counts
        
        # Masks
        is_zero = (y < 0.5).float()  # y == 0
        is_pos = 1.0 - is_zero       # y > 0
        
        # NOTA: Nella nostra convenzione π = P(occupied)
        # Quindi P(structural zero) = 1 - π
        
        # log P(y=0) = log( (1-π) + π·e^{-λ} )
        # dove (1-π) è prob structural zero, π·e^{-λ} è sampling zero
        p_structural_zero = 1.0 - pi
        p_sampling_zero = pi * torch.exp(-lam)
        log_p0 = torch.log(p_structural_zero + p_sampling_zero + self.eps)
        
        # log P(y>0) = log(π) - λ + y·log(λ) - log(y!)
        log_pi = torch.log(pi + self.eps)
        log_fact = torch.lgamma(y + 1.0)  # log(y!)
        log_py = log_pi - lam + y * torch.log(lam + self.eps) - log_fact
        
        # NLL = -log P(y)
        nll = -(is_zero * log_p0 + is_pos * log_py)
        
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        return nll


# =============================================================================
# CROSS ENTROPY LOSS per classificazione (solo blocchi positivi)
# =============================================================================
class PositiveOnlyCELoss(nn.Module):
    """
    Cross-entropy loss calcolata solo sui blocchi con count > 0.
    Questo supervisiona λ branch solo dove ha senso.
    """
    def __init__(self, num_bins=8, bin_edges=None, reduction='mean'):
        super().__init__()
        self.num_bins = num_bins
        if bin_edges is None:
            # Default bin edges
            bin_edges = [0, 1, 2, 3, 4, 5, 6, 10, 9999]
        self.register_buffer('bin_edges', torch.tensor(bin_edges, dtype=torch.float32))
        self.reduction = reduction
    
    def count_to_bin(self, counts):
        """Convert continuous counts to bin indices."""
        bins = torch.zeros_like(counts, dtype=torch.long)
        for i in range(len(self.bin_edges) - 1):
            mask = (counts >= self.bin_edges[i]) & (counts < self.bin_edges[i+1])
            bins[mask] = i
        return bins
    
    def forward(self, logits, target_counts):
        """
        Args:
            logits: [B, num_bins, H, W] - class logits
            target_counts: [B, 1, H, W] - ground truth counts
        """
        # Find positive blocks (count > 0)
        mask_pos = (target_counts > 0.5).squeeze(1)  # [B, H, W]
        
        if mask_pos.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        
        # Convert counts to bin indices
        target_bins = self.count_to_bin(target_counts.squeeze(1))  # [B, H, W]
        
        # Select only positive blocks
        logits_pos = logits.permute(0, 2, 3, 1)[mask_pos]  # [N, num_bins]
        targets_pos = target_bins[mask_pos]  # [N]
        
        loss = F.cross_entropy(logits_pos, targets_pos, reduction=self.reduction)
        return loss


# =============================================================================
# COMBINED STAGE1 LOSS (ZIP-style)
# =============================================================================
class ZIPStage1Loss(nn.Module):
    """
    Loss combinata per Stage 1 in stile ZIP:
    
    L_total = ω·L_CE + L_NLL + β·L_count
    
    dove:
    - L_CE: Cross-entropy sui blocchi positivi (supervisiona λ)
    - L_NLL: ZIP NLL loss (supervisiona π e λ jointly)
    - L_count: MAE sul conteggio totale
    """
    def __init__(
        self,
        ce_weight=0.75,      # ω nel paper
        count_weight=0.1,    # β 
        num_bins=8,
        bin_edges=None
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.count_weight = count_weight
        
        self.zip_nll = ZIPNLLLoss()
        self.ce_loss = PositiveOnlyCELoss(num_bins=num_bins, bin_edges=bin_edges)
    
    def forward(self, pi, lam, lambda_logits, target_counts, gt_total_count):
        """
        Args:
            pi: [B, 1, H, W] - P(occupied)
            lam: [B, 1, H, W] - Poisson rate
            lambda_logits: [B, num_bins, H, W] - class logits for λ
            target_counts: [B, 1, H, W] - GT counts per block
            gt_total_count: [B] - GT total count per image
        """
        # 1. ZIP NLL Loss
        loss_nll = self.zip_nll(pi, lam, target_counts)
        
        # 2. Cross-Entropy Loss (solo su positivi)
        loss_ce = torch.tensor(0.0, device=pi.device)
        if lambda_logits is not None:
            loss_ce = self.ce_loss(lambda_logits, target_counts)
        
        # 3. Count Loss (MAE)
        # Predicted density = (1 - P(structural_zero)) * λ = π * λ
        pred_density = pi * lam
        pred_count = pred_density.sum(dim=[1, 2, 3])  # [B]
        loss_count = F.l1_loss(pred_count, gt_total_count)
        
        # Total
        total_loss = self.ce_weight * loss_ce + loss_nll + self.count_weight * loss_count
        
        return total_loss, {
            'nll': loss_nll.item(),
            'ce': loss_ce.item() if isinstance(loss_ce, torch.Tensor) else loss_ce,
            'count': loss_count.item(),
            'pred_count_mean': pred_count.mean().item()
        }


# =============================================================================
# SIMPLIFIED APPROACH: Binary + Focal for Stage 1
# =============================================================================
class SimplifiedZIPStage1Loss(nn.Module):
    """
    Versione semplificata per Stage 1:
    Usa solo π (binary classification) con supervisione ZIP-aware.
    
    L = L_focal + α·L_sparsity_kl
    
    dove L_sparsity_kl è una KL divergence tra predicted e expected sparsity.
    """
    def __init__(
        self,
        focal_alpha=0.25,
        focal_gamma=2.0,
        sparsity_kl_weight=0.5,
        expected_empty_ratio=0.84  # ~84% vuoti tipico per SHA
    ):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.sparsity_kl_weight = sparsity_kl_weight
        self.expected_empty_ratio = expected_empty_ratio
    
    def forward(self, pi_logits, target, eps=1e-8):
        """
        Args:
            pi_logits: [B, 1, H, W] - raw logits
            target: [B, 1, H, W] - binary target (1=occupied, 0=empty)
        """
        probs = torch.sigmoid(pi_logits)
        
        # 1. Focal Loss
        p_t = probs * target + (1 - probs) * (1 - target)
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t).pow(self.focal_gamma)
        bce = F.binary_cross_entropy_with_logits(pi_logits, target, reduction='none')
        loss_focal = (focal_weight * bce).mean()
        
        # 2. Sparsity KL Divergence
        # Penalizza se la distribuzione predicted è troppo diversa dalla expected
        pred_empty_ratio = (1 - probs).mean()
        
        # KL(expected || predicted)
        p = self.expected_empty_ratio
        q = pred_empty_ratio.clamp(eps, 1 - eps)
        kl = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
        loss_kl = kl
        
        total = loss_focal + self.sparsity_kl_weight * loss_kl
        
        return total, {
            'focal': loss_focal.item(),
            'kl': loss_kl.item(),
            'pred_empty': pred_empty_ratio.item()
        }


# =============================================================================
# UTILITIES
# =============================================================================
def crowd_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'density': torch.stack([item['density'] for item in batch]),
        'points': [item['points'] for item in batch],
        'img_path': [item['img_path'] for item in batch]
    }


def save_checkpoint(state, is_best, save_dir, filename='last_model.pth'):
    last_path = os.path.join(save_dir, filename)
    torch.save(state, last_path)
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        shutil.copyfile(last_path, best_path)


def validate_with_thresholds(loader, model, device, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """
    Validate and find best threshold.
    
    FIXED: Accumula TP/FP/FN/TN batch-by-batch invece di concatenare tensori
    (che fallisce con immagini di dimensioni diverse).
    """
    model.eval()
    
    # Inizializza contatori per ogni threshold
    counters = {thresh: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0} for thresh in thresholds}
    
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            images = batch['image'].to(device)
            gt_density = batch['density'].to(device)
            
            outputs = model.forward_stage1(images)
            probs = torch.sigmoid(outputs['pi_logits'])
            
            h_out, w_out = probs.shape[2:]
            scale_h, scale_w = images.shape[2]/h_out, images.shape[3]/w_out
            ds_count = F.adaptive_avg_pool2d(gt_density, (h_out, w_out)) * (scale_h * scale_w)
            target = (ds_count >= 0.5).float()
            
            # Accumula metriche per ogni threshold
            for thresh in thresholds:
                preds = (probs > thresh).float()
                counters[thresh]['tp'] += (preds * target).sum().item()
                counters[thresh]['fp'] += (preds * (1 - target)).sum().item()
                counters[thresh]['fn'] += ((1 - preds) * target).sum().item()
                counters[thresh]['tn'] += ((1 - preds) * (1 - target)).sum().item()
    
    best_f1 = 0
    best_metrics = None
    best_thresh = 0.5
    
    for thresh in thresholds:
        c = counters[thresh]
        tp, fp, fn, tn = c['tp'], c['fp'], c['fn'], c['tn']
        
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = {
                'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc,
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'threshold': thresh
            }
    
    return best_metrics, best_thresh


# =============================================================================
# MAIN
# =============================================================================
def train_stage1_zip():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_sha.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=32)
    
    # Loss options
    parser.add_argument('--loss_mode', type=str, default='simplified',
                        choices=['full_zip', 'simplified'],
                        help='full_zip uses ZIP NLL, simplified uses Focal+KL')
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--sparsity_kl_weight', type=float, default=0.5)
    parser.add_argument('--expected_empty_ratio', type=float, default=0.84)
    
    # Scheduler
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--lr_head', type=float, default=5e-4)
    parser.add_argument('--T_0', type=int, default=10)
    parser.add_argument('--T_mult', type=int, default=2)
    
    parser.add_argument('--val_interval', type=int, default=10)
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    seed_everything(config.get('SEED', 42))
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    dataset_name = config.get('DATASET', 'sha')
    save_dir = os.path.join('./checkpoints', dataset_name, f'stage1')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"🚀 Stage 1 ZIP Training ({args.loss_mode})")
    print(f"   Device: {device}")
    print(f"   Save: {save_dir}")
    
    # Data
    data_cfg = config['DATA']
    train_dataset = SHA(root=data_cfg['ROOT'], split='train',
                        transforms=build_transforms(data_cfg, True))
    val_dataset = SHA(root=data_cfg['ROOT'], split='val',
                      transforms=build_transforms(data_cfg, False))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, collate_fn=crowd_collate, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=4, collate_fn=crowd_collate)
    
    # Model
    model = ZIPCLIPEBCModel(config).to(device)
    for p in model.clip_ebc_head.parameters():
        p.requires_grad = False
    for p in model.pi_head.parameters():
        p.requires_grad = True
    for p in model.backbone.parameters():
        p.requires_grad = True
    
    # Optimizer
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': args.lr_backbone},
        {'params': model.pi_head.parameters(), 'lr': args.lr_head}
    ], weight_decay=1e-4)
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult)
    
    # Loss
    if args.loss_mode == 'simplified':
        criterion = SimplifiedZIPStage1Loss(
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            sparsity_kl_weight=args.sparsity_kl_weight,
            expected_empty_ratio=args.expected_empty_ratio
        )
        print(f"   Loss: Focal + KL (α={args.focal_alpha}, γ={args.focal_gamma})")
    else:
        criterion = ZIPNLLLoss()
        print(f"   Loss: Full ZIP NLL")
    
    best_f1 = 0.0
    best_thresh = 0.5
    
    for epoch in range(args.epochs):
        model.train()
        losses = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for batch in pbar:
            if batch is None:
                continue
            
            images = batch['image'].to(device)
            gt_density = batch['density'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model.forward_stage1(images)
            pi_logits = outputs['pi_logits']
            pi = torch.sigmoid(pi_logits)
            
            h_out, w_out = pi_logits.shape[2:]
            scale_h, scale_w = images.shape[2]/h_out, images.shape[3]/w_out
            ds_count = F.adaptive_avg_pool2d(gt_density, (h_out, w_out)) * (scale_h * scale_w)
            target = (ds_count >= 0.5).float()
            
            if args.loss_mode == 'simplified':
                loss, loss_dict = criterion(pi_logits, target)
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 
                                  'empty': f"{loss_dict['pred_empty']*100:.1f}%"})
            else:
                # For full ZIP, use pi and lambda
                lam = outputs.get('lambda_', torch.ones_like(pi))
                loss = criterion(pi, lam, ds_count)
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            losses.update(loss.item())
        
        scheduler.step()
        
        # Validation
        if (epoch + 1) % args.val_interval == 0 or (epoch + 1) == args.epochs:
            metrics, opt_thresh = validate_with_thresholds(val_loader, model, device)
            
            print(f"\n📊 Epoch {epoch+1}: Loss={losses.avg:.4f}")
            print(f"   Thresh={opt_thresh:.2f} | Prec={metrics['precision']:.2%} | "
                  f"Rec={metrics['recall']:.2%} | F1={metrics['f1']:.2%}")
            print(f"   TP={int(metrics['tp']):,} | FP={int(metrics['fp']):,} | "
                  f"FN={int(metrics['fn']):,} | TN={int(metrics['tn']):,}")
            
            is_best = metrics['f1'] > best_f1
            if is_best:
                best_f1 = metrics['f1']
                best_thresh = opt_thresh
                print(f"   🌟 New Best!")
            
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_f1': best_f1,
                'best_threshold': best_thresh,
                'optimizer': optimizer.state_dict()
            }, is_best, save_dir)
    
    print(f"\n✅ Done! Best F1: {best_f1:.2%} @ threshold {best_thresh:.2f}")


if __name__ == '__main__':
    train_stage1_zip()