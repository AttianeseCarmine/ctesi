#!/usr/bin/env python3
"""
============================================================
CLIP-EBC Stage 2 Training Script
============================================================
Replica il training del repository ufficiale CLIP-EBC.

Usa:
- CLIPEBCModel (CLIP ResNet50 visual encoder)
- DACELoss (Cross-Entropy + Count Loss)

Usage:
    python train_stage2.py --config configs/config_sha.yaml
============================================================
"""

import os
import sys
import yaml
import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import numpy as np

# Import model
from models.clip_ebc_model import CLIPEBCModel

# Import dataset
from datasets.sha import SHA
from datasets.transforms import build_transforms


# ============================================================
# DACE Loss (come nel repo ufficiale)
# ============================================================

class DACELoss(nn.Module):
    """
    DACE Loss = Cross-Entropy + λ * Count Loss
    
    Discretization-Aware Cross-Entropy Loss dal paper CLIP-EBC.
    
    Args:
        bins: Lista di tuple (min, max) per ogni bin
        bin_centers: Anchor points per ogni bin
        weight_count: λ peso per count loss
        label_smoothing: Label smoothing per CE
    """
    
    def __init__(
        self,
        bins,
        bin_centers,
        weight_count: float = 1.0,
        label_smoothing: float = 0.0,
        block_size: int = 16,
    ):
        super().__init__()
        
        self.bins = [tuple(b) for b in bins]
        self.num_bins = len(bins)
        self.weight_count = weight_count
        self.block_size = block_size
        
        # Cross-Entropy
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction="mean"
        )
        
        # Anchor points
        self.register_buffer(
            "bin_centers",
            torch.tensor(bin_centers, dtype=torch.float32)
        )
    
    def _get_bin_labels(self, block_counts: torch.Tensor) -> torch.Tensor:
        """Converte conteggi per blocco in indici di bin."""
        if block_counts.dim() == 4:
            block_counts = block_counts.squeeze(1)
        
        labels = torch.zeros_like(block_counts, dtype=torch.long)
        
        for idx, (low, high) in enumerate(self.bins):
            high_val = float('inf') if high > 9000 else high
            mask = (block_counts >= low) & (block_counts <= high_val)
            labels[mask] = idx
        
        return labels
    
    def _density_to_blocks(self, density: torch.Tensor, target_size) -> torch.Tensor:
        """Riduce density map a conteggi per blocco."""
        B, C, H, W = density.shape
        tH, tW = target_size
        
        if H == tH and W == tW:
            return density
        
        scale_h = H // tH
        scale_w = W // tW
        
        if scale_h > 0 and scale_w > 0 and H % tH == 0 and W % tW == 0:
            return F.avg_pool2d(density, kernel_size=(scale_h, scale_w)) * (scale_h * scale_w)
        else:
            return F.adaptive_avg_pool2d(density, (tH, tW)) * (H * W) / (tH * tW)
    
    def forward(self, outputs: dict, gt_density: torch.Tensor, points=None):
        """
        Args:
            outputs: Dict dal modello con 'ebc_logits', 'ebc_density', etc.
            gt_density: [B, 1, H_full, W_full] GT density map
            points: List[Tensor] punti GT per count preciso
            
        Returns:
            total_loss, loss_dict
        """
        logits = outputs['ebc_logits']  # [B, num_bins, H, W]
        B, C, H, W = logits.shape
        device = logits.device
        
        # Riduci density a dimensione output
        gt_blocks = self._density_to_blocks(gt_density, (H, W))
        
        # === Cross-Entropy Loss ===
        target_labels = self._get_bin_labels(gt_blocks)  # [B, H, W]
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        labels_flat = target_labels.reshape(-1)  # [B*H*W]
        
        ce_loss = self.ce_loss(logits_flat, labels_flat)
        
        # === Count Loss (MAE) ===
        bin_probs = outputs.get('bin_probs', F.softmax(logits, dim=1))
        centers = self.bin_centers.view(1, -1, 1, 1).to(device)
        pred_density = (bin_probs * centers).sum(dim=1, keepdim=True)
        pred_count = pred_density.sum(dim=(1, 2, 3))
        
        if points is not None:
            gt_count = torch.tensor([len(p) for p in points], dtype=torch.float32, device=device)
        else:
            gt_count = gt_blocks.sum(dim=(1, 2, 3))
        
        count_loss = F.l1_loss(pred_count, gt_count)
        
        # === Total Loss ===
        total_loss = ce_loss + self.weight_count * count_loss
        
        # === Metrics ===
        with torch.no_grad():
            pred_labels = logits_flat.argmax(dim=1)
            accuracy = (pred_labels == labels_flat).float().mean()
            mae = torch.abs(pred_count - gt_count).mean()
        
        loss_dict = {
            'ce_loss': ce_loss.item(),
            'count_loss': count_loss.item(),
            'total_loss': total_loss.item(),
            'accuracy': accuracy.item(),
            'mae': mae.item(),
            'pred_count_mean': pred_count.mean().item(),
            'gt_count_mean': gt_count.mean().item(),
        }
        
        return total_loss, loss_dict


# ============================================================
# Collate Function
# ============================================================

def crowd_collate(batch):
    """Custom collate per gestire batch con elementi None."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'density': torch.stack([item['density'] for item in batch]),
        'points': [item['points'] for item in batch],
        'img_path': [item['img_path'] for item in batch]
    }


# ============================================================
# Training Functions
# ============================================================

def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, config):
    """Training di un'epoca."""
    model.train()
    
    total_loss = 0
    total_ce = 0
    total_count = 0
    total_acc = 0
    total_mae = 0
    num_batches = 0
    
    train_cfg = config.get('TRAIN_STAGE2', {})
    use_amp = train_cfg.get('AMP', False)
    clip_grad = train_cfg.get('CLIP_GRAD_NORM', None)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        if batch is None:
            continue
        
        images = batch['image'].to(device)
        density = batch['density'].to(device)
        points = batch['points']
        
        optimizer.zero_grad()
        
        with autocast('cuda', enabled=use_amp):
            outputs = model(images)
            loss, metrics = criterion(outputs, density, points)
        
        if use_amp:
            scaler.scale(loss).backward()
            if clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
        
        total_loss += metrics['total_loss']
        total_ce += metrics['ce_loss']
        total_count += metrics['count_loss']
        total_acc += metrics['accuracy']
        total_mae += metrics['mae']
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{metrics['total_loss']:.4f}",
            'mae': f"{metrics['mae']:.1f}",
            'acc': f"{metrics['accuracy']*100:.1f}%"
        })
    
    return {
        'loss': total_loss / num_batches,
        'ce_loss': total_ce / num_batches,
        'count_loss': total_count / num_batches,
        'accuracy': total_acc / num_batches,
        'mae': total_mae / num_batches,
    }


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Valutazione sul validation set."""
    model.eval()
    
    mae, mse = 0, 0
    total_gt, total_pred = 0, 0
    num_images = 0
    
    results = []  # Per analisi dettagliata
    
    for batch in tqdm(val_loader, desc="Evaluating"):
        if batch is None:
            continue
        
        images = batch['image'].to(device)
        points = batch['points']
        img_paths = batch['img_path']
        
        outputs = model(images)
        pred_count = outputs['final_count']
        
        for i, pts in enumerate(points):
            gt = len(pts)
            pred = pred_count[i].item()
            
            err = abs(pred - gt)
            mae += err
            mse += (pred - gt) ** 2
            total_gt += gt
            total_pred += pred
            num_images += 1
            
            results.append({
                'img': img_paths[i],
                'gt': gt,
                'pred': pred,
                'error': err,
            })
    
    # Trova worst cases
    results.sort(key=lambda x: x['error'], reverse=True)
    
    return {
        'mae': mae / num_images,
        'rmse': (mse / num_images) ** 0.5,
        'total_gt': total_gt,
        'total_pred': total_pred,
        'num_images': num_images,
        'worst_cases': results[:5],
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Train CLIP-EBC Stage 2')
    parser.add_argument('--config', type=str, default='configs/config_sha.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    # ========================
    # Load Config
    # ========================
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Output directory - sempre checkpoints/sha/stage2
    out_dir = Path('./checkpoints/sha/stage2')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output: {out_dir}")
    
    # Save config
    with open(out_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Seed
    seed = config.get('SEED', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # ========================
    # Build Model
    # ========================
    print("\n" + "="*60)
    print("Building CLIP-EBC Model")
    print("="*60)
    
    model = CLIPEBCModel(config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # ========================
    # Dataset
    # ========================
    data_cfg = config['DATA']
    
    train_transforms = build_transforms(data_cfg, is_train=True)
    val_transforms = build_transforms(data_cfg, is_train=False)
    
    train_dataset = SHA(
        root=data_cfg['ROOT'],
        split='train',
        transforms=train_transforms
    )
    val_dataset = SHA(
        root=data_cfg['ROOT'],
        split='val',
        transforms=val_transforms
    )
    
    train_cfg = config['TRAIN_STAGE2']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['BATCH_SIZE'],
        shuffle=True,
        num_workers=train_cfg['NUM_WORKERS'],
        collate_fn=crowd_collate,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=crowd_collate,
    )
    
    print(f"📂 Train: {len(train_dataset)} images")
    print(f"📂 Val: {len(val_dataset)} images")
    
    # ========================
    # Loss
    # ========================
    loss_cfg = config.get('LOSS_STAGE2', {})
    
    criterion = DACELoss(
        bins=config['BINS'],
        bin_centers=config['BIN_CENTERS'],
        weight_count=loss_cfg.get('WEIGHT_COUNT_LOSS', 1.0),
        label_smoothing=loss_cfg.get('LABEL_SMOOTHING', 0.0),
        block_size=data_cfg.get('ZIP_BLOCK_SIZE', 16),
    ).to(device)
    
    print(f"\n🎯 Loss: DACE (CE + {loss_cfg.get('WEIGHT_COUNT_LOSS', 1.0)} * CountLoss)")
    
    # ========================
    # Optimizer
    # ========================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg['LR'],
        weight_decay=train_cfg.get('WEIGHT_DECAY', 1e-4)
    )
    
    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=train_cfg.get('T_0', 10),
        T_mult=train_cfg.get('T_MULT', 2),
        eta_min=train_cfg.get('ETA_MIN', 1e-8)
    )
    
    # AMP Scaler
    scaler = GradScaler('cuda', enabled=train_cfg.get('AMP', False))
    
    # ========================
    # Resume
    # ========================
    start_epoch = 0
    best_mae = float('inf')
    
    if args.resume:
        print(f"\n📂 Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_mae = ckpt.get('best_mae', float('inf'))
        print(f"   Resumed from epoch {start_epoch}, best MAE: {best_mae:.2f}")
    
    # ========================
    # Training Loop
    # ========================
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    total_epochs = train_cfg['TOTAL_EPOCHS']
    eval_start = train_cfg.get('EVAL_START', 10)
    eval_freq = train_cfg.get('EVAL_FREQ', 5)
    save_freq = train_cfg.get('SAVE_FREQ', 25)
    warmup_epochs = train_cfg.get('WARMUP_EPOCHS', 20)
    warmup_lr = train_cfg.get('WARMUP_LR', 1e-7)
    base_lr = train_cfg['LR']
    
    for epoch in range(start_epoch, total_epochs):
        # Warmup
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            lr = warmup_lr + (base_lr - warmup_lr) * warmup_factor
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, config
        )
        
        # Step scheduler (dopo warmup)
        if epoch >= warmup_epochs:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        print(f"\nEpoch {epoch}/{total_epochs-1} | "
              f"LR: {current_lr:.2e} | "
              f"Loss: {train_metrics['loss']:.4f} | "
              f"CE: {train_metrics['ce_loss']:.4f} | "
              f"Count: {train_metrics['count_loss']:.1f} | "
              f"MAE: {train_metrics['mae']:.1f} | "
              f"Acc: {train_metrics['accuracy']*100:.1f}%")
        
        # ========================
        # Evaluate
        # ========================
        if epoch >= eval_start and (epoch - eval_start) % eval_freq == 0:
            val_metrics = evaluate(model, val_loader, device)
            
            print(f"\n  📊 Validation Results:")
            print(f"     MAE: {val_metrics['mae']:.2f}")
            print(f"     RMSE: {val_metrics['rmse']:.2f}")
            print(f"     Total GT: {val_metrics['total_gt']:.0f}")
            print(f"     Total Pred: {val_metrics['total_pred']:.0f}")
            
            # Worst cases
            print(f"     Worst cases:")
            for wc in val_metrics['worst_cases'][:3]:
                print(f"       {Path(wc['img']).name}: GT={wc['gt']:.0f}, Pred={wc['pred']:.0f}, Err={wc['error']:.0f}")
            
            # Save best
            if val_metrics['mae'] < best_mae:
                best_mae = val_metrics['mae']
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_mae': best_mae,
                    'config': config,
                }, out_dir / 'best_mae.pth')
                print(f"  ✅ New best MAE: {best_mae:.2f}")
        
        # ========================
        # Save Checkpoint
        # ========================
        if (epoch + 1) % save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_mae': best_mae,
                'config': config,
            }, out_dir / f'checkpoint_epoch{epoch}.pth')
            print(f"  💾 Checkpoint saved: epoch {epoch}")
    
    # ========================
    # Final Evaluation
    # ========================
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    
    # Load best model
    best_ckpt_path = out_dir / 'best_mae.pth'
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt['model'])
        print(f"📂 Loaded best model from epoch {best_ckpt['epoch']}")
    
    final_metrics = evaluate(model, val_loader, device)
    
    print(f"\n📊 Final Results:")
    print(f"   MAE: {final_metrics['mae']:.2f}")
    print(f"   RMSE: {final_metrics['rmse']:.2f}")
    print(f"   Total GT: {final_metrics['total_gt']:.0f}")
    print(f"   Total Pred: {final_metrics['total_pred']:.0f}")
    
    print("\n✅ Training completed!")
    print(f"📁 Checkpoints saved to: {out_dir}")
    print(f"🏆 Best MAE: {best_mae:.2f}")


if __name__ == '__main__':
    main()
