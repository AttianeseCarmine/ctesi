#!/usr/bin/env python3
"""
============================================================
CLIP-EBC Stage 2 Training Script - Dynamic Path Version
============================================================
Replica il training del repository ufficiale CLIP-EBC.
Salva automaticamente i checkpoint nella cartella corretta
basandosi sul parametro 'DATASET' del file config.

Usage:
    python train_stage2.py --config configs/config_shb.yaml
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

from models.clip_ebc_model import CLIPEBCModel
from datasets.sha import SHA
from datasets.transforms import build_transforms

# --- LOSS (DACELoss) ---
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
            pred_labels = logits_flat.argmax(dim=1)
            accuracy = (pred_labels == labels_flat).float().mean()
            mae = torch.abs(pred_count - gt_count).mean()
        
        return total_loss, {
            'ce_loss': ce_loss.item(), 'count_loss': count_loss.item(),
            'total_loss': total_loss.item(), 'accuracy': accuracy.item(), 'mae': mae.item()
        }

def crowd_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'density': torch.stack([item['density'] for item in batch]),
        'points': [item['points'] for item in batch],
        'img_path': [item['img_path'] for item in batch]
    }

def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, config):
    model.train()
    metrics_accum = {'loss': 0, 'ce_loss': 0, 'count_loss': 0, 'accuracy': 0, 'mae': 0}
    num_batches = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        if batch is None: continue
        images = batch['image'].to(device)
        density = batch['density'].to(device)
        points = batch['points']
        
        optimizer.zero_grad()
        with autocast('cuda', enabled=config['TRAIN_STAGE2'].get('AMP', False)):
            outputs = model(images)
            loss, metrics = criterion(outputs, density, points)
        
        scaler.scale(loss).backward()
        if config['TRAIN_STAGE2'].get('CLIP_GRAD_NORM'):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['TRAIN_STAGE2']['CLIP_GRAD_NORM'])
        scaler.step(optimizer)
        scaler.update()
        
        for k, v in metrics.items(): metrics_accum[k.replace('total_', '')] += v
        num_batches += 1
        pbar.set_postfix({'loss': f"{metrics['total_loss']:.4f}", 'mae': f"{metrics['mae']:.1f}"})
    
    return {k: v / num_batches for k, v in metrics_accum.items()}

# --- VALUTAZIONE STANDARD (NO SLIDING WINDOW) ---
@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    mae, mse = 0, 0
    total_gt, total_pred = 0, 0
    num_images = 0
    results = []
    
    for batch in tqdm(val_loader, desc="Evaluating"):
        if batch is None: continue
        images = batch['image'].to(device)
        points = batch['points']
        img_paths = batch['img_path']
        
        outputs = model(images)
        pred_counts = outputs['final_count']
        
        for i, pts in enumerate(points):
            gt = len(pts)
            pred = pred_counts[i].item()
            err = abs(pred - gt)
            mae += err
            mse += (pred - gt) ** 2
            total_gt += gt
            total_pred += pred
            num_images += 1
            results.append({'img': img_paths[i], 'gt': gt, 'pred': pred, 'error': err})
            
    results.sort(key=lambda x: x['error'], reverse=True)
    return {
        'mae': mae / num_images, 'rmse': (mse / num_images) ** 0.5,
        'total_gt': total_gt, 'total_pred': total_pred, 'worst_cases': results[:5]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_sha.yaml')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    device = torch.device(f'cuda:{args.gpu}')
    
    # --- MODIFICA DINAMICA DEL PERCORSO ---
    dataset_name = config.get('DATASET', 'sha')
    base_out_dir = config.get('EXP', {}).get('OUT_DIR', './checkpoints')
    out_dir = Path(base_out_dir) / dataset_name / 'stage2'
    
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Checkpoints will be saved to: {out_dir}")
    # -------------------------------------
    
    model = CLIPEBCModel(config).to(device)
    
    # Dataset
    train_ds = SHA(config['DATA']['ROOT'], 'train', build_transforms(config['DATA'], True))
    val_ds = SHA(config['DATA']['ROOT'], 'val', build_transforms(config['DATA'], False))
    
    train_loader = DataLoader(train_ds, batch_size=config['TRAIN_STAGE2']['BATCH_SIZE'], shuffle=True, 
                              num_workers=config['TRAIN_STAGE2']['NUM_WORKERS'], collate_fn=crowd_collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=crowd_collate)
    
    loss_cfg = config.get('LOSS_STAGE2', {})
    criterion = DACELoss(config['BINS'], config['BIN_CENTERS'], weight_count=loss_cfg.get('WEIGHT_COUNT_LOSS', 1.0), label_smoothing=loss_cfg.get('LABEL_SMOOTHING', 0.0), block_size=config['DATA']['ZIP_BLOCK_SIZE']).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),lr=config['TRAIN_STAGE2']['LR'],weight_decay=config['TRAIN_STAGE2']['WEIGHT_DECAY'])

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler('cuda', enabled=config['TRAIN_STAGE2'].get('AMP', True))
    
    best_mae = float('inf')
    start_epoch = 0
    
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_mae = ckpt.get('best_model', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best MAE: {best_mae}")

    print("🚀 Starting Training")
    for epoch in range(start_epoch, config['TRAIN_STAGE2']['TOTAL_EPOCHS']):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, config)
        scheduler.step()
        
        print(f"Ep {epoch} | Loss: {train_metrics['loss']:.4f} | MAE: {train_metrics['mae']:.2f}")
        
        if epoch % config['TRAIN_STAGE2']['EVAL_FREQ'] == 0:
            val_metrics = evaluate(model, val_loader, device)
            print(f"📊 Val MAE: {val_metrics['mae']:.2f} | RMSE: {val_metrics['rmse']:.2f} | Pred: {val_metrics['total_pred']:.0f} (GT: {val_metrics['total_gt']:.0f})")
            
            if val_metrics['mae'] < best_mae:
                best_mae = val_metrics['mae']
                torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_model': best_mae}, out_dir / 'best_model.pth')
                print(f"🌟 New Best Model Saved!")
                
        torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_model': best_mae}, out_dir / 'last_model.pth')

if __name__ == '__main__':
    main()