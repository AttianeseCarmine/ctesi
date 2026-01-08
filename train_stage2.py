#!/usr/bin/env python3
import os
import yaml
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from models.clip_ebc_model import CLIPEBCModel
from datasets.sha import SHA
from datasets.transforms import build_transforms

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

def crowd_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'density': torch.stack([item['density'] for item in batch]),
        'points': [item['points'] for item in batch],
        'img_path': [item['img_path'] for item in batch]
    }

@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    mae = 0
    num_images = 0
    for batch in tqdm(val_loader, desc="Evaluating"):
        if batch is None: continue
        images = batch['image'].to(device)
        points = batch['points']
        outputs = model(images)
        pred_counts = outputs['final_count']
        for i, pts in enumerate(points):
            mae += abs(pred_counts[i].item() - len(pts))
            num_images += 1
    return {'mae': mae / num_images if num_images > 0 else 0}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    device = torch.device(f'cuda:{args.gpu}')
    
    # --- PATH DINAMICI ---
    dataset_name = config.get('DATASET', 'unknown')
    base_out_dir = config.get('EXP', {}).get('OUT_DIR', './checkpoints')
    out_dir = Path(base_out_dir) / dataset_name / 'stage2'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Stage 2 Training | Dataset: {dataset_name}")
    print(f"📂 Saving to: {out_dir}")
    
    model = CLIPEBCModel(config).to(device)
    
    # =========================================================================
    # CARICAMENTO AUTOMATICO STAGE 1 (QUESTO MANCAVA NELLA TUA VERSIONE!)
    # =========================================================================
    if config['TRAIN_STAGE2'].get('PRETRAINED_STAGE1', False):
        path_cfg = config['TRAIN_STAGE2'].get('STAGE1_PATH', 'auto')
        
        if path_cfg == 'auto':
            stage1_ckpt = Path(base_out_dir) / dataset_name / 'stage1' / 'best_model.pth'
        else:
            stage1_ckpt = Path(path_cfg)
            
        if stage1_ckpt.exists():
            print(f"🔄 Loading Stage 1 from: {stage1_ckpt}")
            ckpt = torch.load(stage1_ckpt, map_location=device)
            state_dict = ckpt if 'model' not in ckpt else ckpt['model']
            # strict=False è essenziale perché Stage 2 ha più teste di Stage 1
            model.load_state_dict(state_dict, strict=False)
            print("✅ Weights loaded successfully.")
        else:
            print(f"⚠️ Warning: Stage 1 checkpoint not found at {stage1_ckpt}")
            print("   Training will start from random weights (Not recommended for ZIP-CLIP).")
    # =========================================================================
    
    # Dataset e Training (Standard)
    train_ds = SHA(config['DATA']['ROOT'], 'train', build_transforms(config['DATA'], True))
    val_ds = SHA(config['DATA']['ROOT'], 'val', build_transforms(config['DATA'], False))
    
    train_loader = DataLoader(train_ds, batch_size=config['TRAIN_STAGE2']['BATCH_SIZE'], 
                              shuffle=True, num_workers=8, collate_fn=crowd_collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, 
                            num_workers=2, collate_fn=crowd_collate)
    
    loss_cfg = config.get('LOSS_STAGE2', {})
    criterion = DACELoss(config['BINS'], config['BIN_CENTERS'], 
                         weight_count=loss_cfg.get('WEIGHT_COUNT_LOSS', 1.0),
                         label_smoothing=loss_cfg.get('LABEL_SMOOTHING', 0.0)).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['TRAIN_STAGE2']['LR']), weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler('cuda', enabled=config['TRAIN_STAGE2'].get('AMP', True))
    
    best_mae = float('inf')
    
    resume_ckpt = out_dir / 'last_model.pth'
    start_epoch = 0
    if resume_ckpt.exists():
        print(f"🔄 Resuming training from: {resume_ckpt}")
        checkpoint = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model'])
        
        # Usa .get() per sicurezza se il file è vecchio
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, config['TRAIN_STAGE2']['TOTAL_EPOCHS']):
        model.train()
        pbar = tqdm(train_loader, desc=f"Ep {epoch}")
        
        for batch in pbar:
            if batch is None: continue
            images = batch['image'].to(device)
            density = batch['density'].to(device)
            points = batch['points']
            
            optimizer.zero_grad()
            with autocast('cuda', enabled=config['TRAIN_STAGE2'].get('AMP', True)):
                outputs = model(images)
                loss, metrics = criterion(outputs, density, points)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix({'mae': f"{metrics['mae']:.1f}"})
        
        scheduler.step()
        
        if epoch % config['TRAIN_STAGE2']['EVAL_FREQ'] == 0:
            val_metrics = evaluate(model, val_loader, device)
            print(f"📊 Val MAE: {val_metrics['mae']:.2f}")
            
            if val_metrics['mae'] < best_mae:
                best_mae = val_metrics['mae']
                torch.save({'epoch': epoch, 'model': model.state_dict(), 'best_model': best_mae}, out_dir / 'best_model.pth')
                print("🌟 Saved Best")

            
            torch.save({
                'epoch': epoch, 
                'model': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), # <--- AGGIUNTO
                'scheduler_state_dict': scheduler.state_dict(), # <--- AGGIUNTO
            }, out_dir / 'last_model.pth')

if __name__ == '__main__':
    main()