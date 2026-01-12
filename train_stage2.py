#!/usr/bin/env python3
"""
Train Stage 2: CLIP-EBC Official Replica
"""
import os
import yaml
import argparse
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from models.clip_ebc_model import CLIPEBCModel
from datasets.sha import SHA # O SHB, assicurati di usare la classe giusta
from datasets.transforms import build_transforms
from losses.clip_ebc_loss import DACELoss

# --- SCHEDULER UFFICIALE (Warmup + Cosine) ---
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args['lr_head']
    # Warmup
    if epoch < args['warmup_epochs']:
        lr_ratio = (epoch + 1) / (args['warmup_epochs'] + 1e-8)
    else:
        # Cosine Decay
        den = max(1, args['epochs'] - args['warmup_epochs'])
        progress = (epoch - args['warmup_epochs']) / den
        lr_ratio = 0.5 * (1. + math.cos(math.pi * progress))
    
    # Applica i learning rate differenziati
    for param_group in optimizer.param_groups:
        if "backbone" in param_group['name']:
            param_group['lr'] = args['lr_backbone'] * lr_ratio
        else:
            param_group['lr'] = args['lr_head'] * lr_ratio
            
    return optimizer.param_groups[0]['lr']

def crowd_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'density': torch.stack([item['density'] for item in batch]),
        'points': [item['points'] for item in batch],
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/config_sha.yaml")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out', type=str, default="checkpoints/sha/stage2")
    cmd_args = parser.parse_args()

    device = torch.device(f'cuda:{cmd_args.gpu}')
    os.makedirs(cmd_args.out, exist_ok=True)
    
    with open(cmd_args.config, 'r') as f: config = yaml.safe_load(f)
    print(f"🚀 Training Stage 2 on {config['DATASET']} (Official Replica)")

    # 1. Model
    model = CLIPEBCModel(config).to(device)

    # 2. Optimizer Groups (Split Backbone vs Head)
    backbone_params = []
    head_params = []
    
    # Identifica i parametri del backbone (visual_encoder di CLIP)
    for name, param in model.named_parameters():
        if "visual_encoder" in name or "clip_model" in name:
            backbone_params.append(param)
        else:
            head_params.append(param) # Decoder, Projections, Logit Scale
            
    optimizer = AdamW([
        {'params': backbone_params, 'lr': config['TRAIN_STAGE2']['LR_BACKBONE'], 'name': 'backbone'},
        {'params': head_params, 'lr': config['TRAIN_STAGE2']['LR_HEAD'], 'name': 'head'}
    ], weight_decay=config['TRAIN_STAGE2']['WEIGHT_DECAY'])

    scaler = GradScaler('cuda', enabled=config['TRAIN_STAGE2']['AMP'])

    # 3. Loss
    criterion = DACELoss(
        bins=config['BINS'],
        reduction=config['CLIP_EBC_HEAD']['REDUCTION'],
        weight_count=config['LOSS_STAGE2']['WEIGHT_COUNT_LOSS'],
        weight_ot=config['LOSS_STAGE2']['WEIGHT_OT'],
        weight_tv=config['LOSS_STAGE2']['WEIGHT_TV']
    ).to(device)

    # 4. Data
    # Nota: Assicurati che SHA/SHB dataset carichi i dati correttamente
    train_ds = SHA(config['DATA']['ROOT'], 'train', build_transforms(config['DATA'], True))
    val_ds = SHA(config['DATA']['ROOT'], 'val', build_transforms(config['DATA'], False))
    
    train_loader = DataLoader(train_ds, batch_size=config['TRAIN_STAGE2']['BATCH_SIZE'], 
                              shuffle=True, num_workers=config['TRAIN_STAGE2']['NUM_WORKERS'], 
                              collate_fn=crowd_collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=crowd_collate)

    # Scheduler Args
    sched_args = {
        'lr_head': config['TRAIN_STAGE2']['LR_HEAD'],
        'lr_backbone': config['TRAIN_STAGE2']['LR_BACKBONE'],
        'epochs': config['TRAIN_STAGE2']['TOTAL_EPOCHS'],
        'warmup_epochs': config['TRAIN_STAGE2']['WARMUP_EPOCHS']
    }

    best_mae = float('inf')

    # --- TRAIN LOOP ---
    for epoch in range(config['TRAIN_STAGE2']['TOTAL_EPOCHS']):
        # Adjust LR
        curr_lr = adjust_learning_rate(optimizer, epoch, sched_args)
        
        model.train()
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1} | LR {curr_lr:.2e}")
        
        loss_avg = 0
                
        for batch in pbar:
            if batch is None: 
                continue
            images = batch['image'].to(device)
            gt_density = batch['density'].to(device)
            points = [p.to(device) for p in batch['points']]  # ✅ FIX

            optimizer.zero_grad()

            with autocast('cuda', enabled=config['TRAIN_STAGE2']['AMP']):
                out = model(images)
                loss, loss_dict = criterion(out['ebc_logits'], out['ebc_density'], gt_density, points)
            
            scaler.scale(loss).backward()
            
            # Gradient Clipping (Importante per CLIP)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['TRAIN_STAGE2']['CLIP_GRAD_NORM'])
            
            scaler.step(optimizer)
            scaler.update()
            
            loss_avg += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.2f}"})
    # --- VALIDATION ---
        model.eval()
        val_abs = 0.0
        val_sq = 0.0

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                img = batch['image'].to(device)
                gt_count = len(batch['points'][0])

                out = model(img)
                pred_count = out['final_count'].item()

                diff = pred_count - gt_count
                val_abs += abs(diff)
                val_sq += diff ** 2

        N = len(val_loader)  # batch_size=1 quindi ok
        val_mae = val_abs / max(1, N)
        val_rmse = math.sqrt(val_sq / max(1, N))

        print(f"\n📊 Ep {epoch+1} | Val MAE: {val_mae:.2f} | Val RMSE: {val_rmse:.2f} (Best MAE: {best_mae:.2f})")

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "mae": best_mae,
                    "rmse": val_rmse,
                    "config": config,
                },
                os.path.join(cmd_args.out, "best_model.pth")
            )
            print("🌟 Saved Best Model")

if __name__ == "__main__":
    main()