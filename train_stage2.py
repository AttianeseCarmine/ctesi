#!/usr/bin/env python3
"""
Train Stage 2: CLIP-EBC Official Replica (Fixed)
"""
import os
import yaml
import json
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast # Import corretto per Mixed Precision
from tqdm import tqdm

# --- IMPORTS AGGIORNATI ---
# Assicurati che i file siano in models/clip/model.py e losses/clip_ebc_loss.py
from models.clip.model import CLIP_EBC
from losses.clip_ebc_loss import DACELoss

# Assumi che tu abbia un dataset loader funzionante
# Se non hai datasets/sha.py, devi usare il tuo loader custom
try:
    from datasets.sha import SHA 
    from datasets.transforms import build_transforms
except ImportError:
    print("⚠️ Dataset SHA non trovato, assicurati di avere il file dataset corretto.")

def adjust_learning_rate(optimizer, epoch, args):
    """Cosine schedule con Warmup"""
    lr_max = args['TRAIN_STAGE2']['LR_HEAD']
    warmup_epochs = args['TRAIN_STAGE2']['WARMUP_EPOCHS']
    max_epochs = args['TRAIN_STAGE2']['EPOCHS']

    if epoch < warmup_epochs:
        lr = lr_max * (epoch + 1) / (warmup_epochs + 1e-8)
    else:
        # Cosine annealing
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        lr = lr_max * 0.5 * (1. + math.cos(math.pi * progress))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main(config_path):
    # 1. Load Config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg['DEVICE'])
    os.makedirs(f"checkpoints/{cfg['RUN_NAME']}", exist_ok=True)

    # 2. Load Bins & Anchors (CRUCIALE)
    bins_path = cfg['CLIP_EBC_MODEL']['BINS_JSON']
    if not os.path.exists(bins_path):
        raise FileNotFoundError(f"❌ File bin non trovato: {bins_path}. Copia 'configs/reduction_16.json' dal repo originale.")
    
    with open(bins_path, 'r') as f:
        bins_data = json.load(f)
        # Il json ha chiavi stringa "0", "1"... li convertiamo
        # bins format: [[0,0], [1,1], ..., [m, inf]]
        bins_list = bins_data['bins']
        anchor_points = bins_data['anchor_points']
        
        # Conversione "inf" stringa a float('inf') se necessario
        cleaned_bins = []
        for b in bins_list:
            start = b[0]
            end = float('inf') if b[1] == "inf" else b[1]
            cleaned_bins.append((float(start), float(end)))
        
        cleaned_anchors = [float(a) for a in anchor_points]

    print(f"✅ Loaded {len(cleaned_bins)} bins from {bins_path}")

    # 3. Initialize Model
    print(f"🏗️ Building CLIP_EBC ({cfg['CLIP_EBC_MODEL']['BACKBONE']})...")
    model = CLIP_EBC(
        backbone=cfg['CLIP_EBC_MODEL']['BACKBONE'],
        bins=cleaned_bins,
        anchor_points=cleaned_anchors,
        reduction=cfg['CLIP_EBC_MODEL']['REDUCTION'],
        input_size=cfg['CLIP_EBC_MODEL']['INPUT_SIZE'],
        num_vpt=cfg['CLIP_EBC_MODEL']['NUM_VPT'],
        deep_vpt=cfg['CLIP_EBC_MODEL']['DEEP_VPT'],
        vpt_drop=cfg['CLIP_EBC_MODEL']['VPT_DROP'],
        prompt_type=cfg['CLIP_EBC_MODEL']['PROMPT_TYPE']
    ).to(device)

    # 4. Dataset & Dataloader
    # Nota: Adatta questo pezzo al tuo dataset loader specifico
    # SHA deve ritornare: image, density_map, point_list
    train_transform = build_transforms(cfg, is_train=True)
    val_transform = build_transforms(cfg, is_train=False)
    
    train_dataset = SHA(root=cfg['DATASET']['ROOT'], split=cfg['DATASET']['TRAIN_SPLIT'], transform=train_transform)
    val_dataset = SHA(root=cfg['DATASET']['ROOT'], split=cfg['DATASET']['VAL_SPLIT'], transform=val_transform)
    
    # Collate function custom per gestire liste di punti di lunghezza variabile
    def collate_fn(batch):
        imgs = torch.stack([item[0] for item in batch])
        densities = torch.stack([item[1] for item in batch])
        points = [item[2] for item in batch] # Lista di tensor
        # Se il dataset ritorna anche i nomi, gestiscili qui
        return imgs, densities, points

    train_loader = DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], shuffle=True, num_workers=cfg['TRAIN']['NUM_WORKERS'], collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=cfg['TRAIN']['NUM_WORKERS'], collate_fn=collate_fn)

    # 5. Optimizer & Loss
    # Filtriamo i parametri: Backone ViT è congelato, alleniamo VPT e decoder
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params_to_optimize, lr=cfg['TRAIN']['LR_HEAD'], weight_decay=cfg['TRAIN']['WEIGHT_DECAY'])
    
    criterion = DACELoss(
        bins=cleaned_bins,
        reduction=cfg['CLIP_EBC_MODEL']['REDUCTION'],
        weight_count_loss=cfg['LOSS_STAGE2']['WEIGHT_COUNT'],
        count_loss=cfg['LOSS_STAGE2']['COUNT_LOSS'], # "dmcount"
        weight_ot=cfg['LOSS_STAGE2']['WEIGHT_OT'],
        weight_tv=cfg['LOSS_STAGE2']['WEIGHT_TV'],
        input_size=cfg['CLIP_EBC_MODEL']['INPUT_SIZE']
    ).to(device)

    scaler = GradScaler() # Per Mixed Precision

    # 6. Training Loop
    best_mae = float('inf')
    
    for epoch in range(cfg['TRAIN']['EPOCHS']):
        adjust_learning_rate(optimizer, epoch, cfg)
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{cfg['TRAIN']['EPOCHS']}")
        for imgs, gt_density, gt_points in pbar:
            imgs = imgs.to(device)
            gt_density = gt_density.to(device)
            # gt_points è una lista di tensori, li spostiamo su device dentro la loss o qui se serve
            
            optimizer.zero_grad()
            
            with autocast():
                # CLIP_EBC in training ritorna (logits, density_map)
                pred_logits, pred_density = model(imgs)
                
                # Calcolo Loss
                loss, loss_dict = criterion(pred_logits, pred_density, gt_density, gt_points)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'ce': loss_dict.get('ce_loss', 0).item()})

        # 7. Validation Loop
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for imgs, gt_density, gt_points in val_loader:
                imgs = imgs.to(device)
                gt_count = len(gt_points[0]) # Batch size 1 in validation
                
                # In eval, CLIP_EBC ritorna solo density_map (o expected count map)
                pred_map = model(imgs)
                pred_count = pred_map.sum().item()
                
                val_mae += abs(pred_count - gt_count)
        
        val_mae /= len(val_dataset)
        print(f"📊 Epoch {epoch+1} Result: Val MAE: {val_mae:.2f} (Best: {best_mae:.2f})")
        
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), f"checkpoints/{cfg['RUN_NAME']}/best_model.pth")
            print("💾 Model Saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_vit_sha.yaml')
    args = parser.parse_args()
    main(args.config)