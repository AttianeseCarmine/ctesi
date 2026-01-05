#!/usr/bin/env python3
"""
Train Stage 3: ZIP-CLIP Joint Fine-Tuning
=========================================
Carica Stage 1 e Stage 2 pre-addestrati e li ottimizza insieme
usando il Soft Gating per massimizzare la precisione di conteggio.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import yaml
import argparse
import time
import numpy as np
import random

# --- FIX IMPORTAZIONI ---
# Aggiungiamo la directory corrente al path per evitare errori di modulo
sys.path.append(os.getcwd())

from models.joint_model import ZIPCLIPJointModel
from losses.joint_loss import JointLoss
from losses.clip_ebc_loss import CLIPEBCLoss
from datasets.sha import SHA
from datasets.transforms import build_transforms

# Importiamo i builder dei modelli singoli
from models.clip_ebc_model import CLIPEBCModel # Per Stage 2
from models.zip_clip_ebc_model import ZIPCLIPEBCModel # Per Stage 1

# ============================================================
# UTILITIES
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class AverageMeter(object):
    def __init__(self): self.reset()
    def reset(self): self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

def train_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    images = torch.stack([b['image'] for b in batch]).contiguous()
    densities = torch.stack([b['density'] for b in batch]).contiguous()
    return {'image': images, 'density': densities}

# ============================================================
# MAIN TRAINING LOOP
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_sha.yaml')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--ckpt_stage1', required=True, help='Path checkpoint Stage 1 (.pth)')
    parser.add_argument('--ckpt_stage2', required=True, help='Path checkpoint Stage 2 (.pth)')
    args = parser.parse_args()

    # Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    cfg3 = config.get('TRAIN_STAGE3', {})
    if not cfg3:
        raise ValueError("Manca la sezione TRAIN_STAGE3 nel config.yaml!")

    device = torch.device(f'cuda:{args.gpu}')
    set_seed(config.get('SEED', 42))
    
    # Output Directory
    ckpt_dir = os.path.join("checkpoints", config['RUN_NAME'], "stage3")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"🚀 Starting Stage 3 Joint Training | Output: {ckpt_dir}")

    # --------------------------------------------------------
    # 1. BUILD MODELS (Stage 1 e Stage 2)
    # --------------------------------------------------------
    print("\n🏗️  Building & Loading Models...")

    # --- STAGE 1 (Il Filtro) ---
    print(f"   -> Loading Stage 1 from: {args.ckpt_stage1}")
    # Istanziamo il modello completo usato nello stage 1
    stage1_full = ZIPCLIPEBCModel(config).to(device)
    
    # Caricamento Pesi Robusto
    ckpt1 = torch.load(args.ckpt_stage1, map_location=device)
    state_dict1 = ckpt1['state_dict'] if 'state_dict' in ckpt1 else ckpt1
    # Rimuoviamo prefissi 'module.' se presenti
    state_dict1 = {k.replace('module.', ''): v for k, v in state_dict1.items()}
    
    try:
        stage1_full.load_state_dict(state_dict1, strict=False)
        print("   ✅ Stage 1 weights loaded successfully.")
    except Exception as e:
        print(f"   ⚠️ Warning loading Stage 1: {e}")

 # --- STAGE 2 (Il Contatore) ---
    print(f"   -> Loading Stage 2 from: {args.ckpt_stage2}")
    
    # FIX: Passiamo l'intero dizionario config, come vuole il modello
    stage2_model = CLIPEBCModel(config).to(device)

    # Caricamento Pesi Robusto Stage 2
    ckpt2 = torch.load(args.ckpt_stage2, map_location=device)
    state_dict2 = ckpt2['state_dict'] if 'state_dict' in ckpt2 else \
                  ckpt2['model'] if 'model' in ckpt2 else ckpt2
    state_dict2 = {k.replace('module.', ''): v for k, v in state_dict2.items()}
    
    try:
        stage2_model.load_state_dict(state_dict2, strict=True)
        print("   ✅ Stage 2 weights loaded successfully.")
    except Exception as e:
        print(f"   ⚠️ Warning loading Stage 2: {e}")

    # --------------------------------------------------------
    # 2. JOINT MODEL (Stage 3)
    # --------------------------------------------------------
    # Creiamo il wrapper che unisce i due modelli
    model = ZIPCLIPJointModel(stage1_full, stage2_model).to(device)
    
    # --------------------------------------------------------
    # 3. OPTIMIZER & LOSS
    # --------------------------------------------------------
    # LR molto basso per non distruggere i pesi già appresi (Fine-Tuning)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg3['LR']), weight_decay=float(cfg3['WEIGHT_DECAY']))
    
    clip_loss_fn = CLIPEBCLoss(bins=config['BINS'], bin_centers=config['BIN_CENTERS'])
    
    criterion = JointLoss(
        clip_loss_fn=clip_loss_fn,
        lambda_zip=cfg3['LAMBDA_ZIP'],
        lambda_clip=cfg3['LAMBDA_CLIP'],
        lambda_count=cfg3['LAMBDA_COUNT']
    ).to(device)

    # --------------------------------------------------------
    # 4. DATASETS
    # --------------------------------------------------------
    train_transform = build_transforms(config['DATA'], is_train=True)
    val_transform = build_transforms(config['DATA'], is_train=False)
    
    train_dataset = SHA(config['DATA']['ROOT'], split='train', transforms=train_transform)
    val_dataset = SHA(config['DATA']['ROOT'], split='val', transforms=val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg3['BATCH_SIZE'], 
        shuffle=True, 
        num_workers=cfg3['NUM_WORKERS'], 
        collate_fn=train_collate,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    scaler = GradScaler()
    best_mae = float('inf')

    # ============================================================
    # TRAINING LOOP
    # ============================================================
    print("\n" + "="*40)
    print(f"🚀 STARTING TRAINING (Epochs: {cfg3['EPOCHS']})")
    print("="*40)

    for epoch in range(cfg3['EPOCHS']):
        model.train()
        losses = AverageMeter()
        loss_zip = AverageMeter()
        loss_clip = AverageMeter()
        loss_count = AverageMeter()
        mae_meter = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{cfg3['EPOCHS']}")
        
        for batch in pbar:
            if batch is None: continue
            images = batch['image'].to(device)
            gt_density = batch['density'].to(device)
            
            # --- Prepare Targets ---
            block_size = config['DATA']['ZIP_BLOCK_SIZE']
            
            # 1. Target Counts (Downsample density)
            # Questo crea una mappa 28x28 con il conteggio reale per ogni blocco
            gt_counts_map = F.avg_pool2d(gt_density, block_size, stride=block_size) * (block_size**2)
            
            # 2. Target Mask (0 = Empty, 1 = Crowd)
            gt_mask = (gt_counts_map > 0.001).float() 
            
            targets = {
                'mask': gt_mask,
                'counts': gt_counts_map,
                
                # --- MODIFICA QUESTA RIGA ---
                # PRIMA: 'density': gt_density  <-- ERRORE (448x448)
                # ORA:
                'density': gt_counts_map       # <-- CORRETTO (28x28)
            }
            # --- Optimization Step ---
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss, loss_dict = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # --- Logging ---
            losses.update(loss.item())
            loss_zip.update(loss_dict['l_zip'])
            loss_clip.update(loss_dict['l_clip'])
            loss_count.update(loss_dict['l_count'])
            
            # MAE on-the-fly (sul batch corrente)
            with torch.no_grad():
                pred_count = outputs['final_density'].sum().item()
                gt_count = gt_density.sum().item()
                mae_meter.update(abs(pred_count - gt_count))
            
            pbar.set_postfix({
                'L_tot': f"{losses.avg:.3f}", 
                'L_Cnt': f"{loss_count.avg:.3f}", # Questo deve scendere!
                'MAE': f"{mae_meter.avg:.1f}"
            })

        # ============================================================
        # VALIDATION
        # ============================================================
        if (epoch + 1) % cfg3.get('EVAL_FREQ', 1) == 0:
            val_mae, val_rmse = validate(model, val_loader, device)
            print(f"📊 Val Ep {epoch+1}: MAE={val_mae:.2f} | RMSE={val_rmse:.2f}")
            
            # Save Checkpoint
            if val_mae < best_mae:
                best_mae = val_mae
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_mae': best_mae,
                }, os.path.join(ckpt_dir, "best_model.pth"))
                print("   🌟 New Best Model Saved!")

def validate(model, loader, device):
    model.eval()
    mae_meter = AverageMeter()
    mse_meter = AverageMeter()
    
    with torch.no_grad():
        for batch in loader:
            if batch is None: continue
            img = batch['image'].to(device)
            gt_count = batch['density'].sum().item()
            
            # Forward Stage 3
            out = model(img)
            
            # Usiamo final_density che è già il prodotto di (Mask * Density)
            pred_count = out['final_density'].sum().item()
            
            diff = pred_count - gt_count
            mae_meter.update(abs(diff))
            mse_meter.update(diff**2)
            
    return mae_meter.avg, np.sqrt(mse_meter.avg)

if __name__ == '__main__':
    main()

    #python3 train_stage3.py --config configs/config_sha.yaml --ckpt_stage1 checkpoints/sha/stage1/best_model.pth --ckpt_stage2 checkpoints/sha/stage2/best_model.pth