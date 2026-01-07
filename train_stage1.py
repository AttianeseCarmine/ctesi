#!/usr/bin/env python3
"""
============================================================
STAGE 1: ZIP HEAD TRAINING (Binary Segmentation)
============================================================
Obiettivo: Addestrare SOLO la 'zip_head' (pi) a distinguere 
background (vuoto) da foreground (folla).

Approccio Semplificato (Official Style):
- Input: Immagine
- Output: Mappa di probabilità (pi)
- Target: Maschera binaria (1 se c'è gente, 0 se vuoto)
- Loss: BCEWithLogitsLoss (pesata per sbilanciamento)
============================================================
"""

import argparse
import yaml
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# Assicurati che questi import funzionino
from models.zip_model import ZIPModel
from datasets.sha import SHA
from datasets.transforms import build_transforms
from utils.train_utils import seed_everything

# =============================================================================
# UTILS
# =============================================================================
def crowd_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'density': torch.stack([item['density'] for item in batch]),
        'img_path': [item['img_path'] for item in batch]
    }

def save_checkpoint(state, is_best, save_dir, filename='last_model.pth'):
    last_path = os.path.join(save_dir, filename)
    torch.save(state, last_path)
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        shutil.copyfile(last_path, best_path)

# =============================================================================
# EVALUATION FUNCTION (Binary Metrics)
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    
    tp, tn, fp, fn = 0, 0, 0, 0
    
    # Soglia per decidere se è folla o no (0.5 su sigmoid = 0 su logits)
    threshold = 0.3
    
    for batch in tqdm(loader, desc="Eval"):
        if batch is None: continue
        
        images = batch['image'].to(device)
        gt_density = batch['density'].to(device)
        
        # 1. Forward standard
        outputs = model(images)
        pi_logits = outputs['pi_logits'] # [B, 1, H/16, W/16]
        probs = torch.sigmoid(pi_logits)
        
        # 2. Prepara Ground Truth Binaria (match dimensions)
        h_out, w_out = pi_logits.shape[2:]
        # Downsample della densità alla risoluzione dell'output (H/16)
        # Usiamo adaptive_avg_pool2d * area per conservare il conteggio
        scale_factor = (images.shape[2] * images.shape[3]) / (h_out * w_out)
        gt_down = F.adaptive_avg_pool2d(gt_density, (h_out, w_out)) * scale_factor
        
        # Se nel blocco c'è anche solo mezza persona, è "Folla" (1)
        # Nota: per SHB sparso, possiamo usare una soglia molto bassa (es. 0.001)
        gt_binary = (gt_down > 0.001).float()
        
        # 3. Predizione Binaria
        pred_binary = (probs > threshold).float()
        
        # 4. Metriche
        tp += ((pred_binary == 1) & (gt_binary == 1)).sum().item()
        tn += ((pred_binary == 0) & (gt_binary == 0)).sum().item()
        fp += ((pred_binary == 1) & (gt_binary == 0)).sum().item()
        fn += ((pred_binary == 0) & (gt_binary == 1)).sum().item()
        
    # Calcolo F1-Score (che bilancia Precision e Recall)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    return {'f1': f1, 'acc': acc, 'prec': precision, 'rec': recall}

# =============================================================================
# TRAINING LOOP
# =============================================================================
def train_stage1_simple():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_shb.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    seed_everything(config.get('SEED', 42))
    device = torch.device(f'cuda:{args.gpu}')
    
    # Setup
    dataset_name = config.get('DATASET', 'sha')
    save_dir = os.path.join('./checkpoints', dataset_name, 'stage1')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"🚀 Stage 1: Binary Classifier Training (Simple)")
    print(f"   Dataset: {dataset_name} | Save to: {save_dir}")
    
    # Dataset
    data_cfg = config['DATA']
    train_dataset = SHA(data_cfg['ROOT'], 'train', build_transforms(data_cfg, True))
    val_dataset = SHA(data_cfg['ROOT'], 'val', build_transforms(data_cfg, False))
    
    # Batch size ridotto se necessario per SHB (immagini grandi)
    bs = config['TRAIN_STAGE1'].get('BATCH_SIZE', 16)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, 
                              num_workers=8, collate_fn=crowd_collate, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                            num_workers=4, collate_fn=crowd_collate)
    
    # Modello
    model = ZIPModel(config).to(device)
    
    # --- FREEZE & UNFREEZE ---
    # Congela tutto tranne ZIP Head e Backbone
    for p in model.parameters(): p.requires_grad = False
    
    # Sblocca Backbone
    for p in model.backbone.parameters(): p.requires_grad = True
    
    # Sblocca ZIP Head (Gestione nomi diversi)
    if hasattr(model, 'zip_head'):
        for p in model.zip_head.parameters(): p.requires_grad = True
        head_params = model.zip_head.parameters()
    elif hasattr(model, 'pi_head'):
        for p in model.pi_head.parameters(): p.requires_grad = True
        head_params = model.pi_head.parameters()
    else:
        raise AttributeError("Zip head non trovata (cercato 'zip_head' e 'pi_head')")

    # Optimizer
    lr = float(config['TRAIN_STAGE1']['LR_HEAD'])
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': lr * 0.1}, # Backbone più lento
        {'params': head_params, 'lr': lr}
    ], weight_decay=1e-4)
    
    scaler = GradScaler('cuda')
    
    # Loss: BCEWithLogitsLoss pesata
    # SHB è molto sparso (tanti 0). Diamo più peso ai pixel con folla (1).
    pos_weight_val = float(config['TRAIN_STAGE1'].get('POS_WEIGHT', 15.0))
    pos_weight_tensor = torch.tensor([pos_weight_val]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    best_f1 = 0.0
    epochs = config['TRAIN_STAGE1']['EPOCHS']
    
    print("🔧 Training Start...")
    
    for epoch in range(epochs):
        model.train()
        avg_loss = 0
        steps = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{epochs}")
        for batch in pbar:
            if batch is None: continue
            
            images = batch['image'].to(device)
            gt_density = batch['density'].to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                # 1. Forward (Standard)
                outputs = model(images)
                pi_logits = outputs['pi_logits']
                
                # 2. Target Binario
                h_out, w_out = pi_logits.shape[2:]
                # Adatta la density map alla dimensione dell'output
                # Sum pooling approssimato (avg * area) per vedere se c'è gente
                scale = (images.shape[2] * images.shape[3]) / (h_out * w_out)
                gt_down = F.adaptive_avg_pool2d(gt_density, (h_out, w_out)) * scale
                
                # Maschera: 1 se > 0.001 (c'è gente), 0 altrimenti
                target_binary = (gt_down > 0.001).float()
                
                # 3. Loss
                loss = criterion(pi_logits, target_binary)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            avg_loss += loss.item()
            steps += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # Validation
        if (epoch + 1) % config['TRAIN_STAGE1']['VAL_INTERVAL'] == 0:
            metrics = evaluate(model, val_loader, device)
            print(f"\n📊 Val Ep {epoch+1}: F1={metrics['f1']:.2%} | Acc={metrics['acc']:.2%} | Prec={metrics['prec']:.2%} | Rec={metrics['rec']:.2%}")
            
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                save_checkpoint(model.state_dict(), True, save_dir, 'last_model.pth')
                print("🌟 New Best Saved!")
            else:
                save_checkpoint(model.state_dict(), False, save_dir, 'last_model.pth')

if __name__ == '__main__':
    train_stage1_simple()