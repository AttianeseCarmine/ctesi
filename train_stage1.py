#!/usr/bin/env python3
"""
============================================================
STAGE 1: ZIP FILTER TRAINING
============================================================
Addestra il filtro ZIP (Zero-Inflated Poisson) per distinguere
background (muro/alberi) da foreground (folla).

Backbone: VGG16 (Pretrained)
Head: ZIPHead (stima pi e lambda)
Loss: ZIP Negative Log Likelihood (zip_nll)
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

# Imports dai tuoi modelli
from models.zip_clip_ebc_model import ZIPCLIPEBCModel
from datasets.sha import SHA
from datasets.transforms import build_transforms
from losses.zip_nll import zip_nll

def crowd_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'density': torch.stack([item['density'] for item in batch]),
        'points': [item['points'] for item in batch],
        'img_path': [item['img_path'] for item in batch]
    }

def train_epoch(model, loader, optimizer, scaler, device, epoch, config):
    model.train()
    
    # Congeliamo l'EBC Head (Stage 2) perché qui alleniamo solo ZIP (Stage 1)
    # FIX: Uso il nome corretto 'ebc_head'
    if hasattr(model, 'ebc_head'):
        for p in model.ebc_head.parameters():
            p.requires_grad = False
    
    avg_loss = 0
    steps = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        if batch is None: continue
        
        images = batch['image'].to(device)
        gt_density = batch['density'].to(device)
        
        # Generiamo i target per ZIP
        # Downsampling densità a H/16, W/16 per matchare l'output VGG
        # ZIP lavora a blocchi: somma della densità nel blocco = conteggio
        B, C, H, W = images.shape
        target_counts = F.avg_pool2d(gt_density, kernel_size=16, stride=16) * (16*16)
        
        optimizer.zero_grad()
        
        with autocast('cuda', enabled=True):
            outputs = model(images)
            
            # Output del modello
            # pi_logits -> Probabilità che il blocco sia VUOTO (o pieno, dipende dalla head)
            # lambda_logits -> Rate Poisson
            pi_logits = outputs['pi_logits']
            lambda_logits = outputs['lambda_logits']
            
            # Calcolo Probabilità pi e Lambda
            # ZIPHead usa Sigmoid internamente o logits?
            # Solitamente la loss zip_nll prende probabilità e rate
            pi = torch.sigmoid(pi_logits)
            lam = torch.exp(lambda_logits)
            
            # Loss NLL
            loss = zip_nll(pi, lam, target_counts)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        avg_loss += loss.item()
        steps += 1
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return avg_loss / steps if steps > 0 else 0

@torch.no_grad()
def evaluate(model, loader, device, threshold=0.3):
    model.eval()
    
    # Metriche di classificazione binaria (Muro vs Folla)
    tp, tn, fp, fn = 0, 0, 0, 0
    
    for batch in tqdm(loader, desc="Eval"):
        if batch is None: continue
        images = batch['image'].to(device)
        gt_density = batch['density'].to(device)
        
        outputs = model(images)
        pi = torch.sigmoid(outputs['pi_logits']) # Probabilità VUOTO (o PIENO?)
        # Nota: Solitamente in ZIP: 
        # pi = Probabilità che sia "Structural Zero" (VUOTO)
        # Quindi 1 - pi = Probabilità che ci sia gente.
        
        # Target Reali
        target_counts = F.avg_pool2d(gt_density, kernel_size=16, stride=16) * (16*16)
        is_empty_gt = (target_counts < 0.001).float() # 1 se vuoto, 0 se gente
        
        # Predizioni
        # Se pi > threshold -> Predetto Vuoto
        is_empty_pred = (pi > threshold).float()
        
        tp += ((is_empty_pred == 1) & (is_empty_gt == 1)).sum().item()
        tn += ((is_empty_pred == 0) & (is_empty_gt == 0)).sum().item()
        fp += ((is_empty_pred == 1) & (is_empty_gt == 0)).sum().item() # Predetto vuoto ma c'era gente (Grave!)
        fn += ((is_empty_pred == 0) & (is_empty_gt == 1)).sum().item() # Predetto gente ma era vuoto (Meno grave)
        
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    
    return {'acc': accuracy, 'prec': precision, 'rec': recall}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_shb.yaml')
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    device = torch.device(f'cuda:{args.gpu}')
    
    # Setup Paths
    dataset_name = config.get('DATASET', 'sha')
    out_dir = Path(f'checkpoints/{dataset_name}/stage1')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Model
    print("🏗️  Building ZIP Model...")
    model = ZIPCLIPEBCModel(config).to(device)
    
    # Optimizer (Allena solo Backbone + ZIPHead)
    # EBC Head è congelata/ignorata
    params = list(model.backbone.parameters()) + list(model.zip_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(config['TRAIN_STAGE1']['LR_HEAD']))
    
    scaler = GradScaler('cuda')
    
    # Dataset
    train_ds = SHA(config['DATA']['ROOT'], 'train', build_transforms(config['DATA'], True))
    val_ds = SHA(config['DATA']['ROOT'], 'val', build_transforms(config['DATA'], False))
    
    train_loader = DataLoader(train_ds, batch_size=config['TRAIN_STAGE1']['BATCH_SIZE'], shuffle=True, 
                              num_workers=8, collate_fn=crowd_collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=crowd_collate)
    
    best_acc = 0.0
    
    print(f"🚀 Starting Stage 1 Training on {dataset_name}")
    for epoch in range(config['TRAIN_STAGE1']['EPOCHS']):
        loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch, config)
        print(f"Ep {epoch} | Loss: {loss:.4f}")
        
        if epoch % config['TRAIN_STAGE1']['VAL_INTERVAL'] == 0:
            metrics = evaluate(model, val_loader, device)
            print(f"📊 Val Acc: {metrics['acc']:.4f} | Prec: {metrics['prec']:.4f} | Rec: {metrics['rec']:.4f}")
            
            if metrics['acc'] > best_acc:
                best_acc = metrics['acc']
                torch.save({'model': model.state_dict(), 'epoch': epoch, 'acc': best_acc}, out_dir / 'best_model.pth')
                print("🌟 New Best Model Saved!")
                
        # Save last
        torch.save({'model': model.state_dict(), 'epoch': epoch}, out_dir / 'last_model.pth')

if __name__ == '__main__':
    main()