#!/usr/bin/env python3
import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

# --- IMPORT DAI TUOI SCRIPT ESISTENTI ---
from models.zip_model import ZIPModel
from models.clip_ebc_model import CLIPEBCModel
from losses.joint_loss import JointLoss
from losses.clip_ebc_loss import CLIPEBCLoss
from datasets.sha import SHA
from datasets.transforms import build_transforms

# ==============================================================================
# 1. MODELLO CONGIUNTO (WRAPPER)
# ==============================================================================
class ZIPCLIPJointModel(nn.Module):
    def __init__(self, config, s1_path, s2_path):
        super().__init__()
        
        # 1. Carica i Modelli Base
        print("🏗️  Building Base Models...")
        self.stage1 = ZIPModel(config)
        self.stage2 = CLIPEBCModel(config)
        
        # 2. Carica i Pesi
        print(f"📥 Loading Stage 1: {s1_path}")
        s1_ckpt = torch.load(s1_path, map_location='cpu')
        st1 = s1_ckpt['model'] if 'model' in s1_ckpt else s1_ckpt
        self.stage1.load_state_dict(st1, strict=False)
        
        print(f"📥 Loading Stage 2: {s2_path}")
        s2_ckpt = torch.load(s2_path, map_location='cpu')
        st2 = s2_ckpt['model'] if 'model' in s2_ckpt else s2_ckpt
        self.stage2.load_state_dict(st2, strict=False)

        # 3. Gestione Gradienti (Refined Strategy)
        # Stage 1: Sblocchiamo la testa e il backbone (per adattarsi alla risoluzione)
        for p in self.stage1.parameters(): p.requires_grad = True
        
        # Stage 2: Blocchiamo il backbone CLIP (prezioso), sblocchiamo solo le teste
        for p in self.stage2.parameters(): p.requires_grad = False
        for p in self.stage2.projection.parameters(): p.requires_grad = True
        if hasattr(self.stage2, 'image_decoder'):
            for p in self.stage2.image_decoder.parameters(): p.requires_grad = True

    def forward(self, x):
        # --- Stage 1: Maschera ---
        out1 = self.stage1(x)
        # Probabilità di Sfondo (pi)
        if 'pi' in out1:
            pi = out1['pi']
        else:
            pi = torch.sigmoid(out1['pi_logits'])
        
        # Probabilità di "Persona" (Foreground)
        prob_fg = 1.0 - pi 

        # --- Stage 2: Densità ---
        out2 = self.stage2(x)
        density_raw = out2['ebc_density'] # [B, 1, H_out, W_out]
        logits_ebc = out2['ebc_logits']
        
        # --- Stage 3: Soft Refinement & Interpolazione ---
        # Allinea risoluzione Stage 1 (32x32) a Stage 2 (64x64 o 28x28) se diverse
        if prob_fg.shape[-2:] != density_raw.shape[-2:]:
            prob_fg = F.interpolate(
                prob_fg, 
                size=density_raw.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
            # Aggiorniamo 'pi' interpolato per la loss ZIP
            pi = 1.0 - prob_fg

        # Soft Masking: Densità * Probabilità Presenza
        final_density = density_raw * prob_fg
        
        return {
            'pi_logits': torch.logit(pi + 1e-6), # Riconverte in logits per BCEWithLogits
            'pi': pi,
            'ebc_logits': logits_ebc,
            'final_density': final_density,
            'prob_fg': prob_fg
        }

# ==============================================================================
# 2. UTILS
# ==============================================================================
def crowd_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    images = torch.stack([item['image'] for item in batch])
    points = [item['points'] for item in batch]
    densities = torch.stack([item['density'] for item in batch])
    return {'image': images, 'points': points, 'density': densities}

# ==============================================================================
# 3. MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/config_shb.yaml")
    parser.add_argument('--s1', type=str, required=True, help="Checkpoint Stage 1")
    parser.add_argument('--s2', type=str, required=True, help="Checkpoint Stage 2")
    parser.add_argument('--out', type=str, default="checkpoints/shb/stage3_final")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}')
    os.makedirs(args.out, exist_ok=True)
    
    print(f"🔧 Config: {args.config} | Device: {device}")
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    
    # --- 1. MODELLO ---
    model = ZIPCLIPJointModel(config, args.s1, args.s2).to(device)
    
    # --- 2. CONFIGURAZIONE LOSS (Fix Lettura Config) ---
    # Legge direttamente dalla root (config_shb.yaml style)
    if 'BINS' in config:
        bins = config['BINS']
        centers = config['BIN_CENTERS']
    else:
        # Fallback (config_sha.yaml style o BINS_CONFIG)
        ds_name = config.get('DATASET', 'shb')
        bins = config['BINS_CONFIG'][ds_name]['bins']
        centers = config['BINS_CONFIG'][ds_name]['bin_centers']
        
    print(f"✅ Loss Config: {len(bins)} bins caricati.")

    # A) Loss CLIP
    clip_loss_fn = CLIPEBCLoss(
        bins=bins, 
        bin_centers=centers,
        count_weight=0.1
    ).to(device)
    
    # B) Loss Congiunta
    loss_cfg = config.get('LOSS_STAGE3', {})
    criterion = JointLoss(
        clip_loss_fn=clip_loss_fn,
        lambda_zip=loss_cfg.get('ALPHA_PI', 1.0),
        lambda_clip=loss_cfg.get('ALPHA_EBC', 1.0),
        lambda_count=loss_cfg.get('COUNT_WEIGHT', 1.0)
    ).to(device)
    
    # Optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Dataloaders
    train_ds = SHA(config['DATA']['ROOT'], 'train', build_transforms(config['DATA'], True))
    val_ds = SHA(config['DATA']['ROOT'], 'val', build_transforms(config['DATA'], False))
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, collate_fn=crowd_collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=crowd_collate)
    
    best_mae = float('inf')
    
    print("🚀 Start Training Stage 3...")
    
    for epoch in range(args.epochs):
        model.train()
        loss_epoch = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
        
        for batch in pbar:
            if batch is None: continue
            
            images = batch['image'].to(device)     # [B, 3, 448, 448]
            gt_density = batch['density'].to(device) # [B, 1, 448, 448]
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(images) # Output ha size ridotta (es. 28x28)
            
            # --- FIX CRITICO: RIDIMENSIONAMENTO TARGET ---
            # Ridimensioniamo la Ground Truth per matchare l'output del modello
            out_h, out_w = outputs['pi_logits'].shape[-2:]
            
            if gt_density.shape[-1] != out_w:
                # Interpoliamo la densità GT
                gt_density_resized = F.interpolate(
                    gt_density, 
                    size=(out_h, out_w), 
                    mode='bilinear', 
                    align_corners=False
                )
                # Conserviamo la somma (il conteggio) moltiplicando per il fattore di scala quadrato
                scale_factor = (gt_density.shape[-1] / out_w) ** 2
                gt_density_resized = gt_density_resized * scale_factor
                
                # Creiamo la maschera target sulla versione ridotta
                target_mask = (gt_density_resized > 0.001).float()
            else:
                gt_density_resized = gt_density
                target_mask = (gt_density > 0.001).float()
            # ---------------------------------------------
            
            targets = {
                'mask': target_mask,          # Per ZIP
                'counts': gt_density,         # Per CLIP (usa pooling interno, ok full res)
                'density': gt_density_resized # Per Count Loss (L1) -> DEVE essere resized
            }
            
            loss, loss_dict = criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            loss_epoch += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.2f}"})
            
        scheduler.step()
        
        # --- VALIDATION ---
        model.eval()
        mae = 0
        count = 0
        with torch.no_grad():
            for batch in val_loader:
                img = batch['image'].to(device)
                gt = len(batch['points'][0])
                
                out = model(img)
                # Somma sulla mappa finale per il conteggio
                pred = out['final_density'].sum().item()
                
                mae += abs(pred - gt)
                count += 1
        
        val_mae = mae / count
        print(f"📊 Ep {epoch+1} | Val MAE: {val_mae:.4f} (Best: {best_mae:.4f})")
        
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'mae': best_mae
            }, f"{args.out}/best_model.pth")
            print("🌟 Saved Best!")

if __name__ == "__main__":
    main()