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
from pathlib import Path

# Import dei tuoi modelli esistenti
from models.zip_model import ZIPModel
from models.clip_ebc_model import CLIPEBCModel
from datasets.sha import SHA
from datasets.transforms import build_transforms

# ==============================================================================
# 1. MODELLO CONGIUNTO "REFINED" (Gestisce Risoluzione e Soft Masking)
# ==============================================================================
class RefinedStage3(nn.Module):
    def __init__(self, config, s1_weights=None, s2_weights=None):
        super().__init__()
        
        # Inizializza i sotto-modelli
        print("🏗️  Building Sub-Models...")
        self.stage1 = ZIPModel(config)
        self.stage2 = CLIPEBCModel(config)
        
        # Carica Pesi Stage 1
        if s1_weights:
            print(f"📥 Loading Stage 1: {s1_weights}")
            ckpt = torch.load(s1_weights, map_location='cpu')
            st = ckpt['model'] if 'model' in ckpt else ckpt
            self.stage1.load_state_dict(st, strict=False)
            
        # Carica Pesi Stage 2
        if s2_weights:
            print(f"📥 Loading Stage 2: {s2_weights}")
            ckpt = torch.load(s2_weights, map_location='cpu')
            st = ckpt['model'] if 'model' in ckpt else ckpt
            self.stage2.load_state_dict(st, strict=False)

        # Freeze parziale (Opzionale: sblocca tutto per massimo adattamento)
        # Per ora teniamo CLIP (Stage 2) freezato nel backbone per non distruggerlo subito
        for p in self.stage2.visual_encoder.parameters():
            p.requires_grad = False 
            
    def forward(self, x):
        # --- 1. Stage 1: Mask Prediction ---
        out1 = self.stage1(x)
        # ZIP predice solitamente 'pi' (probabilità di zero/sfondo) o logits
        if 'pi' in out1:
            prob_bg = out1['pi'] 
        else:
            prob_bg = torch.sigmoid(out1['pi_logits'])
            
        # La probabilità di "Essere Persona" è (1 - prob_sfondo)
        prob_fg = 1.0 - prob_bg

        # --- 2. Stage 2: Density Prediction ---
        out2 = self.stage2(x)
        density_raw = out2['ebc_density'] # [B, 1, 64, 64] (se img 1024)
        
        # --- 3. Refinement & Fusion ---
        # Upsample della maschera (da 32x32 a 64x64) per matchare la densità
        if prob_fg.shape[-2:] != density_raw.shape[-2:]:
            prob_fg = F.interpolate(
                prob_fg, 
                size=density_raw.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # SOFT MASKING (Differenziabile!)
        # Invece di tagliare con una soglia, moltiplichiamo.
        # Se prob_fg è alta (persona) -> passiamo la densità.
        # Se prob_fg è bassa (sfondo) -> sopprimiamo la densità verso zero.
        density_refined = density_raw * prob_fg
        
        # Calcolo conteggio totale
        count = density_refined.sum(dim=(1, 2, 3))
        
        return {
            'final_density': density_refined,
            'raw_density': density_raw,
            'mask_fg': prob_fg,
            'pred_count': count
        }

# ==============================================================================
# 2. UTILS TRAIN
# ==============================================================================
def crowd_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'points': [item['points'] for item in batch] # Lista di tensori punti
    }

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mae = 0
    count = 0
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        if batch is None: continue
        img = batch['image'].to(device)
        gt_counts = [len(p) for p in batch['points']]
        
        out = model(img)
        preds = out['pred_count']
        
        for p, g in zip(preds, gt_counts):
            mae += abs(p.item() - g)
            count += 1
    return mae / count if count > 0 else 0

# ==============================================================================
# 3. MAIN TRAINING LOOP
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--s1_weights', type=str, required=True)
    parser.add_argument('--s2_weights', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="checkpoints/shb/stage3_refined")
    parser.add_argument('--lr', type=float, default=1e-5) # LR basso per fine-tuning
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}')
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    
    # Init Modello
    model = RefinedStage3(config, args.s1_weights, args.s2_weights).to(device)
    
    # Dataset
    train_ds = SHA(config['DATA']['ROOT'], 'train', build_transforms(config['DATA'], True))
    val_ds = SHA(config['DATA']['ROOT'], 'val', build_transforms(config['DATA'], False))
    
    train_loader = DataLoader(train_ds, batch_size=config['TRAIN_STAGE2']['BATCH_SIZE'], 
                              shuffle=True, num_workers=4, collate_fn=crowd_collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, 
                            num_workers=2, collate_fn=crowd_collate)
    
    # Optimizer (Allena tutto o solo le teste)
    # Suggerimento: Allena Mask Head e Projection Head, tieni fermi i backbone
    optimizer = AdamW([
        {'params': model.stage1.zip_head.parameters(), 'lr': args.lr},       # Mask Head
        {'params': model.stage2.projection.parameters(), 'lr': args.lr},     # CLIP Projection
        {'params': model.stage2.image_decoder.parameters(), 'lr': args.lr},  # CLIP Decoder
    ], lr=args.lr, weight_decay=1e-4)
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_mae = float('inf')
    
    print(f"🚀 Start Refined Training. Saving to {args.output_dir}")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            if batch is None: continue
            img = batch['image'].to(device)
            # GT Count (per MAE Loss)
            gt_counts = torch.tensor([len(p) for p in batch['points']], dtype=torch.float32, device=device)
            
            optimizer.zero_grad()
            
            # Forward
            out = model(img)
            pred_counts = out['pred_count']
            
            # Loss: L1 Loss diretta sul conteggio finale mascherato
            # Questo forza la maschera a spegnere lo sfondo per ridurre l'errore
            loss = F.l1_loss(pred_counts, gt_counts)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.2f}"})
            
        scheduler.step()
        
        # Eval
        val_mae = evaluate(model, val_loader, device)
        print(f"📊 Epoch {epoch+1} | Loss: {epoch_loss/len(train_loader):.4f} | Val MAE: {val_mae:.4f}")
        
        # Save Best
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(), # Salva TUTTO il modello wrappato
                'best_mae': best_mae
            }, f"{args.output_dir}/best_model.pth")
            print(f"🌟 New Best Saved: {best_mae:.4f}")
            
    print(f"🏁 Training Finished. Best MAE: {best_mae:.4f}")

if __name__ == "__main__":
    main()


    # python train_stage3_refined.py --config configs/config_shb.yaml --s1_weights checkpoints/shb/stage1/best_model.pth --s2_weights checkpoints/shb/stage2/best_model.pth --lr 1e-5 --epochs 50 --output_dir checkpoints/shb/stage3