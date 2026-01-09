#!/usr/bin/env python3
import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from pathlib import Path

# --- IMPORT DAI TUOI SCRIPT ---
from models.zip_model import ZIPModel
from models.clip_ebc_model import CLIPEBCModel
from losses.joint_loss import JointLoss
from losses.clip_ebc_loss import CLIPEBCLoss # Usa la loss unificata
from datasets.sha import SHA
from datasets.transforms import build_transforms
from eval_patchwise import patchwise_count

# ==============================================================================
# 1. MODELLO CONGIUNTO (WRAPPER)
# ==============================================================================
# (Il codice di joint_model.py va bene, lo importiamo o lo incolliamo qui)
# Assumo che tu abbia il file joint_model.py nella cartella models/
from models.joint_model import ZIPCLIPJointModel 

# ==============================================================================
# 2. UTILS
# ==============================================================================
def crowd_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'density': torch.stack([item['density'] for item in batch]),
        'points': [item['points'] for item in batch], # <--- Cruciale per OT Loss
        'img_path': [item['img_path'] for item in batch]
    }

# ==============================================================================
# 3. MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/config_shb.yaml")
    parser.add_argument('--s1', type=str, required=True, help="Path best_model Stage 1")
    parser.add_argument('--s2', type=str, required=True, help="Path best_model Stage 2")
    parser.add_argument('--out', type=str, default="checkpoints/shb/stage3_final")
    # Rimuoviamo il default hardcoded delle epoche qui, lo prendiamo dal config se possibile
    parser.add_argument('--epochs', type=int, default=None) 
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}')
    os.makedirs(args.out, exist_ok=True)
    
    print(f"🔧 Config: {args.config} | Device: {device}")
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    
    # --- LETTURA PARAMETRI DAL YAML (FIX) ---
    t3_conf = config.get('TRAIN_STAGE3', {})
    
    # Priorità: Argomento da riga di comando > Config YAML > Default
    epochs = args.epochs if args.epochs is not None else t3_conf.get('EPOCHS', 50)
    lr = float(t3_conf.get('LR', 1e-6))
    l_zip = float(t3_conf.get('LAMBDA_ZIP', 1.0))
    l_clip = float(t3_conf.get('LAMBDA_CLIP', 1.0))
    l_count = float(t3_conf.get('LAMBDA_COUNT', 1.0))
    
    print(f"⚙️  Params: LR={lr} | λ_Zip={l_zip} | λ_Clip={l_clip} | λ_Count={l_count}")

    # --- 1. MODELLO ---
    model = ZIPCLIPJointModel(config, args.s1, args.s2).to(device)
    
    # --- 2. CONFIGURAZIONE LOSS ---
    crop_size = config['DATA'].get('CROP_SIZE', 448)
    reduction = config['CLIP_EBC_HEAD'].get('REDUCTION', 16)

    # A) Loss CLIP
    clip_loss_fn = CLIPEBCLoss(
        bins=config['BINS'],
        input_size=crop_size,
        reduction=reduction,
        weight_ot=0.1,
        weight_tv=0.01
    ).to(device)
    
    # B) Loss Congiunta (ORA COLLEGATA AL YAML!)
    criterion = JointLoss(
        clip_loss_fn=clip_loss_fn,
        lambda_zip=l_zip,      # <--- Preso dal config
        lambda_clip=l_clip,    # <--- Preso dal config
        lambda_count=l_count   # <--- Preso dal config
    ).to(device)
    
    # Optimizer (ORA COLLEGATO AL YAML!)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                      lr=lr, # <--- Preso dal config
                      weight_decay=1e-4)
                      
    scaler = GradScaler('cuda', enabled=True)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Dataloaders
    train_ds = SHA(config['DATA']['ROOT'], 'train', build_transforms(config['DATA'], True))
    val_ds = SHA(config['DATA']['ROOT'], 'val', build_transforms(config['DATA'], False))
    
    train_loader = DataLoader(train_ds, batch_size=t3_conf.get('BATCH_SIZE', 4), shuffle=True, num_workers=4, collate_fn=crowd_collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=crowd_collate)
    
    best_mae = float('inf')
    
    print(f"🚀 Start Training Stage 3 for {epochs} epochs...")
    
    for epoch in range(epochs): # Usa la variabile epochs corretta
        model.train()
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
        
        for batch in pbar:
            if batch is None: continue
            
            images = batch['image'].to(device)
            gt_density = batch['density'].to(device)
            points = [p.to(device) for p in batch['points']] 
            
            optimizer.zero_grad()
            
            with autocast('cuda', enabled=True):
                outputs = model(images)
                
                # --- PREPARAZIONE TARGET ---
                # Dobbiamo creare la maschera target per lo Stage 1
                # Ridimensioniamo la densità GT alla dimensione dell'output di Stage 1
                out_h, out_w = outputs['pi_logits'].shape[-2:]
                
                if gt_density.shape[-1] != out_w:
                     # Interpolazione per creare maschera corretta
                     gt_resized = F.interpolate(gt_density, size=(out_h, out_w), mode='bilinear', align_corners=False)
                     # Scala valore per mantenere somma (approssimata)
                     scale = (gt_density.shape[-1] / out_w)**2
                     gt_resized = gt_resized * scale
                else:
                     gt_resized = gt_density

                # Maschera binaria: 1 dove c'è almeno un po' di densità
                mask_gt = (gt_resized > 0.001).float()
                
                targets = {
                    'mask': mask_gt,            # Target per ZIP (Stage 1)
                    'density': gt_density,      # Target per CLIP (Full Resolution)
                    'points': points            # Target per OT
                }
                
                # Calcolo Loss
                loss, loss_dict = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix({
                'L': f"{loss.item():.2f}", 
                'Zip': f"{loss_dict.get('loss_zip',0):.2f}",
                'MAE': f"{loss_dict.get('count_loss',0):.1f}"
            })
        
        scheduler.step()
        
        # --- VALIDATION ---
        model.eval()
        mae = 0
        count = 0

        # prendi config eval (se esiste)
        eval_cfg = config.get("EVAL_STAGE3", {})
        use_patch_eval = bool(eval_cfg.get("ENABLED", True))  # se vuoi, metti False di default

        patch_size = int(eval_cfg.get("PATCH_SIZE", 448))
        stride     = int(eval_cfg.get("STRIDE", patch_size))
        thr        = float(eval_cfg.get("THRESHOLD", 0.35))

        with torch.no_grad():
            for batch in val_loader:
                img = batch['image'].to(device)
                gt = len(batch['points'][0])

                if use_patch_eval:
                    pred, dbg = patchwise_count(
                        model,
                        img,
                        patch_size=patch_size,
                        stride=stride,
                        threshold=thr,
                        presence_reduce="max",   # "max" = più sicuro
                    )
                else:
                    out = model(img)
                    pred = out['final_density'].sum().item()

                mae += abs(pred - gt)
                count += 1

        val_mae = mae / count
        print(f"📊 Ep {epoch+1} | Val MAE: {val_mae:.2f} (Best: {best_mae:.2f})")
        
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
 #   python3 train_stage3.py --config configs/config_shb.yaml --s1 checkpoints/shb/stage1/best_model.pth --s2 checkpoints/shb/stage2/best_model.pth --out checkpoints/shb/stage3_final --epochs 150 --gpu 0