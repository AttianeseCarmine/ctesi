#!/usr/bin/env python3
"""
Train Stage 3: ZIP-CLIP Joint Fine-Tuning (FINAL CORRECTED)
===========================================================
Strategia: P2R-ZIP Style (Soft Gating + Partial Unfreeze).
Usa le classi originali del tuo progetto (ZIPModel, CLIPEBCModel).
"""

import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# --- IMPORT MODELLI CORRETTI ---
from models.joint_model import ZIPCLIPJointModel
from models.clip_ebc_model import CLIPEBCModel
from models.zip_model import ZIPModel  # <--- USIAMO QUESTA! Esiste già nel tuo progetto.
from losses.clip_ebc_loss import CLIPEBCLoss 
from losses.zip_nll import ZIPNLLLoss
from datasets.sha import SHA
from datasets.transforms import build_transforms

# ==============================================================================
# UTILS
# ==============================================================================
def freeze_parameters(model, mode="p2r_style"):
    """
    Congela i parametri per il fine-tuning delicato.
    """
    print("\n🔒 Freezing Strategy: P2R-ZIP Style")
    
    # 1. Congela TUTTO inizialmente
    for param in model.parameters():
        param.requires_grad = False

    trainable_params = []

    # --- A. Sblocca ALIGNER (Fondamentale per allineare le feature) ---
    for p in model.mask_aligner.parameters():
        p.requires_grad = True
    trainable_params.append({'params': model.mask_aligner.parameters(), 'lr_mult': 10.0}) # LR alto
    print("   ✅ Aligner: Unfrozen")

    # --- B. Sblocca ZIP HEAD (Stage 1 Head) ---
    # zip_head è dentro model.stage1 (che è un ZIPModel)
    for p in model.stage1.zip_head.parameters():
        p.requires_grad = True
    trainable_params.append({'params': model.stage1.zip_head.parameters(), 'lr_mult': 1.0})
    print("   ✅ ZIP Head: Unfrozen")

    # --- C. Sblocca CLIP DECODER (Stage 2 Head) ---
    # Nel CLIPEBCModel, le parti allenabili sono image_decoder, projection, ecc.
    # Il backbone (visual_encoder) resta congelato o sbloccato parzialmente.
    clip_model = model.stage2
    
    # Sblocca Decoder e Proiezione
    for module_name in ['image_decoder', 'projection', 'logit_scale']:
        if hasattr(clip_model, module_name):
            mod = getattr(clip_model, module_name)
            if isinstance(mod, torch.Tensor): # logit_scale a volte è un parametro diretto
                mod.requires_grad = True
                trainable_params.append({'params': [mod], 'lr_mult': 1.0})
            else:
                for p in mod.parameters():
                    p.requires_grad = True
                trainable_params.append({'params': mod.parameters(), 'lr_mult': 1.0})
            print(f"   ✅ CLIP {module_name}: Unfrozen")

    # --- D. Backbone (Opzionale: Partial Unfreeze) ---
    # P2R suggerisce di sbloccare solo gli ultimissimi layer del backbone se necessario.
    # Per ora lo lasciamo congelato per stabilità, dato che SHA è piccolo.
    print("   🔒 Backbones: Frozen (Safety for small datasets)")
    
    return trainable_params

def crowd_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'density': torch.stack([item['density'] for item in batch]),
        'points': [item['points'] for item in batch],
    }

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/config_sha.yaml")
    parser.add_argument('--s1', type=str, required=True, help="Path best model Stage 1")
    parser.add_argument('--s2', type=str, required=True, help="Path best model Stage 2")
    parser.add_argument('--out', type=str, default="checkpoints/sha/stage3")
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}')
    os.makedirs(args.out, exist_ok=True)
    

     # --- SALVATAGGIO CONFIG ---
    # Salva una copia esatta del config usato per questo training
    saved_config_path = os.path.join(args.out, "config.yaml")
    shutil.copy(args.config, saved_config_path)
    print(f"📄 Configuration saved to: {saved_config_path}")
    # -------------------------------------------------

    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    # Estrazione dinamica dal config (TRAIN_STAGE3)
    t3 = config.get("TRAIN_STAGE3", {})
    EPOCHS = int(t3.get("EPOCHS", 400))
    BASE_LR = float(t3.get("LR", 1.0e-5))
    
    # Lambda per le Loss
    l_zip_w = float(t3.get("LAMBDA_ZIP", 0.5))
    l_clip_w = float(t3.get("LAMBDA_CLIP", 1.0))
    l_cons_w = float(t3.get("LAMBDA_CONS", 0.2))
    
    # Parametri Steepness
    s_start = float(t3.get("STEEPNESS_START", 1.0))
    s_end = float(t3.get("STEEPNESS_END", 20.0))
    
    # Soglia Maschera
    mask_eps = float(t3.get("MASK_EPS", 0.001))

    print(f"🚀 Stage 3 REFINED | Epochs: {EPOCHS} | LR: {BASE_LR}")
    print(f"⚖️  Loss Weights: CLIP={l_clip_w}, ZIP={l_zip_w}, CONS={l_cons_w}")

    
    
    # --- 1. CARICAMENTO MODELLI ---
    print("📦 Loading Models...")
    
    # Stage 1: ZIPModel (Usa la classe dal file zip_model.py)
    stage1 = ZIPModel(config).to(device)
    ckpt1 = torch.load(args.s1, map_location=device)
    stage1.load_state_dict(ckpt1['model'] if 'model' in ckpt1 else ckpt1, strict=False)
    print("   -> Stage 1 Loaded")

    # Stage 2: CLIPEBCModel (Usa la classe dal file clip_ebc_model.py)
    # Forziamo parametri se necessario
    if 'CLIP_EBC_HEAD' not in config: config['CLIP_EBC_HEAD'] = {}
    
    # Imposta il DECODER_DIM solo se non è già presente nel config
    if 'DECODER_DIM' not in config['CLIP_EBC_HEAD']:
        if 'RN' in config['CLIP_EBC_HEAD'].get('CLIP_MODEL', ''):
            config['CLIP_EBC_HEAD']['DECODER_DIM'] = 2048
        else:
            config['CLIP_EBC_HEAD']['DECODER_DIM'] = 768
    
    stage2 = CLIPEBCModel(config).to(device)
    ckpt2 = torch.load(args.s2, map_location=device)
    stage2.load_state_dict(ckpt2['model'] if 'model' in ckpt2 else ckpt2, strict=False)
    print("   -> Stage 2 Loaded")

    # Joint Model (Wrapper)
    # Iniziamo con steepness=1.0 per Soft Gating
    model = ZIPCLIPJointModel(stage1, stage2, steepness=1.0).to(device)

    # --- 2. OPTIMIZER & FREEZING ---
    # Ottimizziamo solo le teste e l'aligner
    optim_groups = freeze_parameters(model)
    
    # Costruiamo lista piatta per l'optimizer
    final_groups = []
    for g in optim_groups:
        final_groups.append({'params': g['params'], 'lr': BASE_LR * g['lr_mult']})

    optimizer = AdamW(final_groups, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = GradScaler('cuda')

    # --- 3. LOSSES ---
    clip_loss_fn = CLIPEBCLoss(
        bins=config['BINS'],
        input_size=config['DATA']['CROP_SIZE'],
        reduction=config['CLIP_EBC_HEAD']['REDUCTION'], # Spesso 16 o 8
        weight_ot=0.1, weight_tv=0.01, weight_count=1.0
    ).to(device)
    
    # ZIP Loss: BCE semplice sulla maschera è più stabile della NLL in questa fase
    zip_loss_fn = nn.BCEWithLogitsLoss()

    # --- 4. DATA ---
    train_ds = SHA(config['DATA']['ROOT'], 'train', build_transforms(config['DATA'], True))
    val_ds = SHA(config['DATA']['ROOT'], 'val', build_transforms(config['DATA'], False))
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, collate_fn=crowd_collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=crowd_collate)

    # --- 5. LOOP ---
    best_mae = float('inf')
    epochs = config['TRAIN_STAGE3']['EPOCHS']

    for epoch in range(EPOCHS):
        model.train()
        
        # STEEPNESS ANNEALING DINAMICO
        current_steepness = s_start + (epoch / EPOCHS) * (s_end - s_start)
        model.steepness = current_steepness
        pbar = tqdm(train_loader, desc=f"Ep {epoch}")
        epoch_loss = 0
        
        for batch in pbar:
            if batch is None: continue
            imgs = batch['image'].to(device)
            gt_density = batch['density'].to(device)
            points = [p.to(device) for p in batch['points']]
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                out = model(imgs)
                
                # 1. CLIP Loss
                l_clip, _ = clip_loss_fn(out['ebc_logits'], out['final_density'], gt_density, points)
                
                # 2. ZIP Loss con soglia da config
                pi_logits = out['pi_logits']
                with torch.no_grad():
                    gt_resized = F.interpolate(gt_density, size=pi_logits.shape[-2:], mode='bilinear')
                    gt_resized = gt_resized * ((gt_density.shape[-1]/pi_logits.shape[-1])**2)
                    mask_target = (gt_resized > mask_eps).float() # USA MASK_EPS
                
                l_zip = zip_loss_fn(pi_logits, mask_target)
                
                # 3. Consistency Regularization
                clip_raw_density = out['raw_density'].detach()
                prob_zip = out['pi_prob']
                l_cons = (clip_raw_density * (1 - prob_zip)).mean()

                # TOTALE PESATA DAI LAMBDA DELLO YAML
                loss = (l_clip_w * l_clip) + (l_zip_w * l_zip) + (l_cons_w * l_cons)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'L': f"{loss.item():.2f}", 'L_ZIP': f"{l_zip.item():.2f}"})

        # --- VALIDATION ---
        model.eval()
        model.steepness = 20.0 # Hard Gating in validation (Pulisce tutto il background)
        
        val_mae = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                points = batch['points']
                out = model(imgs)
                
                # Somma della mappa finale
                pred = out['final_density'].sum().item()
                gt = len(points[0])
                val_mae += abs(pred - gt)
                
        val_mae /= len(val_loader)
        scheduler.step(val_mae)
        
        print(f"📊 Val MAE: {val_mae:.2f} (Best: {best_mae:.2f})")

        if val_mae < best_mae:
            best_mae = val_mae
            save_dict = {
                'model': model.state_dict(),
                'epoch': epoch,
                'mae': val_mae,
                'config': config
            }
            torch.save(save_dict, os.path.join(args.out, "best_model.pth"))
            print(f"🌟 Saved Best Model (MAE: {val_mae:.2f})")

if __name__ == "__main__":
    main()