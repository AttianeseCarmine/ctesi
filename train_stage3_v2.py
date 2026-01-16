#!/usr/bin/env python3
"""
Train Stage 3: ZIP-CLIP Joint Fine-Tuning (Refined Strategy)
============================================================
Miglioramenti:
1. Steepness Annealing: La maschera diventa più "dura" progressivamente.
2. Partial Unfreeze: Sblocca Layer4 del backbone.
3. Consistency Loss: Penalizza se ZIP maschera zone dove CLIP vede folla.
"""

import os
import yaml
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import shutil
# --- IMPORTS ---
from models.joint_model import ZIPCLIPJointModel
from models.clip_ebc_model import CLIPEBCModel
from models.zip_model import ZIPModel
from losses.clip_ebc_loss import CLIPEBCLoss 
from datasets.sha import SHA
from datasets.transforms import build_transforms

# ==============================================================================
# UTILS
# ==============================================================================
def freeze_parameters_refined(model):
    """
    Strategia Avanzata: Sblocca Aligner, Teste e Layer4 del Backbone.
    """
    print("\n🔒 Freezing Strategy: Refined (Backbone Layer4 Unfrozen)")
    
    # 1. Congela TUTTO
    for param in model.parameters():
        param.requires_grad = False

    trainable_params = []

    # --- A. Aligner & Heads ---
    head_params = []
    head_params += list(model.mask_aligner.parameters())
    head_params += list(model.stage1.zip_head.parameters())
    
    # CLIP Heads (Decoder + Projection)
    clip = model.stage2
    for mod_name in ['image_decoder', 'projection', 'logit_scale']:
        if hasattr(clip, mod_name):
            mod = getattr(clip, mod_name)
            if isinstance(mod, torch.Tensor):
                mod.requires_grad = True
                head_params.append(mod)
            else:
                for p in mod.parameters():
                    p.requires_grad = True
                    head_params.append(p)
    
    trainable_params.append({'params': head_params, 'lr_scale': 1.0})
    print("   ✅ Heads & Aligner: Unfrozen")

    # --- B. Backbone Layer 4 (ResNet) ---
    # Sblocchiamo solo l'ultimo blocco convoluzionale per adattare le feature
    backbone_params = []
    if hasattr(model.stage1.backbone, 'features') and hasattr(model.stage1.backbone.features, 'layer4'):
        # Caso ResNet standard
        for p in model.stage1.backbone.features.layer4.parameters():
            p.requires_grad = True
            backbone_params.append(p)
        print("   ✅ ZIP Backbone Layer4: Unfrozen")
    elif hasattr(model.stage1.backbone, 'layer4'):
         # Caso ResNet custom
        for p in model.stage1.backbone.layer4.parameters():
            p.requires_grad = True
            backbone_params.append(p)
        print("   ✅ ZIP Backbone Layer4: Unfrozen")

    trainable_params.append({'params': backbone_params, 'lr_scale': 0.1}) # LR più basso per il backbone

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
    parser.add_argument('--s1', type=str, default="checkpoints/sha/stage1/best_model.pth")
    parser.add_argument('--s2', type=str, default="checkpoints/sha/stage2/best_model.pth")
    parser.add_argument('--out', type=str, default="checkpoints/sha/stage3_refined")
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()  # <--- Qui è definito come args

    device = torch.device(f'cuda:{args.gpu}')
    os.makedirs(args.out, exist_ok=True)
    
    # --- CORREZIONE QUI ---
    # Sostituisci 'cmd_args' con 'args' per coerenza con sopra
    saved_config_path = os.path.join(args.out, "config.yaml") 
    shutil.copy(args.config, saved_config_path)
    print(f"📄 Configuration saved to: {saved_config_path}")

    with open(args.config, 'r') as f: config = yaml.safe_load(f)

    # Parametri Refined
    EPOCHS = 60 # Meno epoche ma più intense
    BASE_LR = 5e-5
    
    print(f"🚀 Stage 3 REFINED | {args.config}")
    
    # --- 1. LOAD MODELS ---
    print("📦 Loading Models...")
    stage1 = ZIPModel(config).to(device)
    ckpt1 = torch.load(args.s1, map_location=device)
    # Controlla se 'model' è una chiave, altrimenti usa l'intero oggetto
    if isinstance(ckpt1, dict) and 'model' in ckpt1:
        stage1.load_state_dict(ckpt1['model'], strict=False)
    else:
        stage1.load_state_dict(ckpt1, strict=False)
    print("   -> Stage 1 Loaded")
    
    # Config hack per CLIP se necessario
    if 'CLIP_EBC_HEAD' not in config: config['CLIP_EBC_HEAD'] = {}
    config['CLIP_EBC_HEAD']['DECODER_DIM'] = 2048 
    
    stage2 = CLIPEBCModel(config).to(device)
    ckpt2 = torch.load(args.s2, map_location=device)
    # Stesso controllo per Stage 2
    if isinstance(ckpt2, dict) and 'model' in ckpt2:
        stage2.load_state_dict(ckpt2['model'], strict=False)
    else:
        stage2.load_state_dict(ckpt2, strict=False)
    print("   -> Stage 2 Loaded")

    # Inizializza con steepness bassa
    model = ZIPCLIPJointModel(stage1, stage2, steepness=1.0).to(device)

    # --- 2. OPTIMIZER ---
    param_groups = freeze_parameters_refined(model)
    final_groups = []
    for g in param_groups:
        final_groups.append({'params': g['params'], 'lr': BASE_LR * g['lr_scale']})

    optimizer = AdamW(final_groups, weight_decay=1e-4)
    
    # OneCycleLR per convergenza migliore
    train_ds = SHA(config['DATA']['ROOT'], 'train', build_transforms(config['DATA'], True))
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, collate_fn=crowd_collate, drop_last=True)
    
    scheduler = OneCycleLR(optimizer, max_lr=BASE_LR, steps_per_epoch=len(train_loader), epochs=EPOCHS, pct_start=0.1)
    scaler = GradScaler('cuda')

    # --- 3. LOSSES ---
    clip_loss_fn = CLIPEBCLoss(
        bins=config['BINS'],
        input_size=config['DATA']['CROP_SIZE'],
        reduction=config['CLIP_EBC_HEAD']['REDUCTION'],
        weight_ot=0.1, weight_tv=0.01, weight_count=1.0
    ).to(device)
    
    zip_loss_fn = nn.BCEWithLogitsLoss()

    val_ds = SHA(config['DATA']['ROOT'], 'val', build_transforms(config['DATA'], False))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=crowd_collate)

    best_mae = float('inf')

    # --- 4. TRAINING LOOP ---
    for epoch in range(EPOCHS):
        model.train()
        
        # STEEPNESS ANNEALING: Da 1.0 a 20.0 progressivamente
        # Questo aiuta il modello ad abituarsi alla maschera binaria
        current_steepness = 1.0 + (epoch / EPOCHS) * 19.0
        model.steepness = current_steepness
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch} (Steep={current_steepness:.1f})")
        
        for batch in pbar:
            if batch is None: continue
            imgs = batch['image'].to(device)
            gt_density = batch['density'].to(device)
            points = [p.to(device) for p in batch['points']]
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                # Forward
                out = model(imgs)
                
                # 1. CLIP Loss (sul risultato finale già mascherato)
                l_clip, _ = clip_loss_fn(
                    out['ebc_logits'], 
                    out['final_density'], # Density mascherata
                    gt_density, 
                    points
                )
                
                # 2. ZIP Loss (Supervisione Diretta Maschera)
                pi_logits = out['pi_logits']
                h, w = pi_logits.shape[-2:]
                
                # Crea target maschera dalla densità GT (downsampled)
                with torch.no_grad():
                    gt_resized = F.interpolate(gt_density, size=(h, w), mode='bilinear')
                    # Normalizza per area per non perdere picchi
                    gt_resized = gt_resized * ((gt_density.shape[-1]/w)**2)
                    mask_target = (gt_resized > 0.005).float() # Soglia leggermente più alta per pulizia
                
                l_zip = zip_loss_fn(pi_logits, mask_target)
                
                # 3. Consistency Regularization (Nuovo!)
                # Se CLIP predice alta densità, ZIP non dovrebbe predire 0.
                # Penalizziamo: Density_CLIP * (1 - Prob_ZIP)
                clip_raw_density = out['raw_density'].detach() # Non backproghiamo su clip qui
                prob_zip = out['pi_prob']
                l_cons = (clip_raw_density * (1 - prob_zip)).mean()

                # Totale
                loss = l_clip + (0.5 * l_zip) + (0.1 * l_cons)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            pbar.set_postfix({'L': f"{loss.item():.2f}", 'Clp': f"{l_clip.item():.2f}", 'Zip': f"{l_zip.item():.2f}"})

        # --- VALIDATION ---
        model.eval()
        model.steepness = 20.0 # Hard Gating per validazione
        
        val_mae = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                points = batch['points']
                out = model(imgs)
                
                pred = out['final_density'].sum().item()
                gt = len(points[0])
                val_mae += abs(pred - gt)
                
        val_mae /= len(val_loader)
        
        print(f"📊 Val MAE: {val_mae:.2f} (Best: {best_mae:.2f})")
        
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({'model': model.state_dict(), 'epoch': epoch}, os.path.join(args.out, "best_model.pth"))
            print("🌟 Saved Best Refined")

if __name__ == "__main__":
    main()