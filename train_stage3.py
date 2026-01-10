#!/usr/bin/env python3
"""
Train Stage 3: ZIP-CLIP Joint Fine-Tuning (FIXED & FINAL)
=========================================================
Fix del bug "Generator Consumed" nell'Optimizer.
Ora i parametri vengono passati correttamente come liste persistenti.
"""

import os
import sys
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

# Aggiungi path corrente
sys.path.append(os.getcwd())

# --- IMPORTAZIONI MODELLI ---
from models.joint_model import ZIPCLIPJointModel
from models.clip_ebc_model import CLIPEBCModel
from models.backbone import build_backbone
from models.pi_head import ZIPHead
from losses.joint_loss import JointLoss
from losses.clip_ebc_loss import CLIPEBCLoss 
from datasets.sha import SHA
from datasets.transforms import build_transforms
from eval_patchwise import patchwise_count

# --- CLASSE WRAPPER SICURA ---
class SafeZIPModel(nn.Module):
    """Assicura compatibilità chiavi (pi_logits)."""
    def __init__(self, config):
        super().__init__()
        self.backbone = build_backbone(config)
        
        if hasattr(self.backbone, 'out_channels'):
            in_channels = self.backbone.out_channels
        else:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 256, 256)
                in_channels = self.backbone(dummy).shape[1]

        zip_cfg = config.get('ZIP_HEAD', {})
        hidden_dim = zip_cfg.get('HIDDEN_DIM', 256)
        self.pi_head = ZIPHead(in_channels=in_channels, hidden_dim=hidden_dim)

    def forward(self, x):
        features = self.backbone(x)
        out = self.pi_head(features)
        
        if isinstance(out, dict):
            if 'pi_logits' in out: return out
            # Adattatori per chiavi diverse
            if 'logit_pi' in out: out['pi_logits'] = out['logit_pi']
            elif 'logits' in out: out['pi_logits'] = out['logits']
            else: out['pi_logits'] = list(out.values())[0]
            return out
        return {'pi_logits': out}

# ==============================================================================
# UTILITIES
# ==============================================================================
def smart_load_state_dict(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint non trovato: {checkpoint_path}")
        return False

    print(f"📂 Loading: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        if 'clip_model_full' in name:
            name = name.replace('clip_model_full', 'clip_model')
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    print("   ✅ Weights loaded.")
    return True

def crowd_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'density': torch.stack([item['density'] for item in batch]),
        'points': [item['points'] for item in batch],
        'img_path': [item['img_path'] for item in batch]
    }

def validate_sliding_window(model, loader, device, window_size=448, stride=448):
    model.eval()
    mae_sum = 0
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            if batch is None: continue
            img_full = batch['image'].to(device)
            gt_points = len(batch['points'][0])
            B, C, H, W = img_full.shape
            
            density_map = torch.zeros((H, W), device=device)
            count_map = torch.zeros((H, W), device=device)
            
            for y in range(0, H, stride):
                for x in range(0, W, stride):
                    y_end = min(y + window_size, H)
                    x_end = min(x + window_size, W)
                    y_start = max(y_end - window_size, 0)
                    x_start = max(x_end - window_size, 0)
                    
                    crop = img_full[:, :, y_start:y_end, x_start:x_end]
                    out = model(crop)
                    crop_d = out['final_density'] if isinstance(out, dict) else out
                    
                    if crop_d.shape[-2:] != crop.shape[-2:]:
                        crop_d = F.interpolate(crop_d, size=crop.shape[-2:], mode='bilinear', align_corners=False)
                    
                    density_map[y_start:y_end, x_start:x_end] += crop_d.squeeze()
                    count_map[y_start:y_end, x_start:x_end] += 1.0
            
            final_pred = (density_map / count_map).sum().item()
            mae_sum += abs(final_pred - gt_points)
            count += 1
            
    return mae_sum / count

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/config_shb.yaml")
    parser.add_argument('--s1', type=str, required=True)
    parser.add_argument('--s2', type=str, required=True)
    parser.add_argument('--out', type=str, default="checkpoints/shb/stage3_final")
    parser.add_argument('--epochs', type=int, default=None) 
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}')
    os.makedirs(args.out, exist_ok=True)
    
    print(f"🔧 Config: {args.config} | GPU: {args.gpu}")
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    
    t3_conf = config.get('TRAIN_STAGE3', {})
    epochs = args.epochs if args.epochs is not None else t3_conf.get('EPOCHS', 50)
    
    # Parametri
    l_zip = float(t3_conf.get('LAMBDA_ZIP', 1.0))
    l_clip = float(t3_conf.get('LAMBDA_CLIP', 1.0))
    l_count = float(t3_conf.get('LAMBDA_COUNT', 100.0))
    lr_base = float(t3_conf.get('LR', 1e-5))
    lr_clip = float(t3_conf.get('LR_CLIP', 1e-6))
    lr_aligner = float(t3_conf.get('LR_ALIGNER', 1e-4))
    wd = float(t3_conf.get('WEIGHT_DECAY', 1e-4))

    print(f"⚙️  PARAMS: Ep={epochs} | LR_Aligner={lr_aligner} | L_Count={l_count}")

    # --- 1. BUILD MODELS ---
    # Gestione VGG/ResNet mismatch nel config
    if config['BACKBONE']['TYPE'] == 'vgg16_bn':
        print("      ⚠️ Config VGG rilevato. Se usi checkpoint ResNet, questo verrà gestito.")
        # Non forziamo qui, lasciamo che Smart Load provi a caricare
        
    stage1_full = SafeZIPModel(config).to(device)
    smart_load_state_dict(stage1_full, args.s1, device)

    # Forziamo Decoder 2048 per Stage 2
    if 'CLIP_EBC_HEAD' not in config: config['CLIP_EBC_HEAD'] = {}
    config['CLIP_EBC_HEAD']['DECODER_DIM'] = 2048 
    
    stage2_model = CLIPEBCModel(config).to(device)
    smart_load_state_dict(stage2_model, args.s2, device)

    # Joint Model
    model = ZIPCLIPJointModel(stage1_full, stage2_model, steepness=10.0).to(device)
    
    # --- 2. OPTIMIZER (CORRETTO) ---
    print("🔓 Sblocco parametri e creazione gruppi...")
    for param in model.parameters():
        param.requires_grad = True

    # [FIX] Usiamo list() per consumare i generatori e renderli liste vere
    p_aligner = list(model.mask_aligner.parameters()) if hasattr(model, 'mask_aligner') else []
    p_stage1 = list(model.stage1.parameters())
    p_stage2 = list(model.stage2.parameters())

    optimizer_grouped_parameters = [
        {'params': p_aligner, 'lr': lr_aligner},
        {'params': p_stage1, 'lr': lr_base},
        {'params': p_stage2, 'lr': lr_clip}
    ]
    # Filtriamo i gruppi vuoti basandoci sulle liste
    optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if len(g['params']) > 0]

    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=wd)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler('cuda', enabled=True)

    # --- 3. LOSS & DATA ---
    crop_size = config['DATA'].get('CROP_SIZE', 448)
    reduction = config['CLIP_EBC_HEAD'].get('REDUCTION', 16)

    clip_loss_fn = CLIPEBCLoss(
        bins=config['BINS'],
        input_size=crop_size,
        reduction=reduction,
        weight_ot=0.1,
        weight_tv=0.01
    ).to(device)
    
    criterion = JointLoss(
        clip_loss_fn=clip_loss_fn,
        lambda_zip=l_zip,
        lambda_clip=l_clip,
        lambda_count=l_count
    ).to(device)
    
    train_ds = SHA(config['DATA']['ROOT'], 'train', build_transforms(config['DATA'], True))
    val_ds = SHA(config['DATA']['ROOT'], 'val', build_transforms(config['DATA'], False))
    
    train_loader = DataLoader(train_ds, batch_size=t3_conf.get('BATCH_SIZE', 4), shuffle=True, num_workers=4, collate_fn=crowd_collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=crowd_collate)
    
    # --- 4. TRAIN LOOP ---
    best_mae = float('inf')
    print("🚀 Inizio Training Stage 3...")
    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{epochs}")
        
        for batch in pbar:
            if batch is None: continue
            
            images = batch['image'].to(device)
            gt_density = batch['density'].to(device)
            points = [p.to(device) for p in batch['points']] 
            
            optimizer.zero_grad()
            
            with autocast('cuda', enabled=True):
                outputs = model(images)
                
                # Maschera ZIP Target
                out_h, out_w = outputs['pi_logits'].shape[-2:]
                if gt_density.shape[-1] != out_w:
                     gt_resized = F.interpolate(gt_density, size=(out_h, out_w), mode='bilinear', align_corners=False)
                     scale = (gt_density.shape[-1] / out_w)**2
                     gt_resized = gt_resized * scale
                else:
                     gt_resized = gt_density
                mask_gt = (gt_resized > 0.001).float()
                
                targets = {
                    'mask': mask_gt,
                    'density': gt_density,
                    'points': points,
                    'counts': gt_resized # Uso la density ridimensionata come count proxy
                }
                
                loss, loss_dict = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            
            # [FIX EXTRA] Unscale prima di clip e step per sicurezza
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            l_cnt = loss_dict.get('l_count', loss_dict.get('l_final', 0.0))
            pbar.set_postfix({'L': f"{loss.item():.2f}", 'Count': f"{l_cnt:.2f}"})
        
        scheduler.step()
        
        # Validation
        val_mae = validate_sliding_window(model, val_loader, device)
        print(f"📊 Ep {epoch+1} | Val MAE: {val_mae:.2f} (Best: {best_mae:.2f})")
        
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'mae': best_mae
            }, os.path.join(args.out, "best_model.pth"))
            print("🌟 Saved Best!")

if __name__ == "__main__":
    main()