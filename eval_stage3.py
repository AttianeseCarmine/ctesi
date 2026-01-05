#!/usr/bin/env python3
"""
============================================================
EVALUATION STAGE 3 (Joint + Sliding Window)
============================================================
Valuta il modello congiunto (ZIP + CLIP) usando la sliding window.
Fondamentale per vedere il vero guadagno di performance.
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

# Imports
from models.zip_clip_ebc_model import ZIPCLIPEBCModel
from models.clip_ebc_model import CLIPEBCModel
from models.joint_model import ZIPCLIPJointModel
from datasets.sha import SHA
from datasets.transforms import build_transforms

def sliding_window_predict_joint(model, image, window_size, stride):
    """Sliding Window per il modello Joint (ritorna final_density)."""
    image_height, image_width = image.shape[-2:]
    
    # Pad immagine per divisibilità (importante per allineare ZIP e CLIP)
    # CLIP reduction=16, ZIP stride=16. Window deve essere multiplo di 16.
    
    num_rows = int(np.ceil((image_height - window_size[0]) / stride[0]) + 1)
    num_cols = int(np.ceil((image_width - window_size[1]) / stride[1]) + 1)
    
    # Mappe di accumulo
    out_h, out_w = image_height // 16, image_width // 16 # Output size (stride 16)
    
    # Nota: Usiamo una size leggermente abbondante per gestire i bordi, poi croppiamo
    pred_map = torch.zeros((1, 1, out_h + 16, out_w + 16)).cuda()
    count_map = torch.zeros((1, 1, out_h + 16, out_w + 16)).cuda()
    
    model.eval()
    
    with torch.no_grad():
        for i in range(num_rows):
            for j in range(num_cols):
                x_start = i * stride[0]
                y_start = j * stride[1]
                x_end = min(x_start + window_size[0], image_height)
                y_end = min(y_start + window_size[1], image_width)
                
                # Aggiusta start se end tocca il bordo
                if x_end == image_height: x_start = x_end - window_size[0]
                if y_end == image_width: y_start = y_end - window_size[1]
                
                x_start, y_start = max(0, x_start), max(0, y_start)
                
                patch = image[:, :, x_start:x_end, y_start:y_end]
                
                # Forward Joint
                out = model(patch)
                # Output è 1/16 della size del patch
                density = out['final_density'] # [1, 1, h, w]
                
                # Coordinate nella mappa di output
                ox_start, ox_end = x_start // 16, x_start // 16 + density.shape[2]
                oy_start, oy_end = y_start // 16, y_start // 16 + density.shape[3]
                
                pred_map[:, :, ox_start:ox_end, oy_start:oy_end] += density
                count_map[:, :, ox_start:ox_end, oy_start:oy_end] += 1.0
                
    # Media e Crop alla dimensione originale corretta
    final_map = pred_map / (count_map + 1e-6)
    final_map = final_map[:, :, :out_h, :out_w]
    
    return final_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_sha.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    device = torch.device(f'cuda:{args.gpu}')
    
    # Ricostruisci il modello Joint
    print("🏗️  Rebuilding Joint Model...")
    stage1 = ZIPCLIPEBCModel(config).to(device)
    stage2 = CLIPEBCModel(config).to(device)
    model = ZIPCLIPJointModel(stage1, stage2).to(device)
    
    # Carica pesi
    print(f"📥 Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'model' in ckpt: ckpt = ckpt['model']
    model.load_state_dict(ckpt, strict=False)
    
    # Dataset
    val_dataset = SHA(config['DATA']['ROOT'], 'val', build_transforms(config['DATA'], False))
    # Collate function inline per semplicità
    def collate(b): return b[0] if b and b[0] else None
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate)
    
    mae, mse, gt_tot, pred_tot = 0, 0, 0, 0
    
    print("🚀 Evaluating Stage 3...")
    for batch in tqdm(val_loader):
        if batch is None: continue
        img = batch['image'].unsqueeze(0).to(device)
        gt_count = len(batch['points'])
        
        # Sliding Window Predict
        pred_map = sliding_window_predict_joint(model, img, (448, 448), (224, 224))
        pred_count = pred_map.sum().item()
        
        mae += abs(pred_count - gt_count)
        mse += (pred_count - gt_count)**2
        gt_tot += gt_count
        pred_tot += pred_count
        
    print("\n📊 STAGE 3 RESULTS")
    print(f"   MAE:  {mae/len(val_dataset):.2f}")
    print(f"   RMSE: {(mse/len(val_dataset))**.5:.2f}")
    print(f"   Pred: {pred_tot:.0f} (GT: {gt_tot:.0f})")

if __name__ == '__main__':
    main()