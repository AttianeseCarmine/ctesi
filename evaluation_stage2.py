#!/usr/bin/env python3
"""
============================================================
EVALUATION OFFICIAL STYLE (Sliding Window)
============================================================
Replica la logica di 'sliding_window_predict' del repo ufficiale.
Gestisce immagini di grandi dimensioni e sovrapposizioni.

Usage:
    python evaluation_stage2.py --config configs/config_sha.yaml --checkpoint checkpoints/sha/stage2/best_model.pth
        python evaluation_stage2.py --config configs/config_shb.yaml --checkpoint checkpoints/shb/stage2/best_model.pth
============================================================
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
import argparse
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

# Imports dai tuoi file
from models.clip_ebc_model import CLIPEBCModel
from datasets.sha import SHA
from datasets.transforms import build_transforms

# ============================================================
# OFFICIAL UTILS (Adapted)
# ============================================================

def sliding_window_predict(
    model: nn.Module,
    image: torch.Tensor,
    window_size: tuple,
    stride: tuple,
) -> torch.Tensor:
    """
    Genera la density map usando sliding window con media delle sovrapposizioni.
    Adattato da utils/eval_utils.py del repo ufficiale.
    """
    assert len(image.shape) == 4, f"Image must be (1, c, h, w), got {image.shape}"
    
    image_height, image_width = image.shape[-2:]
    window_height, window_width = window_size
    stride_height, stride_width = stride
    
    # Calcola righe e colonne necessarie
    num_rows = int(np.ceil((image_height - window_height) / stride_height) + 1)
    num_cols = int(np.ceil((image_width - window_width) / stride_width) + 1)
    
    # Reduction factor del modello (es. 16)
    reduction = model.reduction if hasattr(model, "reduction") else 16
    
    windows = []
    # 1. Estrai tutte le finestre
    for i in range(num_rows):
        for j in range(num_cols):
            x_start = i * stride_height
            y_start = j * stride_width
            x_end = x_start + window_height
            y_end = y_start + window_width
            
            # Gestione bordi (se esce fuori, torna indietro)
            if x_end > image_height:
                x_start = image_height - window_height
                x_end = image_height
            if y_end > image_width:
                y_start = image_width - window_width
                y_end = image_width
                
            window = image[:, :, x_start:x_end, y_start:y_end]
            windows.append(window)
            
    # 2. Batch Processing
    # Attenzione: se l'immagine è enorme, potresti dover processare a chunk
    # Qui assumiamo che le finestre stiano in memoria GPU
    windows = torch.cat(windows, dim=0).to(image.device) # (num_windows, c, h, w)
    
    model.eval()
    with torch.no_grad():
        # Il tuo modello ritorna un dict
        outputs = model(windows)
        preds = outputs['ebc_density'] # [num_windows, 1, h/16, w/16]
        
    preds = preds.cpu().detach().numpy()
    
    # 3. Re-assemble Density Map
    out_h = image_height // reduction
    out_w = image_width // reduction
    
    pred_map = np.zeros((1, out_h, out_w), dtype=np.float32)
    count_map = np.zeros((1, out_h, out_w), dtype=np.float32)
    
    idx = 0
    for i in range(num_rows):
        for j in range(num_cols):
            x_start = i * stride_height
            y_start = j * stride_width
            x_end = x_start + window_height
            y_end = y_start + window_width
            
            if x_end > image_height:
                x_start = image_height - window_height
                x_end = image_height
            if y_end > image_width:
                y_start = image_width - window_width
                y_end = image_width
            
            # Coordinate nello spazio ridotto (feature map)
            sx_start, sx_end = x_start // reduction, x_end // reduction
            sy_start, sy_end = y_start // reduction, y_end // reduction
            
            # Accumula predizioni e conta sovrapposizioni
            pred_map[:, sx_start:sx_end, sy_start:sy_end] += preds[idx, 0, :, :]
            count_map[:, sx_start:sx_end, sy_start:sy_end] += 1.0
            idx += 1
            
    # Media sulle sovrapposizioni
    pred_map /= count_map
    
    return torch.tensor(pred_map).unsqueeze(0) # [1, 1, H, W]

# ============================================================
# MAIN EVALUATION
# ============================================================

def crowd_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'density': torch.stack([item['density'] for item in batch]),
        'points': [item['points'] for item in batch],
        'img_path': [item['img_path'] for item in batch]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_sha.yaml')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--gpu', default=0, type=int)
    # Parametri Sliding Window (Default dal paper per VGG/ResNet)
    parser.add_argument('--window_size', default=448, type=int) 
    parser.add_argument('--stride', default=224, type=int, help="Stride < WindowSize crea sovrapposizione")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    device = torch.device(f'cuda:{args.gpu}')
    
    print("🏗️  Building Model...")
    model = CLIPEBCModel(config).to(device)
    
    print(f"📥 Loading Checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'model' in ckpt: ckpt = ckpt['model']
    model.load_state_dict(ckpt, strict=False)
    
    # Dataset Val
    val_dataset = SHA(config['DATA']['ROOT'], 'val', build_transforms(config['DATA'], False))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=crowd_collate)
    
    mae_acc = 0.0
    mse_acc = 0.0
    total_pred = 0.0
    total_gt = 0.0
    
    print(f"\n🚀 Starting Sliding Window Evaluation")
    print(f"   Window: {args.window_size}x{args.window_size}")
    print(f"   Stride: {args.stride}x{args.stride} (Overlap: {args.window_size - args.stride})")
    print("-" * 50)
    
    model.eval()
    
    for batch in tqdm(val_loader):
        if batch is None: continue
        
        img = batch['image'].to(device)
        points = batch['points'][0]
        gt_count = len(points)
        
        # Gestione immagini piccole
        _, _, H, W = img.shape
        if H < args.window_size or W < args.window_size:
            # Resize o pad minimo
            pad_h = max(0, args.window_size - H)
            pad_w = max(0, args.window_size - W)
            if pad_h > 0 or pad_w > 0:
                img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))
        
        # Sliding Window Predict
        # Nota: window_size e stride devono essere tuple
        ws = (args.window_size, args.window_size)
        st = (args.stride, args.stride)
        
        pred_map = sliding_window_predict(model, img, ws, st)
        
        pred_count = pred_map.sum().item()
        
        # Metriche
        mae_acc += abs(pred_count - gt_count)
        mse_acc += (pred_count - gt_count) ** 2
        total_pred += pred_count
        total_gt += gt_count
        
    final_mae = mae_acc / len(val_dataset)
    final_rmse = (mse_acc / len(val_dataset)) ** 0.5
    
    print("\n" + "="*50)
    print("📊 OFFICIAL STYLE RESULTS")
    print("="*50)
    print(f"   MAE:  {final_mae:.2f}")
    print(f"   RMSE: {final_rmse:.2f}")
    print(f"   Total GT:   {total_gt:.0f}")
    print(f"   Total Pred: {total_pred:.0f}")
    print("="*50)

if __name__ == '__main__':
    main()