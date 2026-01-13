#!/usr/bin/env python3
"""
Evaluate Stage 3: Joint Model Evaluation (Hard Gating Strategy)
===============================================================
Loads the trained ZIP-CLIP Joint Model and evaluates it on the test set.
Uses 'Steepness' control to enforce Hard Gating (cleaning background noise).
"""

import os
import yaml
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

# --- IMPORTS ---
from models.zip_model import ZIPModel
from models.clip_ebc_model import CLIPEBCModel
from models.joint_model import ZIPCLIPJointModel
from datasets.sha import SHA  # Or SHB, depends on config
from datasets.transforms import build_transforms

def sliding_window_predict(model, image, window_size=448, stride=448, device='cuda'):
    """
    Performs sliding window prediction to handle large images and maintain
    feature resolution. Recombines the density map at the end.
    """
    model.eval()
    B, C, H, W = image.shape
    
    # Init canvas for density map
    density_map = torch.zeros((H, W), device=device)
    count_map = torch.zeros((H, W), device=device) # To average overlapping regions
    
    with torch.no_grad():
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y_end = min(y + window_size, H)
                x_end = min(x + window_size, W)
                
                # Adjust start to ensure crop is exactly window_size (if possible)
                y_start = max(y_end - window_size, 0)
                x_start = max(x_end - window_size, 0)
                
                crop = image[:, :, y_start:y_end, x_start:x_end].to(device)
                
                # Forward Pass
                out = model(crop)
                
                # Get final density (already gated by ZIP)
                pred_crop = out['final_density'] # Shape: [1, 1, h, w]
                
                # Accumulate
                density_map[y_start:y_end, x_start:x_end] += pred_crop.squeeze()
                count_map[y_start:y_end, x_start:x_end] += 1.0
                
    # Normalize overlapping regions
    final_density = density_map / count_map
    return final_density

def evaluate(args):
    # 1. Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device} | Dataset: {config['DATASET']}")

    # 2. Build Joint Model
    print("🏗️  Building Joint Model...")
    
    # Initialize sub-models
    stage1 = ZIPModel(config).to(device)
    stage2 = CLIPEBCModel(config).to(device)
    
    # Initialize Joint Model
    # Steepness=20.0 activates HARD GATING (binary-like mask)
    model = ZIPCLIPJointModel(stage1, stage2, steepness=20.0).to(device)
    
    # 3. Load Checkpoint
    print(f"📥 Loading Checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle different saving formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0:
        print(f"⚠️  Missing keys (usually safe if just aux heads): {len(missing)}")
        # print(missing) 
    else:
        print("✅ Weights loaded perfectly.")

    # 4. Data
    # Ensure val set is used
    dataset_name = config.get('DATASET', 'sha').lower()
    root_dir = config['DATA']['ROOT']
    
    # Transform: False = No Augmentation (Just Resize/Normalize)
    val_transforms = build_transforms(config['DATA'], is_train=False) 
    
    if 'sha' in dataset_name or 'shb' in dataset_name:
        dataset = SHA(root_dir, 'val', val_transforms)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported in this script yet.")
        
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # 5. Evaluation Loop
    mae_accum = 0.0
    mse_accum = 0.0
    model.eval()
    
    print(f"🚀 Starting Evaluation (Hard Gating: Steepness={model.steepness})...")
    
    pbar = tqdm(loader)
    for batch in pbar:
        img = batch['image']
        # Points is a list of tensors because number of points varies
        gt_count = len(batch['points'][0]) 
        
        # Strategy: Sliding Window vs Full Image
        # If image is too large, use sliding window
        if img.shape[2] > 1024 or img.shape[3] > 1024:
            pred_density = sliding_window_predict(model, img, device=device)
            pred_count = pred_density.sum().item()
        else:
            img = img.to(device)
            with torch.no_grad():
                out = model(img)
                pred_count = out['final_density'].sum().item()
        
        # Metrics
        error = abs(pred_count - gt_count)
        mae_accum += error
        mse_accum += error ** 2
        
        pbar.set_postfix({'GT': gt_count, 'Pred': f"{pred_count:.1f}", 'Err': f"{error:.1f}"})

    # 6. Final Results
    final_mae = mae_accum / len(dataset)
    final_mse = (mse_accum / len(dataset)) ** 0.5
    
    print("\n" + "="*40)
    print(f"🏆 FINAL RESULTS: {config['DATASET'].upper()}")
    print(f"   MAE: {final_mae:.2f}")
    print(f"   RMSE: {final_mse:.2f}")
    print("="*40 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/config_shb.yaml", help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best_model.pth from Stage 3')
    args = parser.parse_args()
    
    evaluate(args)