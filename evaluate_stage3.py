import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import yaml
import numpy as np
import math

from models.zip_clip_ebc_model import build_model
from losses import build_stage3_loss
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def sliding_window_predict(model, image, window_size=448, stride=128, device="cuda"):
    """
    Sliding window classica con stride denso (128px) per massima precisione.
    """
    B, _, H, W = image.shape
    assert B == 1
    
    # Padding
    if H < window_size or W < window_size:
        pad_h = max(0, window_size - H)
        pad_w = max(0, window_size - W)
        image = F.pad(image, (0, pad_w, 0, pad_h))
        H, W = image.shape[2], image.shape[3]

    downsample_ratio = 16
    h_out, w_out = H // downsample_ratio, W // downsample_ratio
    
    density_map = torch.zeros((1, 1, h_out, w_out), device=device)
    count_map = torch.zeros((1, 1, h_out, w_out), device=device)
    
    h_steps = math.ceil((H - window_size) / stride) + 1
    w_steps = math.ceil((W - window_size) / stride) + 1
    
    for h in range(h_steps):
        for w in range(w_steps):
            y1 = min(h * stride, H - window_size)
            x1 = min(w * stride, W - window_size)
            y2 = y1 + window_size
            x2 = x1 + window_size
            
            crop = image[:, :, y1:y2, x1:x2].to(device)
            
            with torch.no_grad():
                preds = model(crop)
                crop_density = preds["density_map"] 
                
                # NESSUNA MAGIA STRANA QUI.
                # Lasciamo che sia il modello (allenato meglio) a decidere.
                # L'unico aiuto è l'averaging dello sliding window.
                weighted_density = crop_density
            
            y1_out, x1_out = y1 // downsample_ratio, x1 // downsample_ratio
            y2_out = y1_out + crop_density.shape[2]
            x2_out = x1_out + crop_density.shape[3]
            
            density_map[:, :, y1_out:y2_out, x1_out:x2_out] += weighted_density
            count_map[:, :, y1_out:y2_out, x1_out:x2_out] += 1.0
            
    final_density = density_map / torch.clamp(count_map, min=1.0)
    return final_density.sum()

@torch.no_grad()
def validate_stage3(model, criterion, dataloader, device, config, checkpoint_path):
    model.eval()
    
    # IMPORTANTE: Con il nuovo training scale-aware
    if hasattr(model, 'pi_thresh'):
        model.pi_thresh = 0.60 
        print(f"⚡ PI_THRESH: {model.pi_thresh}")

    total_mae = 0.0
    total_mse = 0.0
    zero_mae = 0.0
    zero_samples = 0
    crowd_mae = 0.0
    crowd_samples = 0
    
    w_size = config['DATA']['CROP_SIZE']
    stride = 128 
    
    print(f"\n===== DIAGNOSTICA FINALE (Clean Slide) =====")
    print(f"Window: {w_size} | Stride: {stride}")
    
    progress_bar = tqdm(dataloader, desc="Eval Stage 3")

    for idx, batch in enumerate(progress_bar):
        if isinstance(batch, dict):
            images = batch['image']
            gt_density = batch['density']
        else:
            images, gt_density = batch[0], batch[1]

        gt_density = gt_density.to(device)
        pred_count = sliding_window_predict(model, images, window_size=w_size, stride=stride, device=device).item()
        
        gt_count = gt_density.sum().item()
        err = abs(pred_count - gt_count)
        
        total_mae += err
        total_mse += (pred_count - gt_count) ** 2
        
        if gt_count < 1.0:
            zero_mae += err
            zero_samples += 1
            t = "🟢 ZERO"
        else:
            crowd_mae += err
            crowd_samples += 1
            t = "🔴 CROWD"

        if idx % 20 == 0:
             print(f"[{idx:03d}] {t} | GT: {gt_count:6.1f} | Pred: {pred_count:6.1f} | Err: {err:5.1f}")

    N = len(dataloader.dataset)
    avg_mae = total_mae / N
    avg_rmse = (total_mse / N) ** 0.5
    avg_zero = zero_mae / max(zero_samples, 1)
    avg_crowd = crowd_mae / max(crowd_samples, 1)

    print("\n" + "="*50)
    print(f"🚀 RISULTATI FINALI: {os.path.basename(checkpoint_path)}")
    print(f"Global MAE:  {avg_mae:.3f}")
    print(f"Global RMSE: {avg_rmse:.3f}")
    print(f"Zero MAE:    {avg_zero:.3f}")
    print(f"Crowd MAE:   {avg_crowd:.3f}")
    print("="*50 + "\n")

def main(config_path, checkpoint_path):
    config = load_config(config_path)
    device = torch.device(config['DEVICE'])
    init_seeds(config['SEED'])

    model = build_model(config).to(device)
    
    if not checkpoint_path:
        base_dir = os.path.join(config['EXP']['OUT_DIR'], config['RUN_NAME'], "stage3")
        checkpoint_path = os.path.join(base_dir, "best_stage3_model.pth")
    
    if os.path.exists(checkpoint_path):
        print(f"📂 Loading: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt, strict=False)
    else:
        print("❌ Checkpoint non trovato!")
        return

    criterion = build_stage3_loss(config).to(device)
    DatasetClass = get_dataset(config['DATASET'])
    val_tf = build_transforms(config['DATA'], is_train=False)
    val_dataset = DatasetClass(
        root=config['DATA']['ROOT'],
        split=config['DATA']['VAL_SPLIT'],
        block_size=config['DATA']['ZIP_BLOCK_SIZE'],
        transforms=val_tf,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    validate_stage3(model, criterion, val_loader, device, config, checkpoint_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_sha.yaml")
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()
    main(args.config, args.checkpoint)