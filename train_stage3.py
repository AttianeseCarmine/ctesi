import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import math # <--- Necessario per sqrt

# Imports
from models.zip_model import ZIPModel
from models.clip_ebc_model import CLIPEBCModel
from models.joint_model import ZIPCLIPJointModel
from datasets.builder import build_dataset
from datasets.transforms import build_transforms

# --- FUNZIONE SLIDING WINDOW (Adattata per Joint Model) ---
def sliding_window_predict_joint(model, image, window_size, stride, device='cuda'):
    """
    Versione per Stage 3 (Joint Model).
    """
    model.eval()
    _, _, H, W = image.shape
    
    count_map = torch.zeros((H, W), device=device)
    divisor_map = torch.zeros((H, W), device=device)
    
    with torch.no_grad():
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y_end = min(y + window_size, H)
                x_end = min(x + window_size, W)
                y_start = max(y_end - window_size, 0)
                x_start = max(x_end - window_size, 0)
                
                crop = image[:, :, y_start:y_end, x_start:x_end].to(device)
                
                # Forward Pass Joint Model
                out = model(crop) 
                
                # In Stage 3, ci interessa la densità finale filtrata
                pred_density = out['final_density'] 
                
                # Interpolazione per accumulo
                pred_density = torch.nn.functional.interpolate(
                    pred_density, 
                    size=(crop.shape[2], crop.shape[3]), 
                    mode='bilinear', 
                    align_corners=False
                )
                
                count_map[y_start:y_end, x_start:x_end] += pred_density.squeeze()
                divisor_map[y_start:y_end, x_start:x_end] += 1.0
                
    final_density = count_map / divisor_map
    return final_density.sum().item()

def validate(model, val_loader, device, config):
    model.eval()
    mae_accum = 0.0
    mse_accum = 0.0
    
    # Prende il crop size dal config per adattarsi alla backbone (VGG o ViT)
    crop_size = config['DATA'].get('CROP_SIZE', 448)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating Stage 3"):
            img = batch['image'] # CPU tensor initially
            gt_count = len(batch['points'][0])
            
            # Predizione precisa
            pred_count = sliding_window_predict_joint(
                model, img, 
                window_size=crop_size, 
                stride=crop_size, 
                device=device
            )
            
            # Metriche
            diff = pred_count - gt_count
            mae_accum += abs(diff)
            mse_accum += diff ** 2 # Quadrato per RMSE
            
    N = len(val_loader.dataset)
    mae = mae_accum / N
    rmse = math.sqrt(mse_accum / N) # RMSE Calculation
    
    return mae, rmse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--s1', type=str, default=None, help="Override Stage 1 Checkpoint path")
    parser.add_argument('--s2', type=str, default=None, help="Override Stage 2 Checkpoint path")
    args = parser.parse_args()

    # 1. Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['DEVICE'])
    os.makedirs(os.path.join(config['EXP']['OUT_DIR'], config['RUN_NAME'], 'stage3_refined'), exist_ok=True)

    # 2. Data
    train_loader = DataLoader(
        build_dataset(config, 'train', build_transforms(config['DATA'], True)),
        batch_size=config['TRAIN_STAGE3']['BATCH_SIZE'],
        shuffle=True,
        num_workers=config['TRAIN_STAGE3']['NUM_WORKERS']
    )
    
    # Batch 1 per validazione accurata
    val_loader = DataLoader(
        build_dataset(config, 'val', build_transforms(config['DATA'], False)),
        batch_size=1, 
        shuffle=False,
        num_workers=4
    )

    # 3. Models Setup
    print("🏗️  Building Models...")
    stage1_model = ZIPModel(config).to(device)
    stage2_model = CLIPEBCModel(config).to(device)
    
    # Load Checkpoints
    s1_path = args.s1 if args.s1 else config['TRAIN_STAGE3']['CHECKPOINT_STAGE1']
    s2_path = args.s2 if args.s2 else config['TRAIN_STAGE3']['CHECKPOINT_STAGE2']
    
    print(f"📥 Loading Stage 1: {s1_path}")
    stage1_model.load_state_dict(torch.load(s1_path, map_location=device)['model'], strict=False)
    
    print(f"📥 Loading Stage 2: {s2_path}")
    stage2_model.load_state_dict(torch.load(s2_path, map_location=device)['model'], strict=False)

    # Joint Model
    joint_model = ZIPCLIPJointModel(stage1_model, stage2_model).to(device)
    
    # Freeze logic (Solo se desiderato, qui assumiamo di voler raffinare)
    # Per default in Stage 3 si allenano o raffinano i moduli.
    
    # 4. Optimizer
    optimizer = optim.AdamW(
        joint_model.parameters(), 
        lr=float(config['TRAIN_STAGE3']['LR']),
        weight_decay=float(config['TRAIN_STAGE3']['WEIGHT_DECAY'])
    )

    # 5. Loop
    best_mae = float('inf')
    best_rmse = float('inf')
    
    print(f"🚀 Starting Stage 3 Refinement ({config['DATASET']})")
    
    for epoch in range(config['TRAIN_STAGE3']['EPOCHS']):
        joint_model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
        for batch in pbar:
            img = batch['image'].to(device)
            # In Stage 3 spesso si usa il conteggio globale o mappe, qui assumiamo conteggio globale per semplicità
            # Se il dataset ritorna punti, calcoliamo count
            gt_count = torch.tensor([len(p) for p in batch['points']], device=device, dtype=torch.float32)
            
            optimizer.zero_grad()
            
            out = joint_model(img)
            pred_count = out['final_count']
            
            # Loss: L1 tra conteggio predetto e GT (più eventuali loss di regolarizzazione)
            # Qui usiamo L1 semplice sul conteggio finale come loss principale di raffinamento
            loss = torch.abs(pred_count - gt_count).mean() * config['TRAIN_STAGE3']['LAMBDA_COUNT']
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        # Validation
        mae, rmse = validate(joint_model, val_loader, device, config)
        print(f"📊 Ep {epoch+1} | Val MAE: {mae:.2f} | Val RMSE: {rmse:.2f} (Best MAE: {best_mae:.2f})")
        
        if mae < best_mae:
            best_mae = mae
            best_rmse = rmse
            save_path = os.path.join(config['EXP']['OUT_DIR'], config['RUN_NAME'], 'stage3_refined', 'best_model_refined.pth')
            torch.save({
                'epoch': epoch,
                'model': joint_model.state_dict(),
                'mae': best_mae,
                'rmse': best_rmse
            }, save_path)
            print("🌟 Saved Best Joint Model")

if __name__ == "__main__":
    main()