#!/usr/bin/env python3
import torch
import yaml
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

# Imports dei modelli specifici
from models.zip_model import ZIPModel
from models.clip_ebc_model import CLIPEBCModel
from models.joint_model import DivideAndConquerStage3 
from datasets.sha import SHA
from datasets.transforms import build_transforms

def evaluate_hard_gating(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # --- 1. Inizializzazione Modelli Base ---
    print("🏗️ Building Base Models...")
    zip_model = ZIPModel(config).to(device)
    clip_model = CLIPEBCModel(config).to(device)
    
    # --- 2. Caricamento Pesi (Logica Corretta per train_stage3.py) ---
    print(f"📥 Loading Checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    
    # Prepariamo i dizionari puliti
    zip_dict = {}
    clip_dict = {}
    
    print("🔑 Mapping weights from Joint Model to single modules...")
    for key, value in state_dict.items():
        # train_stage3.py chiama i modelli 'stage1' (ZIP) e 'stage2' (CLIP)
        if key.startswith('stage1.'):
            new_key = key.replace('stage1.', '')
            zip_dict[new_key] = value
        elif key.startswith('stage2.'):
            new_key = key.replace('stage2.', '')
            clip_dict[new_key] = value
            
    # Carichiamo i pesi mappati
    msg_z = zip_model.load_state_dict(zip_dict, strict=False)
    msg_c = clip_model.load_state_dict(clip_dict, strict=False)
    
    print(f"✅ ZIP Weights Loaded (Missing: {len(msg_z.missing_keys)})")
    print(f"✅ CLIP Weights Loaded (Missing: {len(msg_c.missing_keys)})")

    # --- 3. Creazione Pipeline Hard Gating ---
    model = DivideAndConquerStage3(
        zip_model=zip_model,
        clip_ebc_model=clip_model,
        tile_size=224,      # Dimensione standard CLIP
        threshold=0.5       # Soglia confidenza (modificabile)
    ).to(device)
    
    # --- 4. Dataset & Dataloader ---
    val_dataset = SHA(config['DATA']['ROOT'], 'val', build_transforms(config['DATA'], False))
    # Batch size 1 obbligatorio per immagini di dimensioni variabili
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    mae, mse = 0, 0
    total_blocks_proc = 0
    total_blocks_kept = 0
    
    print(f"🚀 Evaluating Hard Gating Strategy (Threshold: {model.threshold})...")
    
    model.eval()
    for batch in tqdm(val_loader):
        img = batch['image'].to(device)
        # GT points shape: [batch, N_points, 2] -> len gives counts
        gt_count = len(batch['points'][0]) 
        
        # Forward pass (Hard Gating)
        pred_count_tensor, n_kept, n_total = model(img)
        pred_count = pred_count_tensor.item()
        
        # Statistiche
        total_blocks_kept += n_kept
        total_blocks_proc += n_total
        
        # Errori
        err = abs(pred_count - gt_count)
        mae += err
        mse += err ** 2
        
    # --- 5. Risultati ---
    N = len(val_dataset)
    mae = mae / N
    mse = (mse / N) ** 0.5
    eff = 100 * (1 - total_blocks_kept/total_blocks_proc) if total_blocks_proc > 0 else 0
    
    print("\n" + "="*50)
    print(f"🏆 RESULTS - HARD GATING (T={model.threshold})")
    print(f"   MAE: {mae:.2f}")
    print(f"   MSE: {mse:.2f}")
    print(f"   Efficiency (Skipped Blocks): {eff:.1f}%")
    print("="*50 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to stage3 checkpoint')
    args = parser.parse_args()
    
    evaluate_hard_gating(args)