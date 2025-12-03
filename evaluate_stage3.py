import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import yaml
import numpy as np

from models.zip_clip_ebc_model import build_model
from losses import build_stage3_loss
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

@torch.no_grad()
def validate_stage3(model, criterion, dataloader, device, config, checkpoint_path):
    model.eval()
    
    total_mae = 0.0
    total_mse = 0.0
    
    # Metriche differenziate
    zero_mae = 0.0
    zero_samples = 0
    crowd_mae = 0.0
    crowd_samples = 0
    
    pi_activations = [] # Per monitorare quanto è attiva la Pi-head
    
    print("\n===== DIAGNOSTICA STAGE 3 (Joint Model) =====")
    print("Monitoraggio: Performance Globale + Capacità di soppressione background (Zero-MAE)")
    
    progress_bar = tqdm(dataloader, desc="Eval Stage 3")

    for idx, batch in enumerate(progress_bar):
        if isinstance(batch, dict):
            images = batch['image']
            gt_density = batch['density']
        else:
            images, gt_density = batch[0], batch[1]

        images = images.to(device)
        gt_density = gt_density.to(device)

        # Full Forward
        preds = model(images)
        
        # Metriche
        pred_count = preds["pred_count"].item()
        gt_count = gt_density.sum().item()
        err = abs(pred_count - gt_count)
        
        total_mae += err
        total_mse += (pred_count - gt_count) ** 2
        
        # Diagnostica Pi-Head
        # Media della probabilità di attivazione su tutta l'immagine
        pi_mean = preds["pi_prob"].mean().item()
        pi_activations.append(pi_mean)

        # Analisi differenziata (Immagini vuote vs Folla)
        if gt_count < 1.0:
            zero_mae += err
            zero_samples += 1
            type_str = "🟢 EMPTY"
        else:
            crowd_mae += err
            crowd_samples += 1
            type_str = "🔴 CROWD"

        if idx % 20 == 0:
            print(f"[IMG {idx:03d}] {type_str} | GT: {gt_count:6.1f} | Pred: {pred_count:6.1f} | "
                  f"Err: {err:5.1f} | Pi-Act: {pi_mean:.3f}")

    # Aggregazione
    N = len(dataloader.dataset)
    avg_mae = total_mae / N
    avg_rmse = (total_mse / N) ** 0.5
    
    avg_zero_mae = zero_mae / max(zero_samples, 1)
    avg_crowd_mae = crowd_mae / max(crowd_samples, 1)
    avg_pi = sum(pi_activations) / len(pi_activations)

    print("\n" + "="*50)
    print(f"🚀 RISULTATI FINALI STAGE 3 - {os.path.basename(checkpoint_path)}")
    print("="*50)
    print(f"Global MAE:        {avg_mae:.3f}")
    print(f"Global RMSE:       {avg_rmse:.3f}")
    print("-" * 30)
    print(f"Analisi Zero-Inflated:")
    print(f"   Avg Pi Activity: {avg_pi:.3f} (Se ~1.0, la Pi-head non filtra nulla)")
    print(f"   MAE su Vuoti:    {avg_zero_mae:.3f} (Dovrebbe essere vicino a 0)")
    print(f"   MAE su Folla:    {avg_crowd_mae:.3f}")
    print("="*50 + "\n")

def main(config_path, checkpoint_path):
    config = load_config(config_path)
    device = torch.device(config['DEVICE'])
    init_seeds(config['SEED'])

    print(f"🚀 Avvio Valutazione Stage 3 (Joint)")
    model = build_model(config).to(device)

    # Cerca il checkpoint stage 3
    if not checkpoint_path:
        base_dir = os.path.join(config['EXP']['OUT_DIR'], config['RUN_NAME'], "stage3")
        checkpoint_path = os.path.join(base_dir, "best_stage3_model.pth")
    
    print(f"📂 Loading: {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
    else:
        print("⚠️ Checkpoint non trovato. Valutazione modello non inizializzato.")

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
    parser.add_argument("--checkpoint", default="", help="Override path checkpoint")
    args = parser.parse_args()
    main(args.config, args.checkpoint)