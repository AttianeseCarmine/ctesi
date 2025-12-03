import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import yaml
import numpy as np

# --- IMPORT ---
from models.zip_clip_ebc_model import build_model
from losses import build_stage2_loss
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

@torch.no_grad()
def validate_stage2(model, criterion, dataloader, device, config, checkpoint_path):
    """
    Validazione Stage 2: Focus sulla EBC-Head e sulla precisione del conteggio.
    """
    model.eval()
    
    total_loss = 0.0
    mae = 0.0
    mse = 0.0
    
    # Metriche specifiche per i Bin
    bin_centers = torch.tensor(config['BINS_CONFIG']['sha']['bin_centers']).to(device)
    total_bin_accuracy = 0.0
    
    print("\n===== DIAGNOSTICA STAGE 2 (EBC-Head / Distribution Matching) =====")
    
    progress_bar = tqdm(dataloader, desc="Validating Stage 2")

    for idx, batch in enumerate(progress_bar):
        if isinstance(batch, dict):
            images = batch['image']
            gt_density = batch['density']
        else:
            images, gt_density = batch[0], batch[1]

        images = images.to(device)
        gt_density = gt_density.to(device)

        # Forward pass (Stage 2 usa idealmente forward_ebc_only, ma forward completo va bene)
        # Usiamo forward completo per vedere l'output finale integrato
        preds = model(images, return_intermediates=True)
        
        # Loss
        loss, loss_dict = criterion(preds, gt_density)
        total_loss += loss.item()

        # --- METRICHE CONTEGGIO ---
        pred_count = preds["pred_count"].item()
        gt_count = gt_density.sum().item()
        
        err = abs(pred_count - gt_count)
        mae += err
        mse += (pred_count - gt_count) ** 2

        # --- DIAGNOSTICA BINS ---
        # Vediamo se il modello è "sicuro" della sua scelta (entropia bassa)
        # bin_probs: [B, num_bins, H, W]
        bin_probs = preds["bin_probs"] 
        
        # Media della max probability (Confidenza)
        max_prob, bin_idx = torch.max(bin_probs, dim=1)
        avg_confidence = max_prob.mean().item()
        
        # Log dettagliato per campioni
        if idx % 20 == 0:
            logit_ce_loss = loss_dict.get('ebc_ce_loss', 0)
            if isinstance(logit_ce_loss, torch.Tensor): logit_ce_loss = logit_ce_loss.item()
            
            print(f"[IMG {idx:03d}] GT: {gt_count:6.1f} | Pred: {pred_count:6.1f} | Diff: {err:5.1f} "
                  f"| Conf: {avg_confidence:.2f} | Loss(KL/CE): {logit_ce_loss:.4f}")

        progress_bar.set_postfix({'loss': f"{loss.item():.3f}", 'err': f"{err:.1f}"})

    # Statistiche finali
    N = len(dataloader.dataset)
    avg_loss = total_loss / N
    avg_mae = mae / N
    avg_rmse = (mse / N) ** 0.5

    print("\n" + "="*40)
    print(f"📊 RISULTATI STAGE 2 - {os.path.basename(checkpoint_path)}")
    print("="*40)
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"MAE:             {avg_mae:.3f}  (Obiettivo: < 65.0)")
    print(f"RMSE:            {avg_rmse:.3f}")
    print("="*40 + "\n")

def main(config_path, checkpoint_path):
    config = load_config(config_path)
    device = torch.device(config['DEVICE'])
    init_seeds(config['SEED'])

    print(f"🚀 Avvio Valutazione Stage 2 (EBC Count)")
    
    # Costruzione Modello
    model = build_model(config).to(device)

    # Caricamento Pesi
    if not os.path.exists(checkpoint_path):
        # Fallback automatico
        base_dir = os.path.join(config['EXP']['OUT_DIR'], config['RUN_NAME'], "stage2")
        checkpoint_path = os.path.join(base_dir, "best_stage2_model.pth")
        
    print(f"📂 Loading: {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
    else:
        print("❌ Errore: Checkpoint non trovato!")
        return

    # Loss e Dati
    criterion = build_stage2_loss(config).to(device)
    
    DatasetClass = get_dataset(config['DATASET'])
    val_tf = build_transforms(config['DATA'], is_train=False)
    val_dataset = DatasetClass(
        root=config['DATA']['ROOT'],
        split=config['DATA']['VAL_SPLIT'],
        block_size=config['DATA']['ZIP_BLOCK_SIZE'],
        transforms=val_tf,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    validate_stage2(model, criterion, val_loader, device, config, checkpoint_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_sha.yaml")
    parser.add_argument("--checkpoint", default="", help="Path manuale (opzionale)")
    args = parser.parse_args()
    main(args.config, args.checkpoint)