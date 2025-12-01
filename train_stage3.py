#!/usr/bin/env python3
# ============================================================
# ZIP-CLIP-EBC: Stage 3 Training - Joint Fine-tuning (REVISED)
# ============================================================
# Obiettivo: Ottimizzare tutto il modello insieme.
# Correzione: Passaggio dei 'points' alla JointLoss per 
#             supportare la componente DACELoss/Optimal Transport.
# ============================================================

import argparse
import os
import sys
import yaml
import random
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# --- I TUOI IMPORT ---
from models.zip_clip_ebc_model import build_model
from losses import build_stage3_loss
from train_utils import init_seeds, get_optimizer, get_scheduler, collate_fn, resume_if_exists
from datasets import get_dataset
from datasets.transforms import build_transforms

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_one_epoch(model, dataloader, criterion, optimizer, device, config, epoch):
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    # Clip grad
    clip_grad = config.get("TRAIN_STAGE3", {}).get("CLIP_GRAD_NORM", 1.0)
    
    pbar = tqdm(dataloader, desc=f"Train Stage 3 Epoch {epoch}")
    
    for batch in pbar:
        # 1. Estrazione Dati (con Punti!)
        images = batch['image'].to(device)
        gt_density = batch['density'].to(device)
        points = batch['points'] # Lista di tensori per OT Loss
        
        # Sposta punti su device se necessario (dipende dalla loss interna)
        points = [p.to(device) for p in points]
        
        optimizer.zero_grad()
        
        # 2. Forward Pass (Completo: Pi + EBC)
        outputs = model(images)
        
        # 3. Loss Calculation
        # La JointLoss ora si aspetta (predictions, gt_density, target_points)
        # perché internamente chiama la DACELoss che vuole i punti.
        loss, loss_dict = criterion(outputs, gt_density, points)
        
        # 4. Backward
        loss.backward()
        
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Log
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "pi": f"{loss_dict.get('joint_pi_bce', 0):.3f}",
            "ebc": f"{loss_dict.get('joint_ebc_total_loss', 0):.3f}"
        })
        
    return total_loss / max(num_batches, 1)

@torch.no_grad()
def validate(model, dataloader, criterion, device, config):
    model.eval()
    
    total_mae = 0.0
    total_mse = 0.0
    num_samples = 0
    
    for batch in tqdm(dataloader, desc="Validating", leave=False):
        images = batch['image'].to(device)
        gt_density = batch['density'].to(device)
        # In validation non serve passare points alla loss se calcoliamo solo MAE manualmente,
        # ma se la loss interna lo richiede per loggare, passiamoli.
        points = batch['points'] 
        
        # Inference completa
        # Il modello restituisce un dizionario. 'pred_count' è la stima finale integrata.
        outputs = model(images)
        
        # Recupera il conteggio predittivo finale
        # Se il modello non calcola pred_count internamente, lo facciamo qui:
        if "pred_count" in outputs:
            pred_count = outputs["pred_count"]
        else:
            # Fallback: Densità pesata (1-p_empty) * lambda
            pi_prob = outputs["pi_prob"]
            lambda_maps = outputs["lambda_maps"]
            pred_density = pi_prob * lambda_maps
            pred_count = pred_density.sum(dim=[1, 2, 3])
            
        gt_count = gt_density.sum(dim=[1, 2, 3])
        
        # Metriche
        mae = torch.abs(pred_count - gt_count).sum().item()
        mse = ((pred_count - gt_count) ** 2).sum().item()
        
        total_mae += mae
        total_mse += mse
        num_samples += images.shape[0]
        
    return {
        "val_mae": total_mae / max(num_samples, 1),
        "val_rmse": np.sqrt(total_mse / max(num_samples, 1))
    }

def main(config_path: str):
    config = load_config(config_path)
    
    device = torch.device(config.get("DEVICE", "cuda"))
    init_seeds(config.get("SEED", 42))
    
    run_name = config["RUN_NAME"]
    train_cfg = config["TRAIN_STAGE3"]
    
    output_dir = os.path.join(config["EXP"]["OUT_DIR"], run_name, "stage3")
    os.makedirs(output_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))
    
    print(f"=" * 60)
    print(f"🚀 ZIP-CLIP-EBC: Stage 3 (Joint Fine-tuning)")
    print(f"   Device: {device}")
    print(f"   Output: {output_dir}")
    print(f"=" * 60)
    
    # 1. Modello
    model = build_model(config).to(device)
    
    # 2. Caricamento Pesi (Cruciale)
    # Cerchiamo di caricare i migliori pesi degli stage precedenti
    stage1_path = os.path.join(config["EXP"]["OUT_DIR"], run_name, "stage1", "best_stage1_model.pth")
    stage2_path = os.path.join(config["EXP"]["OUT_DIR"], run_name, "stage2", "best_stage2_model.pth")
    
    # Carica Stage 1 (Backbone + Pi)
    if os.path.exists(stage1_path):
        print(f"📥 Carico pesi STAGE 1 da: {stage1_path}")
        ckpt1 = torch.load(stage1_path, map_location=device)
        # Se è un dizionario completo, estrai 'model'
        state1 = ckpt1['model'] if (isinstance(ckpt1, dict) and 'model' in ckpt1) else ckpt1
        model.load_state_dict(state1, strict=False)
    else:
        print("⚠️  Stage 1 checkpoint mancante! Pi-Head non inizializzata.")

    # Carica Stage 2 (EBC Head) - Questo sovrascrive il backbone se lo stage 2 lo ha toccato? 
    # No, Stage 2 aveva backbone congelato. Carica solo EBC Head.
    if os.path.exists(stage2_path):
        print(f"📥 Carico pesi STAGE 2 da: {stage2_path}")
        ckpt2 = torch.load(stage2_path, map_location=device)
        state2 = ckpt2['model'] if (isinstance(ckpt2, dict) and 'model' in ckpt2) else ckpt2
        # Carica con strict=False per prendere solo i pesi della EBC head
        model.load_state_dict(state2, strict=False)
    else:
        print("⚠️  Stage 2 checkpoint mancante! EBC-Head non inizializzata.")
        print("    Il modello userà l'inizializzazione casuale per EBC.")

    # 3. Scongela tutto (Fine-tuning)
    # Nello stage 3 addestriamo tutto, ma con LR diversi
    for param in model.parameters():
        param.requires_grad = True
        
    print("🔓 Scongelo TUTTO il modello (Backbone, Pi, EBC)...")
    
    # 4. Dataset
    data_cfg = config["DATA"]
    train_tf = build_transforms(config, is_train=True) # Stage 3 usa default transforms
    val_tf = build_transforms(config, is_train=False)
    
    DatasetClass = get_dataset(config["DATASET"])
    
    train_set = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["TRAIN_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=train_tf
    )
    val_set = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=val_tf
    )
    
    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg.get("BATCH_SIZE", 4),
        shuffle=True,
        num_workers=train_cfg.get("NUM_WORKERS", 4),
        collate_fn=collate_fn, # NECESSARIO per gestire i punti
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=1, 
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # 5. Loss & Optimizer
    criterion = build_stage3_loss(config, device)
    
    # Parametri differenziati
    params = [
        {"params": model.backbone.parameters(), "lr": train_cfg.get("LR_BACKBONE", 1e-6)},
        {"params": model.pi_head.parameters(), "lr": train_cfg.get("LR_HEADS", 1e-5)},
        {"params": model.ebc_head.parameters(), "lr": train_cfg.get("LR_HEADS", 1e-5)},
    ]
    
    optimizer = torch.optim.AdamW(
        params,
        weight_decay=train_cfg.get("WEIGHT_DECAY", 1e-4)
    )
    
    num_epochs = train_cfg.get("EPOCHS", 50)
    scheduler = get_scheduler(optimizer, train_cfg, num_epochs)
    
    # 6. Loop
    best_val_mae = float('inf')
    
    # Resume opzionale per Stage 3
    if train_cfg.get("RESUME_LAST", False):
        last_path = os.path.join(output_dir, "last_stage3_model.pth")
        if os.path.exists(last_path):
            print(f"🔄 Resume Stage 3...")
            ckpt = torch.load(last_path, map_location=device)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['opt'])
            best_val_mae = ckpt['best_val']
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config, epoch
        )
        
        if scheduler:
            scheduler.step()
        
        writer.add_scalar("train/loss", train_loss, epoch)
        
        if epoch % train_cfg.get("VAL_INTERVAL", 5) == 0 or epoch == num_epochs:
            metrics = validate(model, val_loader, criterion, device, config)
            
            val_mae = metrics["val_mae"]
            val_rmse = metrics["val_rmse"]
            
            print(f"📉 Epoch {epoch}: Train Loss={train_loss:.4f} | Val MAE={val_mae:.2f} | RMSE={val_rmse:.2f}")
            
            writer.add_scalar("val/mae", val_mae, epoch)
            writer.add_scalar("val/rmse", val_rmse, epoch)
            
            # Save Best
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                print(f"⭐ New Best Stage 3 Model! (MAE: {best_val_mae:.2f})")
                torch.save(model.state_dict(), os.path.join(output_dir, "best_stage3_model.pth"))
            
            # Save Last
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'opt': optimizer.state_dict(),
                'best_val': best_val_mae
            }, os.path.join(output_dir, "last_stage3_model.pth"))
            
    writer.close()
    print("✅ Stage 3 Completato.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_sha.yaml")
    args = parser.parse_args()
    main(args.config)