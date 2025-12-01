#!/usr/bin/env python3
# ============================================================
# ZIP-CLIP-EBC: Stage 2 Training - EBC-Head (CORRETTO)
# ============================================================
# Obiettivo: Addestrare l'EBC-head a contare usando DACELoss
#            (Classificazione + Optimal Transport).
# ============================================================

import argparse
import os
import sys
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# --- I TUOI IMPORT ---
from models.zip_clip_ebc_model import build_model
from losses import build_stage2_loss
from train_utils import init_seeds, get_scheduler, collate_fn
from datasets import get_dataset
from datasets.transforms import build_transforms

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_one_epoch(model, dataloader, criterion, optimizer, device, config, epoch):
    model.train()
    # Congeliamo backbone e pi_head (Stage 2)
    model.backbone.eval()
    model.pi_head.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    clip_grad = config.get("TRAIN_STAGE2", {}).get("CLIP_GRAD_NORM", 1.0)
    
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    
    for batch in pbar:
        # 1. Estrazione Dati (con Punti per DACELoss)
        images = batch['image'].to(device)
        gt_density = batch['density'].to(device)
        points = batch['points'] # Lista di tensori (necessaria per Optimal Transport)
        
        # Sposta i punti su device (se necessario per la loss specifica)
        points = [p.to(device) for p in points]

        optimizer.zero_grad()
        
        # 2. Forward Pass
        # EBC Only: Ritorna un dizionario con logit e lambda
        outputs = model.forward_ebc_only(images)
        
        # 3. Loss Calculation (DACELoss richiede argomenti specifici)
        # Scompatta l'output del modello per la loss
        pred_class = outputs['logit_bin_maps']
        pred_density = outputs['lambda_maps']
        
        # Chiamata corretta alla DACELoss
        loss, loss_dict = criterion(pred_class, pred_density, gt_density, points)
        
        # 4. Backward & Step
        loss.backward()
        
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.ebc_head.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Log
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "ce": f"{loss_dict.get('ebc_ce_loss', 0):.4f}", # Logga parti della loss
            "cnt": f"{loss_dict.get('ebc_count_loss', 0):.4f}"
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
        points = batch['points']
        points = [p.to(device) for p in points]
        
        # Inference
        outputs = model.forward_ebc_only(images)
        
        # Per la validazione usiamo MAE/MSE sul conteggio puro
        # pred_count è calcolato sommando la mappa di densità lambda
        pred_density = outputs['lambda_maps']
        pred_count = pred_density.sum(dim=[1, 2, 3])
        
        gt_count = gt_density.sum(dim=[1, 2, 3])
        
        mae = torch.abs(pred_count - gt_count).sum().item()
        mse = ((pred_count - gt_count) ** 2).sum().item()
        
        total_mae += mae
        total_mse += mse
        num_samples += images.shape[0]
    
    return {
        "val_mae": total_mae / max(num_samples, 1),
        "val_rmse": np.sqrt(total_mse / max(num_samples, 1)),
    }

def main(config_path: str):
    config = load_config(config_path)
    
    device = torch.device(config.get("DEVICE", "cuda"))
    init_seeds(config.get("SEED", 2025))
    
    train_cfg = config.get("TRAIN_STAGE2", {})
    run_name = config.get("RUN_NAME", "experiment")
    
    output_dir = os.path.join(config["EXP"]["OUT_DIR"], run_name, "stage2")
    os.makedirs(output_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))
    
    print(f"=" * 60)
    print(f"🚀 ZIP-CLIP-EBC: Stage 2 Training (EBC-Head) - REVISED")
    print(f"   Loss: DACELoss (Classification + Optimal Transport)")
    print(f"   Output: {output_dir}")
    print(f"=" * 60)
    
    # 1. Modello
    model = build_model(config).to(device)
    
    # 2. Caricamento Stage 1
    stage1_dir = os.path.join(config["EXP"]["OUT_DIR"], run_name, "stage1")
    ckpt_path = os.path.join(stage1_dir, "best_stage1_model.pth")
    
    if not os.path.exists(ckpt_path):
        # Fallback names
        ckpt_path = os.path.join(stage1_dir, "last_stage1_model.pth")
    
    if os.path.exists(ckpt_path):
        print(f"📥 Caricamento pesi Stage 1 da: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        # Gestione dict vs state_dict
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model.load_state_dict(ckpt['model'], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
    else:
        print(f"⚠️  ATTENZIONE: Checkpoint Stage 1 non trovato in {stage1_dir}!")
    
    # 3. Freeze & Unfreeze
    model.freeze_backbone()
    model.freeze_pi_head()
    model.unfreeze_ebc_head()
    
    print(f"🔒 Backbone e π-Head CONGELATI")
    print(f"🔓 EBC-Head SCONGELATA per training")
    
    # 4. Dataset
    data_cfg = config["DATA"]
    
    # --- CORREZIONE TRANSFORMS ---
    # Usiamo il parametro 'stage' per dire a build_transforms di leggere 
    # CROP_SIZE_STAGE2 dal config
    train_tf = build_transforms(config, is_train=True, stage="stage2")
    val_tf = build_transforms(config, is_train=False, stage="stage2")
    
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
        batch_size=train_cfg.get("BATCH_SIZE", 8),
        shuffle=True,
        num_workers=train_cfg.get("NUM_WORKERS", 4),
        collate_fn=collate_fn, # Importante per gestire la lista di punti!
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
    # Nota: build_stage2_loss ora ritorna DACELoss come configurato
    criterion = build_stage2_loss(config, device)
    
    lr_ebc = train_cfg.get("LR_EBC_HEAD", 1e-4)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_ebc,
        weight_decay=train_cfg.get("WEIGHT_DECAY", 1e-4)
    )
    
    num_epochs = train_cfg.get("EPOCHS", 100)
    scheduler = get_scheduler(optimizer, train_cfg, num_epochs)
    
    # 6. Loop
    best_val_mae = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config, epoch
        )
        
        if scheduler:
            scheduler.step()
        
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("lr/ebc", optimizer.param_groups[0]["lr"], epoch)
        
        if epoch % train_cfg.get("VAL_INTERVAL", 5) == 0 or epoch == num_epochs:
            metrics = validate(model, val_loader, criterion, device, config)
            
            val_mae = metrics["val_mae"]
            val_rmse = metrics["val_rmse"]
            
            print(f"📉 Epoch {epoch}: Train Loss={train_loss:.4f} | Val MAE={val_mae:.2f} | RMSE={val_rmse:.2f}")
            
            writer.add_scalar("val/mae", val_mae, epoch)
            
            # Save Best
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                print(f"⭐ New Best Model! (MAE: {best_val_mae:.2f})")
                torch.save(model.state_dict(), os.path.join(output_dir, "best_stage2_model.pth"))
            
            # Save Last
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'opt': optimizer.state_dict(),
                'best_val': best_val_mae
            }, os.path.join(output_dir, "last_stage2_model.pth"))

    writer.close()
    print("✅ Stage 2 Completato.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_sha.yaml")
    args = parser.parse_args()
    main(args.config)