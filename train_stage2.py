#!/usr/bin/env python3
# ============================================================
# ZIP-CLIP-EBC: Stage 2 Training - EBC-Head
# ============================================================
# Obiettivo: Addestrare l'EBC-head a contare persone nei blocchi
#            non-vuoti usando similarity CLIP.
# Cosa viene addestrato: EBC-head
# Cosa è congelato: π-head + backbone
# ============================================================

#!/usr/bin/env python3
import argparse
import os
import sys
import yaml
import random
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# --- I TUOI IMPORT ---
from models.zip_clip_ebc_model import build_model
from losses import build_stage2_loss
# Importiamo le utility necessarie
from train_utils import init_seeds, get_optimizer, get_scheduler, collate_fn, resume_if_exists
from datasets import get_dataset
from datasets.transforms import build_transforms

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_one_epoch(model, dataloader, criterion, optimizer, device, config, epoch):
    model.train()
    # Congeliamo BNorm e Dropout di backbone e pi_head
    model.backbone.eval()
    model.pi_head.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    # Clip grad specifico per EBC (dal config)
    clip_grad = config.get("TRAIN_STAGE2", {}).get("CLIP_GRAD_NORM", 1.0)
    
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    
    for batch in pbar:
        # Gestione batch (dict o list)
        if isinstance(batch, dict):
            images = batch['image']
            gt_density = batch['density']
        else:
            images, gt_density = batch[0], batch[1]

        images = images.to(device)
        gt_density = gt_density.to(device)
        
        optimizer.zero_grad()
        
        # Forward: EBC Only
        # Nota: Qui usiamo la maschera generata da Pi (che è congelata).
        # Se Pi è accurata (come abbiamo visto), va benissimo.
        outputs = model.forward_ebc_only(images)
        
        # Loss: Stage 2 (Cross Entropy sui bin)
        loss, loss_dict = criterion(outputs, gt_density)
        
        loss.backward()
        
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.ebc_head.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Log veloce
        ce_loss = loss_dict.get('ebc_ce_loss', 0)
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "ce": f"{ce_loss:.4f}",
        })
    
    return total_loss / max(num_batches, 1)

@torch.no_grad()
def validate(model, dataloader, criterion, device, config):
    model.eval()
    
    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    num_samples = 0
    
    for batch in tqdm(dataloader, desc="Validating", leave=False):
        if isinstance(batch, dict):
            images = batch['image']
            gt_density = batch['density']
        else:
            images, gt_density = batch[0], batch[1]

        images = images.to(device)
        gt_density = gt_density.to(device)
        
        outputs = model.forward_ebc_only(images)
        
        # Loss
        loss, _ = criterion(outputs, gt_density)
        total_loss += loss.item()
        
        # Metriche conteggio
        pred_count = outputs["pred_count"]
        gt_count = gt_density.sum(dim=[1, 2, 3])
        
        mae = torch.abs(pred_count - gt_count).sum().item()
        mse = ((pred_count - gt_count) ** 2).sum().item()
        
        total_mae += mae
        total_mse += mse
        num_samples += images.shape[0]
    
    num_batches = len(dataloader)
    
    return {
        "val_loss": total_loss / max(num_batches, 1),
        "val_mae": total_mae / max(num_samples, 1),
        "val_rmse": np.sqrt(total_mse / max(num_samples, 1)),
    }

def main(config_path: str):
    config = load_config(config_path)
    
    device = torch.device(config.get("DEVICE", "cuda"))
    init_seeds(config.get("SEED", 2025))
    
    train_cfg = config.get("TRAIN_STAGE2", {})
    run_name = config.get("RUN_NAME", "experiment")
    
    # Percorsi output corretti
    output_dir = os.path.join(config["EXP"]["OUT_DIR"], run_name, "stage2")
    os.makedirs(output_dir, exist_ok=True)
    
    # Logging tensorboard
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))
    
    print(f"=" * 60)
    print(f"🚀 ZIP-CLIP-EBC: Stage 2 Training (EBC-Head)")
    print(f"   Device: {device}")
    print(f"   Output: {output_dir}")
    print(f"=" * 60)
    
    # 1. Modello
    model = build_model(config).to(device)
    
    # 2. Carica Stage 1 (Gestione nomi file corretta)
    stage1_dir = os.path.join(config["EXP"]["OUT_DIR"], run_name, "stage1")
    
    # Cerchiamo il checkpoint in ordine di preferenza
    candidates = [
        "best_stage1_model.pth", 
        "last_stage1_model.pth", 
        "best_model.pth" # Fallback vecchi nomi
    ]
    
    loaded = False
    for filename in candidates:
        ckpt_path = os.path.join(stage1_dir, filename)
        if os.path.exists(ckpt_path):
            print(f"📥 Caricamento pesi Stage 1 da: {ckpt_path}")
            # Fix per PyTorch 2.6+
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            
            # Gestione se il checkpoint è dict completo o solo state_dict
            if isinstance(ckpt, dict) and 'model' in ckpt:
                model.load_state_dict(ckpt['model'], strict=False)
            else:
                model.load_state_dict(ckpt, strict=False)
            loaded = True
            break
    
    if not loaded:
        print(f"⚠️ ATTENZIONE: Nessun checkpoint Stage 1 trovato in {stage1_dir}!")
        print("   Il training partirà da zero (sconsigliato).")
    
    # 3. Congelamento (Cruciale per Stage 2)
    model.freeze_backbone()
    model.freeze_pi_head()
    model.unfreeze_ebc_head()
    
    # Verifica parametri
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ℹ️  Parametri trainabili (EBC only): {trainable:,}")
    
    # 4. Dataset Reale (Copiato da Stage 1)
    data_cfg = config["DATA"]
    
    # Stage 2 usa crop normali (256) come definito nel config
    train_tf = build_transforms(
        data_cfg, 
        is_train=True, 
        override_crop_size=data_cfg.get("CROP_SIZE_STAGE2"),
        override_crop_scale=data_cfg.get("CROP_SCALE_STAGE2")
    )
    val_tf = build_transforms(data_cfg, is_train=False)
    
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
        collate_fn=collate_fn,
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
    
    # 5. Loss, Optimizer, Scheduler
    criterion = build_stage2_loss(config).to(device)
    
    # Optimizer solo per EBC (usiamo la tua funzione helper se esiste, o config manuale)
    # Qui usiamo la logica diretta per sicurezza
    lr_ebc = train_cfg.get("LR_EBC_HEAD", 1e-4)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_ebc,
        weight_decay=train_cfg.get("WEIGHT_DECAY", 1e-4)
    )
    
    num_epochs = train_cfg.get("EPOCHS", 100)
    scheduler = get_scheduler(optimizer, train_cfg, num_epochs)
    
    # 6. Training Loop
    best_val_mae = float('inf')
    start_epoch = 1
    
    # Resume Stage 2 (se esiste un checkpoint interrotto di Stage 2)
    if train_cfg.get("RESUME_LAST", True):
        last_s2 = os.path.join(output_dir, "last_stage2_model.pth")
        if os.path.exists(last_s2):
            ckpt = torch.load(last_s2, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'], strict=False)
            optimizer.load_state_dict(ckpt['opt'])
            start_epoch = ckpt['epoch'] + 1
            best_val_mae = ckpt['best_val']
            print(f"🔄 Resume Stage 2 dall'epoca {start_epoch}")

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config, epoch
        )
        
        if scheduler:
            scheduler.step()
        
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("lr/ebc_head", optimizer.param_groups[0]["lr"], epoch)
        
        # Validazione
        if epoch % train_cfg.get("VAL_INTERVAL", 5) == 0 or epoch == num_epochs:
            metrics = validate(model, val_loader, criterion, device, config)
            
            val_mae = metrics["val_mae"]
            val_rmse = metrics["val_rmse"]
            
            print(f"📉 Epoch {epoch}: Train Loss={train_loss:.4f} | Val MAE={val_mae:.2f} | RMSE={val_rmse:.2f}")
            
            writer.add_scalar("val/mae", val_mae, epoch)
            writer.add_scalar("val/rmse", val_rmse, epoch)
            
            # Save Checkpoints
            ckpt_dict = {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "best_val": best_val_mae
            }
            
            # Save Last
            torch.save(ckpt_dict, os.path.join(output_dir, "last_stage2_model.pth"))
            
            # Save Best
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                print(f"⭐ New Best Stage 2 Model! (MAE: {best_val_mae:.2f})")
                torch.save(model.state_dict(), os.path.join(output_dir, "best_stage2_model.pth"))
    
    writer.close()
    print("✅ Stage 2 Completato!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZIP-CLIP-EBC Stage 2")
    parser.add_argument("--config", type=str, default="config_sha.yaml")
    args = parser.parse_args()
    
    main(args.config)

    #python train_stage2.py --config config_sha.yaml
    #