#!/usr/bin/env python3
# ============================================================
# ZIP-CLIP-EBC: Stage 3 Training - Joint Fine-tuning
# ============================================================
# Obiettivo: Ottimizzare tutto il modello insieme per coerenza
#            tra π-head e EBC-head.
# Cosa viene addestrato: Tutto (π + EBC + backbone con LR ridotto)
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
    
    # Clip grad dal config
    clip_grad = config.get("TRAIN_STAGE3", {}).get("CLIP_GRAD_NORM", 1.0)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} (Joint)")
    
    for batch in pbar:
        if isinstance(batch, dict):
            images = batch['image']
            gt_density = batch['density']
        else:
            images, gt_density = batch[0], batch[1]

        images = images.to(device)
        gt_density = gt_density.to(device)
        
        optimizer.zero_grad()
        
        # Full forward (Pi + EBC + Backbone)
        outputs = model(images)
        
        # Loss Joint (Pi BCE + EBC CE + Count L1)
        loss, loss_dict = criterion(outputs, gt_density)
        
        loss.backward()
        
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Log: mostriamo le 3 componenti della loss
        pi_loss = loss_dict.get('joint_pi_bce_loss', 0)
        ebc_loss = loss_dict.get('joint_ebc_ce_loss', 0)
        cnt_loss = loss_dict.get('joint_count_loss', 0)
        
        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "pi": f"{pi_loss:.3f}",
            "ebc": f"{ebc_loss:.3f}",
            "cnt": f"{cnt_loss:.3f}"
        })
    
    return total_loss / max(num_batches, 1)
@torch.no_grad()
def validate(model, dataloader, criterion, device, config):
    model.eval()
    
    total_mae = 0.0
    total_mse = 0.0
    total_samples = 0
    
    # --- DEBUG: Contatore per stampare solo i primi batch ---
    debug_steps = 0
    
    for batch in tqdm(dataloader, desc="Validating", leave=False):
        if isinstance(batch, dict):
            images = batch['image']
            gt_density = batch['labels'] # O 'density', controlla la chiave nel tuo dataset!
        else:
            images, gt_density = batch[0], batch[1]

        images = images.to(device)
        gt_density = gt_density.to(device)
        
        # Forward
        outputs = model(images)
        
        # Predizione
        pred_count = outputs["pred_count"] 
        # Se pred_count non c'è, calcolalo dalla mappa:
        if pred_count is None:
             pred_count = outputs['density_map'].sum(dim=[1, 2, 3])
        
        # GT Reale
        gt_count = gt_density.sum(dim=[1, 2, 3])
        
        # --- DEBUG PRINT ---
        if debug_steps < 5:
            print(f"\n[DEBUG IMG {debug_steps}]")
            print(f"   GT Tensor Sum: {gt_count.item():.4f}")
            print(f"   Pred Tensor Sum: {pred_count.item():.4f}")
            print(f"   Max Val in GT Map: {gt_density.max().item():.6f}")
            debug_steps += 1
        # -------------------

        mae = torch.abs(pred_count - gt_count).sum().item()
        mse = ((pred_count - gt_count) ** 2).sum().item()
        
        total_mae += mae
        total_mse += mse
        total_samples += images.shape[0]
    
    avg_mae = total_mae / max(total_samples, 1)
    avg_rmse = np.sqrt(total_mse / max(total_samples, 1))
    
    return avg_mae, avg_rmse

def main(config_path: str):
    config = load_config(config_path)
    
    device = torch.device(config.get("DEVICE", "cuda"))
    init_seeds(config.get("SEED", 2025))
    
    train_cfg = config.get("TRAIN_STAGE3", {})
    run_name = config.get("RUN_NAME", "experiment")
    
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
    
    # 2. Caricamento Pesi (Logica Combinata)
    stage1_dir = os.path.join(config["EXP"]["OUT_DIR"], run_name, "stage1")
    stage2_dir = os.path.join(config["EXP"]["OUT_DIR"], run_name, "stage2")
    
    # A) Carica Stage 1 (Fondamenta: Backbone + Pi-Head)
    stage1_path = os.path.join(stage1_dir, "best_stage1_model.pth")
    if not os.path.exists(stage1_path):
        stage1_path = os.path.join(stage1_dir, "last_stage1_model.pth")
        
    if os.path.exists(stage1_path):
        print(f"📥 Carico pesi STAGE 1 da: {stage1_path}")
        ckpt1 = torch.load(stage1_path, map_location=device)
        state1 = ckpt1['model'] if (isinstance(ckpt1, dict) and 'model' in ckpt1) else ckpt1
        model.load_state_dict(state1, strict=False)
    else:
        print("⚠️  ATTENZIONE: Nessun checkpoint Stage 1 trovato!")

    # B) Carica Stage 2 (EBC-Head specializzata)
    #    Questo sovrascriverà i pesi EBC e potenzialmente Backbone se Stage 2 l'ha toccato
    stage2_path = os.path.join(stage2_dir, "best_stage2_model.pth")
    if not os.path.exists(stage2_path):
        print("⚠️  Best Stage 2 non trovato, cerco 'last'...")
        stage2_path = os.path.join(stage2_dir, "last_stage2_model.pth")

    if os.path.exists(stage2_path):
        print(f"📥 Carico pesi STAGE 2 da: {stage2_path}")
        ckpt2 = torch.load(stage2_path, map_location=device)
        state2 = ckpt2['model'] if (isinstance(ckpt2, dict) and 'model' in ckpt2) else ckpt2
        # Carichiamo con strict=False per sicurezza, ma dovrebbe matchare tutto
        model.load_state_dict(state2, strict=False)
    else:
        print("⚠️  ATTENZIONE: Nessun checkpoint Stage 2 trovato!")
        print("    Il training partirà con l'EBC-Head inizializzata da Stage 1 (o random).")

    # 3. Scongelamento Totale
    print("🔓 Scongelo TUTTO il modello (Backbone, Pi, EBC)...")
    model.unfreeze_backbone()
    model.unfreeze_pi_head()
    model.unfreeze_ebc_head()
    
    # Verifica parametri
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ℹ️  Parametri totali in training: {trainable:,}")
    
    # 4. Dataset Reale (Configurazione Stage 2/3 usa crop normali)
    data_cfg = config["DATA"]
    train_tf = build_transforms(
        data_cfg, is_train=True, 
        override_crop_size=data_cfg.get("CROP_SIZE_STAGE2"), # 256
        override_crop_scale=data_cfg.get("CROP_SCALE_STAGE2")
    )
    val_tf = build_transforms(data_cfg, is_train=False)
    
    DatasetClass = get_dataset(config["DATASET"])
    train_set = DatasetClass(root=data_cfg["ROOT"], split=data_cfg["TRAIN_SPLIT"], 
                             block_size=data_cfg["ZIP_BLOCK_SIZE"], transforms=train_tf)
    val_set = DatasetClass(root=data_cfg["ROOT"], split=data_cfg["VAL_SPLIT"], 
                           block_size=data_cfg["ZIP_BLOCK_SIZE"], transforms=val_tf)

    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg.get("BATCH_SIZE", 8),
        shuffle=True, num_workers=4, collate_fn=collate_fn, 
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, 
        num_workers=4, collate_fn=collate_fn
    )
    
    # 5. Optimizer Differenziato (Backbone lento, Teste veloci)
    lr_heads = train_cfg.get("LR_HEADS", 5e-5)
    lr_backbone = train_cfg.get("LR_BACKBONE", 5e-6) # Molto basso per non rompere CLIP
    
    param_groups = model.get_param_groups(
        lr_backbone=lr_backbone,
        lr_pi_head=lr_heads,
        lr_ebc_head=lr_heads
    )
    optimizer = get_optimizer(param_groups, train_cfg) # La tua funzione helper supporta i gruppi? 
    # Se no, usiamo torch.optim.AdamW diretto per sicurezza:
    optimizer = torch.optim.AdamW(param_groups, weight_decay=train_cfg.get("WEIGHT_DECAY", 1e-4))

    num_epochs = train_cfg.get("EPOCHS", 100)
    scheduler = get_scheduler(optimizer, train_cfg, num_epochs)
    
    # 6. Loss Combinata
    criterion = build_stage3_loss(config).to(device)
    
    # 7. Training Loop
    best_val_mae = float('inf')
    start_epoch = 1
    
    # Resume se esiste
    if train_cfg.get("RESUME_LAST", True):
        last_s3 = os.path.join(output_dir, "last_stage3_model.pth")
        if os.path.exists(last_s3):
            ckpt = torch.load(last_s3, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'], strict=False)
            optimizer.load_state_dict(ckpt['opt'])
            start_epoch = ckpt['epoch'] + 1
            best_val_mae = ckpt['best_val']
            print(f"🔄 Resume Stage 3 dall'epoca {start_epoch}")

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config, epoch
        )
        
        if scheduler:
            scheduler.step()
        
        writer.add_scalar("train/loss", train_loss, epoch)
        
        # Validazione
        if epoch % train_cfg.get("VAL_INTERVAL", 5) == 0 or epoch == num_epochs:
            val_mae, val_rmse = validate(model, val_loader, criterion, device, config)
            
            print(f"📉 Epoch {epoch}: Train Loss={train_loss:.4f} | Val MAE={val_mae:.2f} | RMSE={val_rmse:.2f}")
            
            writer.add_scalar("val/mae", val_mae, epoch)
            
            ckpt_dict = {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "best_val": best_val_mae
            }
            torch.save(ckpt_dict, os.path.join(output_dir, "last_stage3_model.pth"))
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                print(f"⭐ New Best Joint Model! (MAE: {best_val_mae:.2f})")
                torch.save(model.state_dict(), os.path.join(output_dir, "best_stage3_model.pth"))
    
    writer.close()
    print("✅ Stage 3 Completato!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_sha.yaml")
    args = parser.parse_args()
    main(args.config)

        #python train_stage3.py --config config_sha.yaml | tee logs/train_stage3.txt