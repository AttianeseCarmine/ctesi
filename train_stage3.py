#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZIP-CLIP-EBC: Stage 3 Training - Joint Fine-tuning

Obiettivo: Fine-tuning congiunto di tutto il modello.
           Bilancia la loss ZIP e CLIP-EBC per ottimizzazione end-to-end.

Moduli addestrati:
    - Backbone (con LR molto basso)
    - π-Head (con LR medio)
    - EBC-Head (con LR medio)

Loss:
    L_total = L_ZIP + α * L_CLIP_EBC

Pre-requisiti:
    - Stage 1 e Stage 2 completati

Usage:
    python train_stage3.py --config configs/config_sha.yaml
"""

import os
import sys
import argparse
import time
import yaml
import json
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from tqdm import tqdm

from models import ZIPCLIPEBCModel
from losses import ZIPCLIPEBCLoss
from datasets import Crowd, collate_fn
from datasets.transforms import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomApply


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: dict,
    scaler: GradScaler = None,
):
    """Training per una epoca."""
    model.train()
    
    total_loss = 0.0
    total_zip = 0.0
    total_ebc = 0.0
    total_mae = 0.0
    n_batches = 0
    
    use_amp = config["TRAIN_STAGE3"].get("AMP", False) and scaler is not None
    grad_clip = config["TRAIN_STAGE3"].get("CLIP_GRAD_NORM", 1.0)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, points, densities) in enumerate(pbar):
        images = images.to(device)
        densities = densities.to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=use_amp):
            outputs = model(images, return_intermediates=True)
            
            # Prepara predictions dict per la loss
            predictions = {
                "pi": outputs["pi"],
                "lambda_zip": outputs["lambda_zip"],
                "logits": outputs["ebc_outputs"]["logits"],
                "bin_probs": outputs["bin_probs"],
            }
            
            loss, loss_dict = criterion(predictions, densities)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        # Metriche
        total_loss += loss.item()
        total_zip += loss_dict.get("loss_zip", torch.tensor(0)).item()
        total_ebc += loss_dict.get("loss_ebc", torch.tensor(0)).item()
        n_batches += 1
        
        # MAE
        with torch.no_grad():
            pred_count = outputs["pred_count"]
            for i, pts in enumerate(points):
                gt = len(pts) if pts is not None else 0
                total_mae += abs(pred_count[i].item() - gt)
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "zip": f"{loss_dict.get('loss_zip', 0):.4f}",
            "ebc": f"{loss_dict.get('loss_ebc', 0):.4f}",
        })
    
    n_samples = len(dataloader.dataset)
    
    return {
        "loss": total_loss / n_batches,
        "loss_zip": total_zip / n_batches,
        "loss_ebc": total_ebc / n_batches,
        "mae": total_mae / n_samples,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: dict,
):
    """Validazione."""
    model.eval()
    
    total_loss = 0.0
    n_batches = 0
    
    pred_counts = []
    gt_counts = []
    
    for images, points, densities in tqdm(dataloader, desc="Validating"):
        images = images.to(device)
        densities = densities.to(device)
        
        outputs = model(images, return_intermediates=True)
        
        predictions = {
            "pi": outputs["pi"],
            "lambda_zip": outputs["lambda_zip"],
            "logits": outputs["ebc_outputs"]["logits"],
            "bin_probs": outputs["bin_probs"],
        }
        
        loss, loss_dict = criterion(predictions, densities)
        
        total_loss += loss.item()
        n_batches += 1
        
        pred_count = outputs["pred_count"].cpu().numpy()
        for i, pts in enumerate(points):
            gt = len(pts) if pts is not None else 0
            pred_counts.append(pred_count[i])
            gt_counts.append(gt)
    
    pred_counts = np.array(pred_counts)
    gt_counts = np.array(gt_counts)
    
    mae = np.abs(pred_counts - gt_counts).mean()
    mse = ((pred_counts - gt_counts) ** 2).mean()
    rmse = np.sqrt(mse)
    nae = (np.abs(pred_counts - gt_counts) / (gt_counts + 1e-6)).mean() * 100
    
    return {
        "loss": total_loss / n_batches,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "nae": nae,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Stage 3 - Joint Fine-tuning")
    parser.add_argument("--config", type=str, default="configs/config_sha.yaml")
    parser.add_argument("--stage2_ckpt", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config.get("DEVICE", "cuda"))
    torch.manual_seed(config.get("SEED", 42))
    np.random.seed(config.get("SEED", 42))
    
    print("=" * 60)
    print("🚀 STAGE 3: Joint Fine-tuning")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Dataset: {config['DATASET']}")
    
    # Directories
    run_name = config.get("RUN_NAME", "experiment")
    output_dir = os.path.join(config["EXP"]["OUT_DIR"], run_name, "stage3")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}")
    
    # Dataset
    dataset_name = config["DATASET"]
    data_cfg = config["DATA"]
    
    crop_size = data_cfg.get("CROP_SIZE", 448)
    crop_scale = data_cfg.get("CROP_SCALE", [0.5, 1.5])
    
    train_transforms = RandomApply([
        RandomResizedCrop(size=(crop_size, crop_size), scale=tuple(crop_scale)),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ], p=[1.0, 0.5, 0.3])
    
    train_dataset = Crowd(
        dataset=dataset_name,
        split="train",
        transforms=train_transforms,
        num_crops=1,
    )
    
    val_dataset = Crowd(
        dataset=dataset_name,
        split="val",
        transforms=None,
        num_crops=1,
    )
    
    train_cfg = config["TRAIN_STAGE3"]
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["BATCH_SIZE"],
        shuffle=True,
        num_workers=train_cfg["NUM_WORKERS"],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=train_cfg["NUM_WORKERS"],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Model
    model_cfg = config["MODEL"]
    bins_cfg = config["BINS_CONFIG"][dataset_name]
    
    model = ZIPCLIPEBCModel(
        backbone=model_cfg.get("BACKBONE", "vgg16_bn"),
        pretrained_backbone=True,
        clip_model=model_cfg.get("BACKBONE", "ViT-B-16"),
        clip_pretrained=model_cfg.get("CLIP_PRETRAINED", "openai"),
        bins=bins_cfg["bins"],
        bin_centers=bins_cfg["bin_centers"],
        zip_hidden_dim=config.get("PI_HEAD", {}).get("HIDDEN_DIM", 256),
        temperature=config.get("EBC_HEAD", {}).get("TEMPERATURE", 0.1),
        learnable_temperature=config.get("EBC_HEAD", {}).get("LEARNABLE_TEMP", True),
        density_mode="zip_gated_ebc",
    ).to(device)
    
    # Carica checkpoint Stage 2
    stage2_ckpt = args.stage2_ckpt
    if stage2_ckpt is None:
        stage2_ckpt = os.path.join(
            config["EXP"]["OUT_DIR"], run_name, "stage2", "best_stage2_model.pth"
        )
    
    if os.path.exists(stage2_ckpt):
        print(f"📂 Loading Stage 2 checkpoint: {stage2_ckpt}")
        ckpt = torch.load(stage2_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"   Stage 2 MAE: {ckpt.get('best_mae', 'N/A')}")
    else:
        # Prova Stage 1
        stage1_ckpt = os.path.join(
            config["EXP"]["OUT_DIR"], run_name, "stage1", "best_stage1_model.pth"
        )
        if os.path.exists(stage1_ckpt):
            print(f"⚠️ Stage 2 not found, loading Stage 1: {stage1_ckpt}")
            ckpt = torch.load(stage1_ckpt, map_location=device)
            model.load_state_dict(ckpt["model"], strict=False)
        else:
            print("⚠️ No previous checkpoints found. Training from scratch.")
    
    # Scongela tutto per joint training
    model.unfreeze_backbone()
    model.unfreeze_zip_head()
    model.unfreeze_ebc_head()
    
    print(f"Trainable params: {model.get_trainable_params():,}")
    
    # Loss
    loss_cfg = config.get("LOSS_STAGE3", {})
    criterion = ZIPCLIPEBCLoss(
        bins=bins_cfg["bins"],
        bin_centers=bins_cfg["bin_centers"],
        alpha=loss_cfg.get("ALPHA_EBC", 1.0),
        label_smoothing=loss_cfg.get("LABEL_SMOOTHING", 0.1),
        count_weight=loss_cfg.get("COUNT_WEIGHT", 0.1),
        block_size=data_cfg.get("ZIP_BLOCK_SIZE", 16),
    )
    
    # Optimizer con LR differenziati
    param_groups = model.get_param_groups(
        lr_backbone=train_cfg.get("LR_BACKBONE", 5e-6),
        lr_zip_head=train_cfg.get("LR_HEADS", 1e-5),
        lr_ebc_head=train_cfg.get("LR_HEADS", 1e-5),
    )
    
    optimizer = AdamW(
        param_groups,
        weight_decay=train_cfg.get("WEIGHT_DECAY", 1e-4),
    )
    
    # Scheduler
    warmup_epochs = train_cfg.get("WARMUP_EPOCHS", 5)
    total_epochs = train_cfg["EPOCHS"]
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    main_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-8)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
    
    # AMP
    scaler = GradScaler() if train_cfg.get("AMP", False) else None
    
    # Resume
    start_epoch = 1
    best_mae = float("inf")
    
    if args.resume or train_cfg.get("RESUME_LAST", False):
        ckpt_path = os.path.join(output_dir, "last.pth")
        if os.path.exists(ckpt_path):
            print(f"📂 Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"], strict=False)
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            best_mae = ckpt.get("best_mae", float("inf"))
    
    # Early stopping
    patience = train_cfg.get("EARLY_STOPPING_PATIENCE", 50)
    epochs_without_improvement = 0
    
    # Training loop
    print(f"\n🏋️ Training for {total_epochs} epochs...")
    print(f"   Alpha (EBC weight): {loss_cfg.get('ALPHA_EBC', 1.0)}")
    
    for epoch in range(start_epoch, total_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{total_epochs} | LR backbone: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*60}")
        
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config, scaler
        )
        
        scheduler.step()
        
        print(f"Train - Loss: {train_metrics['loss']:.4f} "
              f"(ZIP: {train_metrics['loss_zip']:.4f}, EBC: {train_metrics['loss_ebc']:.4f})")
        print(f"        MAE: {train_metrics['mae']:.2f}")
        
        # Validate
        val_interval = train_cfg.get("VAL_INTERVAL", 2)
        if epoch % val_interval == 0 or epoch == total_epochs:
            val_metrics = validate(model, val_loader, criterion, device, config)
            
            print(f"Val - Loss: {val_metrics['loss']:.4f}")
            print(f"      MAE: {val_metrics['mae']:.2f}, RMSE: {val_metrics['rmse']:.2f}, NAE: {val_metrics['nae']:.1f}%")
            
            # Save best
            if val_metrics["mae"] < best_mae:
                best_mae = val_metrics["mae"]
                epochs_without_improvement = 0
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_mae": best_mae,
                    "metrics": val_metrics,
                }, os.path.join(output_dir, "best_stage3_model.pth"))
                print(f"⭐ New best MAE: {best_mae:.2f}")
            else:
                epochs_without_improvement += val_interval
            
            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch}")
                break
        
        # Save last
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_mae": best_mae,
        }, os.path.join(output_dir, "last.pth"))
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("📊 Final Evaluation")
    print(f"{'='*60}")
    
    # Carica best model per eval finale
    best_ckpt = torch.load(os.path.join(output_dir, "best_stage3_model.pth"), map_location=device)
    model.load_state_dict(best_ckpt["model"])
    
    final_metrics = validate(model, val_loader, criterion, device, config)
    
    print(f"\n🏆 FINAL RESULTS:")
    print(f"   MAE:  {final_metrics['mae']:.2f}")
    print(f"   RMSE: {final_metrics['rmse']:.2f}")
    print(f"   NAE:  {final_metrics['nae']:.1f}%")
    
    # Salva risultati
    results = {
        "dataset": dataset_name,
        "best_mae": float(best_mae),
        "final_metrics": {k: float(v) for k, v in final_metrics.items()},
        "config": config,
    }
    
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"✅ Stage 3 Training Complete!")
    print(f"Best MAE: {best_mae:.2f}")
    print(f"Checkpoint: {output_dir}/best_stage3_model.pth")
    print(f"Results: {output_dir}/results.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
