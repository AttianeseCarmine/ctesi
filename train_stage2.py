#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZIP-CLIP-EBC: Stage 2 Training - CLIP-EBC Head

Obiettivo: Addestrare la EBC-head a classificare blocchi nei bins di conteggio.
           Il backbone e la π-head sono CONGELATI.

Moduli addestrati:
    - CLIP-EBC Head (visual projector, temperature)
    - Il CLIP text encoder rimane sempre congelato

Loss:
    - Cross-Entropy sui bins + Count Loss (L1)

Pre-requisiti:
    - Stage 1 completato (checkpoint π-head disponibile)

Usage:
    python train_stage2.py --config configs/config_sha.yaml
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
from losses import CLIPEBCLoss
from datasets import Crowd, collate_fn
from datasets.transforms import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomApply


# =============================================================================
# LOSS STAGE 2
# =============================================================================

class Stage2Loss(nn.Module):
    """
    Loss per Stage 2: Cross-Entropy sui bins + Count Loss.
    
    Basata sulla loss del paper CLIP-EBC ufficiale.
    """
    
    def __init__(
        self,
        bins: list,
        bin_centers: list,
        block_size: int = 16,
        label_smoothing: float = 0.1,
        count_weight: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size
        
        self.clip_ebc_loss = CLIPEBCLoss(
            bins=bins,
            bin_centers=bin_centers,
            label_smoothing=label_smoothing,
            count_weight=count_weight,
        )
    
    def compute_block_counts(self, gt_density: torch.Tensor) -> torch.Tensor:
        """Calcola conteggi GT per blocco."""
        block_counts = F.avg_pool2d(
            gt_density,
            kernel_size=self.block_size,
            stride=self.block_size
        ) * (self.block_size ** 2)
        return block_counts
    
    def forward(self, predictions, gt_density):
        """
        Args:
            predictions: Dict con 'ebc_outputs' o 'logits', 'bin_probs'
            gt_density: [B, 1, H, W]
        """
        target_counts = self.compute_block_counts(gt_density)
        
        # Estrai output EBC
        if "ebc_outputs" in predictions:
            logits = predictions["ebc_outputs"]["logits"]
            bin_probs = predictions["ebc_outputs"]["bin_probs"]
        else:
            logits = predictions.get("logits")
            bin_probs = predictions.get("bin_probs")
        
        # Se logits non disponibili, usa log(probs)
        if logits is None and bin_probs is not None:
            logits = torch.log(bin_probs + 1e-8)
        
        loss, loss_dict = self.clip_ebc_loss(logits, target_counts, bin_probs)
        
        # Aggiungi MAE
        with torch.no_grad():
            pred_count = predictions["pred_count"]
            gt_count = gt_density.sum(dim=[1, 2, 3])
            mae = torch.abs(pred_count - gt_count).mean()
            loss_dict["mae"] = mae
        
        return loss, loss_dict


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
    
    # Assicurati che backbone e π-head rimangano in eval mode
    model.backbone.eval()
    model.zip_head.eval()
    
    total_loss = 0.0
    total_ce = 0.0
    total_count = 0.0
    total_acc = 0.0
    total_mae = 0.0
    n_batches = 0
    
    use_amp = config["TRAIN_STAGE2"].get("AMP", False) and scaler is not None
    grad_clip = config["TRAIN_STAGE2"].get("CLIP_GRAD_NORM", 1.0)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, points, densities) in enumerate(pbar):
        images = images.to(device)
        densities = densities.to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=use_amp):
            outputs = model(images, return_intermediates=True)
            loss, loss_dict = criterion(outputs, densities)
        
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
        
        total_loss += loss.item()
        total_ce += loss_dict.get("ebc_ce_loss", torch.tensor(0)).item()
        total_count += loss_dict.get("ebc_count_loss", torch.tensor(0)).item()
        total_acc += loss_dict.get("ebc_accuracy", torch.tensor(0)).item()
        total_mae += loss_dict.get("mae", torch.tensor(0)).item()
        n_batches += 1
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{loss_dict.get('ebc_accuracy', 0):.3f}",
            "mae": f"{loss_dict.get('mae', 0):.1f}",
        })
    
    return {
        "loss": total_loss / n_batches,
        "ce_loss": total_ce / n_batches,
        "count_loss": total_count / n_batches,
        "accuracy": total_acc / n_batches,
        "mae": total_mae / n_batches,
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
    total_acc = 0.0
    n_batches = 0
    
    pred_counts = []
    gt_counts = []
    
    for images, points, densities in tqdm(dataloader, desc="Validating"):
        images = images.to(device)
        densities = densities.to(device)
        
        outputs = model(images, return_intermediates=True)
        loss, loss_dict = criterion(outputs, densities)
        
        total_loss += loss.item()
        total_acc += loss_dict.get("ebc_accuracy", torch.tensor(0)).item()
        n_batches += 1
        
        # Accumula conteggi
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
    
    return {
        "loss": total_loss / n_batches,
        "accuracy": total_acc / n_batches,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Stage 2 - EBC-Head")
    parser.add_argument("--config", type=str, default="configs/config_sha.yaml")
    parser.add_argument("--stage1_ckpt", type=str, default=None, 
                        help="Path to stage 1 checkpoint (auto-detected if not provided)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config.get("DEVICE", "cuda"))
    torch.manual_seed(config.get("SEED", 42))
    np.random.seed(config.get("SEED", 42))
    
    print("=" * 60)
    print("🚀 STAGE 2: CLIP-EBC Head Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Dataset: {config['DATASET']}")
    
    # Directories
    run_name = config.get("RUN_NAME", "experiment")
    output_dir = os.path.join(config["EXP"]["OUT_DIR"], run_name, "stage2")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}")
    
    # Dataset
    dataset_name = config["DATASET"]
    data_cfg = config["DATA"]
    
    crop_size = data_cfg.get("CROP_SIZE_STAGE2", data_cfg.get("CROP_SIZE", 448))
    crop_scale = data_cfg.get("CROP_SCALE_STAGE2", data_cfg.get("CROP_SCALE", [0.5, 1.5]))
    
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
    
    train_cfg = config["TRAIN_STAGE2"]
    
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
    
    # Carica checkpoint Stage 1
    stage1_ckpt = args.stage1_ckpt
    if stage1_ckpt is None:
        stage1_ckpt = os.path.join(
            config["EXP"]["OUT_DIR"], run_name, "stage1", "best_stage1_model.pth"
        )
    
    if os.path.exists(stage1_ckpt):
        print(f"📂 Loading Stage 1 checkpoint: {stage1_ckpt}")
        ckpt = torch.load(stage1_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"   Stage 1 F1: {ckpt.get('best_f1', 'N/A')}")
    else:
        print(f"⚠️ Stage 1 checkpoint not found: {stage1_ckpt}")
        print("   Training from scratch (non-optimal)")
    
    # Congela backbone e π-head
    model.freeze_backbone()
    model.freeze_zip_head()
    
    # Scongela solo EBC head
    model.unfreeze_ebc_head()
    
    print(f"Trainable params: {model.get_trainable_params():,}")
    
    # Loss
    loss_cfg = config.get("LOSS_STAGE2", {})
    criterion = Stage2Loss(
        bins=bins_cfg["bins"],
        bin_centers=bins_cfg["bin_centers"],
        block_size=data_cfg.get("ZIP_BLOCK_SIZE", 16),
        label_smoothing=loss_cfg.get("LABEL_SMOOTHING", 0.1),
        count_weight=loss_cfg.get("COUNT_WEIGHT", 0.1),
    )
    
    # Optimizer - solo parametri EBC
    ebc_params = [p for n, p in model.ebc_head.named_parameters() 
                  if p.requires_grad and "text_encoder" not in n]
    
    optimizer = AdamW(
        ebc_params,
        lr=train_cfg.get("LR_EBC_HEAD", 1e-4),
        weight_decay=1e-4,
    )
    
    # Scheduler
    warmup_epochs = train_cfg.get("WARMUP_EPOCHS", 5)
    total_epochs = train_cfg["EPOCHS"]
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    main_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-7)
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
    patience = train_cfg.get("EARLY_STOPPING_PATIENCE", 100)
    epochs_without_improvement = 0
    
    # Training loop
    print(f"\n🏋️ Training for {total_epochs} epochs...")
    
    for epoch in range(start_epoch, total_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{total_epochs} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*60}")
        
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config, scaler
        )
        
        scheduler.step()
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.3f}, MAE: {train_metrics['mae']:.2f}")
        
        # Validate
        val_interval = train_cfg.get("VAL_INTERVAL", 10)
        if epoch % val_interval == 0 or epoch == total_epochs:
            val_metrics = validate(model, val_loader, criterion, device, config)
            
            print(f"Val - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.3f}")
            print(f"      MAE: {val_metrics['mae']:.2f}, RMSE: {val_metrics['rmse']:.2f}")
            
            # Save best
            if val_metrics["mae"] < best_mae:
                best_mae = val_metrics["mae"]
                epochs_without_improvement = 0
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_mae": best_mae,
                    "metrics": val_metrics,
                }, os.path.join(output_dir, "best_stage2_model.pth"))
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
    
    print(f"\n{'='*60}")
    print(f"✅ Stage 2 Training Complete!")
    print(f"Best MAE: {best_mae:.2f}")
    print(f"Checkpoint: {output_dir}/best_stage2_model.pth")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
