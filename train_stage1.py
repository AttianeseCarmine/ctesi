#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZIP-CLIP-EBC: Stage 1 Training - π-Head (Zero-Inflation)

Obiettivo: Addestrare la π-head a classificare blocchi vuoti vs pieni.
           Il backbone viene fine-tuned con learning rate più basso.

Moduli addestrati:
    - Backbone (VGG16-BN) con LR basso
    - π-Head con LR alto

Loss:
    - BCE con pos_weight per bilanciare classi sbilanciate

Usage:
    python train_stage1.py --config configs/config_sha.yaml
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
from datasets import Crowd, collate_fn
from datasets.transforms import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomApply


# =============================================================================
# LOSS STAGE 1
# =============================================================================

class Stage1Loss(nn.Module):
    """
    Loss per Stage 1: BCE per classificazione π (vuoto/pieno).
    
    Args:
        pos_weight: Peso per la classe "pieno" (blocchi pieni sono pochi)
        block_size: Dimensione del blocco (stride del backbone)
    """
    
    def __init__(self, pos_weight: float = 3.0, block_size: int = 16):
        super().__init__()
        self.pos_weight = pos_weight
        self.block_size = block_size
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]),
            reduction='mean'
        )
    
    def compute_gt_occupancy(self, gt_density: torch.Tensor) -> torch.Tensor:
        """Calcola GT occupancy per blocco."""
        gt_counts_per_block = F.avg_pool2d(
            gt_density,
            kernel_size=self.block_size,
            stride=self.block_size
        ) * (self.block_size ** 2)
        return (gt_counts_per_block > 0.5).float()
    
    def forward(self, predictions, gt_density):
        """
        Args:
            predictions: Dict con 'logit_pi' o 'zip_outputs'
            gt_density: [B, 1, H, W]
        """
        # Estrai logit_pi
        if "zip_outputs" in predictions:
            logit_pi = predictions["zip_outputs"]["logit_pi"]
        else:
            logit_pi = predictions.get("logit_pi", predictions.get("pi"))
        
        # Se π è già sigmoid, riconverti a logit (approssimazione)
        if logit_pi.min() >= 0 and logit_pi.max() <= 1:
            logit_pi = torch.logit(logit_pi.clamp(1e-6, 1-1e-6))
        
        # Se è [B, 2, H, W], prendi canale 1 (prob "pieno")
        if logit_pi.shape[1] == 2:
            logit_occupied = logit_pi[:, 1:2, :, :]
        else:
            # Se è [B, 1, H, W] rappresenta P(vuoto), inverti
            logit_occupied = -logit_pi
        
        gt_occupancy = self.compute_gt_occupancy(gt_density)
        
        # Allinea dimensioni
        if gt_occupancy.shape[-2:] != logit_occupied.shape[-2:]:
            gt_occupancy = F.interpolate(
                gt_occupancy,
                size=logit_occupied.shape[-2:],
                mode='nearest'
            )
        
        # Sposta pos_weight sul device corretto
        if self.bce.pos_weight.device != logit_occupied.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logit_occupied.device)
        
        loss = self.bce(logit_occupied, gt_occupancy)
        
        # Metriche
        with torch.no_grad():
            pred_occupied = (torch.sigmoid(logit_occupied) > 0.5).float()
            accuracy = (pred_occupied == gt_occupancy).float().mean()
            
            # Confusion matrix
            tp = ((pred_occupied == 1) & (gt_occupancy == 1)).sum()
            tn = ((pred_occupied == 0) & (gt_occupancy == 0)).sum()
            fp = ((pred_occupied == 1) & (gt_occupancy == 0)).sum()
            fn = ((pred_occupied == 0) & (gt_occupancy == 1)).sum()
            
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        loss_dict = {
            "loss": loss.detach(),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        
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
    
    total_loss = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    n_batches = 0
    
    use_amp = config["TRAIN_STAGE1"].get("AMP", False) and scaler is not None
    grad_clip = config["TRAIN_STAGE1"].get("CLIP_GRAD_NORM", 1.0)
    log_interval = config.get("EXP", {}).get("LOG_INTERVAL", 50)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, points, densities) in enumerate(pbar):
        images = images.to(device)
        densities = densities.to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=use_amp):
            # Forward
            outputs = model(images, return_intermediates=True)
            loss, loss_dict = criterion(outputs, densities)
        
        # Backward
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
        
        # Accumula metriche
        total_loss += loss.item()
        total_acc += loss_dict["accuracy"].item()
        total_f1 += loss_dict["f1"].item()
        n_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{loss_dict['accuracy'].item():.3f}",
            "f1": f"{loss_dict['f1'].item():.3f}",
        })
    
    return {
        "loss": total_loss / n_batches,
        "accuracy": total_acc / n_batches,
        "f1": total_f1 / n_batches,
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
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    n_batches = 0
    
    # Per MAE indicativo
    total_mae = 0.0
    n_samples = 0
    
    for images, points, densities in tqdm(dataloader, desc="Validating"):
        images = images.to(device)
        densities = densities.to(device)
        
        outputs = model(images, return_intermediates=True)
        loss, loss_dict = criterion(outputs, densities)
        
        total_loss += loss.item()
        total_acc += loss_dict["accuracy"].item()
        total_precision += loss_dict["precision"].item()
        total_recall += loss_dict["recall"].item()
        total_f1 += loss_dict["f1"].item()
        n_batches += 1
        
        # MAE indicativo
        pred_count = outputs["pred_count"]
        for i, pts in enumerate(points):
            gt = len(pts) if pts is not None else 0
            pred = pred_count[i].item()
            total_mae += abs(pred - gt)
            n_samples += 1
    
    return {
        "loss": total_loss / n_batches,
        "accuracy": total_acc / n_batches,
        "precision": total_precision / n_batches,
        "recall": total_recall / n_batches,
        "f1": total_f1 / n_batches,
        "mae": total_mae / n_samples if n_samples > 0 else 0,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Stage 1 - π-Head")
    parser.add_argument("--config", type=str, default="configs/config_sha.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    
    # Carica config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device(config.get("DEVICE", "cuda"))
    torch.manual_seed(config.get("SEED", 42))
    np.random.seed(config.get("SEED", 42))
    
    print("=" * 60)
    print("🚀 STAGE 1: π-Head Training (Zero-Inflation)")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Dataset: {config['DATASET']}")
    
    # Directories
    run_name = config.get("RUN_NAME", "experiment")
    output_dir = os.path.join(config["EXP"]["OUT_DIR"], run_name, "stage1")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}")
    
    # Dataset
    dataset_name = config["DATASET"]
    data_cfg = config["DATA"]
    
    crop_size = data_cfg.get("CROP_SIZE_STAGE1", data_cfg.get("CROP_SIZE", 448))
    crop_scale = data_cfg.get("CROP_SCALE_STAGE1", data_cfg.get("CROP_SCALE", [0.5, 1.5]))
    
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
    
    train_cfg = config["TRAIN_STAGE1"]
    
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
        density_mode="zip_gated_ebc",
    ).to(device)
    
    # Congela EBC head per Stage 1
    model.freeze_ebc_head()
    
    print(f"Trainable params: {model.get_trainable_params():,}")
    
    # Loss
    loss_cfg = config.get("LOSS_STAGE1", {})
    criterion = Stage1Loss(
        pos_weight=loss_cfg.get("POS_WEIGHT", 3.0),
        block_size=data_cfg.get("ZIP_BLOCK_SIZE", 16),
    )
    
    # Optimizer
    param_groups = [
        {
            "params": [p for p in model.backbone.parameters() if p.requires_grad],
            "lr": train_cfg.get("LR_BACKBONE", 1e-5),
            "name": "backbone",
        },
        {
            "params": [p for p in model.zip_head.parameters() if p.requires_grad],
            "lr": train_cfg.get("LR_PI_HEAD", 1e-4),
            "name": "pi_head",
        },
    ]
    
    optimizer = AdamW(param_groups, weight_decay=1e-4)
    
    # Scheduler
    warmup_epochs = train_cfg.get("WARMUP_EPOCHS", 5)
    total_epochs = train_cfg["EPOCHS"]
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    main_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
    
    # AMP
    scaler = GradScaler() if train_cfg.get("AMP", False) else None
    
    # Resume
    start_epoch = 1
    best_f1 = 0.0
    
    if args.resume or train_cfg.get("RESUME_LAST", False):
        ckpt_path = os.path.join(output_dir, "last.pth")
        if os.path.exists(ckpt_path):
            print(f"📂 Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"], strict=False)
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            best_f1 = ckpt.get("best_f1", 0.0)
    
    # Training loop
    print(f"\n🏋️ Training for {total_epochs} epochs...")
    
    for epoch in range(start_epoch, total_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{total_epochs} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config, scaler
        )
        
        scheduler.step()
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.3f}, F1: {train_metrics['f1']:.3f}")
        
        # Validate
        val_interval = train_cfg.get("VAL_INTERVAL", 10)
        if epoch % val_interval == 0 or epoch == total_epochs:
            val_metrics = validate(model, val_loader, criterion, device, config)
            
            print(f"Val - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.3f}, F1: {val_metrics['f1']:.3f}")
            print(f"      Precision: {val_metrics['precision']:.3f}, "
                  f"Recall: {val_metrics['recall']:.3f}, MAE: {val_metrics['mae']:.2f}")
            
            # Save best
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_f1": best_f1,
                    "metrics": val_metrics,
                }, os.path.join(output_dir, "best_stage1_model.pth"))
                print(f"⭐ New best F1: {best_f1:.4f}")
        
        # Save last
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_f1": best_f1,
        }, os.path.join(output_dir, "last.pth"))
    
    print(f"\n{'='*60}")
    print(f"✅ Stage 1 Training Complete!")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Checkpoint: {output_dir}/best_stage1_model.pth")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
