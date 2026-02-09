
"""
---------------------------------------------------------------------------
TRAIN STAGE 1: Gating Network Pre-training (Zero-Inflated Branch)
---------------------------------------------------------------------------
Author: Carmine Attianese
Thesis: Enhancing Crowd Counting in Complex Scenes via Zero-Inflated Vision-Language Models

Description:
This script performs the first stage of training, focusing on the binary 
classification task (Background vs. Crowd). It trains the 'pi_head' (Gating Network) 
to identify informative regions in the image.

Key Objectives:
1. Optimize Binary Focal Loss to handle class imbalance (mostly background).
2. Maximize Recall (minimize False Negatives) to ensure the subsequent 
   counting branch receives all valid crowd regions.
---------------------------------------------------------------------------
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import os
import yaml
import sys
import shutil
import logging

# Import dai tuoi moduli
from datasets import standardize_dataset_name
from models.zip_model import ZIPModel 
from utils import setup, cleanup, init_seeds, get_logger, get_config, barrier, reduce_mean
from utils import get_dataloader, load_checkpoint, get_writer, update_train_result, log

# =============================================================================
# 1. LOSS: Binary Focal Loss (Bilanciata per Recall)
# =============================================================================
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=4.0):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha  
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# =============================================================================
# 2. CARICAMENTO CONFIGURAZIONE
# =============================================================================
def load_config_and_update_args(args):
    if not args.config: return args, {}
    with open(args.config, 'r') as f: cfg = yaml.safe_load(f)
    def is_passed(k): return f"--{k}" in sys.argv
    mapping = {
        "dataset": ["DATASET"], "input_size": ["INPUT_SIZE"], "reduction": ["REDUCTION"],
        "model": ["BACKBONE", "TYPE"], "batch_size": ["TRAIN_STAGE1", "BATCH_SIZE"],
        "num_workers": ["TRAIN_STAGE1", "NUM_WORKERS"], "lr": ["TRAIN_STAGE1", "LR_HEAD"],
        "lr_backbone": ["TRAIN_STAGE1", "LR_BACKBONE"], "total_epochs": ["TRAIN_STAGE1", "TOTAL_EPOCHS"],
    }
    for arg, path in mapping.items():
        if not is_passed(arg):
            try:
                v = cfg
                for k in path: v = v[k]
                setattr(args, arg, v)
            except: pass
    return args, cfg

# =============================================================================
# 3. ROBUST LOGIT EXTRACTION
# =============================================================================
def extract_logits(outputs):
    if isinstance(outputs, dict):
        # Cerchiamo logit_pi (nuovo) o pi_logits (vecchio)
        for key in ['logit_pi', 'pi_logits', 'pi']:
            if key in outputs: return outputs[key]
        raise KeyError(f"Output del modello non riconosciuto. Chiavi: {list(outputs.keys())}")
    return outputs

# =============================================================================
# 4. TRAINING & VALIDATION ENGINES
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, rank):
    model.train()
    total_loss = 0.0
    it = tqdm(loader, desc="Training", disable=(rank != 0))
    for batch in it:
        images, _, target_density = batch if not isinstance(batch, dict) else (batch['image'], None, batch['density'])
        images, target_density = images.to(device), target_density.to(device)
        optimizer.zero_grad()
        with autocast(device_type='cuda', enabled=True):
            outputs = model(images)
            pi_logits = extract_logits(outputs)
            h_out, w_out = pi_logits.shape[-2:]
            H, W = target_density.shape[-2:]
            r_h, r_w = H // h_out, W // w_out
            gt_block = F.max_pool2d(target_density, kernel_size=(r_h, r_w), stride=(r_h, r_w))
            target_binary = (gt_block > 0.05).float() 
            loss = criterion(pi_logits, target_binary)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        if rank == 0: it.set_postfix({'loss': f"{loss.item():.4f}"})
    return total_loss / len(loader)

@torch.no_grad()
def validate_combined(model, loader, device, min_recall=0.80):
    """
    Validates the model by scanning multiple thresholds.
    Strategy: Prioritizes models that maintain high Recall (safety) while maximizing F1.
    """
    model.eval()
    all_logits, all_targets = [], []
    for batch in loader:
        images, _, target_density = batch if not isinstance(batch, dict) else (batch['image'], None, batch['density'])
        images, target_density = images.to(device), target_density.to(device)
        outputs = model(images)
        pi_logits = extract_logits(outputs)
        h_out, w_out = pi_logits.shape[-2:]
        H, W = target_density.shape[-2:]
        r_h, r_w = H // h_out, W // w_out
        gt_block = F.max_pool2d(target_density, kernel_size=(r_h, r_w), stride=(r_h, r_w))
        gt_binary = (gt_block > 1e-3).float()
        all_logits.append(pi_logits.flatten().cpu())
        all_targets.append(gt_binary.flatten().cpu())

    probs = torch.sigmoid(torch.cat(all_logits))
    targets = torch.cat(all_targets)
    
    candidates = []
    for t in np.linspace(0.05, 0.7, 15): # Scansione ampia per trovare il gate perfetto
        preds = (probs > t).float()
        tp = ((preds == 1) & (targets == 1)).sum().item()
        fn = ((preds == 0) & (targets == 1)).sum().item()
        fp = ((preds == 1) & (targets == 0)).sum().item()
        
        rec = tp / (tp + fn + 1e-7)
        prec = tp / (tp + fp + 1e-7)
        f1 = 2 * (prec * rec) / (prec + rec + 1e-7)
        candidates.append({'rec': rec, 'f1': f1, 'prec': prec, 'threshold': t})
            
    # LOGICA DI SELEZIONE:
    # 1. Filtra quelli con Recall > min_recall (80%)
    valid_candidates = [c for c in candidates if c['rec'] >= min_recall]
    
    if valid_candidates:
        # 2. Tra quelli sicuri, prendi quello con la F1 migliore
        best_res = max(valid_candidates, key=lambda x: x['f1'])
    else:
        # Fallback: se nessuno arriva all'80%, prendi quello con la Recall più alta
        best_res = max(candidates, key=lambda x: x['rec'])
        
    return best_res

# =============================================================================
# 5. MAIN RUNNER
# =============================================================================
def run(rank, nprocs, args, full_cfg):
    setup(rank, nprocs)
    init_seeds(args.seed + rank)
    device = torch.device(f"cuda:{rank}")

    model_cfg = full_cfg.copy()
    model_cfg["BACKBONE"] = {"TYPE": args.model}
    model = ZIPModel(model_cfg).to(device)

    is_vit = 'vit' in args.model.lower()
    lr_backbone = args.lr_backbone if args.lr_backbone > 0 else (2e-5 if is_vit else 1e-5)
    lr_head = args.lr if args.lr else 1e-4
    
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': lr_backbone},
        {'params': (model.zip_head if hasattr(model, 'zip_head') else model.pi_head).parameters(), 'lr': lr_head}
    ], weight_decay=0.05 if is_vit else 1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_epochs)
    scaler = GradScaler(enabled=args.amp)
    criterion = BinaryFocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)

    train_loader, sampler = get_dataloader(args, split="train", ddp=(nprocs > 1))
    val_loader = None
    if rank == 0:
        val_loader = get_dataloader(args, split="val", ddp=False)
        logger = get_logger(os.path.join(args.out, "train.log"))
        with open(os.path.join(args.out, "config_stage1.yaml"), 'w') as f:
            yaml.dump(vars(args), f)
        logger.info(f"🚀 RECALL & F1 TARGET TRAINING | Backbone: {args.model}")

    if nprocs > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank])

    # Tracking dei Best
    best_f1 = 0.0
    best_recall_at_f1 = 0.0
    best_epoch = 0

    for epoch in range(1, args.total_epochs + 1):
        if sampler: sampler.set_epoch(epoch)
        loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, rank)
        scheduler.step()

        if rank == 0:
            stats = validate_combined(model.module if nprocs > 1 else model, val_loader, device, min_recall=0.80)
            
            # Stampa livedo nel .out
            print(f"\n[Epoch {epoch}] Loss: {loss:.4f} | Recall: {stats['rec']:.4f} | F1: {stats['f1']:.3f} | Threshold: {stats['threshold']:.2f}")
            logger.info(f"Epoch {epoch:03d} | Loss: {loss:.4f} | R: {stats['rec']:.4f} | F1: {stats['f1']:.3f} | Best F1: {best_f1:.3f}")

            # Salvataggio: Priorità F1 (ma la funzione validate_combined ha già filtrato per Recall > 80%)
            if stats['f1'] > best_f1:
                best_f1 = stats['f1']
                best_recall_at_f1 = stats['rec']
                best_epoch = epoch
                torch.save((model.module if nprocs > 1 else model).state_dict(), os.path.join(args.out, "best_model.pth"))
                
                print(f"🌟 NUOVO BEST! Epoca {epoch}: F1 {best_f1:.3f} (Recall {best_recall_at_f1:.3f})")
                logger.info(f"🌟 Saved New Best Model (F1: {best_f1:.3f}, R: {best_recall_at_f1:.3f})")

    if rank == 0:
        print("\n" + "="*50)
        print(f"🏁 TRAINING COMPLETATO")
        print(f"🏆 Miglior Risultato (Epoca {best_epoch}):")
        print(f"   - F1-Score: {best_f1:.4f}")
        print(f"   - Recall:   {best_recall_at_f1:.4f}")
        print("="*50)

    cleanup()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="sha")
    parser.add_argument("--model", type=str, default="vit_b_16")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--total_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1.0e-5)
    parser.add_argument("--focal_alpha", type=float, default=0.65)
    parser.add_argument("--focal_gamma", type=float, default=3.0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=str, default="12355")
    # Parametri Augmentation completi per evitare crash
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--reduction", type=int, default=8)
    parser.add_argument("--min_scale", type=float, default=1.0)
    parser.add_argument("--max_scale", type=float, default=2.0)
    parser.add_argument("--num_crops", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--brightness", type=float, default=0.1)
    parser.add_argument("--contrast", type=float, default=0.1)
    parser.add_argument("--saturation", type=float, default=0.1)
    parser.add_argument("--hue", type=float, default=0.0)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--jitter_prob", type=float, default=0.2)
    parser.add_argument("--blur_prob", type=float, default=0.2)
    parser.add_argument("--noise_prob", type=float, default=0.3)
    parser.add_argument("--saltiness", type=float, default=1e-3)
    parser.add_argument("--spiciness", type=float, default=1e-3)
    parser.add_argument("--sliding_window", action="store_true")
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--window_size", type=int, default=None)

    args, _ = parser.parse_known_args()
    args, full_cfg = load_config_and_update_args(args)
    os.makedirs(args.out, exist_ok=True)
    nprocs = torch.cuda.device_count()
    if nprocs > 1: mp.spawn(run, nprocs=nprocs, args=(nprocs, args, full_cfg))
    else: run(0, 1, args, full_cfg)