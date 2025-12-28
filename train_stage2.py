"""
Train Stage 2: CLIP-EBC Head Training
======================================
Replica del training CLIP-EBC ufficiale.

Modifiche rispetto alla versione originale:
- DACELoss (CE + Count Loss) invece di CLIPEBCLoss semplice
- CosineAnnealingWarmRestarts con warmup
- Forward stage2 (senza ZIP gating)
- Passa i punti alla loss per count preciso

Uso:
    python train_stage2.py --config ./config_sha.yaml
"""

import argparse
import yaml
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from functools import partial

# --- IMPORTS ---
from models.zip_clip_ebc_model import ZIPCLIPEBCModel
from datasets.sha import SHA
from datasets.transforms import build_transforms
from utils.train_utils import AverageMeter, seed_everything


# =============================================================================
# DACE LOSS (Official CLIP-EBC Style)
# =============================================================================

class DACELoss(nn.Module):
    """
    DACE Loss = Cross-Entropy + Count Loss (MAE)
    
    Versione semplificata della loss ufficiale CLIP-EBC.
    """
    
    def __init__(self, bins, bin_centers, weight_count: float = 1.0, reduction: int = 8):
        super().__init__()
        
        self.bins = bins
        self.num_bins = len(bins)
        self.reduction = reduction
        self.weight_count = weight_count
        
        # Cross-Entropy
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")
        
        # Anchor points
        self.register_buffer(
            "bin_centers",
            torch.tensor(bin_centers, dtype=torch.float32)
        )
    
    def _get_bin_labels(self, block_counts):
        """Converte conteggi per blocco in indici di bin."""
        if block_counts.dim() == 4:
            block_counts = block_counts.squeeze(1)  # [B, H, W]
        
        labels = torch.zeros_like(block_counts, dtype=torch.long)
        
        for idx, (low, high) in enumerate(self.bins):
            high_val = float('inf') if high > 9000 else high
            mask = (block_counts >= low) & (block_counts <= high_val)
            labels[mask] = idx
        
        return labels
    
    def _density_to_blocks(self, density, target_size):
        """Riduce density map a conteggi per blocco."""
        B, C, H, W = density.shape
        tH, tW = target_size
        
        if H == tH and W == tW:
            return density
        
        # Sum pooling
        scale_h = H // tH
        scale_w = W // tW
        
        if scale_h > 0 and scale_w > 0 and H % tH == 0 and W % tW == 0:
            return F.avg_pool2d(density, kernel_size=(scale_h, scale_w)) * (scale_h * scale_w)
        else:
            return F.adaptive_avg_pool2d(density, (tH, tW)) * (H * W) / (tH * tW)
    
    def forward(self, logits, gt_density, points=None):
        """
        Args:
            logits: [B, num_bins, H, W] - logits classificazione
            gt_density: [B, 1, H_full, W_full] - GT density map
            points: List[Tensor] - punti GT (opzionale, per count loss preciso)
        """
        B, C, H, W = logits.shape
        device = logits.device
        
        # Riduci density a dimensione output
        gt_blocks = self._density_to_blocks(gt_density, (H, W))
        
        # === Cross-Entropy Loss ===
        target_labels = self._get_bin_labels(gt_blocks)  # [B, H, W]
        
        # Flatten per CE
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        labels_flat = target_labels.reshape(-1)  # [B*H*W]
        
        ce_loss = self.ce_loss(logits_flat, labels_flat)
        
        # === Count Loss (MAE) ===
        bin_probs = F.softmax(logits, dim=1)
        centers = self.bin_centers.view(1, -1, 1, 1).to(device)
        pred_density = (bin_probs * centers).sum(dim=1, keepdim=True)
        
        pred_count = pred_density.sum(dim=(1, 2, 3))
        
        if points is not None:
            gt_count = torch.tensor([len(p) for p in points], dtype=torch.float32, device=device)
        else:
            gt_count = gt_blocks.sum(dim=(1, 2, 3))
        
        count_loss = F.l1_loss(pred_count, gt_count)
        
        # === Total ===
        total_loss = ce_loss + self.weight_count * count_loss
        
        return total_loss, {
            'ce_loss': ce_loss.detach(),
            'count_loss': count_loss.detach(),
            'pred_count': pred_count.mean().detach(),
            'gt_count': gt_count.mean().detach()
        }


# =============================================================================
# LR SCHEDULER (Warmup + CosineAnnealingWarmRestarts)
# =============================================================================

def get_lr_lambda(epoch, base_lr, warmup_epochs, warmup_lr, T_0, T_mult, eta_min):
    """LR con warmup lineare + cosine annealing warm restarts."""
    if epoch < warmup_epochs:
        return (warmup_lr + (base_lr - warmup_lr) * epoch / warmup_epochs) / base_lr
    else:
        epoch_shifted = epoch - warmup_epochs
        if T_mult == 1:
            T_cur = epoch_shifted % T_0
            T_i = T_0
        else:
            n = int(math.log((epoch_shifted / T_0 * (T_mult - 1) + 1), T_mult))
            T_cur = epoch_shifted - T_0 * (T_mult ** n - 1) / (T_mult - 1)
            T_i = T_0 * T_mult ** n
        lr = eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
        return lr / base_lr


# =============================================================================
# UTILITIES
# =============================================================================

def crowd_collate(batch):
    """Gestisce batch con dimensioni variabili."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'density': torch.stack([item['density'] for item in batch]),
        'points': [item['points'] for item in batch],
        'img_path': [item['img_path'] for item in batch]
    }


# =============================================================================
# TRAIN FUNCTION
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, writer):
    model.train()
    losses = AverageMeter()
    ce_losses = AverageMeter()
    count_losses = AverageMeter()
    
    pbar = tqdm(loader, desc=f"Ep {epoch+1} Train", leave=False)
    
    for i, batch in enumerate(pbar):
        if batch is None:
            continue
        
        images = batch['image'].to(device)
        gt_density = batch['density'].to(device)
        points = batch['points']  # Per count loss preciso
        
        # Forward - SOLO EBC HEAD (senza ZIP gating)
        outputs = model.forward_stage2(images)
        logits = outputs['ebc_logits']
        
        # Loss con punti per count preciso
        loss, loss_info = criterion(logits, gt_density, points)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update meters
        losses.update(loss.item(), images.size(0))
        ce_losses.update(loss_info['ce_loss'].item(), images.size(0))
        count_losses.update(loss_info['count_loss'].item(), images.size(0))
        
        pbar.set_postfix({
            'loss': f"{losses.avg:.4f}",
            'ce': f"{ce_losses.avg:.4f}",
            'cnt': f"{count_losses.avg:.2f}"
        })
        
        # Tensorboard
        global_step = epoch * len(loader) + i
        if i % 20 == 0:
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            writer.add_scalar("Train/CE_Loss", loss_info['ce_loss'].item(), global_step)
            writer.add_scalar("Train/Count_Loss", loss_info['count_loss'].item(), global_step)

    return losses.avg


# =============================================================================
# VALIDATION FUNCTION
# =============================================================================

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    mae_sum = 0.0
    mse_sum = 0.0
    count = 0
    
    for batch in tqdm(loader, desc="Validation", leave=False):
        if batch is None:
            continue
        
        images = batch['image'].to(device)
        gt_counts = torch.tensor([len(p) for p in batch['points']], dtype=torch.float32, device=device)
        
        # Forward EBC only
        outputs = model.forward_stage2(images)
        logits = outputs['ebc_logits']
        
        # Calcola count da logits
        bin_probs = F.softmax(logits, dim=1)
        bin_centers = model.clip_ebc_head.bin_centers.view(1, -1, 1, 1).to(device)
        pred_density = (bin_probs * bin_centers).sum(dim=1, keepdim=True)
        pred_counts = pred_density.sum(dim=(1, 2, 3))
        
        # Accumula
        diff = pred_counts - gt_counts
        mae_sum += torch.abs(diff).sum().item()
        mse_sum += (diff ** 2).sum().item()
        count += len(gt_counts)
    
    mae = mae_sum / count if count > 0 else 0
    rmse = math.sqrt(mse_sum / count) if count > 0 else 0
    
    return mae, rmse


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    args = parser.parse_args()

    # Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    seed_everything(config.get('SEED', 42))
    
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  STAGE 2: CLIP-EBC HEAD TRAINING")
    print(f"{'='*60}")
    print(f"🖥️  Device: {device}")
    
    dataset_name = config['DATASET']
    
    # Paths
    train_cfg = config['TRAIN_STAGE2']
    save_dir = train_cfg.get('SAVE_DIR', f"./checkpoints/{dataset_name}/stage2")
    log_dir = f"./logs/{dataset_name}/stage2"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    
    # =================================================================
    # DATASET
    # =================================================================
    data_cfg = config['DATA']
    print(f"\n📂 Dataset: {data_cfg['ROOT']}")
    
    train_dataset = SHA(data_cfg['ROOT'], split='train', 
                       transforms=build_transforms(data_cfg, is_train=True))
    val_dataset = SHA(data_cfg['ROOT'], split='val', 
                     transforms=build_transforms(data_cfg, is_train=False))
    
    batch_size = train_cfg.get('BATCH_SIZE', 8)
    num_workers = train_cfg.get('NUM_WORKERS', 4)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=crowd_collate,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1,
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=crowd_collate
    )
    
    print(f"   Train: {len(train_dataset)} | Val: {len(val_dataset)} | Batch: {batch_size}")
    
    # =================================================================
    # MODEL
    # =================================================================
    print(f"\n🤖 Building model...")
    model = ZIPCLIPEBCModel(config).to(device)
    
    # Carica Stage 1 (se esiste) - OPZIONALE per Stage 2
    stage1_ckpt = f"./checkpoints/{dataset_name}/stage1/best_model.pth"
    if os.path.exists(stage1_ckpt):
        print(f"📦 Loading Stage 1: {stage1_ckpt}")
        ckpt = torch.load(stage1_ckpt, map_location=device)
        model.load_state_dict(ckpt['state_dict'], strict=False)
    else:
        print("⚠️  Stage 1 checkpoint not found (training from scratch - OK for Stage 2)")

    # Freeze/Unfreeze
    model.freeze_for_stage(2)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable params: {trainable:,}")
    
    # =================================================================
    # LOSS
    # =================================================================
    bins = [tuple(b) for b in config['BINS']]
    bin_centers = config['BIN_CENTERS']
    reduction = config.get('REDUCTION', 8)
    
    loss_cfg = config.get('LOSS_STAGE2', {})
    weight_count = loss_cfg.get('WEIGHT_COUNT_LOSS', 1.0)
    
    criterion = DACELoss(
        bins=bins,
        bin_centers=bin_centers,
        weight_count=weight_count,
        reduction=reduction
    ).to(device)
    
    print(f"\n🎯 Loss: DACELoss (CE + MAE)")
    print(f"   Bins: {len(bins)} | Weight count: {weight_count}")
    
    # =================================================================
    # OPTIMIZER
    # =================================================================
    lr = float(train_cfg.get('LR', 1e-4))
    weight_decay = float(train_cfg.get('WEIGHT_DECAY', 1e-4))
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # =================================================================
    # SCHEDULER (CosineAnnealingWarmRestarts)
    # =================================================================
    warmup_epochs = train_cfg.get('WARMUP_EPOCHS', 50)
    warmup_lr = float(train_cfg.get('WARMUP_LR', 1e-6))
    T_0 = train_cfg.get('T_0', 5)
    T_mult = train_cfg.get('T_MULT', 2)
    eta_min = float(train_cfg.get('ETA_MIN', 1e-7))
    
    lr_lambda = partial(
        get_lr_lambda,
        base_lr=lr,
        warmup_epochs=warmup_epochs,
        warmup_lr=warmup_lr,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=eta_min
    )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    print(f"\n📈 Scheduler: CosineAnnealingWarmRestarts")
    print(f"   Warmup: {warmup_epochs} epochs | T_0={T_0}, T_mult={T_mult}")
    
    # =================================================================
    # TRAINING LOOP
    # =================================================================
    total_epochs = train_cfg.get('TOTAL_EPOCHS', 2600)
    eval_start = train_cfg.get('EVAL_START', 50)
    eval_freq = train_cfg.get('EVAL_FREQ', 1)
    
    best_mae = float('inf')
    best_rmse = float('inf')
    
    print(f"\n🚀 Training for {total_epochs} epochs (eval from epoch {eval_start})")
    print(f"{'='*60}\n")
    
    for epoch in range(total_epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        
        # Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{total_epochs} | Loss: {train_loss:.4f} | LR: {current_lr:.2e}")
        
        # Validation
        do_eval = (epoch >= eval_start) and ((epoch - eval_start) % eval_freq == 0)
        do_eval = do_eval or (epoch + 1 == total_epochs)
        
        if do_eval:
            mae, rmse = validate(model, val_loader, device)
            print(f"  📊 Val | MAE: {mae:.2f} | RMSE: {rmse:.2f}")
            
            writer.add_scalar("Val/MAE", mae, epoch)
            writer.add_scalar("Val/RMSE", rmse, epoch)
            
            # Save best MAE
            if mae < best_mae:
                best_mae = mae
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_mae': best_mae,
                    'best_rmse': best_rmse,
                    'config': config
                }, os.path.join(save_dir, "best_mae.pth"))
                print(f"  ⭐ New Best MAE: {best_mae:.2f}")
            
            # Save best RMSE
            if rmse < best_rmse:
                best_rmse = rmse
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_mae': best_mae,
                    'best_rmse': best_rmse,
                    'config': config
                }, os.path.join(save_dir, "best_rmse.pth"))
                print(f"  ⭐ New Best RMSE: {best_rmse:.2f}")
        
        # Save last
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_mae': best_mae,
            'best_rmse': best_rmse,
        }, os.path.join(save_dir, "last.pth"))
    
    writer.close()
    
    print(f"\n{'='*60}")
    print(f"  ✅ Training completed!")
    print(f"     Best MAE: {best_mae:.2f}")
    print(f"     Best RMSE: {best_rmse:.2f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()