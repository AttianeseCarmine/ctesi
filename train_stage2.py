import argparse
import yaml
import os
import sys
import shutil  # <--- AGGIUNTO L'IMPORT MANCANTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math

# --- IMPORTS ---
from models.zip_clip_ebc_model import ZIPCLIPEBCModel
from datasets.shha import SHHA
from datasets.transforms import build_transforms
from losses.clip_ebc_loss import CLIPEBCLoss 
from utils.train_utils import AverageMeter, seed_everything

# =============================================================================
# UTILITIES
# =============================================================================

def crowd_collate(batch):
    """Gestisce batch con dimensioni variabili (punti)"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    images = torch.stack([item['image'] for item in batch])
    densities = torch.stack([item['density'] for item in batch])
    # I punti rimangono una lista perché hanno lunghezza variabile
    points = [item['points'] for item in batch]
    paths = [item['img_path'] for item in batch]
    return {'image': images, 'density': densities, 'points': points, 'img_path': paths}

def get_optimizer(model, config):
    """Costruisce l'optimizer basandosi sul config"""
    train_cfg = config['TRAIN_STAGE2']
    
    # Parametri separati per backbone e testa
    params = [
        {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': float(train_cfg.get('LR_BACKBONE', 1e-5))},
        {'params': [p for p in model.clip_ebc_head.parameters() if p.requires_grad], 'lr': float(train_cfg.get('LR_EBC_HEAD', 1e-4))}
    ]
    
    opt_name = train_cfg.get('OPTIMIZER', 'adamw').lower()
    weight_decay = float(train_cfg.get('WEIGHT_DECAY', 1e-4))
    
    if opt_name == 'sgd':
        return optim.SGD(params, momentum=0.9, weight_decay=weight_decay)
    elif opt_name == 'adam':
        return optim.Adam(params, weight_decay=weight_decay)
    else:
        return optim.AdamW(params, weight_decay=weight_decay)

def get_scheduler(optimizer, config, steps_per_epoch):
    """Costruisce lo scheduler (opzionale)"""
    train_cfg = config['TRAIN_STAGE2']
    epochs = train_cfg['EPOCHS']
    sched_name = train_cfg.get('SCHEDULER', 'cosine').lower()
    
    if sched_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)
    elif sched_name == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=train_cfg.get('LR_STEP', 30), gamma=0.1)
    return None

# =============================================================================
# TRAIN LOOP (Singola Epoca)
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch, writer):
    model.train()
    # Importante: Pi-Head e BatchNorm del backbone devono restare in eval se congelati/già trainati
    model.pi_head.eval() 
    
    losses = AverageMeter()
    
    loader_bar = tqdm(loader, desc=f"Train Ep {epoch}", leave=False)
    
    for i, batch in enumerate(loader_bar):
        if batch is None: continue
        
        images = batch['image'].to(device)
        gt_density = batch['density'].to(device)

        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Selezione output corretta
        if 'ebc_logits' in outputs: logits = outputs['ebc_logits']
        elif 'clip_ebc_logits' in outputs: logits = outputs['clip_ebc_logits']
        else: logits = outputs.get('logits') # Fallback

        # Loss calculation
        loss_out = criterion(logits, gt_density)
        
        # Gestione Tupla (loss, stats)
        if isinstance(loss_out, (tuple, list)):
            loss = loss_out[0]
        else:
            loss = loss_out
            
        loss.backward()
        
        # Clip Gradients (opzionale ma consigliato per VLM)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler: scheduler.step()

        losses.update(loss.item(), images.size(0))
        loader_bar.set_postfix(loss=f"{losses.avg:.4f}")
        
        # Log frequente su TensorBoard
        global_step = epoch * len(loader) + i
        if i % 10 == 0:
            writer.add_scalar("Train/Loss_Step", loss.item(), global_step)
            writer.add_scalar("Train/LR", optimizer.param_groups[0]['lr'], global_step)

    return losses.avg

# =============================================================================
# VALIDATION LOOP
# =============================================================================
def validate(model, loader, bin_centers, device):
    model.eval()
    mae_meter = AverageMeter()
    rmse_meter = AverageMeter()
    
    # Reshape bins per broadcasting: [1, N_bins, 1, 1]
    bins_view = bin_centers.view(1, -1, 1, 1).to(device)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            if batch is None: continue
            
            images = batch['image'].to(device)
            gt_density = batch['density'].to(device)
            
            outputs = model(images)
            
            if 'ebc_logits' in outputs: logits = outputs['ebc_logits']
            else: logits = outputs.get('clip_ebc_logits')
            
            # Decoding: Expected Value = Sum( Prob_i * Value_i )
            probs = torch.softmax(logits, dim=1) 
            pred_map = (probs * bins_view).sum(dim=1)
            
            # Conteggi totali
            pred_count = pred_map.sum().item()
            gt_count = gt_density.sum().item()
            
            # Aggiorna metriche
            error = abs(gt_count - pred_count)
            mae_meter.update(error, images.size(0))
            rmse_meter.update(error**2, images.size(0))
            
    rmse = math.sqrt(rmse_meter.avg)
    return mae_meter.avg, rmse

# =============================================================================
# MAIN
# =============================================================================
def train_stage2():
    parser = argparse.ArgumentParser(description='Stage 2: Train CLIP-EBC')
    parser.add_argument('--config', type=str, default='configs/config_sha.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    # 1. Config & Setup
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    seed_everything(config.get('SEED', 42))
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    dataset_name = config.get('DATASET', 'dataset')
    
    # Directory Setup
    base_save_dir = config.get('save_dir', './checkpoints')
    save_dir = os.path.join(base_save_dir, dataset_name, 'stage2')
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    print(f"🚀 [Stage 2] Training VLM on {dataset_name}")
    print(f"📂 Checkpoints: {save_dir}")
    print(f"📈 Logs: {log_dir}")

    # 2. Data
    data_cfg = config['DATA']
    t_cfg = config['TRAIN_STAGE2']
    
    train_ds = SHHA(root=data_cfg['ROOT'], split='train', transforms=build_transforms(data_cfg, True)) 
    val_ds = SHHA(root=data_cfg['ROOT'], split='val', transforms=build_transforms(data_cfg, False))

    train_loader = DataLoader(train_ds, batch_size=t_cfg['BATCH_SIZE'], shuffle=True, 
                              num_workers=t_cfg.get('NUM_WORKERS', 4), collate_fn=crowd_collate, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, 
                            num_workers=t_cfg.get('NUM_WORKERS', 4), collate_fn=crowd_collate)

    # 3. Model
    # NOTA: Questo inizializza il modello con pesi di default (Backbone ImageNet + CLIP OpenAI).
    # NON carica lo Stage 1, il che è corretto per questo step.
    model = ZIPCLIPEBCModel(config).to(device)
    
    # Congelamento Strategico
    for p in model.pi_head.parameters(): p.requires_grad = False
    for p in model.backbone.parameters(): p.requires_grad = True # Sblocchiamo backbone
    for p in model.clip_ebc_head.parameters(): p.requires_grad = True

    # 4. Optimizer & Loss
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config, len(train_loader))

    # --- Setup Loss (Fix: Bins from Config) ---
    if 'BIN_CENTERS' not in config: raise ValueError("BIN_CENTERS missing in config")
    
    bin_centers = torch.tensor(config['BIN_CENTERS'], dtype=torch.float32).to(device)
    bins_list = config['BINS']

    print(f"⚙️ Loss setup: {len(bins_list)} bins, centers={bin_centers.shape}")
    criterion = CLIPEBCLoss(config, bin_centers).to(device)
    criterion.bins = bins_list # Inject python list
    # ------------------------------------------

    best_mae = float('inf')
    start_epoch = 0
    epochs = t_cfg['EPOCHS']
    val_interval = t_cfg.get('VAL_INTERVAL', 1)

    # 5. Training Loop Completo
    for epoch in range(start_epoch, epochs):
        
        # --- TRAIN ---
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, writer)
        
        writer.add_scalar("Train/Loss_Epoch", train_loss, epoch)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f}")

        # --- VALIDATE ---
        if (epoch + 1) % val_interval == 0 or (epoch + 1) == epochs:
            val_mae, val_rmse = validate(model, val_loader, bin_centers, device)
            
            print(f"📊 Validation: MAE {val_mae:.2f} | RMSE {val_rmse:.2f}")
            writer.add_scalar("Val/MAE", val_mae, epoch)
            writer.add_scalar("Val/RMSE", val_rmse, epoch)

            # --- SAVE ---
            is_best = val_mae < best_mae
            if is_best: 
                best_mae = val_mae
                print(f"⭐ New Best MAE: {best_mae:.2f}")
            
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mae': best_mae,
                'config': config
            }
            
            # Save Last
            torch.save(state, os.path.join(save_dir, 'last_model.pth'))
            
            # Save Best
            if is_best:
                shutil.copyfile(os.path.join(save_dir, 'last_model.pth'), 
                              os.path.join(save_dir, 'best_model.pth'))

    writer.close()
    print("✅ Training Stage 2 Completed.")

if __name__ == '__main__':
    train_stage2()