import os
import sys
import json
import yaml
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# --- IMPORTS ---
# Assumiamo che i file siano al loro posto
from datasets import standardize_dataset_name
from models import get_model
from models.zip_model import ZIPModel
from models.joint_model import ZIPCLIPJointModel 
from losses.joint_loss import JointLoss
from utils import setup, cleanup, init_seeds, get_dataloader, barrier

# ==============================================================================
# 1. FREEZING STRATEGY
# ==============================================================================
def freeze_parameters_refined(model):
    print("\n🔒 Freezing Strategy: Refined (P2R Style)")
    
    # Congela tutto
    for param in model.parameters():
        param.requires_grad = False
    
    # Norm layers in eval mode
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            module.eval() 

    trainable_params = []
    seen_params = set()

    # --- Sblocca le Teste (Heads) ---
    head_params = []
    head_params += list(model.stage1.zip_head.parameters()) # ZIP Head
    
    # CLIP Heads (cerca decoder/regressor)
    clip = model.stage2
    for name, module in clip.named_modules():
        if any(x in name for x in ['decoder', 'head', 'proj', 'regressor']):
             for p in module.parameters(): 
                 if id(p) not in seen_params:
                     head_params.append(p)
                     seen_params.add(id(p))
                     p.requires_grad = True
    
    if head_params:
        trainable_params.append({'params': head_params, 'lr_scale': 1.0})
        print(f"   ✅ Heads: {len(head_params)} params unfrozen")

    # --- Sblocca Ultimo Blocco Backbone (Opzionale) ---
    backbone_params = []
    # Gestione ViT vs ResNet
    if hasattr(model.stage1.backbone, 'blocks'): # ViT
        for p in model.stage1.backbone.blocks[-1].parameters():
            if id(p) not in seen_params:
                backbone_params.append(p)
                seen_params.add(id(p))
                p.requires_grad = True
    elif hasattr(model.stage1.backbone, 'layer4'): # ResNet
        for p in model.stage1.backbone.layer4.parameters():
            if id(p) not in seen_params:
                backbone_params.append(p)
                seen_params.add(id(p))
                p.requires_grad = True

    if backbone_params:
        trainable_params.append({'params': backbone_params, 'lr_scale': 0.1})
        print(f"   ✅ Backbone (Last Block): {len(backbone_params)} params unfrozen")

    return trainable_params

# ==============================================================================
# 2. TRAINING LOOP
# ==============================================================================
def train_epoch(model, loader, optimizer, scaler, criterion, device, rank):
    model.train()
    for m in model.modules(): # Force eval per BN/LN
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)): m.eval()
        
    pbar = tqdm(loader, desc="Train") if rank == 0 else loader
    avg_loss = 0
    
    for batch in pbar:
        # Gestione input flessibile (tuple o dict)
        if isinstance(batch, dict):
            imgs = batch["image"].to(device)
            gt_density = batch["density"].to(device)
        else:
            imgs, _, gt_density = batch
            imgs = imgs.to(device)
            gt_density = gt_density.to(device)
            
        optimizer.zero_grad()
        
        with autocast(enabled=(scaler is not None)):
            outputs = model(imgs)
            loss, loss_dict = criterion(outputs, {'density': gt_density})
            
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        avg_loss += loss.item()
        if rank == 0:
            pbar.set_postfix(L=f"{loss.item():.3f}", C=f"{loss_dict['l_count']:.3f}")
            
    return avg_loss / len(loader)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    mae = 0
    mse = 0
    total = 0
    
    for batch in loader:
        if isinstance(batch, dict):
            imgs = batch["image"].to(device)
            points = batch.get("points")
        else:
            imgs, points, _ = batch
            imgs = imgs.to(device)
            
        res = model(imgs)
        # Somma della densità finale per il conteggio
        pred_count = res['final_density'].sum(dim=(1,2,3))
        
        if points is not None:
            gt_counts = torch.tensor([len(p) for p in points], device=device)
        else:
            continue
            
        err = torch.abs(pred_count - gt_counts)
        mae += err.sum().item()
        mse += (err ** 2).sum().item()
        total += imgs.size(0)
        
    return mae/total, (mse/total)**0.5

# ==============================================================================
# 3. MAIN RUNNER
# ==============================================================================
def run(rank, nprocs, args):
    setup(rank, nprocs)
    device = torch.device(f"cuda:{rank}")
    
    # --- Config Stage 2 (Bins) ---
    bins = None
    if args.c2:
        if rank == 0: print(f"📂 Reading Stage 2 Config: {args.c2}")
        with open(args.c2) as f: 
            c2 = yaml.full_load(f) # full_load per tuple python
        if 'bins' in c2: bins = c2['bins']
        # Override input size se presente
        if 'input_size' in c2: args.input_size = c2['input_size']

    # --- Dataset ---
    args.bins = bins # Hack per passare bins al dataloader se serve
    train_res = get_dataloader(args, split="train", ddp=(nprocs>1))
    train_loader = train_res[0] if isinstance(train_res, tuple) else train_res
    
    if rank == 0:
        val_res = get_dataloader(args, split="val", ddp=False)
        val_loader = val_res[0] if isinstance(val_res, tuple) else val_res

    # --- Model Building ---
    # 1. ZIP
    with open(args.config) as f: z_cfg = yaml.full_load(f)
    stage1 = ZIPModel(z_cfg).to(device)
    # Load weights ZIP
    s1 = torch.load(args.s1, map_location='cpu')
    stage1.load_state_dict(s1.get('model', s1), strict=False)
    
    # 2. CLIP (Costruzione Esplicita)
    anchor_points = [float(i) for i in range(len(bins))] if bins else None
    
    stage2 = get_model(
        backbone=args.model, 
        input_size=args.input_size, 
        reduction=args.reduction,
        bins=bins, 
        anchor_points=anchor_points, 
        prompt_type=args.prompt_type,
        num_vpt=args.num_vpt, 
        vpt_drop=args.vpt_drop, 
        deep_vpt=not args.shallow_vpt
    ).to(device)
    
    # Load weights CLIP
    s2 = torch.load(args.s2, map_location='cpu')
    stage2.load_state_dict(s2.get('model', s2), strict=False)
    
    # 3. Joint Model
    model = ZIPCLIPJointModel(stage1, stage2).to(device)
    if nprocs > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # --- Optimizer ---
    params = freeze_parameters_refined(model.module if nprocs > 1 else model)
    optimizer = torch.optim.AdamW([
        {'params': g['params'], 'lr': args.lr * g['lr_scale']} for g in params
    ], weight_decay=1e-4)
    
    scaler = GradScaler()
    criterion = JointLoss().to(device)
    
    # --- Loop ---
    best_mae = float('inf')
    if args.out and not os.path.exists(args.out) and rank == 0:
        os.makedirs(args.out)

    for ep in range(args.epochs):
        if nprocs > 1: train_loader.sampler.set_epoch(ep)
        
        if rank == 0: print(f"\n--- Epoch {ep+1}/{args.epochs} ---")
        
        train_epoch(model, train_loader, optimizer, scaler, criterion, device, rank)
        barrier(nprocs>1)
        
        if rank == 0 and (ep+1) % args.eval_freq == 0:
            mae, rmse = validate(model, val_loader, device)
            print(f"📊 Val MAE: {mae:.2f} (Best: {best_mae:.2f})")
            
            if mae < best_mae:
                best_mae = mae
                if args.out:
                    torch.save(model.state_dict(), f"{args.out}/best_model_joint.pth")
                    print("🌟 Model Saved!")

    cleanup(nprocs>1)

# ==============================================================================
# PARSER SETUP
# ==============================================================================
def main():
    parser = argparse.ArgumentParser("Train Stage 3 (P2R Final)")
    
    # Files
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--c2", type=str, default=None)
    parser.add_argument("--s1", type=str, required=True)
    parser.add_argument("--s2", type=str, required=True)
    parser.add_argument("--out", type=str, default="checkpoints/stage3")
    
    # Model Params
    parser.add_argument("--model", type=str, default="clip_resnet50")
    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--reduction", type=int, default=8)
    parser.add_argument("--prompt_type", type=str, default="word")
    parser.add_argument("--num_vpt", type=int, default=32)
    parser.add_argument("--vpt_drop", type=float, default=0.0)
    parser.add_argument("--shallow_vpt", action="store_true")
    
    # Training Params
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    
    # --- AUGMENTATION PARAMS (Mancavano e causavano AttributeError) ---
    parser.add_argument("--num_crops", type=int, default=1)
    parser.add_argument("--min_scale", type=float, default=1.0)
    parser.add_argument("--max_scale", type=float, default=2.0)
    parser.add_argument("--brightness", type=float, default=0.1)
    parser.add_argument("--contrast", type=float, default=0.1)
    parser.add_argument("--saturation", type=float, default=0.1)
    parser.add_argument("--hue", type=float, default=0.0)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--saltiness", type=float, default=1e-3)
    parser.add_argument("--spiciness", type=float, default=1e-3)
    parser.add_argument("--jitter_prob", type=float, default=0.2)
    parser.add_argument("--blur_prob", type=float, default=0.2)
    parser.add_argument("--noise_prob", type=float, default=0.5)
    
    # Dataloader specifics
    parser.add_argument("--truncation", type=int, default=4)
    parser.add_argument("--granularity", type=str, default="fine")
    parser.add_argument("--anchor_points", type=str, default="average")
    parser.add_argument("--sliding_window", action="store_true")
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--resize_to_multiple", action="store_true")
    parser.add_argument("--zero_pad_to_multiple", action="store_true")

    args = parser.parse_args()
    
    # Standardize Dataset
    args.dataset = standardize_dataset_name(args.dataset)
    
    # Launch
    args.nprocs = torch.cuda.device_count()
    if args.nprocs > 1:
        mp.spawn(run, nprocs=args.nprocs, args=(args.nprocs, args))
    else:
        run(0, 1, args)

if __name__ == '__main__':
    main()