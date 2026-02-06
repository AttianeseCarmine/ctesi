# train_stage3_clip_vit_b_16.py
# Stage 3 (Joint ZIP + CLIP) - Full Logging, Sliding Window & Best Metric Tracking

import os, sys, json, yaml, shutil
import logging
from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import math

current_dir = os.path.abspath(os.path.dirname(__file__))

# --- MONKEY PATCH PER CLIP ---
try:
    import models.clip.utils as clip_utils
    def fixed_format_count(val, prompt_type):
        left, right = val
        if prompt_type == "word": return clip_utils.num2word(left), clip_utils.num2word(right)
        return left, right
    clip_utils.format_count = fixed_format_count
except ImportError: pass

from datasets import standardize_dataset_name
from models import get_model
from models.zip_model import ZIPModel
from models.joint_model import ZIPCLIPJointModel
from utils import setup, cleanup, init_seeds, get_config, barrier, get_dataloader, get_writer, update_train_result

try: from utils import get_loss_fn
except Exception: get_loss_fn = None

# --- LOGGER SETUP ---
def setup_logger(output_dir):
    """Configura il logger per scrivere su file e console."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger()
    
    # Rimuovi handler precedenti per evitare duplicati
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # File Handler
    file_handler = logging.FileHandler(os.path.join(output_dir, "train.log"))
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    return logger

# --- SLIDING WINDOW PREDICT (Logica Stage 2) ---
def sliding_window_predict(model, image, window_size, stride):
    """
    Esegue la predizione patch-based per mantenere l'accuratezza del ViT su immagini grandi.
    """
    model.eval()
    B, C, H, W = image.shape
    assert B == 1, "Validation sliding window requires batch_size=1"

    # Se l'immagine è piccola, passiamo diretti
    if H <= window_size and W <= window_size:
        return model(image)['final_density']

    # Calcolo Padding
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
    
    _, _, H_pad, W_pad = image.shape
    
    # Recuperiamo il fattore di riduzione
    reduction = getattr(model, 'reduction', 8)
    if isinstance(model, DDP): reduction = getattr(model.module, 'reduction', 8)

    output_density = torch.zeros((1, 1, H_pad // reduction, W_pad // reduction), device=image.device)
    count_map = torch.zeros((1, 1, H_pad // reduction, W_pad // reduction), device=image.device)

    # Ciclo Sliding Window
    with torch.no_grad():
        for y in range(0, H_pad, stride):
            for x in range(0, W_pad, stride):
                # Crop
                crop = image[:, :, y:y+window_size, x:x+window_size]
                
                # Safety check
                if crop.shape[2] != window_size or crop.shape[3] != window_size:
                    continue

                with autocast(enabled=True):
                    out = model(crop)
                    pred_crop = out['final_density']
                
                # Posizionamento
                y_out = y // reduction
                x_out = x // reduction
                h_c = pred_crop.shape[2]
                w_c = pred_crop.shape[3]
                
                output_density[:, :, y_out:y_out+h_c, x_out:x_out+w_c] += pred_crop
                count_map[:, :, y_out:y_out+h_c, x_out:x_out+w_c] += 1

    # Media sulle sovrapposizioni
    output_density = output_density / (count_map + 1e-6)
    
    # Rimuovi padding finale
    final_h = H // reduction
    final_w = W // reduction
    return output_density[:, :, :final_h, :final_w]

# --- SMART LOADER ---
def smart_load_weights(model, checkpoint_path, logger, model_name="Model"):
    if not os.path.exists(checkpoint_path):
        msg = f"❌ File non trovato: {checkpoint_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)
        
    logger.info(f"📦 [{model_name}] Loading weights from: {checkpoint_path}")
    try: ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except: ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    # Se mancano troppe chiavi, provo a fixare i prefissi
    if len(missing) > 10:
        logger.warning(f"⚠️ [{model_name}] Direct load missed {len(missing)} keys. Attempting prefix fix...")
        new_state = {}
        for k, v in state_dict.items():
            k_new = k
            if k_new.startswith("model."): k_new = k_new.replace("model.", "", 1)
            elif k_new.startswith("backbone."): k_new = k_new.replace("backbone.", "", 1)
            elif k_new.startswith("module."): k_new = k_new.replace("module.", "", 1)
            new_state[k_new] = v
        missing, unexpected = model.load_state_dict(new_state, strict=False)
    
    logger.info(f"📊 [{model_name}] Report -> Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    if len(missing) > 60: 
        logger.error(f"🚨 CRITICAL ERROR: {model_name} load failed. Too many missing keys.")
        raise RuntimeError(f"Weight loading failed for {model_name}")
    else:
        logger.info(f"✅ [{model_name}] Loaded successfully!")

# --- PARSER ---
parser = ArgumentParser("Train Stage 3")
parser.add_argument("--config_s1", type=str, required=True)
parser.add_argument("--config_s2", type=str, required=True)
parser.add_argument("--s1", type=str, required=True)
parser.add_argument("--s2", type=str, required=True)
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--input_size", type=int, default=224)
parser.add_argument("--reduction", type=int, default=8)
parser.add_argument("--dataset", type=str)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-6)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--total_epochs", type=int, default=100)
parser.add_argument("--eval_start", type=int, default=1)
parser.add_argument("--eval_freq", type=int, default=1)
parser.add_argument("--max_steepness", type=float, default=2.0)
parser.add_argument("--base_steepness", type=float, default=1.0)
parser.add_argument("--zip_w", type=float, default=0.001)
parser.add_argument("--cons_w", type=float, default=0.01)
parser.add_argument("--amp", action="store_true")
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--seed", type=int, default=42)
# Dummy args
parser.add_argument("--regression", action="store_true")
parser.add_argument("--truncation", type=int, default=4)
parser.add_argument("--anchor_points", type=str, default="average")
parser.add_argument("--prompt_type", type=str, default="word")
parser.add_argument("--granularity", type=str, default="fine")
parser.add_argument("--num_vpt", type=int, default=32)
parser.add_argument("--vpt_drop", type=float, default=0.0)
parser.add_argument("--shallow_vpt", action="store_true")
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
parser.add_argument("--sliding_window", action="store_true")
parser.add_argument("--stride", type=int, default=None)
parser.add_argument("--window_size", type=int, default=None)
parser.add_argument("--resize_to_multiple", action="store_true")
parser.add_argument("--zero_pad_to_multiple", action="store_true")
parser.add_argument("--weight_count_loss", type=float, default=1.0)
parser.add_argument("--count_loss", type=str, default="dmcount")

def load_configs(args):
    merged = {}
    with open(args.config_s1, "r") as f:
        c1 = yaml.load(f, Loader=yaml.UnsafeLoader)
        if "BACKBONE" in c1: merged["BACKBONE"] = c1["BACKBONE"]
        elif "model" in c1: merged["BACKBONE"] = {"TYPE": c1["model"]}
        if "ZIP_HEAD" in c1: merged["ZIP_HEAD"] = c1["ZIP_HEAD"]
        if "reduction" in c1: merged["reduction"] = c1["reduction"]
    with open(args.config_s2, "r") as f:
        c2 = yaml.load(f, Loader=yaml.UnsafeLoader)
        for k, v in c2.items():
            if k not in ["BACKBONE", "ZIP_HEAD"]: merged[k] = v
    for k, v in merged.items():
        if hasattr(args, k.lower()) and f"--{k.lower()}" not in sys.argv: setattr(args, k.lower(), v)
    if args.out is None: args.out = os.path.join(current_dir, "checkpoints", args.dataset, "stage3_aligned")
    return args, merged

def build_bins(args):
    if args.regression: return None, None
    try:
        with open(os.path.join(current_dir, "configs", f"reduction_{args.reduction}.json"), "r") as f:
            cfg = json.load(f)[str(args.truncation)][args.dataset]
        bins = [(float(b[0]), float(b[1])) for b in cfg["bins"][args.granularity]]
        anchors = [float(p) for p in (cfg["anchor_points"][args.granularity]["average"] if args.anchor_points=="average" else cfg["anchor_points"][args.granularity]["middle"])]
        return bins, anchors
    except:
        if hasattr(args, 'bins') and args.bins: return args.bins, args.anchor_points
        return None, None

def run(rank, nprocs, args, merged_cfg):
    setup(rank, nprocs)
    init_seeds(args.seed + rank)
    device = torch.device(f"cuda:{rank}")
    
    # 1. SETUP OUTPUT E LOGGER
    args.ckpt_dir = args.out
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    logger = None
    if rank == 0:
        logger = setup_logger(args.ckpt_dir)
        logger.info(f"🚀 Training Started. Checkpoints will be saved to: {args.ckpt_dir}")
        
        # Salvataggio Config
        config_save_path = os.path.join(args.ckpt_dir, "config.yaml")
        final_config = vars(args).copy()
        final_config.update(merged_cfg)
        with open(config_save_path, "w") as f:
            yaml.dump(final_config, f)
        logger.info(f"💾 Configuration saved.")

    # 2. BUILD MODELS
    zip_cfg = {"BACKBONE": merged_cfg.get("BACKBONE", {"TYPE": "vit_b_16"}), "ZIP_HEAD": merged_cfg.get("ZIP_HEAD", {}), "REDUCTION": int(args.reduction)}
    stage1 = ZIPModel(zip_cfg).to(device)
    
    stage2 = get_model(
        backbone=args.model, input_size=args.input_size, reduction=args.reduction,
        bins=args.bins, anchor_points=args.anchor_points, prompt_type=args.prompt_type,
        num_vpt=args.num_vpt, vpt_drop=args.vpt_drop, deep_vpt=not args.shallow_vpt
    ).to(device)
    
    if rank == 0:
        logger.info("="*40)
        smart_load_weights(stage1, args.s1, logger, "Stage1 (ZIP)")
        smart_load_weights(stage2, args.s2, logger, "Stage2 (CLIP)")
        logger.info("="*40)
    
    barrier(nprocs > 1) 

    # 3. FREEZE CLIP
    for p in stage2.parameters(): p.requires_grad = False
    for n, p in stage2.named_parameters():
        if any(x in n for x in ["vpt", "head", "prompt"]): p.requires_grad = True
            
    # 4. JOINT MODEL
    model = ZIPCLIPJointModel(stage1, stage2, steepness=args.base_steepness).to(device)
    model.reduction = args.reduction
    
    if nprocs > 1: model = DDP(nn.SyncBatchNorm.convert_sync_batchnorm(model), device_ids=[rank])
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)
    
    train_loader, _ = get_dataloader(args, 'train', (nprocs > 1))
    
    # Validation (batch_size=1)
    import argparse
    args_val = argparse.Namespace(**vars(args))
    args_val.batch_size = 1 
    if rank == 0: val_loader = get_dataloader(args_val, 'val', False)
    
    best_mae = float('inf')
    best_rmse = float('inf')
    best_epoch = 0
    
    # 5. LOOP
    for epoch in range(1, args.total_epochs + 1):
        model.train()
        if args.total_epochs > 1:
            curr_steep = args.base_steepness + (epoch / args.total_epochs) * (args.max_steepness - args.base_steepness)
        else: curr_steep = args.max_steepness
        (model.module if nprocs > 1 else model).steepness = curr_steep

        if rank == 0: pbar = tqdm(train_loader, desc=f"Ep {epoch}")
        else: pbar = train_loader
        
        for batch in pbar:
            if isinstance(batch, (list, tuple)):
                imgs, points, gt = batch
            else:
                imgs = batch['image']
                gt = batch['density']
            
            imgs = imgs.to(device)
            gt = gt.to(device)
            
            with autocast(enabled=args.amp):
                out = model(imgs)
                pred = out['final_density']
                
                if pred.shape[-2:] != gt.shape[-2:]:
                    gt_s = F.interpolate(gt, size=pred.shape[-2:], mode='bilinear', align_corners=False) 
                    gt_s *= (gt.shape[-2]*gt.shape[-1]) / (pred.shape[-2]*pred.shape[-1])
                else: gt_s = gt
                
                l_main = F.mse_loss(pred, gt_s)
                l_cons = (out['raw_density'].detach() * (1 - out['pi_prob'])).mean()
                
                h, w = out['pi_logits'].shape[-2:]
                gt_pi = F.interpolate(gt, size=(h, w), mode='bilinear', align_corners=False)
                mask_target = (gt_pi * ((gt.shape[-2]*gt.shape[-1])/(h*w)) > 0.01).float()
                l_zip = F.binary_cross_entropy_with_logits(out['pi_logits'], mask_target)

                loss = l_main + (args.cons_w * l_cons) + (args.zip_w * l_zip)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            if rank == 0: 
                pbar.set_postfix(mae=f"{abs(pred.sum()-gt_s.sum()).item():.1f}", loss=f"{loss.item():.2f}")
            
        # 6. EVALUATION
        if rank == 0 and epoch >= args.eval_start and (epoch % args.eval_freq == 0):
            model.eval()
            mae, rmse = 0, 0
            count = 0
            
            logger.info(f"🔍 Validating Epoch {epoch} (Sliding Window)...")
            
            for batch in tqdm(val_loader, desc="Validating"):
                if isinstance(batch, (list, tuple)):
                    imgs, points, _ = batch
                else:
                    imgs = batch['image']
                    points = batch['points']
                
                imgs = imgs.to(device)
                gt_cnt = len(points[0])
                
                pred_density = sliding_window_predict(
                    model.module if nprocs > 1 else model, 
                    imgs, 
                    window_size=args.input_size, 
                    stride=args.input_size 
                )
                pred_cnt = pred_density.sum().item()
                
                err = pred_cnt - gt_cnt
                mae += abs(err)
                rmse += err**2
                count += 1
            
            if count > 0:
                mae /= count
                rmse = (rmse / count) ** 0.5
            
            # --- TRACKING BEST ---
            if mae < best_mae:
                best_mae = mae
                best_rmse = rmse
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "best_model.pth"))
                logger.info(f"🌟 NEW BEST MODEL SAVED! (Epoch {epoch})")
            
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "last_model.pth"))
            
            # --- PRINT LOGS ---
            logger.info("-" * 50)
            logger.info(f"📈 Epoch {epoch} Summary:")
            logger.info(f"   Current MAE:  {mae:.4f} | RMSE: {rmse:.4f}")
            logger.info(f"   🏆 BEST MAE:  {best_mae:.4f} | RMSE: {best_rmse:.4f} (at Epoch {best_epoch})")
            logger.info("-" * 50)

    cleanup()

if __name__ == "__main__":
    args, merged = load_configs(parser.parse_args())
    args.bins, args.anchor_points = build_bins(args)
    os.makedirs(args.out, exist_ok=True)
    nprocs = torch.cuda.device_count()
    if nprocs > 1: mp.spawn(run, nprocs=nprocs, args=(nprocs, args, merged))
    else: run(0, 1, args, merged)