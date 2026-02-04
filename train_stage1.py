import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import os
import yaml
import sys
import shutil

current_dir = os.path.abspath(os.path.dirname(__file__))

from datasets import standardize_dataset_name
from models.zip_model import ZIPModel 
from utils import setup, cleanup, init_seeds, get_logger, get_config, barrier, reduce_mean
from utils import get_dataloader, load_checkpoint, get_writer, update_train_result, log

# =============================================================================
# 1. IMPLEMENTAZIONE FOCAL LOSS (Ispirata a p2r_zip)
# =============================================================================
class BinaryFocalLoss(nn.Module):
    """
    Focal Loss per classificazione binaria.
    alpha: Bilanciamento classi (es. 0.25 significa che il background pesa meno).
           Per il tuo caso (poche patch piene), alpha > 0.5 può aiutare la Recall.
    gamma: Fattore di focalizzazione (es. 2.0). Più è alto, più il modello ignora
           gli esempi facili e si concentra su quelli difficili.
    """
    def __init__(self, alpha=0.5, gamma=2.0, logits=True, reduce=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        pt = torch.exp(-bce_loss)
        
        # Se targets=1 usa alpha, se targets=0 usa (1-alpha)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Formula Focal Loss standard
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduce:
            return focal_loss.mean()
        else:
            return focal_loss

# =============================================================================
# ARGUMENT PARSER
# =============================================================================
parser = ArgumentParser(description="Train ZIP Stage 1 (Focal Loss).")

# Configurazione
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--model", type=str, default="vit_b_16")
parser.add_argument("--input_size", type=int, default=448)
parser.add_argument("--reduction", type=int, default=16)
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--dataset", type=str, required=False)
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--resume", type=str, default=None)

# Parametri Training
parser.add_argument("--lr", type=float, default=1e-4, help="LR Head")
parser.add_argument("--lr_backbone", type=float, default=1e-5, help="LR Backbone (basso per ViT)")
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--total_epochs", type=int, default=2000)

# --- FOCAL LOSS PARAMETERS ---
# Consiglio: alpha=0.75 aiuta a mantenere alta la Recall (dà più peso ai positivi)
# Consiglio: gamma=2.0 aiuta la Precision (penalizza i falsi positivi difficili)
parser.add_argument("--focal_alpha", type=float, default=0.75, help="Alpha for Focal Loss")
parser.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma for Focal Loss")

parser.add_argument("--target_recall", type=float, default=0.90, help="Recall target per la selezione del modello")

# Augmentations
parser.add_argument("--num_crops", type=int, default=1)
parser.add_argument("--min_scale", type=float, default=1.0)
parser.add_argument("--max_scale", type=float, default=2.0)
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

# Eval
parser.add_argument("--eval_freq", type=int, default=5)
parser.add_argument("--save_freq", type=int, default=5)
parser.add_argument("--save_best_k", type=int, default=3)
parser.add_argument("--amp", action="store_true")
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--sliding_window", action="store_true")
parser.add_argument("--stride", type=int, default=None)
parser.add_argument("--window_size", type=int, default=None)
parser.add_argument("--resize_to_multiple", action="store_true")
parser.add_argument("--zero_pad_to_multiple", action="store_true")

def load_config_and_update_args(args):
    if not args.config: return args
    with open(args.config, 'r') as f: cfg = yaml.safe_load(f)
    def is_passed(k): return f"--{k}" in sys.argv
    mapping = {
        "dataset": ["DATASET"], "input_size": ["INPUT_SIZE"], "reduction": ["REDUCTION"],
        "model": ["BACKBONE", "TYPE"], "batch_size": ["TRAIN_STAGE1", "BATCH_SIZE"],
        "num_workers": ["TRAIN_STAGE1", "NUM_WORKERS"], "lr": ["TRAIN_STAGE1", "LR_HEAD"],
        "lr_backbone": ["TRAIN_STAGE1", "LR_BACKBONE"], "total_epochs": ["TRAIN_STAGE1", "TOTAL_EPOCHS"],
        "amp": ["TRAIN_STAGE1", "AMP"],
        "reduction": ["TRAIN_STAGE1", "REDUCTION"],
    }

    for arg, path in mapping.items():
        if not is_passed(arg):
            try:
                v = cfg
                for k in path: v = v[k]
                setattr(args, arg, v)
            except: pass

        # Fallback: se TRAIN_STAGE1.REDUCTION non esiste, usa REDUCTION globale
    if (not is_passed("reduction")) and (getattr(args, "reduction", None) is None):
        try:
            print("Usando REDUCTION globale dal file di train.")
            args.reduction = cfg["REDUCTION"]
        except Exception:
            pass

    return args

def save_checkpoint(state, is_best, save_dir, filename='last_model.pth'):
    os.makedirs(save_dir, exist_ok=True)
    last_path = os.path.join(save_dir, filename)
    torch.save(state, last_path)
    if is_best: shutil.copyfile(last_path, os.path.join(save_dir, 'best_model.pth'))

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, rank, nprocs):
    model.train()
    total_loss = 0.0
    it = tqdm(loader, desc="Train Focal") if rank == 0 else loader
    
    for batch in it:
        if isinstance(batch, dict):
            images, target_density = batch['image'].to(device), batch['density'].to(device)
        else:
            images, _, target_density = batch
            images, target_density = images.to(device), target_density.to(device)

        optimizer.zero_grad()

        with autocast(enabled=scaler is not None):
            outputs = model(images)
            pi_logits = outputs['pi_logits'] if isinstance(outputs, dict) else outputs

            # --- CLEAN LABEL GENERATION ---
            h_out, w_out = pi_logits.shape[-2:]
            H, W = target_density.shape[-2:]
            r_h, r_w = H // h_out, W // w_out
            
            # Max pooling per trovare picchi
            gt_block = F.max_pool2d(target_density, kernel_size=(r_h, r_w), stride=(r_h, r_w))
            
            # 1. Usa una soglia di rumore più alta (1e-3 invece di 1e-5) per pulire le etichette
            target_binary = (gt_block > 5e-4).float()

            loss = criterion(pi_logits, target_binary)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        curr_loss = loss.item()
        if nprocs > 1:
            curr_loss = reduce_mean(torch.tensor(curr_loss, device=device), nprocs).item()
        
        total_loss += curr_loss
        if rank == 0: it.set_postfix({'loss': f"{curr_loss:.4f}"})

    return {'loss': total_loss / len(loader)}

@torch.no_grad()
def evaluate_stage1_constrained(model, loader, device, target_recall=0.90):
    model.eval()
    all_logits, all_targets = [], []
    
    for batch in tqdm(loader, desc="Eval S1", disable=False):
        if isinstance(batch, dict):
            images, target_density = batch['image'].to(device), batch['density'].to(device)
        else:
            images, _, target_density = batch
            images, target_density = images.to(device), target_density.to(device)

        outputs = model(images)
        pi_logits = outputs['pi_logits'] if isinstance(outputs, dict) else outputs

        h_out, w_out = pi_logits.shape[-2:]
        H, W = target_density.shape[-2:]
        r_h, r_w = H // h_out, W // w_out
        
        gt_block = F.max_pool2d(target_density, kernel_size=(r_h, r_w), stride=(r_h, r_w))
        # Coerenza con il training: soglia 1e-3
        gt_binary = (gt_block > 1e-3).float()

        all_logits.append(pi_logits.flatten().cpu())
        all_targets.append(gt_binary.flatten().cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    probs = torch.sigmoid(all_logits)

    thresholds = np.arange(0.05, 0.96, 0.01)
    best_res = {'f1': 0.0, 'acc': 0.0, 'prec': 0.0, 'rec': 0.0, 'threshold': 0.5}
    candidates = []
    
    for t in thresholds:
        pred_binary = (probs > t).float()
        tp = ((pred_binary == 1) & (all_targets == 1)).sum().item()
        tn = ((pred_binary == 0) & (all_targets == 0)).sum().item()
        fp = ((pred_binary == 1) & (all_targets == 0)).sum().item()
        fn = ((pred_binary == 0) & (all_targets == 1)).sum().item()
        
        ep = 1e-8
        prec = tp / (tp + fp + ep)
        rec = tp / (tp + fn + ep)
        f1 = 2 * (prec * rec) / (prec + rec + ep)
        acc = (tp + tn) / (tp + tn + fp + fn + ep)
        
        candidates.append({'f1': f1, 'acc': acc, 'prec': prec, 'rec': rec, 'threshold': t})

    # Filtra i candidati che rispettano la recall minima
    valid = [c for c in candidates if c['rec'] >= (target_recall - 0.01)]
    
    if valid:
        # Tra i validi, scegli quello con la MIGLIORE F1 (che implica migliore Precision dato che Recall è fissa)
        best_res = max(valid, key=lambda x: x['f1'])
        status = "✅ Target Met"
    else:
        # Fallback
        best_res = max(candidates, key=lambda x: x['rec'])
        status = "⚠️ Target Missed"

    print(f"\n{status} -> Sel: T={best_res['threshold']:.2f} | F1={best_res['f1']:.4f} | R={best_res['rec']:.4f} | P={best_res['prec']:.4f}")
    return best_res

def run(local_rank, nprocs, args):
    init_seeds(args.seed + local_rank)
    setup(local_rank, nprocs)
    device = f"cuda:{local_rank}" if local_rank != -1 else "cuda:0"
    
    cfg = vars(args)
    cfg["BACKBONE"] = {"TYPE": args.model}
    model = ZIPModel(cfg).to(device)
    
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': args.lr_backbone},
        {'params': (model.zip_head if hasattr(model, 'zip_head') else model.pi_head).parameters(), 'lr': args.lr}
    ], weight_decay=args.weight_decay)

    # --- USA FOCAL LOSS ---
    loss_fn = BinaryFocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)
    scaler = GradScaler() if args.amp else None

    if args.out: args.ckpt_dir = args.out
    else: args.ckpt_dir = os.path.join(current_dir, "checkpoints", args.dataset, args.model, "stage1_focal")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    train_loader, sampler = get_dataloader(args, split="train", ddp=(nprocs > 1))
    if local_rank == 0:
        val_loader = get_dataloader(args, split="val", ddp=False)
        writer = get_writer(args.ckpt_dir)
        logger = get_logger(os.path.join(args.ckpt_dir, "train.log"))
        with open(os.path.join(args.ckpt_dir, "config_stage1.yaml"), 'w') as f:
            yaml.dump(vars(args), f)

    if nprocs > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])

    best_metric = 0.0 

    for epoch in range(1, args.total_epochs + 1):
        if sampler: sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device, local_rank, nprocs)
        barrier(nprocs > 1)

        if local_rank == 0:
            update_train_result(epoch, train_stats, writer)
            log(logger, epoch, args.total_epochs, loss_info=train_stats)

            if epoch % args.eval_freq == 0:
                eval_model = model.module if nprocs > 1 else model
                val_stats = evaluate_stage1_constrained(eval_model, val_loader, device, target_recall=args.target_recall)
                
                log(logger, epoch, args.total_epochs, 
                    message=f"Eval | T={val_stats['threshold']:.2f} F1={val_stats['f1']:.3f} R={val_stats['rec']:.3f} P={val_stats['prec']:.3f}")

                for k, v in val_stats.items(): writer.add_scalar(f"val/{k}", v, epoch)

                if val_stats['f1'] > best_metric:
                    best_metric = val_stats['f1']
                    print("🌟 New Best Model!")
                    save_checkpoint({
                        "epoch": epoch,
                        "model_state_dict": (model.module if nprocs > 1 else model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_f1": best_metric,
                        "optimal_threshold": val_stats['threshold'],
                        "args": vars(args)
                    }, True, args.ckpt_dir)
                else:
                    save_checkpoint({
                        "epoch": epoch,
                        "model_state_dict": (model.module if nprocs > 1 else model).state_dict(),
                        "loss_info": train_stats
                    }, False, args.ckpt_dir)

    cleanup(nprocs > 1)

def main():
    args = load_config_and_update_args(parser.parse_args())
    if not args.dataset: raise ValueError("Dataset missing")
    args.dataset = standardize_dataset_name(args.dataset)
    args.nprocs = torch.cuda.device_count()
    if args.nprocs > 1: mp.spawn(run, nprocs=args.nprocs, args=(args.nprocs, args))
    else: run(0, 1, args)

if __name__ == "__main__":
    main()