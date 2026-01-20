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
import json
import shutil  # Necessario per copyfile

# Ottieni la directory corrente per i path relativi
current_dir = os.path.abspath(os.path.dirname(__file__))

# Import specifici del progetto
from datasets import standardize_dataset_name
from models.zip_model import ZIPModel 
from utils import setup, cleanup, init_seeds, get_logger, get_config, barrier, reduce_mean
from utils import get_dataloader, load_checkpoint, get_writer, update_train_result, log

# =============================================================================
# ARGUMENT PARSER (Stage 1)
# =============================================================================
parser = ArgumentParser(description="Train ZIP Stage 1 (Binary Classification).")

# Configurazione
parser.add_argument("--config", type=str, default=None, help="Path to the .yaml configuration file.")

# Parametri Modello / Backbone
# NOTA: Il default qui è vit_b_16, ma verrà sovrascritto dal valore nel YAML (BACKBONE -> TYPE)
parser.add_argument("--model", type=str, default="vit_b_16", help="Backbone model name.")
parser.add_argument("--input_size", type=int, default=448, help="Input image size.")
parser.add_argument("--reduction", type=int, default=16, help="Reduction factor (16 for ViT).")
parser.add_argument("--out",type=str,default=None,help="checkpoints/<dataset>/<model>/stage1.")
# Parametri Dataset
parser.add_argument("--dataset", type=str, required=False, help="Dataset name (sha, shb, qnrf).")
parser.add_argument("--data_dir", type=str, default="./data", help="Root directory of data.")
parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
parser.add_argument("--num_workers", type=int, default=8, help="Data loading workers.")
parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint")
# Parametri Training Stage 1
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the Head.")
parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Learning rate for the Backbone (old style: lr*0.1).")
parser.add_argument("--pos_weight", type=float, default=15.0, help="Positive class weight (old style default).")

parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--total_epochs", type=int, default=50)


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
parser.add_argument("--noise_prob", type=float, default=0.5)
parser.add_argument("--saltiness", type=float, default=1e-3, help="Saltiness for pepper salt noise.")
parser.add_argument("--spiciness", type=float, default=1e-3, help="Spiciness for pepper salt noise.")

# Eval & Log
parser.add_argument("--eval_freq", type=int, default=1)
parser.add_argument("--save_freq", type=int, default=5)
parser.add_argument("--save_best_k", type=int, default=3)
parser.add_argument("--amp", action="store_true", help="Use AMP.")
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--seed", type=int, default=42)

# Parametri Sliding Window
parser.add_argument("--sliding_window", action="store_true", help="Use sliding window strategy for evaluation.")
parser.add_argument("--stride", type=int, default=None, help="The stride for sliding window strategy.")
parser.add_argument("--window_size", type=int, default=None, help="The window size for in prediction.")
parser.add_argument("--resize_to_multiple", action="store_true", help="Resize the image to the nearest multiple of the input size.")
parser.add_argument("--zero_pad_to_multiple", action="store_true", help="Zero pad the image to the nearest multiple of the input size.")


# =============================================================================
# HELPER CONFIG (MODIFICATO PER LEGGERE BACKBONE DAL YAML)
# =============================================================================
def load_config_and_update_args(args):
    if not args.config: return args
    print(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f: cfg = yaml.safe_load(f)

    def is_passed_in_cli(arg_name): return f"--{arg_name}" in sys.argv

    mapping = {
        "dataset": ["DATASET"],
        "input_size": ["INPUT_SIZE"],
        "reduction": ["REDUCTION"],
        # --- FIX: Ora leggiamo il modello (backbone) dal config ---
        "model": ["BACKBONE", "TYPE"], 
        # ----------------------------------------------------------
        
        # Train Stage 1
        "batch_size": ["TRAIN_STAGE1", "BATCH_SIZE"],
        "num_workers": ["TRAIN_STAGE1", "NUM_WORKERS"],
        "lr": ["TRAIN_STAGE1", "LR_HEAD"],
        "lr_backbone": ["TRAIN_STAGE1", "LR_BACKBONE"],
        "total_epochs": ["TRAIN_STAGE1", "TOTAL_EPOCHS"],
        "pos_weight": ["TRAIN_STAGE1", "POS_WEIGHT"],
        "amp": ["TRAIN_STAGE1", "AMP"]
    }

    for arg_key, keys_path in mapping.items():
        if not is_passed_in_cli(arg_key):
            try:
                # Naviga nel dizionario annidato (es. BACKBONE -> TYPE)
                val = cfg
                for k in keys_path: val = val[k]
                
                # Assegna il valore
                setattr(args, arg_key, val)
                # print(f"  [Config] {arg_key} -> {val}") # Debug
            except (KeyError, TypeError): pass
    return args

# =============================================================================
# SAVE CHECKPOINT FUNCTION
# =============================================================================
def save_checkpoint(state, is_best, save_dir, filename='last_model.pth'):
    """
    Salva il checkpoint corrente come 'filename' (default: last_model.pth).
    Se is_best è True, crea una copia chiamata 'best_model.pth'.
    """
    os.makedirs(save_dir, exist_ok=True)
    last_path = os.path.join(save_dir, filename)
    torch.save(state, last_path)
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        shutil.copyfile(last_path, best_path)

# =============================================================================
# TRAIN FUNCTION
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, rank, nprocs):
    model.train()
    ddp = nprocs > 1
    total_loss = 0.0
    
    iterator = tqdm(loader, desc="Train S1") if rank == 0 else loader
    
    for batch in iterator:
        if isinstance(batch, dict):
            images = batch['image'].to(device)
            target_density = batch['density'].to(device)
        else:
            images, _, target_density = batch
            images = images.to(device)
            target_density = target_density.to(device)

        optimizer.zero_grad()

        with autocast(enabled=scaler is not None):
            # Forward
            outputs = model(images)
            pi_logits = outputs['pi_logits'] if isinstance(outputs, dict) else outputs

            # Target Binary Generation
            h_out, w_out = pi_logits.shape[2:]
            scale_factor = (images.shape[2] * images.shape[3]) / (h_out * w_out)
            gt_down = F.adaptive_avg_pool2d(target_density, (h_out, w_out)) * scale_factor
            target_binary = (gt_down > 0.001).float()

            # Loss
            loss = criterion(pi_logits, target_binary)

        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        curr_loss = loss.item()
        if ddp:
            loss_tensor = torch.tensor(curr_loss, device=device)
            curr_loss = reduce_mean(loss_tensor, nprocs).item()
            
        total_loss += curr_loss
        if rank == 0:
            iterator.set_postfix({'loss': f"{curr_loss:.4f}"})

    avg_loss = total_loss / len(loader)
    return {'loss': avg_loss}

# =============================================================================
# EVALUATION FUNCTION
# =============================================================================
@torch.no_grad()
def evaluate_stage1(model, loader, device):
    model.eval()
    tp, tn, fp, fn = 0, 0, 0, 0
    
    for batch in tqdm(loader, desc="Eval S1", disable=False):
        if isinstance(batch, dict):
            images = batch['image'].to(device)
            target_density = batch['density'].to(device)
        else:
            images, _, target_density = batch
            images = images.to(device)
            target_density = target_density.to(device)

        outputs = model(images)
        pi_logits = outputs['pi_logits'] if isinstance(outputs, dict) else outputs

        h_out, w_out = pi_logits.shape[2:]
        scale_factor = (images.shape[2] * images.shape[3]) / (h_out * w_out)
        gt_down = F.adaptive_avg_pool2d(target_density, (h_out, w_out)) * scale_factor
        gt_binary = (gt_down > 0.001).float()

        pred_prob = torch.sigmoid(pi_logits)
        pred_binary = (pred_prob > threshold).float()   # threshold in [0,1]


        tp += ((pred_binary == 1) & (gt_binary == 1)).sum().item()
        tn += ((pred_binary == 0) & (gt_binary == 0)).sum().item()
        fp += ((pred_binary == 1) & (gt_binary == 0)).sum().item()
        fn += ((pred_binary == 0) & (gt_binary == 1)).sum().item()

    epsilon = 1e-8
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)

    return {'f1': f1, 'acc': accuracy, 'prec': precision, 'rec': recall}

# =============================================================================
# MAIN RUN
# =============================================================================
def run(local_rank: int, nprocs: int, args: ArgumentParser) -> None:
    print(f"Rank {local_rank} process among {nprocs} processes.")
    init_seeds(args.seed + local_rank)
    setup(local_rank, nprocs)
    
    device = f"cuda:{local_rank}" if local_rank != -1 else "cuda:0"
    print(f"Using device: {device}")
    ddp = nprocs > 1

    # --- MODEL SETUP ---
    cfg = vars(args)
    cfg["BACKBONE"] = {"TYPE": args.model}   # oppure args.backbone se lo chiami così
    model = ZIPModel(cfg).to(device)

    if local_rank == 0:
        print("DEBUG BACKBONE:", cfg.get("BACKBONE", None))


    for p in model.parameters():
        p.requires_grad = True 

    # --- OPTIMIZER ---
    backbone_params = list(model.backbone.parameters())
    head_module = model.zip_head if hasattr(model, 'zip_head') else model.pi_head
    head_params = list(head_module.parameters())

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr_backbone},
        {'params': head_params, 'lr': args.lr}
    ], weight_decay=args.weight_decay)

    # --- LOSS ---
    pos_weight = torch.tensor([args.pos_weight]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

    scaler = GradScaler() if args.amp else None

    # --- DIRECTORIES (Corrected: Reads dataset & backbone from args updated via Config) ---
    # Structure: checkpoints / dataset_name / backbone_name / stage1
    config_name = f"{args.model}_{args.dataset}" # Non usato per la folder ma utile per log
    
    # args.model ora contiene la backbone corretta letta dal YAML (es. vit_b_16 o resnet50)
    # --- DIRECTORIES ---
    # Default: checkpoints/<dataset>/<model>/stage1
    default_ckpt_dir = os.path.join(current_dir, "checkpoints", args.dataset, args.model, "stage1")

    # If --out is provided, use it (absolute or relative to current working dir)
    if args.out is not None and str(args.out).strip() != "":
        args.ckpt_dir = os.path.abspath(args.out)
    else:
        args.ckpt_dir = default_ckpt_dir

    os.makedirs(args.ckpt_dir, exist_ok=True)


    if local_rank == 0:
        print(f"📂 Checkpoints will be saved to: {args.ckpt_dir}")
        config_save_path = os.path.join(args.ckpt_dir, "config_stage1.yaml")
        try:
            with open(config_save_path, 'w') as f:
                yaml.dump(vars(args), f, default_flow_style=False)
            print(f"📄 Configuration saved to {config_save_path}")
        except Exception as e:
            print(f"⚠️ Could not save config yaml: {e}")

    start_epoch = 1
    best_f1 = 0.0 
    
    # --- DATALOADERS ---
    args.regression = False 
    args.prompt_type = None
    args.bins, args.anchor_points = None, None 

    train_loader, sampler = get_dataloader(args, split="train", ddp=ddp)
    
    if local_rank == 0:
        val_loader = get_dataloader(args, split="val", ddp=False)
        writer = get_writer(args.ckpt_dir)
        logger = get_logger(os.path.join(args.ckpt_dir, "train_stage1.log"))
        logger.info(get_config(vars(args), mute=False))

    if ddp:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # --- TRAIN LOOP ---
    for epoch in range(start_epoch, args.total_epochs + 1):
        if sampler is not None: sampler.set_epoch(epoch)
        
        if local_rank == 0:
            log(logger, epoch, args.total_epochs, message=f"Start Train Stage 1 | LR Head: {optimizer.param_groups[1]['lr']:.2e}")

        train_stats = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, local_rank, nprocs
        )
        
        barrier(ddp)

        if local_rank == 0:
            update_train_result(epoch, train_stats, writer)
            log(logger, None, None, loss_info=train_stats)
            # --- EVALUATION ---
            is_best = False
            val_stats = {} 

            if epoch >= 0 and (epoch % args.eval_freq == 0):
                print("Evaluating Stage 1...")
                eval_model = model.module if ddp else model
                
                val_stats = evaluate_stage1(eval_model, val_loader, device)
                
                # STAMPA METRICHE
                print(f"📊 Eval Ep {epoch}: F1={val_stats['f1']:.4f} | Acc={val_stats['acc']:.4f} | Prec={val_stats['prec']:.4f} | Rec={val_stats['rec']:.4f}")
                # STAMPO LE METRICHE SUL LOG
                log(logger, epoch, args.total_epochs,
                   message=(
                        f"Eval Stage1 | "
                        f"F1={val_stats['f1']:.4f} | "
                        f"Acc={val_stats['acc']:.4f} | "
                        f"Prec={val_stats['prec']:.4f} | "
                        f"Rec={val_stats['rec']:.4f}"
                    )
                )

                for k, v in val_stats.items():
                    writer.add_scalar(f"val/{k}", v, epoch)

                if val_stats['f1'] > best_f1:
                    best_f1 = val_stats['f1']
                    is_best = True
                    print(f"🌟 New Best F1: {best_f1:.4f}")

            # --- SAVE CHECKPOINT ---
            # Prepara lo stato del dizionario
            state = {
                "epoch": epoch,
                "model_state_dict": (model.module if ddp else model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": None,
                "grad_scaler_state_dict": scaler.state_dict() if scaler else None,
                "loss_info": train_stats,
                "hist_scores": val_stats,
                "best_scores": {'f1': best_f1},
            }

            # 1. Salva SEMPRE "last_model.pth"
            # 2. Se is_best=True, copia "last_model.pth" in "best_model.pth"
            save_checkpoint(state, is_best, args.ckpt_dir, filename='last_model.pth')

        barrier(ddp)

    if local_rank == 0:
        writer.close()
        print(f"Stage 1 Training Completed. Best F1: {best_f1:.4f}")

    cleanup(ddp)

def main():
    args = parser.parse_args()
    args = load_config_and_update_args(args)
    
    if args.dataset is None:
        raise ValueError("Dataset name must be provided.")

    args.dataset = standardize_dataset_name(args.dataset)
    
    if args.sliding_window:
        args.window_size = args.input_size if args.window_size is None else args.window_size
        args.stride = args.input_size if args.stride is None else args.stride
    else:
        args.window_size = None
        args.stride = None
        args.zero_pad_to_multiple = False
        args.resize_to_multiple = False

    args.nprocs = torch.cuda.device_count()
    print(f"Using {args.nprocs} GPUs.")
    
    if args.nprocs > 1:
        mp.spawn(run, nprocs=args.nprocs, args=(args.nprocs, args))
    else:
        run(0, 1, args)

if __name__ == "__main__":
    main()