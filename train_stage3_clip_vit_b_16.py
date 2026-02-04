# train_stage3_clip_vit_b_16.py
# Stage 3 (Joint ZIP + CLIP) - Dual Config Support
#
# Changes:
# - Accepts --config_s1 AND --config_s2 to auto-merge parameters.
# - Includes "monkey patch" for CLIP utils bug.
# - Auto-tunes behavior for ViT vs ResNet.
# - FIX: Uses yaml.UnsafeLoader to handle !!python/tuple tags in configs.
# - FIX: Sets weights_only=False in torch.load for PyTorch 2.6+ compatibility.

import os, sys, json, yaml, shutil
from argparse import ArgumentParser

import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

current_dir = os.path.abspath(os.path.dirname(__file__))

# =============================================================================
# 🛠️ FIX BUG MONKEY PATCH (Per evitare crash su format_count)
# =============================================================================
try:
    import models.clip.utils as clip_utils
    def fixed_format_count(val, prompt_type):
        left, right = val
        if prompt_type == "word":
            return clip_utils.num2word(left), clip_utils.num2word(right)
        return left, right
    clip_utils.format_count = fixed_format_count
except ImportError:
    pass
# =============================================================================

from datasets import standardize_dataset_name
from models import get_model
from models.zip_model import ZIPModel
from models.joint_model import ZIPCLIPJointModel

from utils import setup, cleanup, init_seeds, get_logger, get_config, barrier
from utils import get_dataloader
from utils import get_writer, update_train_result, log

# Try to import Stage2 loss builder
try:
    from utils import get_loss_fn
except Exception:
    get_loss_fn = None

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

# ------------------------------------------------------------
# Parser
# ------------------------------------------------------------
parser = ArgumentParser("Train Stage 3 (Joint ZIP+CLIP) - Dual Config")

# MODIFICATO: Ora accettiamo due config
parser.add_argument("--config_s1", type=str, required=True, help="Config file used for Stage 1 (defines ZIP Architecture)")
parser.add_argument("--config_s2", type=str, required=True, help="Config file used for Stage 2 (defines Dataset/Bins/CLIP settings)")

parser.add_argument("--s1", type=str, required=True, help="Checkpoint Stage 1 .pth")
parser.add_argument("--s2", type=str, required=True, help="Checkpoint Stage 2 .pth")

# stage2 model name override (optional, usually inferred from config_s2)
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--input_size", type=int, default=224)
parser.add_argument("--reduction", type=int, default=8, choices=[8, 16, 32])

# dataset override
parser.add_argument("--dataset", type=str, required=False)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=4)

# bins cfg overrides
parser.add_argument("--regression", action="store_true")
parser.add_argument("--truncation", type=int, default=4)
parser.add_argument("--anchor_points", type=str, default="average", choices=["average", "middle"])
parser.add_argument("--prompt_type", type=str, default="word", choices=["word", "number"])
parser.add_argument("--granularity", type=str, default="fine", choices=["fine", "dynamic", "coarse"])
parser.add_argument("--num_vpt", type=int, default=32)
parser.add_argument("--vpt_drop", type=float, default=0.0)
parser.add_argument("--shallow_vpt", action="store_true")

# train params overrides
parser.add_argument("--lr", type=float, default=1e-6)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--total_epochs", type=int, default=600)
parser.add_argument("--eval_start", type=int, default=1)
parser.add_argument("--eval_freq", type=int, default=5)
parser.add_argument("--save_freq", type=int, default=1)
parser.add_argument("--amp", action="store_true")
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--seed", type=int, default=42)

# augmentation overrides
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

# output
parser.add_argument("--out", type=str, default=None)

# refined knobs
parser.add_argument("--base_steepness", type=float, default=1.0)
parser.add_argument("--max_steepness", type=float, default=8.0)
parser.add_argument("--zip_w", type=float, default=0.01)
parser.add_argument("--cons_w", type=float, default=0.05)

parser.add_argument("--weight_count_loss", type=float, default=1.0)
parser.add_argument("--count_loss", type=str, default="dmcount")


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_configs_and_update_args(args):
    """
    Legge DUE file di configurazione e li unisce.
    - config_s1: Fornisce l'architettura ZIP (BACKBONE, ZIP_HEAD).
    - config_s2: Fornisce i parametri di CLIP, Dataset, Augmentation, Bins.
    """
    merged_cfg = {}

    # 1. Carica Config Stage 1 (per ZIP architecture)
    if args.config_s1 and os.path.exists(args.config_s1):
        with open(args.config_s1, "r") as f:
            # FIX: Usa UnsafeLoader per supportare !!python/tuple se presente
            c1 = yaml.load(f, Loader=yaml.UnsafeLoader)
            
            # Prendiamo solo ciò che serve per ZIP
            if "BACKBONE" in c1: merged_cfg["BACKBONE"] = c1["BACKBONE"]
            if "ZIP_HEAD" in c1: merged_cfg["ZIP_HEAD"] = c1["ZIP_HEAD"]
            # Se S1 definisce reduction, teniamolo come fallback
            if "reduction" in c1: merged_cfg["reduction"] = c1["reduction"]
    
    # 2. Carica Config Stage 2 (per tutto il resto)
    if args.config_s2 and os.path.exists(args.config_s2):
        with open(args.config_s2, "r") as f:
            # FIX: Usa UnsafeLoader per supportare !!python/tuple (usato per i bins)
            c2 = yaml.load(f, Loader=yaml.UnsafeLoader)
            
            # S2 sovrascrive tutto tranne le chiavi specifiche di ZIP preservate sopra
            for k, v in c2.items():
                if k not in ["BACKBONE", "ZIP_HEAD"]: 
                    merged_cfg[k] = v

    # 3. Aggiorna args con il dizionario unito
    def is_passed(arg_name): return f"--{arg_name}" in sys.argv

    for k, v in merged_cfg.items():
        k_lower = k.lower()
        if hasattr(args, k_lower) and not is_passed(k_lower):
            setattr(args, k_lower, v)

    # Fallback per output dir
    if args.out is None:
        if args.model is None: args.model = "unknown_model"
        if args.dataset is None: args.dataset = "unknown_dataset"
        config_name = f"{args.model}_{args.dataset}"
        args.out = os.path.join(current_dir, "checkpoints", args.dataset, config_name, "stage3_refined")

    return args, merged_cfg


def build_bins_and_anchors(args):
    if args.regression:
        return None, None

    # Load bins from json file based on dataset/reduction
    # Se i bin sono già stati caricati nel config (merged_cfg), potremmo usarli direttamente.
    # Ma per sicurezza usiamo il json standard se esiste, altrimenti fallback.
    json_path = os.path.join(current_dir, "configs", f"reduction_{args.reduction}.json")
    
    try:
        with open(json_path, "r") as f:
            config = json.load(f)[str(args.truncation)][args.dataset]
        bins = config["bins"][args.granularity]
        anchor_points = (
            config["anchor_points"][args.granularity]["average"]
            if args.anchor_points == "average"
            else config["anchor_points"][args.granularity]["middle"]
        )
        bins = [(float(b[0]), float(b[1])) for b in bins]
        anchor_points = [float(p) for p in anchor_points]
        return bins, anchor_points
    except Exception as e:
        # Se fallisce il caricamento dal JSON, proviamo a vedere se sono stati caricati dal config yaml
        # (che ora supporta le tuple grazie a UnsafeLoader)
        if hasattr(args, 'bins') and args.bins is not None:
             print(f"[INFO] Using bins from config file (JSON load failed/skipped).")
             return args.bins, args.anchor_points
        else:
             raise e


def load_weights_only(path, map_location="cpu"):
    try:
        # Tenta con weights_only=False (Necessario per PyTorch 2.6+ se ci sono numpy scalars)
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Fallback per versioni vecchie di PyTorch che non hanno questo argomento
        ckpt = torch.load(path, map_location=map_location)
        
    if isinstance(ckpt, dict):
        # Handle various checkpoint formats
        for k in ["model_state_dict", "state_dict", "model"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                sd = ckpt[k]
                break
        else:
            sd = ckpt
    else:
        sd = ckpt

    if isinstance(sd, dict) and any(kk.startswith("module.") for kk in sd.keys()):
        sd = {kk.replace("module.", "", 1): vv for kk, vv in sd.items()}
    return sd


def is_vit_backbone(model_name: str) -> bool:
    m = (model_name or "").lower()
    return ("vit" in m) or ("visualtransformer" in m)


def freeze_stage2_backbone_for_vit(stage2):
    for p in stage2.parameters():
        p.requires_grad = False
    for name, p in stage2.named_parameters():
        n = name.lower()
        if any(k in n for k in ["vpt", "prompt", "classifier", "regressor", "head", "proj", "projection"]):
            p.requires_grad = True


def build_optimizer_auto(model, args, vit_mode: bool):
    if not vit_mode:
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_heads = args.lr
    lr_backbone = min(args.lr, 1e-7)

    params_backbone, params_other = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        nl = n.lower()
        if nl.startswith("stage2.") and not any(k in nl for k in ["vpt", "prompt", "classifier", "regressor", "head", "proj", "projection"]):
            params_backbone.append(p)
        else:
            params_other.append(p)

    groups = []
    if params_backbone:
        groups.append({"params": params_backbone, "lr": lr_backbone})
    if params_other:
        groups.append({"params": params_other, "lr": lr_heads})

    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


# ------------------------------------------------------------
# Train/Eval Logic
# ------------------------------------------------------------
def train_one_epoch_refined(model, loader, optimizer, scaler, device, rank, nprocs, args,
                            clip_loss_fn=None, vit_mode: bool = False):
    model.train()
    total_loss = 0.0
    it = tqdm(loader, desc="Train S3") if rank == 0 else loader
    pos_w = 5.0 
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))

    lambda_final = 0.1 if vit_mode else 0.0
    cons_w = 0.0 if vit_mode else args.cons_w

    for step, batch in enumerate(it):
        if isinstance(batch, dict):
            imgs = batch["image"].to(device)
            gt_density = batch["density"].to(device)
            points = batch.get("points", None)
            counts = batch.get("counts", None)
        else:
            imgs, points, gt_density = batch
            imgs = imgs.to(device)
            gt_density = gt_density.to(device)
            counts = None

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(scaler is not None)):
            out = model(imgs)

            pred_density = out["final_density"]
            gt_resized = F.interpolate(gt_density, size=pred_density.shape[-2:], mode="bilinear", align_corners=False)
            scale = (gt_density.shape[-2] * gt_density.shape[-1]) / (pred_density.shape[-2] * pred_density.shape[-1])
            gt_resized = gt_resized * scale
            
            # --- AUTO COUNTS for ViT ---
            if vit_mode and (counts is None) and (out.get("ebc_logits", None) is not None):
                B, C, Hc, Wc = out["ebc_logits"].shape
                counts = torch.zeros((B, Hc, Wc), dtype=torch.long, device=device)

                if points is not None and isinstance(points, (list, tuple)) and len(points) == B:
                    img_h, img_w = imgs.shape[-2], imgs.shape[-1]
                    cell_w = img_w / Wc
                    cell_h = img_h / Hc

                    for bi in range(B):
                        pts = points[bi]
                        if pts is None: continue
                        if torch.is_tensor(pts): pts_xy = pts.detach().cpu().tolist()
                        else: pts_xy = pts

                        for p in pts_xy:
                            if p is None or len(p) < 2: continue
                            x, y = float(p[0]), float(p[1])
                            if x < 0 or y < 0: continue
                            j = int(x / cell_w)
                            i = int(y / cell_h)
                            if 0 <= i < Hc and 0 <= j < Wc:
                                counts[bi, i, j] += 1
                    counts = counts.clamp_(0, 4)
                else:
                    gt_c = F.interpolate(gt_density, size=(Hc, Wc), mode="bilinear", align_corners=False)
                    scale_c = (gt_density.shape[-2] * gt_density.shape[-1]) / (Hc * Wc)
                    gt_c = gt_c * scale_c
                    cnt = gt_c.squeeze(1) if gt_c.dim() == 4 and gt_c.size(1) == 1 else gt_c
                    counts = torch.floor(cnt + 1e-6).long().clamp_(0, 4).to(device)

            # ---- Loss Calculation ----
            if vit_mode and clip_loss_fn is not None and out.get("ebc_logits", None) is not None and counts is not None:
                try:
                    c = counts.to(device) if torch.is_tensor(counts) else counts
                    pts = points
                    if points is not None and torch.is_tensor(points): pts = points.to(device)
                    l_clip, clip_logs = clip_loss_fn(out["ebc_logits"], c, gt_density, pts)
                except Exception:
                    l_map = F.mse_loss(pred_density, gt_resized)
                    pred_cnt = pred_density.sum(dim=[1, 2, 3])
                    gt_cnt = gt_density.sum(dim=[1, 2, 3])
                    l_cnt = (pred_cnt - gt_cnt).abs().mean()
                    l_clip = l_map + l_cnt
            else:
                l_map = F.mse_loss(pred_density, gt_resized)
                pred_cnt = pred_density.sum(dim=[1, 2, 3])
                gt_cnt = gt_density.sum(dim=[1, 2, 3])
                l_cnt = (pred_cnt - gt_cnt).abs().mean()
                l_clip = l_map + l_cnt

            if lambda_final > 0:
                l_clip = l_clip + lambda_final * F.l1_loss(pred_density, gt_resized)

            pi_logits = out["pi_logits"]
            h, w = pi_logits.shape[-2:]
            with torch.no_grad():
                gt_pi = F.interpolate(gt_density, size=(h, w), mode="bilinear", align_corners=False)
                scale = (gt_density.shape[-2] * gt_density.shape[-1]) / (h * w)
                gt_pi = gt_pi * scale
                mask_target = (gt_pi > 0.001).float()
            l_zip = bce(pi_logits, mask_target)

            clip_raw = out["raw_density"].detach()
            prob_zip = out["pi_prob"]
            l_cons = (clip_raw * (1.0 - prob_zip)).mean()

            loss = l_clip + args.zip_w * l_zip + cons_w * l_cons

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        if rank == 0:
            it.set_postfix({
                "L": f"{loss.item():.3f}",
                "clip": f"{float(l_clip.item()):.3f}",
                "zip": f"{float(l_zip.item()):.3f}"
            })

    return {"loss": total_loss / max(1, len(loader))}


@torch.no_grad()
def evaluate_mae_rmse(model, loader, device, rank=0):
    model.eval()
    abs_sum = 0.0
    se_sum = 0.0
    n = 0
    for batch in loader:
        if isinstance(batch, dict):
            imgs = batch["image"].to(device)
            points = batch.get("points", None)
        else:
            imgs, points, _ = batch
            imgs = imgs.to(device)

        out = model(imgs)
        pred = out["final_density"].sum().item()

        if points is not None:
            gt = len(points[0]) if isinstance(points, (list, tuple)) else int(points)
        else:
            continue

        err = float(pred - gt)
        abs_sum += abs(err)
        se_sum += err * err
        n += 1

    mae = abs_sum / max(1, n)
    rmse = (se_sum / max(1, n)) ** 0.5
    return mae, rmse


def run(local_rank: int, nprocs: int, args, merged_cfg):
    print(f"Rank {local_rank} process among {nprocs} processes.")
    init_seeds(args.seed + local_rank)
    setup(local_rank, nprocs)

    device = f"cuda:{local_rank}" if local_rank != -1 else "cuda:0"
    ddp = nprocs > 1

    if args.dataset is None:
        raise ValueError("Dataset name must be provided.")
    args.dataset = standardize_dataset_name(args.dataset)

    vit_mode = is_vit_backbone(args.model)

    bins, anchor_points = build_bins_and_anchors(args)
    args.bins = bins
    args.anchor_points = anchor_points

    if local_rank == 0:
        val_loader = get_dataloader(args, split="val", ddp=False)

    args.batch_size = int(args.batch_size / nprocs) if ddp else int(args.batch_size)
    args.num_workers = int(args.num_workers / nprocs) if ddp else int(args.num_workers)
    train_loader, sampler = get_dataloader(args, split="train", ddp=ddp)

    # --- BUILD MODELS USING MERGED CONFIG ---
    # ZIP Model (Stage 1) needs specific config structure
    zip_head_cfg = merged_cfg.get("ZIP_HEAD", {"HIDDEN_DIM": 256})
    backbone_cfg = merged_cfg.get("BACKBONE", {"TYPE": "resnet50"}) # Fallback if missing
    
    zip_cfg = {
        "BACKBONE": backbone_cfg,
        "ZIP_HEAD": zip_head_cfg,
        "REDUCTION": int(args.reduction),
    }
    stage1 = ZIPModel(zip_cfg).to(device)

    # Stage 2 Model
    stage2 = get_model(
        backbone=args.model,
        input_size=args.input_size,
        reduction=args.reduction,
        bins=bins,
        anchor_points=anchor_points,
        prompt_type=args.prompt_type,
        num_vpt=args.num_vpt,
        vpt_drop=args.vpt_drop,
        deep_vpt=not args.shallow_vpt,
    ).to(device)

    # Load Weights
    stage1.load_state_dict(load_weights_only(args.s1), strict=False)
    stage2.load_state_dict(load_weights_only(args.s2), strict=False)

    if vit_mode:
        freeze_stage2_backbone_for_vit(stage2)

    model = ZIPCLIPJointModel(stage1, stage2, steepness=args.base_steepness).to(device)

    if args.out is None:
        config_name = f"{args.model}_{args.dataset}"
        args.ckpt_dir = os.path.join(current_dir, "checkpoints", args.dataset, config_name, "stage3_refined")
    else:
        args.ckpt_dir = args.out
    os.makedirs(args.ckpt_dir, exist_ok=True)

    writer = None
    logger = None
    if local_rank == 0:
        # Save merged config for reproducibility
        with open(os.path.join(args.ckpt_dir, "config_merged.yaml"), "w") as f:
            yaml.dump(merged_cfg, f)
        
        writer = get_writer(args.ckpt_dir)
        logger = get_logger(os.path.join(args.ckpt_dir, "train_stage3.log"))
        logger.info(get_config(vars(args), mute=False))
        if vit_mode:
            logger.info("AUTO MODE: ViT detected -> Stage2-loss + backbone freeze + diff LR groups")

    if ddp:
        model = DDP(nn.SyncBatchNorm.convert_sync_batchnorm(model), device_ids=[local_rank], output_device=local_rank)

    clip_loss_fn = None
    if vit_mode and get_loss_fn is not None:
        try:
            clip_loss_fn = get_loss_fn(args)
        except Exception:
            clip_loss_fn = None

    optimizer = build_optimizer_auto(model, args, vit_mode=vit_mode)
    scaler = GradScaler() if args.amp else None

    # Resume Logic
    resume_path = os.path.join(args.ckpt_dir, "last_model.pth")
    if not os.path.exists(resume_path):
        resume_path = os.path.join(args.ckpt_dir, "best_model.pth")

    start_epoch = 1
    best_mae = float("inf")
    best_val_scores = {"mae": [float("inf")], "rmse": [float("inf")], "epoch": [0]}
    
    if os.path.exists(resume_path):
        if local_rank == 0: print(f"🔄 Resuming from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        (model.module if ddp else model).load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict"): optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if args.amp and checkpoint.get("grad_scaler_state_dict") and scaler: scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        if checkpoint.get("best_val_scores"):
            best_val_scores = checkpoint["best_val_scores"]
            best_mae = float(best_val_scores["mae"][0])

    def save_last_and_best(epoch, is_best: bool, curr_val_scores=None):
        state = {
            "epoch": epoch,
            "model_state_dict": (model.module if ddp else model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "grad_scaler_state_dict": scaler.state_dict() if scaler else None,
            "best_val_scores": best_val_scores,
        }
        torch.save(state, os.path.join(args.ckpt_dir, "last_model.pth"))
        if is_best:
            torch.save(state, os.path.join(args.ckpt_dir, "best_model.pth"))

    for epoch in range(start_epoch, args.total_epochs + 1):
        if sampler is not None: sampler.set_epoch(epoch)

        if args.total_epochs > 1:
            steep = args.base_steepness + (epoch - 1) / (args.total_epochs - 1) * (args.max_steepness - args.base_steepness)
        else:
            steep = args.max_steepness
        (model.module if ddp else model).steepness = float(steep)

        loss_info = train_one_epoch_refined(
            model, train_loader, optimizer, scaler, device, local_rank, nprocs, args,
            clip_loss_fn=clip_loss_fn, vit_mode=vit_mode
        )
        barrier(ddp)

        if local_rank == 0:
            update_train_result(epoch, loss_info, writer)
            eval_now = (epoch >= args.eval_start) and ((epoch - args.eval_start) % args.eval_freq == 0)
            
            if not eval_now:
                log(logger, epoch, args.total_epochs, loss_info=loss_info, message="\n")
                save_last_and_best(epoch, is_best=False)
            else:
                eval_model = model.module if ddp else model
                eval_model.steepness = args.max_steepness
                val_mae, val_rmse = evaluate_mae_rmse(eval_model, val_loader, device)
                curr_val_scores = {"mae": float(val_mae), "rmse": float(val_rmse)}

                if writer:
                    writer.add_scalar("val/mae", float(val_mae), epoch)
                    writer.add_scalar("val/rmse", float(val_rmse), epoch)

                is_best = float(val_mae) < best_mae
                if is_best:
                    best_mae = float(val_mae)
                    best_val_scores["mae"][0] = best_mae
                    best_val_scores["rmse"][0] = float(val_rmse)
                    best_val_scores["epoch"][0] = epoch
                    print(f"🌟 New Best MAE: {best_mae:.4f} | RMSE: {float(val_rmse):.4f}")

                log(logger, epoch, args.total_epochs, None, curr_val_scores, best_val_scores, message="\n")
                save_last_and_best(epoch, is_best=is_best, curr_val_scores=curr_val_scores)

        barrier(ddp)

    if local_rank == 0 and writer: writer.close()
    cleanup(ddp)


def main():
    args = parser.parse_args()
    # LOAD MERGED CONFIG
    args, merged_cfg = load_configs_and_update_args(args)

    args.nprocs = torch.cuda.device_count()
    print(f"Using {args.nprocs} GPUs.")
    if args.nprocs > 1:
        mp.spawn(run, nprocs=args.nprocs, args=(args.nprocs, args, merged_cfg))
    else:
        run(0, 1, args, merged_cfg)

if __name__ == "__main__":
    main()