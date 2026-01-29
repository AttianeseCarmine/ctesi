# train_stage3_v2.py
# Stage 3 (Joint ZIP + CLIP) - auto-tuned behavior for ViT vs ResNet
#
# Goal:
# - Keep current behavior for clip_resnet50 (so you don't break good results)
# - Automatically switch to a safer/more aligned training regime when backbone is clip_vit_b_16 (or any ViT):
#     1) Use Stage2 loss (CE + count + TV + OT) when available (via utils.get_loss_fn)
#     2) Freeze most of CLIP ViT backbone to avoid drift / catastrophic forgetting
#     3) Use differential LR param groups (very low lr for backbone, normal for heads/VPT/ZIP)
#     4) Keep a small stabilizer on final_density (optional, only for ViT)

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

from datasets import standardize_dataset_name
from models import get_model
from models.zip_model import ZIPModel
from models.joint_model import ZIPCLIPJointModel

from utils import setup, cleanup, init_seeds, get_logger, get_config, barrier
from utils import get_dataloader
from utils import get_writer, update_train_result, log
from losses.joint_loss import JointLoss

# Try to import Stage2 loss builder (if your repo has it; it should).
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
parser = ArgumentParser("Train Stage 3 (Joint ZIP+CLIP) - Stage3_v2 (auto ViT safety)")

parser.add_argument("--config", type=str, default="configs/config.yaml")
parser.add_argument("--s1", type=str, required=True)  # ckpt stage1
parser.add_argument("--s2", type=str, required=True)  # ckpt stage2

# stage2 model name (clip_vit_b_16 etc.)
parser.add_argument("--model", type=str, default="clip_vit_b_16")
parser.add_argument("--input_size", type=int, default=224)
parser.add_argument("--reduction", type=int, default=8, choices=[8, 16, 32])

# dataset
parser.add_argument("--dataset", type=str, required=False)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=4)

# bins cfg (come stage2)
parser.add_argument("--regression", action="store_true")  # keep False for Stage2-style bins
parser.add_argument("--truncation", type=int, default=4)
parser.add_argument("--anchor_points", type=str, default="average", choices=["average", "middle"])
parser.add_argument("--prompt_type", type=str, default="word", choices=["word", "number"])
parser.add_argument("--granularity", type=str, default="fine", choices=["fine", "dynamic", "coarse"])
parser.add_argument("--num_vpt", type=int, default=32)
parser.add_argument("--vpt_drop", type=float, default=0.0)
parser.add_argument("--shallow_vpt", action="store_true")

# train
parser.add_argument("--lr", type=float, default=1e-6)          # "main lr" (heads/ZIP)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--total_epochs", type=int, default=600)
parser.add_argument("--eval_start", type=int, default=1)
parser.add_argument("--eval_freq", type=int, default=10)
parser.add_argument("--save_freq", type=int, default=1)
parser.add_argument("--amp", action="store_true")
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--seed", type=int, default=42)

# augmentation (stage2-like)
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
parser.add_argument("--zip_w", type=float, default=0.1)
parser.add_argument("--cons_w", type=float, default=0.05)

# --- needed by utils.get_loss_fn (DACELoss path when bins != None) ---
parser.add_argument("--weight_count_loss", type=float, default=1.0)
parser.add_argument("--count_loss", type=str, default="dmcount")



# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_config_and_update_args(args):
    if not args.config or not os.path.exists(args.config):
        return args

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    def is_passed(arg_name): return f"--{arg_name}" in sys.argv

    mapping = {
        "dataset": ["DATASET"],
        "model": ["MODEL"],
        "input_size": ["INPUT_SIZE"],
        "reduction": ["REDUCTION"],
        "truncation": ["TRUNCATION"],

        "batch_size": ["TRAIN_STAGE3", "BATCH_SIZE"],
        "num_workers": ["TRAIN_STAGE3", "NUM_WORKERS"],
        "lr": ["TRAIN_STAGE3", "LR"],
        "weight_decay": ["TRAIN_STAGE3", "WEIGHT_DECAY"],
        "total_epochs": ["TRAIN_STAGE3", "TOTAL_EPOCHS"],
        "eval_freq": ["TRAIN_STAGE3", "EVAL_FREQ"],
        "amp": ["TRAIN_STAGE3", "AMP"],
    }

    for arg_key, keys_path in mapping.items():
        if is_passed(arg_key):
            continue
        try:
            val = cfg
            for k in keys_path:
                val = val[k]
            setattr(args, arg_key, val)
        except (KeyError, TypeError):
            pass

    if args.out is None:
        config_name = f"{args.model}_{args.dataset}"
        args.out = os.path.join(current_dir, "checkpoints", args.dataset, config_name, "stage3_refined")

    return args


def build_bins_and_anchors(args):
    # identical to stage2
    if args.regression:
        return None, None

    with open(os.path.join(current_dir, "configs", f"reduction_{args.reduction}.json"), "r") as f:
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


def load_weights_only(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict):
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
    """
    Conservative freeze:
    - Freeze everything in Stage2
    - Unfreeze only VPT/prompt/head-like params (heuristic by name)
    This prevents ViT drift while still letting it adapt via prompts/heads.
    """
    for p in stage2.parameters():
        p.requires_grad = False

    for name, p in stage2.named_parameters():
        n = name.lower()
        if any(k in n for k in ["vpt", "prompt", "classifier", "regressor", "head", "proj", "projection"]):
            p.requires_grad = True


def build_optimizer_auto(model, args, vit_mode: bool):
    """
    Keep current behavior for ResNet (single param group with args.lr).
    For ViT: differential LR (very low backbone LR), normal LR for heads/ZIP.
    """
    if not vit_mode:
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ViT-safe defaults (do NOT touch ResNet path)
    lr_heads = args.lr
    lr_backbone = min(args.lr, 1e-7)  # cap backbone lr very low

    params_backbone, params_other = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        nl = n.lower()

        # Anything inside stage2 that is NOT prompt/vpt/head -> treat as backbone
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
# Train/Eval
# ------------------------------------------------------------
def train_one_epoch_refined(model, loader, optimizer, scaler, device, rank, nprocs, args,
                            clip_loss_fn=None, vit_mode: bool = False):
    model.train()
    total_loss = 0.0
    it = tqdm(loader, desc="Train S3") if rank == 0 else loader
    pos_w = 5.0  # parti da 10, poi 5–15
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))

    # ViT-only: small stabilizer on final_density to prevent weird gating solutions
    # (kept off for ResNet to preserve behavior)
    lambda_final = 0.1 if vit_mode else 0.0
    cons_w = 0.0 if vit_mode else args.cons_w  # ViT tends to be more sensitive to consistency pressure

    for step, batch in enumerate(it):
        if isinstance(batch, dict):
            imgs = batch["image"].to(device)
            gt_density = batch["density"].to(device)
            points = batch.get("points", None)
            counts = batch.get("counts", None)
        else:
            # fallback legacy tuple
            imgs, points, gt_density = batch
            imgs = imgs.to(device)
            gt_density = gt_density.to(device)
            counts = None

        if points is not None and isinstance(points, (list, tuple)):
            # some loaders keep points on CPU; keep as-is for eval; for OT loss it may need .to(device) inside loss fn
            pass

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(scaler is not None)):
            out = model(imgs)


            pred_density = out["final_density"]
            gt_resized = F.interpolate(gt_density, size=pred_density.shape[-2:], mode="bilinear", align_corners=False)
            scale = (gt_density.shape[-2] * gt_density.shape[-1]) / (pred_density.shape[-2] * pred_density.shape[-1])
            gt_resized = gt_resized * scale
            # --- AUTO COUNTS for ViT: prefer points -> stable cell counts ---
            if vit_mode and (counts is None) and (out.get("ebc_logits", None) is not None):
                B, C, Hc, Wc = out["ebc_logits"].shape
                counts = torch.zeros((B, Hc, Wc), dtype=torch.long, device=device)

                # points expected as list (len B), each element: Nx2 coords in patch space
                if points is not None and isinstance(points, (list, tuple)) and len(points) == B:
                    img_h, img_w = imgs.shape[-2], imgs.shape[-1]
                    cell_w = img_w / Wc
                    cell_h = img_h / Hc


                    for bi in range(B):
                        pts = points[bi]
                        if pts is None:
                            continue
                        # pts can be numpy, list, or tensor
                        if torch.is_tensor(pts):
                            pts_xy = pts.detach().cpu().tolist()
                        else:
                            pts_xy = pts

                        for p in pts_xy:
                            if p is None or len(p) < 2:
                                continue
                            x, y = float(p[0]), float(p[1])

                            # robust clamp
                            if x < 0 or y < 0:
                                continue
                            j = int(x / cell_w)
                            i = int(y / cell_h)
                            if 0 <= i < Hc and 0 <= j < Wc:
                                counts[bi, i, j] += 1

                    # bin to 0..4 (4 = ">=4")
                    counts = counts.clamp_(0, 4)

                else:
                    # fallback: density-based (keep conservative)
                    gt_c = F.interpolate(gt_density, size=(Hc, Wc), mode="bilinear", align_corners=False)
                    scale_c = (gt_density.shape[-2] * gt_density.shape[-1]) / (Hc * Wc)
                    gt_c = gt_c * scale_c
                    cnt = gt_c.squeeze(1) if gt_c.dim() == 4 and gt_c.size(1) == 1 else gt_c
                    counts = torch.floor(cnt + 1e-6).long().clamp_(0, 4).to(device)

            use_stage2_loss_now = (
                vit_mode
                and (clip_loss_fn is not None)
                and (out.get("ebc_logits", None) is not None)
                and (counts is not None)
            )
            # ---- 1) CLIP-side loss ----
            # For ViT: prefer Stage2 loss (CE + count + TV + OT) when available and batch provides counts.
            if vit_mode and clip_loss_fn is not None and out.get("ebc_logits", None) is not None and counts is not None:
                # try to be compatible with your Stage2 loss signature
                try:
                    c = counts.to(device) if torch.is_tensor(counts) else counts
                    pts = points
                    if points is not None and torch.is_tensor(points):
                        pts = points.to(device)

                    if rank == 0 and step == 0 and points is not None and isinstance(points, (list, tuple)) and len(points) > 0:
                        p0 = points[0]
                        if torch.is_tensor(p0):
                            p0_list = p0.detach().cpu().tolist()
                        else:
                            p0_list = p0
                        if p0_list is not None and len(p0_list) > 0:
                            xs = [float(pp[0]) for pp in p0_list if pp is not None and len(pp) >= 2]
                            ys = [float(pp[1]) for pp in p0_list if pp is not None and len(pp) >= 2]
                            if len(xs) > 0:
                                print(f"[S3] points[0] x_range=({min(xs):.1f},{max(xs):.1f}) y_range=({min(ys):.1f},{max(ys):.1f}) n={len(xs)} input_size={args.input_size}")


                    l_clip, clip_logs = clip_loss_fn(out["ebc_logits"], c, gt_density, pts)
                    # clip_loss_fn already includes dmcount/tv/ot/ce (depending on your utils)
                except Exception:
                    # fallback safe
                    l_map = F.mse_loss(pred_density, gt_resized)
                    pred_cnt = pred_density.sum(dim=[1, 2, 3])
                    gt_cnt = gt_density.sum(dim=[1, 2, 3])
                    l_cnt = (pred_cnt - gt_cnt).abs().mean()
                    l_clip = l_map + l_cnt
            else:
                # ResNet path OR missing Stage2-loss prerequisites -> keep current simplified loss
                l_map = F.mse_loss(pred_density, gt_resized)
                pred_cnt = pred_density.sum(dim=[1, 2, 3])
                gt_cnt = gt_density.sum(dim=[1, 2, 3])
                l_cnt = (pred_cnt - gt_cnt).abs().mean()
                l_clip = l_map + l_cnt

            if lambda_final > 0:
                l_clip = l_clip + lambda_final * F.l1_loss(pred_density, gt_resized)

            # ---- 2) ZIP loss (mask supervision) ----
            pi_logits = out["pi_logits"]
            h, w = pi_logits.shape[-2:]
            with torch.no_grad():
                gt_pi = F.interpolate(gt_density, size=(h, w), mode="bilinear", align_corners=False)
                scale = (gt_density.shape[-2] * gt_density.shape[-1]) / (h * w)
                gt_pi = gt_pi * scale
                mask_target = (gt_pi > 0.001).float()
            l_zip = bce(pi_logits, mask_target)

            # ---- 3) Consistency loss ----
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
                "zip": f"{float(l_zip.item()):.3f}",
                "cons": f"{float(l_cons.item()):.3f}",
            })
        if rank == 0 and (step % 50 == 0):
            print(
                f"[S3] step={step} vit_mode={vit_mode} use_stage2_loss={use_stage2_loss_now} "
                f"has_ebc_logits={out.get('ebc_logits', None) is not None} has_counts={counts is not None}"
            )

            p = out["pi_prob"].detach()
            p_mean = p.mean().item()
            p_lo = (p < 0.1).float().mean().item()
            p_hi = (p > 0.9).float().mean().item()
            print(f"[S3] gate pi_prob: mean={p_mean:.3f}  <0.1={p_lo:.3f}  >0.9={p_hi:.3f}")

            raw_cnt = out["raw_density"].detach().sum(dim=[1,2,3]).mean().item()
            fin_cnt = out["final_density"].detach().sum(dim=[1,2,3]).mean().item()
            ratio = fin_cnt / (raw_cnt + 1e-6)
            print(f"[S3] counts mean: raw={raw_cnt:.2f} final={fin_cnt:.2f} ratio={ratio:.3f}")

            pred_cnt_m = out["final_density"].detach().sum(dim=[1,2,3]).mean().item()
            gt_cnt_m = gt_density.detach().sum(dim=[1,2,3]).mean().item()
            print(f"[S3] pred_cnt_mean={pred_cnt_m:.2f} gt_cnt_mean={gt_cnt_m:.2f} diff={pred_cnt_m-gt_cnt_m:.2f}")

            print(f"[S3] counts_source={'AUTO' if (counts is not None and use_stage2_loss_now) else 'NONE/FALLBACK'}")
            flat = counts.view(-1)
            hist = [(flat == i).float().mean().item() for i in range(5)]
            print(f"[S3] counts_hist 0..4 = {[round(x,3) for x in hist]}")


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
            # points is usually a list-of-lists (batch size 1 in sliding window eval)
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


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
def run(local_rank: int, nprocs: int, args):
    print(f"Rank {local_rank} process among {nprocs} processes.")
    init_seeds(args.seed + local_rank)
    setup(local_rank, nprocs)

    device = f"cuda:{local_rank}" if local_rank != -1 else "cuda:0"
    ddp = nprocs > 1

    if args.dataset is None:
        raise ValueError("Dataset name must be provided.")
    args.dataset = standardize_dataset_name(args.dataset)

    vit_mode = is_vit_backbone(args.model)

    # bins/anchors like stage2
    bins, anchor_points = build_bins_and_anchors(args)
    args.bins = bins
    args.anchor_points = anchor_points

    # dataloaders (stage2-style)
    if local_rank == 0:
        val_loader = get_dataloader(args, split="val", ddp=False)

    args.batch_size = int(args.batch_size / nprocs) if ddp else int(args.batch_size)
    args.num_workers = int(args.num_workers / nprocs) if ddp else int(args.num_workers)
    train_loader, sampler = get_dataloader(args, split="train", ddp=ddp)

    # build Stage1 from YAML
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    zip_head_cfg = cfg.get("ZIP_HEAD", {"HIDDEN_DIM": 256})

    zip_cfg = {
        "BACKBONE": cfg["BACKBONE"],
        "ZIP_HEAD": zip_head_cfg,
        "REDUCTION": int(args.reduction),
    }
    stage1 = ZIPModel(zip_cfg).to(device)

    # stage2
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

    # load weights
    stage1.load_state_dict(load_weights_only(args.s1), strict=False)
    stage2.load_state_dict(load_weights_only(args.s2), strict=False)

    # ViT safety: freeze backbone (only for ViT)
    if vit_mode:
        freeze_stage2_backbone_for_vit(stage2)

    model = ZIPCLIPJointModel(stage1, stage2, steepness=args.base_steepness).to(device)

    # output dir
    if args.out is None:
        config_name = f"{args.model}_{args.dataset}"
        args.ckpt_dir = os.path.join(current_dir, "checkpoints", args.dataset, config_name, "stage3_refined")
    else:
        args.ckpt_dir = args.out
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # logger/writer
    writer = None
    logger = None
    if local_rank == 0:
        shutil.copyfile(args.config, os.path.join(args.ckpt_dir, "config_stage3.yaml"))
        writer = get_writer(args.ckpt_dir)
        logger = get_logger(os.path.join(args.ckpt_dir, "train_stage3.log"))
        logger.info(get_config(vars(args), mute=False))
        if vit_mode:
            logger.info("AUTO MODE: ViT detected -> Stage2-loss (if available) + backbone freeze + diff LR groups")

    # DDP
    if ddp:
        model = DDP(nn.SyncBatchNorm.convert_sync_batchnorm(model), device_ids=[local_rank], output_device=local_rank)

    # Stage2 loss fn (ONLY used in vit_mode; ResNet path unaffected)
    clip_loss_fn = None
    if vit_mode and get_loss_fn is not None:
        try:
            clip_loss_fn = get_loss_fn(args)
        except Exception:
            clip_loss_fn = None
            if local_rank == 0 and logger is not None:
                logger.info("WARNING: get_loss_fn(args) failed -> fallback to simplified Stage3 loss")
    if local_rank == 0:
        print(f"[S3] get_loss_fn available={get_loss_fn is not None} clip_loss_fn_created={clip_loss_fn is not None}")


    # optimizer/scaler (AUTO: ResNet keeps previous behavior; ViT uses diff LR groups)
    optimizer = build_optimizer_auto(model, args, vit_mode=vit_mode)
    scaler = GradScaler() if args.amp else None

    # Resume
    resume_path = os.path.join(args.ckpt_dir, "last_model.pth")
    start_epoch = 1
    best_mae = float("inf")
    best_val_scores = {"mae": [float("inf")], "rmse": [float("inf")], "epoch": [0]}
    hist_val_scores = {}
    loss_info = {}

    if os.path.exists(resume_path):
        if local_rank == 0:
            print(f"🔄 Trovato checkpoint, riprendo da: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        (model.module if ddp else model).load_state_dict(checkpoint["model_state_dict"])

        if checkpoint.get("optimizer_state_dict", None) is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if args.amp and checkpoint.get("grad_scaler_state_dict", None) is not None and scaler is not None:
            scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])

        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        if checkpoint.get("best_val_scores", None) is not None:
            best_val_scores = checkpoint["best_val_scores"]
            best_mae = float(best_val_scores["mae"][0])

        if local_rank == 0:
            print(f"   -> Riprendo dall'epoca {start_epoch}, Best MAE precedente: {best_mae:.4f}")
    else:
        if local_rank == 0:
            print("🚀 Nessun checkpoint trovato, inizio training da zero.")

    def save_last_and_best(epoch, is_best: bool, curr_val_scores=None):
        state = {
            "epoch": epoch,
            "model_state_dict": (model.module if ddp else model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None,
            "grad_scaler_state_dict": scaler.state_dict() if scaler else None,
            "loss_info": loss_info,
            "hist_val_scores": hist_val_scores,
            "best_val_scores": best_val_scores,
            "val_scores": curr_val_scores,
        }
        torch.save(state, os.path.join(args.ckpt_dir, "last_model.pth"))
        if is_best:
            torch.save(state, os.path.join(args.ckpt_dir, "best_model.pth"))

    # Train loop
    for epoch in range(start_epoch, args.total_epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        # steepness annealing
        if args.total_epochs > 1:
            steep = args.base_steepness + (epoch - 1) / (args.total_epochs - 1) * (args.max_steepness - args.base_steepness)
        else:
            steep = args.max_steepness
        (model.module if ddp else model).steepness = float(steep)

        # train
        loss_info = train_one_epoch_refined(
            model, train_loader, optimizer, scaler, device, local_rank, nprocs, args,
            clip_loss_fn=clip_loss_fn, vit_mode=vit_mode
        )
        barrier(ddp)

        if local_rank == 0:
            update_train_result(epoch, loss_info, writer)

            eval_now = (epoch >= args.eval_start) and ((epoch - args.eval_start) % args.eval_freq == 0)
            if not eval_now:
                log(logger, epoch, args.total_epochs, loss_info=loss_info, message="\n" * 2)
                save_last_and_best(epoch, is_best=False, curr_val_scores=None)
            else:
                eval_model = model.module if ddp else model
                eval_model.steepness = args.max_steepness

                val_mae, val_rmse = evaluate_mae_rmse(eval_model, val_loader, device)
                curr_val_scores = {"mae": float(val_mae), "rmse": float(val_rmse)}

                try:
                    writer.add_scalar("val/mae", float(val_mae), epoch)
                    writer.add_scalar("val/rmse", float(val_rmse), epoch)
                except Exception:
                    pass

                hist_val_scores.setdefault("mae", []).append((epoch, float(val_mae)))
                hist_val_scores.setdefault("rmse", []).append((epoch, float(val_rmse)))

                is_best = float(val_mae) < best_mae
                if is_best:
                    best_mae = float(val_mae)
                    best_val_scores["mae"][0] = best_mae
                    best_val_scores["rmse"][0] = float(val_rmse)
                    best_val_scores["epoch"][0] = epoch
                    print(f"🌟 New Best MAE: {best_mae:.4f} | RMSE: {float(val_rmse):.4f} at epoch {epoch}")

                log(logger, epoch, args.total_epochs, None, curr_val_scores, best_val_scores, message="\n" * 3)
                save_last_and_best(epoch, is_best=is_best, curr_val_scores=curr_val_scores)

        barrier(ddp)

    if local_rank == 0 and writer is not None:
        writer.close()
    cleanup(ddp)


def main():
    args = parser.parse_args()
    args = load_config_and_update_args(args)

    args.nprocs = torch.cuda.device_count()
    print(f"Using {args.nprocs} GPUs.")
    if args.nprocs > 1:
        mp.spawn(run, nprocs=args.nprocs, args=(args.nprocs, args))
    else:
        run(0, 1, args)


if __name__ == "__main__":
    main()
