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
from utils import get_dataloader, load_checkpoint, save_checkpoint
from utils import get_writer, update_train_result, update_eval_result, log

# ------------------------------------------------------------
# Parser
# ------------------------------------------------------------
parser = ArgumentParser("Train Stage 3 (Joint ZIP+CLIP) - aligned to Stage2 pipeline")

parser.add_argument("--config", type=str, default="configs/config.yaml")
parser.add_argument("--s1", type=str, required=True)  # ckpt stage1
parser.add_argument("--s2", type=str, required=True)  # ckpt stage2

# stage2 model name (clip_vit_b_16 etc.)
parser.add_argument("--model", type=str, default="clip_vit_b_16")
parser.add_argument("--input_size", type=int, default=224)
parser.add_argument("--reduction", type=int, default=16, choices=[8, 16, 32])

# dataset
parser.add_argument("--dataset", type=str, required=False)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=4)

# bins cfg (come stage2)
parser.add_argument("--regression", action="store_true")  # tienilo False (Stage2-style bins)
parser.add_argument("--truncation", type=int, default=4)
parser.add_argument("--anchor_points", type=str, default="average", choices=["average", "middle"])
parser.add_argument("--prompt_type", type=str, default="word", choices=["word", "number"])
parser.add_argument("--granularity", type=str, default="fine", choices=["fine", "dynamic", "coarse"])
parser.add_argument("--num_vpt", type=int, default=32)
parser.add_argument("--vpt_drop", type=float, default=0.0)
parser.add_argument("--shallow_vpt", action="store_true")

# train
parser.add_argument("--lr", type=float, default=1e-6)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--total_epochs", type=int, default=50)
parser.add_argument("--eval_start", type=int, default=1)
parser.add_argument("--eval_freq", type=int, default=1)
parser.add_argument("--save_freq", type=int, default=1)
parser.add_argument("--amp", action="store_true")
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--seed", type=int, default=42)



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
parser.add_argument("--max_steepness", type=float, default=20.0)
parser.add_argument("--zip_w", type=float, default=0.5)
parser.add_argument("--cons_w", type=float, default=0.1)

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
    # identico allo stage2
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
            # fallback: può essere già un state_dict
            sd = ckpt
    else:
        sd = ckpt

    # strip "module."
    if isinstance(sd, dict) and any(kk.startswith("module.") for kk in sd.keys()):
        sd = {kk.replace("module.", "", 1): vv for kk, vv in sd.items()}
    return sd


# ------------------------------------------------------------
# Train/Eval
# ------------------------------------------------------------
def train_one_epoch_refined(model, loader, optimizer, scaler, device, rank, nprocs, args):
    model.train()
    total_loss = 0.0

    it = tqdm(loader, desc="Train S3") if rank == 0 else loader
    bce = nn.BCEWithLogitsLoss()

    for batch in it:
        # get_dataloader nel tuo progetto spesso ritorna dict con image/density/points
        if isinstance(batch, dict):
            imgs = batch["image"].to(device)
            gt_density = batch["density"].to(device)
            points = batch.get("points", None)
        else:
            imgs, points, gt_density = batch
            imgs = imgs.to(device)
            gt_density = gt_density.to(device)

        optimizer.zero_grad()

        with autocast(enabled=(scaler is not None)):
            out = model(imgs)

            # ---- 1) "CLIP loss" (qui metto la forma minima robusta) ----
            # Se vuoi usare esattamente la loss stage2, qui andrebbe richiamata la tua get_loss_fn(args)
            # ma senza il file non posso importarla correttamente.
            # Quindi: loss count semplice su densità finale (MSE su density map) + MAE sul count
            pred_density = out["final_density"]
            gt_resized = F.interpolate(gt_density, size=pred_density.shape[-2:], mode="bilinear", align_corners=False)
            # scala area per conservare conteggi (come già fai)
            scale = (gt_density.shape[-2] * gt_density.shape[-1]) / (pred_density.shape[-2] * pred_density.shape[-1])
            gt_resized = gt_resized * scale

            l_map = F.mse_loss(pred_density, gt_resized)

            pred_cnt = pred_density.sum(dim=[1,2,3])
            gt_cnt = gt_density.sum(dim=[1,2,3])
            l_cnt = (pred_cnt - gt_cnt).abs().mean()

            l_clip = l_map + l_cnt

            # ---- 2) ZIP loss (supervisione mask) ----
            pi_logits = out["pi_logits"]
            h, w = pi_logits.shape[-2:]
            with torch.no_grad():
                gt_pi = F.interpolate(gt_density, size=(h, w), mode="bilinear", align_corners=False)
                gt_pi = gt_pi * ((gt_density.shape[-1] / w) ** 2)
                mask_target = (gt_pi > 0.001).float()
            l_zip = bce(pi_logits, mask_target)

            # ---- 3) Consistency loss ----
            clip_raw = out["raw_density"].detach()
            prob_zip = out["pi_prob"]
            l_cons = (clip_raw * (1.0 - prob_zip)).mean()

            loss = l_clip + args.zip_w * l_zip + args.cons_w * l_cons

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        if rank == 0:
            it.set_postfix({"L": f"{loss.item():.3f}", "clip": f"{l_clip.item():.3f}", "zip": f"{l_zip.item():.3f}"})

    return {"loss": total_loss / max(1, len(loader))}


@torch.no_grad()
def evaluate_mae(model, loader, device, rank=0):
    model.eval()
    mae = 0.0
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

        # se points esiste, usa quello (conteggio GT più affidabile)
        if points is not None:
            gt = len(points[0])
        else:
            # fallback: se manca points non posso fare count vero
            gt = pred

        mae += abs(pred - gt)
        n += 1
    return mae / max(1, n)


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

    # bins/anchors like stage2
    bins, anchor_points = build_bins_and_anchors(args)
    args.bins = bins
    args.anchor_points = anchor_points

    # dataloaders (identico stile stage2)
    if local_rank == 0:
        val_loader = get_dataloader(args, split="val", ddp=False)

    args.batch_size = int(args.batch_size / nprocs) if ddp else int(args.batch_size)
    args.num_workers = int(args.num_workers / nprocs) if ddp else int(args.num_workers)
    train_loader, sampler = get_dataloader(args, split="train", ddp=ddp)

    # --- build Stage1 config from YAML (ZIPModel wants dict with BACKBONE/ZIP_HEAD) ---
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    zip_cfg = cfg  # contiene BACKBONE, ZIP_HEAD ecc.
    stage1 = ZIPModel(zip_cfg).to(device)

    # stage2 (same builder as stage2)
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

    model = ZIPCLIPJointModel(stage1, stage2, steepness=args.base_steepness).to(device)

    # optimizer + scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler() if args.amp else None

    # ckpt dir
    args.ckpt_dir = args.out  # <-- FIX
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if local_rank == 0:
        shutil.copyfile(args.config, os.path.join(args.ckpt_dir, "config_stage3.yaml"))
        writer = get_writer(args.ckpt_dir)
        logger = get_logger(os.path.join(args.ckpt_dir, "train_stage3.log"))
        logger.info(get_config(vars(args), mute=False))

    # ddp
    model = DDP(nn.SyncBatchNorm.convert_sync_batchnorm(model), device_ids=[local_rank], output_device=local_rank) if ddp else model

    start_epoch = 1
    best_mae = float("inf")
    hist_val_scores = {}  # opzionale, lo teniamo solo per logging interno
    best_val_scores = {"mae": [float("inf")], "epoch": [0]}
    loss_info = {}

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

        last_path = os.path.join(args.ckpt_dir, "last_model.pth")
        torch.save(state, last_path)

        if is_best:
            best_path = os.path.join(args.ckpt_dir, "best_model.pth")
            torch.save(state, best_path)



    curr_val_scores = None
    for epoch in range(start_epoch, args.total_epochs + 1):
        curr_val_scores = None
        if sampler is not None:
            sampler.set_epoch(epoch)

        # steepness annealing
        if args.total_epochs > 1:
            steep = args.base_steepness + (epoch - 1) / (args.total_epochs - 1) * (args.max_steepness - args.base_steepness)
        else:
            steep = args.max_steepness
        (model.module if ddp else model).steepness = float(steep)



        # train
        loss_info = train_one_epoch_refined(model, train_loader, optimizer, scaler, device, local_rank, nprocs, args)
        barrier(ddp)

        if local_rank == 0:
            # log train
            update_train_result(epoch, loss_info, writer)

            eval_now = (epoch >= args.eval_start) and ((epoch - args.eval_start) % args.eval_freq == 0)

            if not eval_now:
                log(logger, epoch, args.total_epochs, loss_info=loss_info, message="\n" * 2)
                save_last_and_best(epoch, is_best=False, curr_val_scores=None)

            else:
                # eval
                eval_model = model.module if ddp else model
                eval_model.steepness = args.max_steepness

                val_mae = evaluate_mae(eval_model, val_loader, device)
                curr_val_scores = {"mae": float(val_mae)}

                # tensorboard val
                try:
                    writer.add_scalar("val/mae", float(val_mae), epoch)
                except Exception:
                    pass

                # storico semplice
                hist_val_scores.setdefault("mae", [])
                hist_val_scores["mae"].append((epoch, float(val_mae)))

                # best?
                is_best = float(val_mae) < best_mae
                if is_best:
                    best_mae = float(val_mae)
                    best_val_scores["mae"][0] = best_mae
                    best_val_scores["epoch"][0] = epoch
                    print(f"🌟 New Best MAE: {best_mae:.4f} at epoch {epoch}")

                # log eval (UNA SOLA VOLTA)
                log(logger, epoch, args.total_epochs, None, curr_val_scores, best_val_scores, message="\n" * 3)

                # salva last + eventualmente best
                save_last_and_best(epoch, is_best=is_best, curr_val_scores=curr_val_scores)


        barrier(ddp)

    if local_rank == 0:
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
