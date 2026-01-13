

import os
import math
import yaml
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from torch.amp import GradScaler, autocast  # ✅ torch.amp (no warning / più robusto)

from models.clip_ebc_model import CLIPEBCModel
from datasets.builder import build_dataset
from datasets.transforms import build_transforms
from losses.clip_ebc_loss import DACELoss


# -----------------------------
# Collate
# -----------------------------
def crowd_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    out = {
        "image": torch.stack([item["image"] for item in batch]),
        "density": torch.stack([item["density"] for item in batch]),
        "points": [item["points"] for item in batch],
    }
    if "img_path" in batch[0]:
        out["img_path"] = [item.get("img_path", "") for item in batch]
    return out


# -----------------------------
# LR schedule: warmup + cosine
# -----------------------------
def adjust_learning_rate(optimizer, epoch, total_epochs, warmup_epochs):
    if warmup_epochs > 0 and epoch < warmup_epochs:
        lr_ratio = (epoch + 1) / (warmup_epochs + 1e-8)
    else:
        den = max(1, total_epochs - warmup_epochs)
        progress = (epoch - warmup_epochs) / den
        lr_ratio = 0.5 * (1.0 + math.cos(math.pi * progress))

    for pg in optimizer.param_groups:
        base = pg.get("_base_lr", pg["lr"])
        pg["lr"] = base * lr_ratio

    return optimizer.param_groups[-1]["lr"]


# -----------------------------
# Optimizer builder (VPT-aware)
# -----------------------------
def build_optimizer(model: CLIPEBCModel, cfg_s2: dict):
    wd = float(cfg_s2.get("WEIGHT_DECAY", 0.0))

    lr_backbone = float(cfg_s2.get("LR_BACKBONE", cfg_s2.get("LR", 1e-6)))
    lr_head = float(cfg_s2.get("LR_HEAD", cfg_s2.get("LR", 1e-4)))
    lr_vpt = float(cfg_s2.get("LR_VPT", lr_head))

    param_groups = []

    if hasattr(model, "get_param_groups"):
        groups = model.get_param_groups()

        if len(groups.get("backbone_visual", [])) > 0:
            param_groups.append(
                {"params": groups["backbone_visual"], "lr": lr_backbone, "_base_lr": lr_backbone, "name": "backbone"}
            )

        if len(groups.get("vpt", [])) > 0:
            param_groups.append(
                {"params": groups["vpt"], "lr": lr_vpt, "_base_lr": lr_vpt, "name": "vpt"}
            )

        if len(groups.get("heads", [])) > 0:
            param_groups.append(
                {"params": groups["heads"], "lr": lr_head, "_base_lr": lr_head, "name": "head"}
            )

    if len(param_groups) == 0:
        backbone_params, head_params = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "clip_model.visual" in n or "visual_encoder" in n:
                backbone_params.append(p)
            else:
                head_params.append(p)

        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": lr_backbone, "_base_lr": lr_backbone, "name": "backbone"})
        if head_params:
            param_groups.append({"params": head_params, "lr": lr_head, "_base_lr": lr_head, "name": "head"})

    optimizer = AdamW(param_groups, weight_decay=wd)

    print("🔧 Optimizer param groups:")
    for i, pg in enumerate(optimizer.param_groups):
        n_params = sum(p.numel() for p in pg["params"])
        print(f"  - [{i}] {pg.get('name','group')} | lr={pg['lr']:.2e} | params={n_params}")
    return optimizer


# -----------------------------
# Validate: MAE + RMSE
# -----------------------------
@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    abs_sum = 0.0
    sq_sum = 0.0
    n = 0

    for batch in tqdm(val_loader, desc="Validating", leave=False):
        if batch is None:
            continue

        images = batch["image"].to(device, non_blocking=True)
        points_list = batch["points"]

        out = model(images)
        pred_counts = out["final_count"]  # [B]

        for i, pts in enumerate(points_list):
            gt = len(pts)
            pred = float(pred_counts[i].item())
            diff = pred - gt
            abs_sum += abs(diff)
            sq_sum += diff * diff
            n += 1

    n = max(1, n)
    mae = abs_sum / n
    rmse = math.sqrt(sq_sum / n)
    return mae, rmse


# -----------------------------
# Checkpoint IO
# -----------------------------
def save_ckpt(path, epoch, model, optimizer, scaler, best_mae):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_mae": best_mae,
        },
        path,
    )


def load_ckpt(path, model, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_mae = float(ckpt.get("best_mae", float("inf")))
    return start_epoch, best_mae


# -----------------------------
# Out dir resolver
# -----------------------------
def resolve_out_dir(config, cli_out):
    if cli_out is not None:
        return cli_out
    base = Path(config.get("EXP", {}).get("OUT_DIR", "./checkpoints"))
    dataset = config.get("DATASET", "unknown")
    return str(base / dataset / "stage2")


def get_stage2_crop_size(data_cfg: dict):
    # ✅ usa il crop “stage2” se presente, altrimenti il generico
    return int(data_cfg.get("CROP_SIZE_STAGE2", data_cfg.get("CROP_SIZE", 448)))


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Device
    device_str = str(config.get("DEVICE", "cuda"))
    if device_str.startswith("cuda"):
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device(device_str)

    # Seed
    seed = int(config.get("SEED", 42))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Out dir
    out_dir = resolve_out_dir(config, args.out)
    print("📁 Output Directory:", out_dir, flush=True)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    print(f"🚀 Stage2 | dataset={config.get('DATASET')} | out={out_dir}", flush=True)
    print(f"🖥️  device={device} | seed={seed}", flush=True)

    # Build datasets/loaders
    train_tf = build_transforms(config["DATA"], is_train=True)
    val_tf = build_transforms(config["DATA"], is_train=False)

    train_ds = build_dataset(config, "train", train_tf)
    val_ds = build_dataset(config, "val", val_tf)

    s2 = config["TRAIN_STAGE2"]
    train_loader = DataLoader(
        train_ds,
        batch_size=int(s2["BATCH_SIZE"]),
        shuffle=True,
        num_workers=int(s2.get("NUM_WORKERS", 4)),
        collate_fn=crowd_collate,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, int(s2.get("NUM_WORKERS", 4)) // 2),
        collate_fn=crowd_collate,
        pin_memory=True,
    )

    # Model
    model = CLIPEBCModel(config).to(device)

    # ✅ crop size reale per Stage2 (serve alla loss OT)
    crop_size = get_stage2_crop_size(config.get("DATA", {}))
    print(f"🧩 Stage2 crop_size (for loss OT grid) = {crop_size}", flush=True)

    # ✅ reduction dal modello (evita mismatch)
    reduction = int(getattr(model.visual_encoder, "reduction", 16))
    print(f"🔎 Using REDUCTION={reduction} (from model.visual_encoder)", flush=True)

    # Loss (IMPORTANTISSIMO: passa input_size=crop_size)
    lcfg = config.get("LOSS_STAGE2", {})
    input_size = int(config["DATA"].get("CROP_SIZE_STAGE2", config["DATA"].get("CROP_SIZE", 448)))

    criterion = DACELoss(
        bins=config["BINS"],
        reduction=reduction,
        input_size=input_size,  # <-- IMPORTANTISSIMO per OT (224 vs 448)
        weight_count=float(lcfg.get("WEIGHT_COUNT_LOSS", 1.0)),
        count_loss=str(lcfg.get("COUNT_LOSS", "dmcount")),
        weight_ot=float(lcfg.get("WEIGHT_OT", 0.1)),
        weight_tv=float(lcfg.get("WEIGHT_TV", 0.01)),
        label_smoothing=float(lcfg.get("LABEL_SMOOTHING", 0.0)),
    ).to(device)

    # Optimizer
    optimizer = build_optimizer(model, s2)

    # AMP
    amp_enabled = bool(s2.get("AMP", True)) and (device.type == "cuda")
    scaler = GradScaler("cuda", enabled=amp_enabled)

    # Resume
    last_path = os.path.join(out_dir, "last.pth")
    best_path = os.path.join(out_dir, "best_model.pth")

    start_epoch = 0
    best_mae = float("inf")
    if os.path.exists(last_path):
        print(f"🔄 Auto-resume from {last_path}", flush=True)
        start_epoch, best_mae = load_ckpt(last_path, model, optimizer, scaler, device)
        print(f"⏩ start_epoch={start_epoch} | best_mae={best_mae:.2f}", flush=True)

    # Train loop
    total_epochs = int(s2["TOTAL_EPOCHS"])
    warmup_epochs = int(s2.get("WARMUP_EPOCHS", 0))
    eval_freq = int(s2.get("EVAL_FREQ", 5))
    clip_grad = float(s2.get("CLIP_GRAD_NORM", 1.0))

    # ✅ one-time shape sanity check
    shape_checked = False

    for epoch in range(start_epoch, total_epochs):
        curr_lr = adjust_learning_rate(optimizer, epoch, total_epochs, warmup_epochs)

        model.train()
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{total_epochs} | LR {curr_lr:.2e}", leave=True)

        for batch in pbar:
            if batch is None:
                continue

            images = batch["image"].to(device, non_blocking=True)
            gt_density = batch["density"].to(device, non_blocking=True)
            points = [p.to(device, non_blocking=True) for p in batch["points"]]

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=amp_enabled):
                out = model(images)

                # ✅ SANITY: pred grid must match crop_size//reduction
                if not shape_checked:
                    pred_h, pred_w = out["ebc_density"].shape[-2:]
                    expected = crop_size // reduction
                    print(f"🔍 pred_grid={pred_h}x{pred_w} | expected={expected}x{expected}", flush=True)
                    if (pred_h != expected) or (pred_w != expected):
                        raise RuntimeError(
                            f"[ShapeMismatch] pred={pred_h}x{pred_w} but expected {expected}x{expected}. "
                            f"Check DATA.CROP_SIZE_STAGE2 ({crop_size}) and reduction ({reduction}) and your transforms."
                        )
                    shape_checked = True

                loss, loss_dict = criterion(out["ebc_logits"], out["ebc_density"], gt_density, points)

            scaler.scale(loss).backward()

            if clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        # Validation
        if (epoch + 1) % eval_freq == 0:
            mae, rmse = validate(model, val_loader, device)
            print(
                f"\n📊 Ep {epoch+1} | Val MAE: {mae:.2f} | Val RMSE: {rmse:.2f} | Best MAE: {best_mae:.2f}",
                flush=True
            )

            if mae < best_mae:
                best_mae = mae
                torch.save(
                    {"epoch": epoch, "model": model.state_dict(), "mae": best_mae, "rmse": rmse, "config": config},
                    best_path,
                )
                print(f"🌟 Saved BEST -> {best_path}", flush=True)

        # Save last (every epoch)
        save_ckpt(last_path, epoch, model, optimizer, scaler, best_mae)

    print("✅ Training completed.", flush=True)
    print("\n Best Val MAE:", best_mae, "Best val RMSE:", best_rmse, flush=True)


if __name__ == "__main__":
    main()
