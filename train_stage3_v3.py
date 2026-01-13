#!/usr/bin/env python3
import os
import math
import yaml
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from models.zip_model import ZIPModel
from models.clip_ebc_model import CLIPEBCModel
from models.joint_model import ZIPCLIPJointModel

from losses.clip_ebc_loss import CLIPEBCLoss

from datasets.transforms import build_transforms
from datasets.sha import SHA


# -------------------------
# Collate (coerente coi tuoi train)
# -------------------------
def crowd_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "density": torch.stack([item["density"] for item in batch]),
        "points": [item["points"] for item in batch],
        "img_path": [item.get("img_path", "") for item in batch],
    }


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


# -------------------------
# ZIP target mask from GT density
# (stessa idea che usi nei tuoi stage)
# -------------------------
def build_zip_mask_gt(gt_density: torch.Tensor, out_hw, eps: float = 1e-3):
    out_h, out_w = out_hw
    if gt_density.shape[-2:] != (out_h, out_w):
        gt_resized = F.interpolate(gt_density, size=(out_h, out_w), mode="bilinear", align_corners=False)
        scale = (gt_density.shape[-1] / out_w) ** 2
        gt_resized = gt_resized * scale
    else:
        gt_resized = gt_density

    mask_gt = (gt_resized > eps).float()
    return mask_gt


# -------------------------
# Debug mask stats
# -------------------------
def mask_stats(prob_presence: torch.Tensor, thr: float = 0.5):
    p = prob_presence.clamp(1e-6, 1 - 1e-6)
    mean = p.mean().item()
    active = (p > thr).float().mean().item()
    entropy = (-(p * torch.log(p) + (1 - p) * torch.log(1 - p))).mean().item()
    return mean, active, entropy


def soft_iou(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    inter = (pred * gt).sum(dim=(1, 2, 3))
    union = (pred + gt - pred * gt).sum(dim=(1, 2, 3))
    return ((inter + eps) / (union + eps)).mean().item()


# -------------------------
# Validation (MAE + RMSE) sul conteggio finale
# -------------------------

@torch.no_grad()
def sliding_window_predict(model, image, window_size=448, stride=448, device='cuda'):
    model.eval()
    B, C, H, W = image.shape

    density_map = torch.zeros((H, W), device=device)
    count_map = torch.zeros((H, W), device=device)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_end = min(y + window_size, H)
            x_end = min(x + window_size, W)

            y_start = max(y_end - window_size, 0)
            x_start = max(x_end - window_size, 0)

            crop = image[:, :, y_start:y_end, x_start:x_end].to(device)

            out = model(crop)
            pred_crop = out['final_density']  # [1,1,h,w]

            density_map[y_start:y_end, x_start:x_end] += pred_crop.squeeze()
            count_map[y_start:y_end, x_start:x_end] += 1.0

    final_density = density_map / count_map
    return final_density

@torch.no_grad()
def validate(model, val_loader, device, window_size=448, stride=448, hard_steepness=20.0):
    model.eval()
    abs_sum, sq_sum, n = 0.0, 0.0, 0

    # forza la stessa steepness dell'eval (salva/ripristina)
    old_steep = getattr(model, "steepness", None)
    if old_steep is not None:
        model.steepness = hard_steepness

    for batch in tqdm(val_loader, desc="Validating", leave=False):
        if batch is None:
            continue

        img = batch["image"]          # resta su CPU come nell'eval
        gt = len(batch["points"][0])

        # stessa politica: se grande -> sliding window
        if img.shape[2] > 1024 or img.shape[3] > 1024:
            pred_density = sliding_window_predict(
                model, img,
                window_size=window_size, stride=stride,
                device=device
            )
            pred = pred_density.sum().item()
        else:
            img = img.to(device)
            out = model(img)
            pred = out["final_density"].sum().item()

        diff = pred - gt
        abs_sum += abs(diff)
        sq_sum += diff * diff
        n += 1

    # ripristina steepness
    if old_steep is not None:
        model.steepness = old_steep

    mae = abs_sum / max(1, n)
    rmse = math.sqrt(sq_sum / max(1, n))
    return mae, rmse


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--s1", type=str, default=None)
    parser.add_argument("--s2", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # device
    if "cuda" in str(config.get("DEVICE", "cuda")).lower() and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    t3 = config.get("TRAIN_STAGE3", {})

    # output dir
    out_dir = args.out
    print("📁 Output Directory:", out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # hyperparams (config-driven)
    epochs = int(t3.get("EPOCHS", 50))
    bs = int(t3.get("BATCH_SIZE", 4))
    num_workers = int(t3.get("NUM_WORKERS", 4))

    lr = float(t3.get("LR", 1e-6))
    wd = float(t3.get("WEIGHT_DECAY", 1e-4))

    # puoi anche separare LR se lo vuoi aggiungendo LR_ZIP/LR_CLIP nel config
    lr_zip = float(t3.get("LR_ZIP", lr))
    lr_clip = float(t3.get("LR_CLIP", lr))

    lambda_zip = float(t3.get("LAMBDA_ZIP", 1.0))
    lambda_clip = float(t3.get("LAMBDA_CLIP", 1.0))
    lambda_count = float(t3.get("LAMBDA_COUNT", 10.0))

    # alternanza (opzionali)
    zip_steps = int(t3.get("ZIP_STEPS", 1))
    clip_steps = int(t3.get("CLIP_STEPS", 1))

    # debug logging (opzionali)
    log_every = int(t3.get("LOG_EVERY", 50))
    mask_thr = float(t3.get("MASK_THR", 0.5))
    mask_eps = float(t3.get("MASK_EPS", 1e-3))

    # AMP + grad clip
    amp_enabled = bool(t3.get("AMP", True)) and (device.type == "cuda")
    clip_grad_norm = float(t3.get("CLIP_GRAD_NORM", 1.0))

    ckpt_s1 = args.s1
    ckpt_s2 = args.s2
    if not ckpt_s1 or not ckpt_s2:
        raise ValueError("TRAIN_STAGE3 deve contenere CHECKPOINT_STAGE1 e CHECKPOINT_STAGE2")

    print(f"🔧 Device: {device}")
    print(f"🚀 Stage3 Intermittenza | epochs={epochs} bs={bs} zip_steps={zip_steps} clip_steps={clip_steps}")
    print(f"   LR_ZIP={lr_zip:.2e} LR_CLIP={lr_clip:.2e} WD={wd:.1e} AMP={amp_enabled}")
    print(f"   λ_zip={lambda_zip} λ_clip={lambda_clip} λ_count={lambda_count}")
    print(f"   debug: log_every={log_every} mask_thr={mask_thr} mask_eps={mask_eps}")

    # -------------------------
    # Build stage1 + stage2 models
    # -------------------------
    stage1 = ZIPModel(config).to(device)
    stage2 = CLIPEBCModel(config).to(device)

    # load checkpoints
    s1_state = torch.load(ckpt_s1, map_location=device)
    s2_state = torch.load(ckpt_s2, map_location=device)

    # supporta sia state_dict puro che dict con chiave "model"
    stage1.load_state_dict(s1_state.get("model", s1_state), strict=True)
    stage2.load_state_dict(s2_state.get("model", s2_state), strict=True)

    # joint model: usa il tuo init reale (stage1_model, stage2_model)
    steepness = float(t3.get("STEEPNESS", 1.0))
    model = ZIPCLIPJointModel(stage1, stage2, steepness=steepness).to(device)

    # losses (coerenti con la tua JointLoss, ma separabili per fase)
    zip_bce = nn.BCEWithLogitsLoss()

    crop_size = int(config["DATA"].get("CROP_SIZE", 448))
    reduction = int(config.get("CLIP_EBC_HEAD", {}).get("REDUCTION", 16))
    loss2_cfg = config.get("LOSS_STAGE2", {})

    clip_loss_fn = CLIPEBCLoss(
        bins=config.get("BINS", []),
        input_size=crop_size,
        reduction=reduction,
        weight_ot=float(loss2_cfg.get("WEIGHT_OT", 0.1)),
        weight_tv=float(loss2_cfg.get("WEIGHT_TV", 0.01)),
        weight_count=float(loss2_cfg.get("WEIGHT_COUNT_LOSS", 1.0)),
        label_smoothing=float(loss2_cfg.get("LABEL_SMOOTHING", 0.0)),
    ).to(device)

    # optimizers separati
    opt_zip = AdamW(model.stage1.parameters(), lr=lr_zip, weight_decay=wd)
    opt_clip = AdamW(model.stage2.parameters(), lr=lr_clip, weight_decay=wd)

    scaler = GradScaler("cuda", enabled=amp_enabled)

    # data (qui: SHA, ma stessa struttura che usi)
    train_ds = SHA(config["DATA"]["ROOT"], "train", build_transforms(config["DATA"], True))
    val_ds = SHA(config["DATA"]["ROOT"], "val", build_transforms(config["DATA"], False))

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True, collate_fn=crowd_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=max(1, num_workers // 2),
        pin_memory=True, collate_fn=crowd_collate
    )

    best_mae = float("inf")

    # alternanza state
    phase = "zip"
    phase_left = zip_steps
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{epochs}")

        for batch in pbar:
            if batch is None:
                continue

            global_step += 1
            images = batch["image"].to(device)
            gt_density = batch["density"].to(device)
            points = batch["points"]
            gt_count = torch.tensor([len(p) for p in points], device=device, dtype=torch.float32)

            # switch phase
            if phase_left <= 0:
                if phase == "zip":
                    phase = "clip"
                    phase_left = clip_steps
                else:
                    phase = "zip"
                    phase_left = zip_steps
            phase_left -= 1

            # set trainable params
            if phase == "zip":
                set_requires_grad(model.stage1, True)
                set_requires_grad(model.stage2, False)
                opt = opt_zip
            else:
                set_requires_grad(model.stage1, False)
                set_requires_grad(model.stage2, True)
                opt = opt_clip

            opt.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=amp_enabled):
                outputs = model(images)

                # ZIP supervision mask_gt
                out_h, out_w = outputs["pi_logits"].shape[-2:]
                mask_gt = build_zip_mask_gt(gt_density, (out_h, out_w), eps=mask_eps)

                # 1) ZIP loss
                l_zip = zip_bce(outputs["pi_logits"], mask_gt)

                # 2) CLIP loss (ATTENZIONE: nel tuo joint_model la densità raw si chiama 'raw_density')
                l_clip, dict_clip = clip_loss_fn(
                    pred_class=outputs["ebc_logits"],
                    pred_density=outputs["raw_density"],
                    target_density=gt_density,
                    target_points=[p.to(device) for p in points],
                )

                # 3) Count loss sul risultato gated (cuore del matrimonio)
                pred_count = outputs["final_density"].sum(dim=(1, 2, 3))
                l_count = F.l1_loss(pred_count, gt_count)

                # phase-specific total
                if phase == "zip":
                    total = (lambda_zip * l_zip) + (lambda_count * l_count)
                else:
                    total = (lambda_clip * l_clip) + (lambda_count * l_count)

            scaler.scale(total).backward()
            if clip_grad_norm > 0:
                scaler.unscale_(opt)
                if phase == "zip":
                    torch.nn.utils.clip_grad_norm_(model.stage1.parameters(), clip_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.stage2.parameters(), clip_grad_norm)

            scaler.step(opt)
            scaler.update()

            # progress bar
            if phase == "zip":
                pbar.set_postfix({"ph": "ZIP", "L": f"{total.item():.3f}", "Lzip": f"{l_zip.item():.3f}", "Lcnt": f"{l_count.item():.3f}"})
            else:
                pbar.set_postfix({"ph": "CLIP", "L": f"{total.item():.3f}", "Lclip": f"{l_clip.item():.3f}", "Lcnt": f"{l_count.item():.3f}"})

            # debug logging
            if log_every > 0 and (global_step % log_every) == 0:
                with torch.no_grad():
                    prob_presence = outputs["pi_prob"]  # già in [0,1] e allineata alla densità raw
                    # per softIoU serve mask_gt alla stessa risoluzione di pi_prob
                    # qui pi_prob è già su raw_density size; mask_gt è su pi_logits size.
                    # quindi riallineiamo mask_gt:
                    if mask_gt.shape[-2:] != prob_presence.shape[-2:]:
                        mask_gt_al = F.interpolate(mask_gt, size=prob_presence.shape[-2:], mode="nearest")
                    else:
                        mask_gt_al = mask_gt

                    mean, active, ent = mask_stats(prob_presence, thr=mask_thr)
                    siou = soft_iou(prob_presence, mask_gt_al)

                    cnt_raw = outputs["raw_density"].sum(dim=(1, 2, 3)).mean().item()
                    cnt_final = outputs["final_density"].sum(dim=(1, 2, 3)).mean().item()
                    gt_mean = gt_count.mean().item()

                    print(
                        f"🧩 step={global_step} ep={epoch} ph={phase} | "
                        f"presence_mean={mean:.3f} active@{mask_thr}={active:.3f} entropy={ent:.3f} softIoU={siou:.3f} | "
                        f"cnt_raw={cnt_raw:.1f} cnt_final={cnt_final:.1f} gt={gt_mean:.1f}"
                    )

        # validation
        val_mae, val_rmse = validate(model, val_loader, device)
        print(f"📊 Ep {epoch} | Val MAE: {val_mae:.2f} | Val RMSE: {val_rmse:.2f} (Best MAE: {best_mae:.2f})")

        # save last
        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "mae": val_mae, "rmse": val_rmse},
            os.path.join(out_dir, "last_model.pth"),
        )

        # save best
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "mae": best_mae, "rmse": val_rmse},
                os.path.join(out_dir, "best_model.pth"),
            )
            print("🌟 Saved Best Model")
    print("\n Best Val MAE:", best_mae, "best RMSE:", val_rmse)
    print("✅ Training completato.")


if __name__ == "__main__":
    main()
