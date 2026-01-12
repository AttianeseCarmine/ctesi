#!/usr/bin/env python3
"""
Visualize Stage3 like paper:
Row 1: Image
Row 2: GT density overlay + GT count text
Row 3: Pred final density overlay + Pred count text

Fixed images: IMG_10.jpg, IMG_20.jpg, IMG_100.jpg (by basename match in dataset split)
"""

import os
import math
import yaml
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from torch.utils.data import DataLoader

from datasets.sha import SHA  # se usi SHB, cambia qui oppure passa --dataset shb
from datasets.transforms import build_transforms

from models.zip_model import ZIPModel
from models.clip_ebc_model import CLIPEBCModel
from models.joint_model import ZIPCLIPJointModel


# --- denormalize utility (ImageNet stats, come nei tuoi train) ---
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def denorm_img(x: torch.Tensor) -> np.ndarray:
    """
    x: [3,H,W] normalized tensor
    returns uint8 RGB image [H,W,3]
    """
    x = x.detach().cpu()
    x = x * IMAGENET_STD + IMAGENET_MEAN
    x = x.clamp(0, 1)
    img = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return img


def overlay_density(ax, img_rgb, density, cmap="jet", alpha=0.55):
    """
    img_rgb: [H,W,3] uint8
    density: [H,W] float (numpy)
    """
    ax.imshow(img_rgb)
    # normalizzazione robusta: evita che poche peak saturino tutto
    vmax = np.percentile(density, 99.5) if density.max() > 0 else 1.0
    vmax = max(vmax, 1e-6)
    ax.imshow(density, cmap=cmap, alpha=alpha, vmin=0.0, vmax=vmax)
    ax.axis("off")


def add_corner_text(ax, text: str):
    # stile simile screenshot: bianco bold con contorno nero
    ax.text(
        8, 22, text,
        color="white",
        fontsize=14,
        fontweight="bold",
        ha="left", va="top",
        path_effects=[pe.withStroke(linewidth=3, foreground="black")]
    )


@torch.no_grad()
def predict_final_density(model, img_tensor, device):
    """
    img_tensor: [1,3,H,W] normalized, on CPU or GPU
    returns final_density numpy [H,W] and pred_count float
    """
    model.eval()
    img_tensor = img_tensor.to(device)
    out = model(img_tensor)
    final_density = out["final_density"]  # [1,1,h,w]
    final_density = final_density.squeeze(0).squeeze(0).detach().cpu().numpy()
    pred_count = float(final_density.sum())
    return final_density, pred_count


def find_samples_by_basenames(dataset, basenames):
    """
    Cerca nel dataset gli indici con basename uguale a uno dei basenames richiesti.
    Funziona anche se il dataset non espone image_list: iteriamo.
    """
    targets = {b: None for b in basenames}
    found = 0

    # tentativo 1: se dataset ha lista immagini
    img_list = getattr(dataset, "img_list", None) or getattr(dataset, "image_list", None) or None
    if img_list is not None:
        for i, p in enumerate(img_list):
            bn = os.path.basename(str(p))
            if bn in targets and targets[bn] is None:
                targets[bn] = i
                found += 1
            if found == len(basenames):
                break
        return targets

    # tentativo 2: fallback, iterazione (lento ma qui sono 3 immagini)
    for i in range(len(dataset)):
        sample = dataset[i]
        img_path = sample.get("img_path", None)
        if isinstance(img_path, str):
            bn = os.path.basename(img_path)
            if bn in targets and targets[bn] is None:
                targets[bn] = i
                found += 1
        if found == len(basenames):
            break

    return targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="stage3 checkpoint (best/last)")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--out", type=str, default="stage3_vis.png")
    parser.add_argument("--dataset", type=str, default="sha", choices=["sha", "shb"])
    parser.add_argument("--images", nargs="+", default=["IMG_10.jpg", "IMG_20.jpg", "IMG_100.jpg"])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # --- build stage3 joint model (stage1 + stage2) ---
    stage1 = ZIPModel(config).to(device)
    stage2 = CLIPEBCModel(config).to(device)
    steep = float(config.get("TRAIN_STAGE3", {}).get("STEEPNESS", 20.0))
    model = ZIPCLIPJointModel(stage1, stage2, steepness=steep).to(device)

    # load checkpoint (supporta dict con 'model' oppure state_dict puro)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    # --- dataset ---
    tf = build_transforms(config["DATA"], is_train=False)

    if args.dataset.lower() == "sha":
        ds = SHA(config["DATA"]["ROOT"], args.split, tf)
    else:
        # se hai SHB come classe separata, cambia import e mettila qui
        ds = SHA(config["DATA"]["ROOT"], args.split, tf)

    idx_map = find_samples_by_basenames(ds, args.images)

    # controlla che le immagini esistano nel dataset
    chosen = []
    for bn in args.images:
        idx = idx_map.get(bn, None)
        if idx is None:
            print(f"⚠️  Non trovata nel dataset split='{args.split}': {bn}")
        else:
            chosen.append((bn, idx))

    if len(chosen) == 0:
        raise RuntimeError("Nessuna delle immagini richieste è stata trovata nel dataset. Controlla nomi e split.")

    # --- figure layout: 3 rows x N cols ---
    N = len(chosen)
    fig, axes = plt.subplots(3, N, figsize=(4.5 * N, 9))
    if N == 1:
        axes = np.array(axes).reshape(3, 1)

    # --- loop images ---
    for col, (bn, idx) in enumerate(chosen):
        sample = ds[idx]
        img_t = sample["image"]          # [3,H,W] normalized
        gt_density_t = sample["density"] # [1,H,W] or [H,W]
        points = sample["points"]        # [N,2] tensor
        gt_count = int(len(points))

        img_rgb = denorm_img(img_t)

        # gt density to numpy [H,W]
        if gt_density_t.ndim == 3:
            gt_den = gt_density_t.squeeze(0).detach().cpu().numpy()
        else:
            gt_den = gt_density_t.detach().cpu().numpy()

        # pred
        pred_den, pred_count = predict_final_density(model, img_t.unsqueeze(0), device=device)

        abs_err = abs(pred_count - gt_count)
        pct_err = (abs_err / max(1, gt_count)) * 100.0

        print(f"{bn}: Pred={pred_count:.1f} | GT={gt_count} | AbsErr={abs_err:.1f} | PctErr={pct_err:.1f}%")

        # row 1: image
        ax0 = axes[0, col]
        ax0.imshow(img_rgb)
        ax0.axis("off")

        # row 2: GT overlay
        ax1 = axes[1, col]
        overlay_density(ax1, img_rgb, gt_den, cmap="jet", alpha=0.55)
        add_corner_text(ax1, f"GT: {gt_count}")

        # row 3: Pred overlay
        ax2 = axes[2, col]
        overlay_density(ax2, img_rgb, pred_den, cmap="jet", alpha=0.55)
        add_corner_text(ax2, f"Pred: {pred_count:.1f}")

    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"✅ Saved visualization to: {args.out}")


if __name__ == "__main__":
    main()
