#!/usr/bin/env python3
# stage1_heatmap.py

import argparse
import os
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import yaml
from PIL import Image
from torchvision import transforms

from models.zip_model import ZIPModel


# -----------------------------
# Helpers
# -----------------------------
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def force_backbone_in_config(config: dict, backbone: str | None):
    if backbone is None:
        backbone = (
            config.get("model")
            or config.get("backbone")
            or config.get("encoder")
            or config.get("zip_backbone")
            or (config.get("BACKBONE", {}) or {}).get("TYPE", None)
        )

    if backbone is None:
        return config

    config["model"] = backbone
    config["BACKBONE"] = config.get("BACKBONE", {}) or {}
    config["BACKBONE"]["TYPE"] = backbone
    return config


def infer_dataset_and_backbone(config: dict):
    dataset_name = config.get("dataset", "unknown")

    backbone_name = config.get("model")
    if not backbone_name:
        bb = config.get("BACKBONE", {}) or {}
        if isinstance(bb, dict):
            backbone_name = bb.get("TYPE", "unknown")
        else:
            backbone_name = "unknown"

    return dataset_name, backbone_name


def is_vit_backbone(backbone_name: str):
    # robust: accetta "vit_b_16", "clip_vit_b_16", ecc.
    return isinstance(backbone_name, str) and ("vit" in backbone_name.lower())


def load_state_dict_safe(model: torch.nn.Module, state_dict: dict):
    model_sd = model.state_dict()
    filtered = {}
    skipped = []

    for k, v in state_dict.items():
        kk = k.replace("module.", "")
        if kk in model_sd and hasattr(v, "shape") and hasattr(model_sd[kk], "shape"):
            if tuple(v.shape) == tuple(model_sd[kk].shape):
                filtered[kk] = v
            else:
                skipped.append((kk, tuple(v.shape), tuple(model_sd[kk].shape)))

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    print(f"[*] Loaded keys: {len(filtered)}/{len(state_dict)}")
    if skipped:
        print(f"[!] Skipped due to shape mismatch: {len(skipped)} (showing up to 8)")
        for s in skipped[:8]:
            print("    ", s)

    if missing:
        print(f"[i] Missing keys after safe load: {len(missing)}")
    if unexpected:
        print(f"[i] Unexpected keys after safe load: {len(unexpected)}")

    return missing, unexpected


def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tensor = tensor.clone().detach().cpu().squeeze().permute(1, 2, 0).numpy()
    img = (tensor * std) + mean
    return np.clip(img, 0, 1)


def get_transform(h, w):
    # qui lasciamo multipli di 16: va bene sia per ViT sia per ResNet (non rompe nulla)
    new_h = math.ceil(h / 16) * 16
    new_w = math.ceil(w / 16) * 16
    return transforms.Compose(
        [
            transforms.Resize((new_h, new_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def minmax_0_1(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if (x_max - x_min) < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - x_min) / (x_max - x_min)).astype(np.float32)


# -----------------------------
# Main
# -----------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Avvio Stage1 heatmap su: {args.image_path}")

    # Config
    if os.path.exists(args.config):
        print(f"📖 Leggo config: {args.config}")
        config = load_config(args.config) or {}
    else:
        print("⚠️ Config non trovato, uso config vuoto (può crashare se ZIPModel richiede campi).")
        config = {}

    # Backbone override
    config = force_backbone_in_config(config, args.backbone)
    dataset_name, backbone_name = infer_dataset_and_backbone(config)

    print(f"ℹ️  Dataset:  {dataset_name}")
    print(f"ℹ️  Backbone: {backbone_name}")

    vit_mode = is_vit_backbone(backbone_name)
    print(f"ℹ️  Heatmap scaling: {'min-max [0,1] (ViT only)' if vit_mode else 'raw probabilities [0,1]'}")

    # Model
    model = ZIPModel(config).to(device)

    # Load weights SAFE
    if not os.path.isfile(args.checkpoint):
        print("❌ Checkpoint non trovato!")
        return

    print(f"📥 Carico checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else state_dict

    load_state_dict_safe(model, state_dict)
    model.eval()

    # Image
    if not os.path.exists(args.image_path):
        print(f"❌ Immagine non trovata: {args.image_path}")
        return

    raw_img = Image.open(args.image_path).convert("RGB")
    orig_w, orig_h = raw_img.size
    transform = get_transform(orig_h, orig_w)
    img_tensor = transform(raw_img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        out = model(img_tensor)
        logits = out["pi_logits"] if isinstance(out, dict) else out
        probs = torch.sigmoid(logits)

    # Stats
    l_min, l_max, l_mean = logits.min().item(), logits.max().item(), logits.mean().item()
    p_min, p_max, p_mean = probs.min().item(), probs.max().item(), probs.mean().item()
    print("\n📊 STATISTICHE:")
    print(f"   LOGITS -> Min {l_min:+.4f} | Max {l_max:+.4f} | Mean {l_mean:+.4f}")
    print(f"   PROBS  -> Min {p_min:.4f}  | Max {p_max:.4f}  | Mean {p_mean:.4f}")
    print("-" * 60)

    # Heatmap (upsample to image size)
    img_vis = denormalize(img_tensor)
    H, W = img_vis.shape[:2]
    heatmap = (
        F.interpolate(probs, size=(H, W), mode="bilinear", align_corners=False)
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    ).astype(np.float32)

    # ✅ ViT-only: min-max normalize for contrast, but keep plot scale 0..1
    if vit_mode:
        heatmap = minmax_0_1(heatmap)

    # Plot: SOLO 2 COLONNE
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(img_vis)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    title = "Heatmap (min-max scaled to [0,1])" if vit_mode else "Heatmap (raw probability [0,1])"
    im = axes[1].imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title(title)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Save
    img_name_clean = os.path.splitext(os.path.basename(args.image_path))[0]
    output_dir = "visualize"
    os.makedirs(output_dir, exist_ok=True)

    suffix = "vitminmax" if vit_mode else "raw"
    out_file = f"{output_dir}/stage1_heatmap_{dataset_name}_{backbone_name}_{img_name_clean}_{suffix}.png"
    out_file = out_file.replace("/", "_")

    plt.tight_layout()
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close()

    print(f"\n✅ Salvato in: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_stage1.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--backbone", type=str, default=None, help="Override backbone (es: vit_b_16, resnet50)")
    args = parser.parse_args()
    main(args)
