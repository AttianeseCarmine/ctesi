
import argparse
import os
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import yaml
from PIL import Image, ImageDraw
from torchvision import transforms

from models.zip_model import ZIPModel


# ==============================================================================
# Config helpers
# ==============================================================================
def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def force_backbone_in_config(config: dict, backbone: str | None):
    """Forza backbone sia in config flat che nested."""
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
        backbone_name = bb.get("TYPE", "unknown") if isinstance(bb, dict) else "unknown"

    return dataset_name, backbone_name


def is_vit_backbone(backbone_name: str) -> bool:
    """Heuristica robusta: se nel nome c'è vit/visiontransformer/clip_vit -> ViT."""
    b = (backbone_name or "").lower()
    return ("vit" in b) or ("visiontransformer" in b) or ("clip_vit" in b)


# ==============================================================================
# Safe load weights
# ==============================================================================
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


# ==============================================================================
# Image helpers
# ==============================================================================
def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    tensor = tensor.clone().detach().cpu()
    img = tensor.permute(1, 2, 0).numpy().astype(np.float32)
    img = (img * std) + mean
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def get_smart_transform(h: int, w: int, multiple: int = 16):
    """
    Ridimensiona l'immagine a multipli di 16:
    - perfetto per ViT (token 16x16)
    - non "rompe" ResNet (solo resize coerente per visual)
    """
    new_h = math.ceil(h / multiple) * multiple
    new_w = math.ceil(w / multiple) * multiple

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose(
        [
            transforms.Resize((new_h, new_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def load_gt_points(image_path: str):
    img_dir = os.path.dirname(image_path)
    base_dir = os.path.dirname(img_dir)
    fname = os.path.basename(image_path)
    name_no_ext = os.path.splitext(fname)[0]
    label_path = os.path.join(base_dir, "labels", f"{name_no_ext}.npy")

    print(f"🔍 [Data] Cerco labels in: {label_path}")
    if not os.path.exists(label_path):
        print("❌ File labels non trovato.")
        return None

    try:
        points = np.load(label_path)
        print(f"✅ [Data] Trovati {len(points)} punti.")
        return points
    except Exception as e:
        print(f"⚠️ Errore lettura .npy: {e}")
        return None


# ==============================================================================
# Gating grids
# ==============================================================================
def _ensure_prob_4d(p: torch.Tensor) -> torch.Tensor:
    if p.dim() == 2:
        p = p.unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
    elif p.dim() == 3:
        p = p.unsqueeze(1)  # [B,1,h,w]
    elif p.dim() == 4 and p.size(1) != 1:
        p = p[:, :1]
    return p


def vit_token_grid_from_prob(pi_prob: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    ViT token grid reale: 16x16 px -> (H//16, W//16)
    Proiettiamo pi_prob con Adaptive MAX pooling (coerente con gating).
    """
    p = _ensure_prob_4d(pi_prob)
    gh = max(1, H // 16)
    gw = max(1, W // 16)
    return F.adaptive_max_pool2d(p, output_size=(gh, gw))  # [B,1,gh,gw]


def coarsen_vit_grid_for_visual(token_grid: torch.Tensor) -> torch.Tensor:
    """
    SOLO VISUAL: raggruppa 2x2 token -> blocchi visivi ~32x32.
    (La griglia reale ViT resta 16x16.)
    """
    g = _ensure_prob_4d(token_grid)
    s = 2  # fisso, niente parametri CLI

    _, _, gh, gw = g.shape
    pad_h = (s - (gh % s)) % s
    pad_w = (s - (gw % s)) % s
    if pad_h or pad_w:
        g = F.pad(g, (0, pad_w, 0, pad_h), mode="replicate")

    return F.max_pool2d(g, kernel_size=s, stride=s)


def draw_grid_gating(image_rgb: np.ndarray, prob_grid: torch.Tensor, threshold: float):
    H, W, _ = image_rgb.shape
    g = _ensure_prob_4d(prob_grid)
    gh, gw = g.shape[-2], g.shape[-1]

    prob_up = F.interpolate(g, size=(H, W), mode="nearest").squeeze().detach().cpu().numpy()
    is_background = prob_up < threshold

    vis_img = image_rgb.copy()
    vis_img[is_background] = (vis_img[is_background] * 0.3).astype(vis_img.dtype)

    pil_img = Image.fromarray(vis_img.astype(np.uint8))
    draw = ImageDraw.Draw(pil_img, "RGBA")

    cell_h = H / gh
    cell_w = W / gw

    prob_small = g.squeeze().detach().cpu().numpy()
    empty_blocks = int((prob_small < threshold).sum())
    total_blocks = int(gh * gw)

    for r in range(gh):
        for c in range(gw):
            if prob_small[r, c] >= threshold:
                x1 = int(c * cell_w)
                y1 = int(r * cell_h)
                x2 = int((c + 1) * cell_w)
                y2 = int((r + 1) * cell_h)
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 180), width=1)

    return np.array(pil_img), empty_blocks, total_blocks, (gh, gw)


# ==============================================================================
# MAIN
# ==============================================================================
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"📂 Caricamento config da: {args.config}")
    if not os.path.exists(args.config):
        print("❌ Config file non trovato!")
        return

    config = load_config(args.config)
    config = force_backbone_in_config(config, args.backbone)
    dataset_name, backbone_name = infer_dataset_and_backbone(config)
    vit_mode = is_vit_backbone(backbone_name)

    print(f"\n🚀 AVVIO VISUALIZZAZIONE STAGE 1")
    print(f"   Dataset : {dataset_name}")
    print(f"   Backbone: {backbone_name}")
    print(f"   Soglia  : {args.threshold}")
    if vit_mode:
        print("   Mode    : ViT real token grid (16x16) + VIS coarsen 2x2 -> blocchi ~32x32")
    else:
        print("   Mode    : Native pi_prob grid (CNN/ResNet)")

    # Model
    model = ZIPModel(config).to(device)

    # Load checkpoint (SAFE)
    if not os.path.isfile(args.checkpoint):
        print("❌ Checkpoint non trovato!")
        return

    print(f"📥 Caricamento pesi da: {args.checkpoint}")
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
    orig_W, orig_H = raw_img.size

    trans = get_smart_transform(orig_H, orig_W, multiple=16)
    img_tensor = trans(raw_img).unsqueeze(0).to(device)
    target_H, target_W = img_tensor.shape[2], img_tensor.shape[3]

    # GT points (optional)
    gt_points = load_gt_points(args.image_path)
    gt_points_scaled = np.empty((0, 2))
    if gt_points is not None and len(gt_points) > 0:
        scale_x = target_W / orig_W
        scale_y = target_H / orig_H
        gt_points_scaled = np.stack([gt_points[:, 0] * scale_x, gt_points[:, 1] * scale_y], axis=1)

    # Inference
    with torch.no_grad():
        out = model(img_tensor)
        pi_logits = out["pi_logits"] if isinstance(out, dict) else out
        pi_prob = torch.sigmoid(pi_logits)

    img_vis = denormalize(img_tensor.squeeze())

    # --- Grid selection ---
    if vit_mode:
        token_grid = vit_token_grid_from_prob(pi_prob, H=target_H, W=target_W)  # REAL 16x16 token grid
        prob_grid = coarsen_vit_grid_for_visual(token_grid)  # VIS 2x2 -> 32x32 blocks
        grid_name = ""
    else:
        prob_grid = _ensure_prob_4d(pi_prob)
        grid_name = "Native pi_prob grid"

    mask_vis, empty_blocks, total_blocks, (gh, gw) = draw_grid_gating(img_vis, prob_grid, threshold=args.threshold)
    empty_pct = 100.0 * empty_blocks / max(1, total_blocks)

    print(f"[*] pi_prob shape : {tuple(_ensure_prob_4d(pi_prob).shape)}")
    print(f"[*] grid used    : {grid_name} -> {gh}x{gw}")

    # Plot
    os.makedirs("visualize", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    axes[0].imshow(img_vis)
    axes[0].set_title(f"Input ({orig_W}x{orig_H})", fontsize=16, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(img_vis)
    blue_overlay = np.zeros_like(img_vis)
    blue_overlay[:, :, 2] = 255
    axes[1].imshow(blue_overlay, alpha=0.2)

    if gt_points_scaled.shape[0] > 0:
        axes[1].scatter(
            gt_points_scaled[:, 0],
            gt_points_scaled[:, 1],
            c="white",
            s=20,
            marker=".",
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
        )
        axes[1].set_title(f"GT: {len(gt_points)} Persone", fontsize=16, fontweight="bold", color="darkblue")
    else:
        axes[1].set_title("GT: N/A", fontsize=16, fontweight="bold", color="gray")
    axes[1].axis("off")

    axes[2].imshow(mask_vis)
    title_str = f"Stage 1  ({backbone_name})\nFiltered: {empty_blocks}/{total_blocks} cells ({empty_pct:.1f}%)"
    axes[2].set_title(title_str, fontsize=16, fontweight="bold", color="darkred")
    axes[2].axis("off")

    img_name_clean = os.path.splitext(os.path.basename(args.image_path))[0]
    out_file = f"visualize/stage1_{dataset_name}_{backbone_name}_{img_name_clean}_thr{args.threshold}.png"
    out_file = out_file.replace("/", "_")

    plt.tight_layout()
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Risultato salvato in: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizza output Stage 1 (auto: ViT=16x16, visual blocks bigger; ResNet=native grid)")
    parser.add_argument("--config", type=str, default="config_stage1.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backbone", type=str, default=None, help="Override backbone (es: vit_b_16, resnet50)")
    args = parser.parse_args()
    main(args)
