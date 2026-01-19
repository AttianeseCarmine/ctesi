import argparse
import os
import json
import yaml
import re
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from torchvision import transforms
from scipy.ndimage import gaussian_filter

# Import moduli del progetto
from models import get_model
from datasets import standardize_dataset_name
from datasets.utils import generate_density_map

# ==============================================================================
# 1. CONFIGURAZIONE & PARAMETRI
# ==============================================================================

def load_config_and_parameters(args):
    """
    Carica config e parametri. 
    Cerca SOLO 'model'/'MODEL' per definire l'architettura Stage 2.
    """
    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config non trovato: {config_path}")
        
    with open(config_path, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)
        except AttributeError:
            config = yaml.load(f, Loader=yaml.Loader)
    
    # --- 1. Determinazione Nome Modello ---
    model_name = None
    
    # A. Override da riga di comando (Priorità Massima)
    if args.model_name is not None:
        model_name = args.model_name
    
    # B. Dal Config (Cerca solo MODEL o model, NON backbone)
    if model_name is None:
        keys_to_check = ['model', 'MODEL']
        for k in keys_to_check:
            if k in config and config[k]:
                val = config[k]
                if isinstance(val, dict):
                    model_name = val.get('TYPE') or val.get('type')
                else:
                    model_name = val
                break
    
    # C. Fallback Euristico dal Path del Checkpoint
    if model_name is None:
        ckpt_path_str = args.checkpoint.lower()
        if "resnet50" in ckpt_path_str:
            model_name = "clip_resnet50"
            print("⚠️ 'model' non trovato nel config. Dedotto dal path: clip_resnet50")
        elif "vit_b_16" in ckpt_path_str:
            model_name = "clip_vit_b_16"
            print("⚠️ 'model' non trovato nel config. Dedotto dal path: clip_vit_b_16")
        else:
            print("⚠️ Impossibile determinare il modello. Uso default: clip_vit_b_16")
            model_name = "clip_vit_b_16"

    args.model = model_name
    
    # --- 2. Altri Parametri ---
    args.dataset = config.get('dataset') or config.get('DATASET', 'sha')
    args.dataset = standardize_dataset_name(args.dataset)
    
    args.input_size = config.get('input_size') or config.get('INPUT_SIZE', 224)
    args.reduction = config.get('reduction') or config.get('REDUCTION', 8)
    args.truncation = config.get('truncation') or config.get('TRUNCATION', 4)
    args.granularity = config.get('granularity') or config.get('GRANULARITY', 'fine')
    args.anchor_points_type = config.get('anchor_points') or 'average'
    args.prompt_type = config.get('prompt_type') or config.get('PROMPT_TYPE', 'word')
    
    args.num_vpt = config.get('num_vpt', 32)
    args.vpt_drop = config.get('vpt_drop', 0.0)
    args.deep_vpt = not config.get('shallow_vpt', False)

    # --- 3. Caricamento Bins & Anchors ---
    if 'bins' in config and 'anchor_points' in config:
        args.bins = config['bins']
        args.anchor_points = config['anchor_points']
    else:
        possible_paths = [
            os.path.join("configs", f"reduction_{args.reduction}.json"),
            os.path.join(os.path.dirname(config_path), f"reduction_{args.reduction}.json"),
            f"reduction_{args.reduction}.json"
        ]
        
        json_path = None
        for p in possible_paths:
            if os.path.exists(p):
                json_path = p
                break
        
        if json_path:
            with open(json_path, "r") as f:
                reduction_cfg = json.load(f)[str(args.truncation)][args.dataset]
            
            raw_bins = reduction_cfg["bins"][args.granularity]
            if args.anchor_points_type == "average":
                raw_anchors = reduction_cfg["anchor_points"][args.granularity]["average"]
            else:
                raw_anchors = reduction_cfg["anchor_points"][args.granularity]["middle"]
                
            args.bins = [(float(b[0]), float(b[1])) for b in raw_bins]
            args.anchor_points = [float(p) for p in raw_anchors]
        else:
            print(f"❌ ERRORE: File reduction_{args.reduction}.json non trovato!")
            args.bins = None
            args.anchor_points = None
    
    return args

# ==============================================================================
# 2. UTILS VISUALIZZAZIONE (CORRETTA)
# ==============================================================================

def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tensor = tensor.clone().detach().cpu()
    if tensor.dim() == 4: tensor = tensor.squeeze(0)
    img = tensor.permute(1, 2, 0).numpy()
    img = (img * std) + mean
    return np.clip(img, 0, 1)

def create_density_overlay(image, density_map, alpha=0.6, sigma=2):
    """
    Crea l'overlay della mappa di densità gestendo correttamente le dimensioni.
    """
    H, W = image.shape[:2]
    
    # --- FIX DIMENSIONI ---
    # Trasforma numpy in tensor
    d_tensor = torch.tensor(density_map).float()
    
    # Normalizza le dimensioni a (1, 1, H_curr, W_curr) per l'interpolazione
    if d_tensor.dim() == 2:   # (H, W)
        d_tensor = d_tensor.unsqueeze(0).unsqueeze(0)
    elif d_tensor.dim() == 3: # (1, H, W) o (C, H, W)
        d_tensor = d_tensor.unsqueeze(0)
        
    # Resize density map se necessario
    if d_tensor.shape[-2:] != (H, W):
        d_tensor = F.interpolate(
            d_tensor,
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        )
    
    # Ritorna a numpy (H, W)
    density_map_resized = d_tensor.squeeze().numpy()
    
    # Smoothing
    if sigma > 0: 
        density_smooth = gaussian_filter(density_map_resized, sigma=sigma)
    else: 
        density_smooth = density_map_resized
    
    # Normalize [0,1] per colormap
    max_val = density_smooth.max()
    density_norm = density_smooth / max_val if max_val > 1e-6 else density_smooth
    
    # Colormap: Blu -> Ciano -> Verde -> Giallo -> Rosso
    colors = [
        (0.0, 0.0, 0.3), (0.0, 0.0, 0.8), (0.0, 0.8, 1.0), 
        (0.0, 1.0, 0.5), (0.5, 1.0, 0.0), (1.0, 1.0, 0.0), 
        (1.0, 0.5, 0.0), (1.0, 0.0, 0.0)
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list('density', colors, N=256)
    density_colored = cmap(density_norm)[:, :, :3]
    
    # Maschera background (trasparenza)
    mask = (density_norm > 0.05)[..., None]
    
    overlay = np.where(mask, (1 - alpha) * image + alpha * density_colored, image)
    return np.clip(overlay, 0, 1)

def get_transform(h, w, model_name):
    # ResNet usa stride 32, ViT usa 16
    stride = 32 if "resnet" in str(model_name).lower() else 16
    new_h = math.ceil(h / stride) * stride
    new_w = math.ceil(w / stride) * stride
    return transforms.Compose([
        transforms.Resize((new_h, new_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_gt_points(image_path):
    try:
        path_no_ext = os.path.splitext(image_path)[0]
        fname = os.path.basename(path_no_ext)
        base_dir = os.path.dirname(os.path.dirname(image_path))
        
        candidates = [
            os.path.join(base_dir, 'labels', fname + '.npy'), # data/sha/val/labels/IMG.npy
            image_path.replace('images', 'labels').replace('.jpg', '.npy') # Stessa cartella ma labels
        ]
        for p in candidates:
            if os.path.exists(p): return np.load(p)
    except:
        pass
    return None

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--sigma', type=float, default=4.0, help="Sigma GT Heatmap")
    
    # Opzione per forzare il modello
    parser.add_argument('--model_name', type=str, default=None)
    
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 1. Caricamento Configurazione
    try:
        cfg = load_config_and_parameters(args)
        print(f"✅ Config Caricato.")
        print(f"   Modello: {cfg.model}")
        print(f"   Dataset: {cfg.dataset}")
    except Exception as e:
        print(f"❌ Errore Config: {e}")
        return

    # 2. Costruzione Modello
    print("🏗️  Costruzione Modello...")
    model = get_model(
        backbone=cfg.model,
        input_size=cfg.input_size,
        reduction=cfg.reduction,
        bins=cfg.bins,
        anchor_points=cfg.anchor_points,
        prompt_type=cfg.prompt_type,
        num_vpt=cfg.num_vpt,
        vpt_drop=cfg.vpt_drop,
        deep_vpt=cfg.deep_vpt
    ).to(device)

    # 3. Caricamento Pesi
    if os.path.exists(args.checkpoint):
        print(f"📥 Loading Weights: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        try:
            model.load_state_dict(new_state_dict, strict=False)
        except RuntimeError as e:
            print(f"❌ ERRORE CARICAMENTO PESI: Mismatch architettura/pesi.\n{e}")
            return
        model.eval()
    else:
        print("❌ Checkpoint non trovato.")
        return

    # 4. Immagine
    if not os.path.exists(args.image_path):
        print("❌ Immagine non trovata.")
        return
    
    raw_img = Image.open(args.image_path).convert('RGB')
    W_orig, H_orig = raw_img.size
    
    # Transform corretto
    trans = get_transform(H_orig, W_orig, cfg.model)
    img_tensor = trans(raw_img).unsqueeze(0).to(device)
    
    # 5. Inferenza
    with torch.no_grad():
        pred_density = model(img_tensor)
        if isinstance(pred_density, (tuple, list)): 
            pred_density = pred_density[-1]

    pred_count = pred_density.sum().item()
    
    # Squeeze robusto: assicura che sia numpy array senza dimensioni batch inutili
    pred_den_np = pred_density.squeeze().cpu().numpy()

    # 6. GT & Plotting
    gt_points = load_gt_points(args.image_path)
    gt_den_np = None
    gt_count = 0
    
    if gt_points is not None:
        gt_count = len(gt_points)
        H_net, W_net = pred_den_np.shape # Usa la dimensione di output della rete
        pts_tensor = torch.from_numpy(gt_points).float()
        pts_tensor[:, 0] *= (W_net / W_orig)
        pts_tensor[:, 1] *= (H_net / H_orig)
        
        # Genera e fai subito squeeze per avere (H, W)
        gt_den_tensor = generate_density_map(pts_tensor, H_net, W_net, sigma=args.sigma)
        gt_den_np = gt_den_tensor.squeeze().numpy()

    # Visualizzazione
    img_vis = denormalize(img_tensor)
    overlay_pred = create_density_overlay(img_vis, pred_den_np, alpha=0.6, sigma=2)
    
    if gt_den_np is not None:
        overlay_gt = create_density_overlay(img_vis, gt_den_np, alpha=0.6, sigma=args.sigma)
    else:
        overlay_gt = np.zeros_like(img_vis)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # P1: Originale
    axes[0].imshow(img_vis)
    axes[0].set_title(f"Input ({W_orig}x{H_orig})", fontsize=16)
    axes[0].axis('off')
    
    # P2: GT
    axes[1].imshow(overlay_gt)
    axes[1].set_title(f"Ground Truth\nCount: {gt_count}", fontsize=16, color='green', fontweight='bold')
    axes[1].axis('off')
    if gt_den_np is None: axes[1].text(0.5,0.5, "GT Missing", ha='center', color='white')
    
    # P3: Prediction
    axes[2].imshow(overlay_pred)
    err = abs(gt_count - pred_count)
    col = 'blue' if err < max(1, gt_count*0.1) else 'red'
    axes[2].set_title(f"Pred ({cfg.model})\nCount: {pred_count:.2f} | Err: {err:.1f}", 
                      fontsize=16, color=col, fontweight='bold')
    axes[2].axis('off')

    # Salvataggio
    base_name = os.path.basename(args.image_path)
    name_no_ext = os.path.splitext(base_name)[0]
    match = re.search(r'\d+', name_no_ext)
    img_num = match.group(0) if match else name_no_ext
    backbone_name = str(cfg.model).replace("clip_", "")
    
    os.makedirs("visualize", exist_ok=True)
    out_path = f"visualize/stage2_{img_num}_{backbone_name}.png"
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Salvato: {out_path}")
    print(f"   GT: {gt_count} | Pred: {pred_count:.2f}")

if __name__ == "__main__":
    main()