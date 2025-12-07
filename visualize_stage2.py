import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
import random
import scipy.ndimage
import torch.nn.functional as F

# --- I TUOI IMPORT ---
from models.zip_clip_ebc_model import build_model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import collate_fn

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def denormalize(tensor):
    """Denormalizza l'immagine per la visualizzazione."""
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * std + mean)
    img = np.clip(img, 0, 1)
    return img

def make_visible_gt(density_map, sigma=1.5):
    """
    GT nitida per il confronto (Stile 'Clear Points').
    """
    if isinstance(density_map, torch.Tensor):
        dmap = density_map.squeeze().cpu().detach().numpy()
    else:
        dmap = density_map
        
    # Smoothing leggero
    heatmap = scipy.ndimage.gaussian_filter(dmap, sigma=sigma)
    
    # Normalizzazione hard per rendere i punti luminosi
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap

def process_density_for_overlay(density_map, target_h, target_w):
    """
    Prepara la density map predetta per l'overlay:
    1. Upsample alla dimensione originale.
    2. Smoothing leggero per estetica.
    3. Normalizzazione per la colormap.
    """
    # Se è un tensore [1, 1, h, w], facciamo upsample
    if density_map.dim() == 4:
        density_up = F.interpolate(
            density_map, 
            size=(target_h, target_w), 
            mode='bilinear', 
            align_corners=False
        )
        dmap_np = density_up.squeeze().cpu().numpy()
    else:
        dmap_np = density_map
        
    # Smoothing per l'overlay (leggermente più morbido dei punti GT per vedere le aree)
    heatmap = scipy.ndimage.gaussian_filter(dmap_np, sigma=3.0)
    
    # Normalizzazione 0-1 per colormap corretta
    eps = 1e-6
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + eps)
    
    return heatmap_norm

def safe_load_weights(model, checkpoint_path, device):
    print(f"📥 Caricamento pesi da: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model_state = model.state_dict()
    new_state = {}
    ignored = []
    
    for k, v in state_dict.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                new_state[k] = v
            else:
                ignored.append(k)
    
    if ignored:
        print(f"⚠️  Ignorati {len(ignored)} layer per mismatch dimensioni: {ignored}")
        
    model.load_state_dict(new_state, strict=False)
    print("✅ Pesi caricati.")

def main(config_path, checkpoint_path):
    config = load_config(config_path)
    device = torch.device("cpu") # CPU per visualizzazione
    
    print(f"="*60)
    print(f"🚀 VISUALIZE STAGE 2 (Stile Overlay Stage 1)")
    print(f"   Config: {config_path}")
    print(f"="*60)

    model = build_model(config).to(device)
    
    if os.path.isfile(checkpoint_path):
        safe_load_weights(model, checkpoint_path, device)
    else:
        print(f"❌ Errore: Checkpoint non trovato: {checkpoint_path}")
        return

    model.eval()

    DatasetClass = get_dataset(config['DATASET'])
    val_tf = build_transforms(config['DATA'], is_train=False)
    
    val_dataset = DatasetClass(
        root=config['DATA']['ROOT'],
        split=config['DATA']['VAL_SPLIT'],
        block_size=config['DATA']['ZIP_BLOCK_SIZE'],
        transforms=val_tf,
    )
    
    indices = random.sample(range(len(val_dataset)), 3)
    
    rows = len(indices)
    fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows))
    
    cols = ["Input Image", "Ground Truth (Sharp Density)", "Prediction Overlay"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=16, fontweight='bold', pad=20)

    print("🖼️  Generazione visualizzazioni...")

    for i, idx in enumerate(indices):
        sample = val_dataset[idx]
        image = sample['image'].unsqueeze(0).to(device)
        gt_density = sample['density'].unsqueeze(0).to(device)

        # Forward Pass
        with torch.no_grad():
            outputs = model(image)
            pred_density = outputs['density_map']
            pred_count = outputs['pred_count'].item()
            gt_count = gt_density.sum().item()

        img_np = denormalize(sample['image'])
        H, W = img_np.shape[:2]
        
        # 1. GT Density (Nitida)
        gt_heat = make_visible_gt(gt_density, sigma=1.5)
        
        # 2. Prediction Overlay (Heatmap smooth sovrapposta)
        pred_overlay = process_density_for_overlay(pred_density, H, W)

        # Colore errore
        err = abs(pred_count - gt_count)
        err_pct = err / max(gt_count, 1)
        err_color = "lime" if err_pct < 0.15 else "red"

        # --- Colonna 1: Immagine Originale ---
        axes[i, 0].imshow(img_np)
        axes[i, 0].axis('off')
        axes[i, 0].text(10, 30, f"ID: {idx}", color="white", fontsize=12, bbox=dict(facecolor='black', alpha=0.7))

        # --- Colonna 2: GT Heatmap (Punti Chiari) ---
        axes[i, 1].imshow(gt_heat, cmap='jet', vmin=0, vmax=1)
        axes[i, 1].axis('off')
        axes[i, 1].text(10, 30, f"GT: {gt_count:.1f}", color="white", fontsize=12, bbox=dict(facecolor='black', alpha=0.7))

        # --- Colonna 3: Overlay (Immagine + Heatmap Predizione) ---
        # Sfondo: Immagine originale
        axes[i, 2].imshow(img_np)
        
        # Overlay: Heatmap predizione semitrasparente
        # vmin=0, vmax=1 fissa la scala colori (Blu -> Rosso)
        im = axes[i, 2].imshow(pred_overlay, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[i, 2].axis('off')
        
        axes[i, 2].text(10, 30, f"Pred: {pred_count:.1f}", color=err_color, fontsize=12, fontweight='bold', bbox=dict(facecolor='black', alpha=0.7))
        
        # Aggiungi colorbar piccola solo alla prima riga
        if i == 0:
            cbar = plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
            cbar.set_label('Density Intensity', rotation=270, labelpad=15)

    plt.tight_layout()
    out_file = "check_stage2.png"
    plt.savefig(out_file, bbox_inches='tight', dpi=150)
    print(f"✅ Immagine salvata: {out_file}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_sha.yaml")
    parser.add_argument("--checkpoint", default="experiments/sha_zip_clip_ebc/stage2/best_stage2_model.pth")
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    main(args.config, args.checkpoint)