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
    """GT nitida (Punti Rossi)."""
    if isinstance(density_map, torch.Tensor):
        dmap = density_map.squeeze().cpu().detach().numpy()
    else:
        dmap = density_map
    heatmap = scipy.ndimage.gaussian_filter(dmap, sigma=sigma)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap

def draw_grid_mask(ax, pi_prob, H_img, W_img, patch_size=16, thresh=0.25):
    """
    Disegna la griglia rossa e oscura i blocchi vuoti.
    """
    # 1. Upsample della mappa Pi (Nearest Neighbor per mantenere i blocchi quadrati)
    pi_map_up = F.interpolate(
        pi_prob, 
        size=(H_img, W_img), 
        mode='nearest' # Importante: Nearest per vedere i blocchi netti
    )
    pi_np = pi_map_up.squeeze().cpu().numpy() # [H, W] con valori 0-1
    
    # 2. Crea l'overlay scuro (Mask)
    # Creiamo un'immagine nera RGBA
    overlay = np.zeros((H_img, W_img, 4))
    overlay[:, :, 0:3] = 0 # Canali RGB neri
    
    # Canale Alpha: 
    # Se Prob < Thresh (Vuoto) -> Alpha 0.7 (Scuro)
    # Se Prob > Thresh (Folla) -> Alpha 0.0 (Trasparente/Acceso)
    is_empty = pi_np < thresh
    overlay[..., 3] = is_empty * 0.75 
    
    # Disegna l'overlay scuro
    ax.imshow(overlay)
    
    # 3. Disegna la Griglia Rossa
    # Linee Verticali
    for x in range(0, W_img, patch_size):
        ax.axvline(x, color='red', linewidth=0.3, alpha=0.3)
    # Linee Orizzontali
    for y in range(0, H_img, patch_size):
        ax.axhline(y, color='red', linewidth=0.3, alpha=0.3)

    return pi_np.mean()

def safe_load_weights(model, checkpoint_path, device):
    print(f"📥 Caricamento pesi da: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    print("✅ Pesi caricati.")

def main(config_path, checkpoint_path):
    config = load_config(config_path)
    device = torch.device("cpu") 
    
    print(f"="*60)
    print(f"🚀 VISUALIZE STAGE 1 (Grid & Mask)")
    print(f"   Config: {config_path}")
    print(f"="*60)

    model = build_model(config).to(device)
    
    if os.path.isfile(checkpoint_path):
        safe_load_weights(model, checkpoint_path, device)
    else:
        print(f"❌ Errore: Checkpoint non trovato: {checkpoint_path}")
        return

    model.eval()
    
    # Recupera la soglia dal config o usa default
    pi_thresh = config['MODEL'].get('PI_THRESH', 0.25)
    print(f"ℹ️  Visualizzazione con soglia attivazione: {pi_thresh}")

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
    fig, axes = plt.subplots(rows, 3, figsize=(18, 6 * rows))
    
    col_titles = ["Input Image", "Ground Truth (Red Points)", "Zip Mask (Dark=Empty, Grid=16px)"]
    for ax, col in zip(axes[0], col_titles):
        ax.set_title(col, fontsize=16, fontweight='bold', pad=20)

    print("🖼️  Generazione visualizzazioni...")

    for i, idx in enumerate(indices):
        sample = val_dataset[idx]
        image = sample['image'].unsqueeze(0).to(device)
        gt_density = sample['density'].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model.forward_pi_only(image)
            pi_prob = outputs['pi_prob']

        img_np = denormalize(sample['image'])
        H, W = img_np.shape[:2]
        
        gt_heat = make_visible_gt(gt_density, sigma=1.5)
        gt_count = gt_density.sum().item()

        # Col 1: Originale
        axes[i, 0].imshow(img_np)
        axes[i, 0].axis('off')
        axes[i, 0].text(10, 30, f"ID: {idx}", color="white", fontsize=12, bbox=dict(facecolor='black', alpha=0.7))

        # Col 2: GT
        axes[i, 1].imshow(gt_heat, cmap='jet', vmin=0, vmax=1)
        axes[i, 1].axis('off')
        axes[i, 1].text(10, 30, f"GT Count: {gt_count:.1f}", color="white", fontsize=12, bbox=dict(facecolor='black', alpha=0.7))

        # Col 3: Grid Mask Overlay
        # Prima l'immagine base
        axes[i, 2].imshow(img_np) 
        # Poi la funzione magica che disegna griglia e maschera
        mean_pi = draw_grid_mask(axes[i, 2], pi_prob, H, W, patch_size=16, thresh=pi_thresh)
        
        axes[i, 2].axis('off')
        axes[i, 2].text(10, 30, f"Avg Prob: {mean_pi:.3f}", color="white", fontsize=12, bbox=dict(facecolor='black', alpha=0.7))

    plt.tight_layout()
    out_file = "check_stage1.png"
    plt.savefig(out_file, bbox_inches='tight', dpi=150)
    print(f"✅ Immagine salvata: {out_file}")
    print("   (Le zone SCURE sono quelle che il modello IGNORA)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_sha.yaml")
    # Usa il best stage 1 di default
    parser.add_argument("--checkpoint", default="experiments/sha_zip_clip_ebc/stage1/best_stage1_model.pth")
    args = parser.parse_args()
    
    # Forza CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    main(args.config, args.checkpoint)
    # python visualize_stage1.py --config config_sha.yaml --ckpt exp/sha_zip_clip_ebc/stage1/last_stage1_model.pth