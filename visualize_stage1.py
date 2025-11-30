import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
import random
from torchvision.transforms import functional as F

# I tuoi moduli
from models.zip_clip_ebc_model import build_model
from datasets import get_dataset
from datasets.transforms import build_transforms

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def denormalize(tensor):
    """Converte un tensore normalizzato CLIP in immagine numpy visualizzabile."""
    # Mean e Std di CLIP
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * std + mean)
    img = np.clip(img, 0, 1)
    return img

def main(config_path, checkpoint_path):
    # 1. Setup
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Modello
    print("🏗️ Costruisco il modello...")
    model = build_model(config).to(device)
    
    # 3. Carica Pesi
    if os.path.exists(checkpoint_path):
        print(f"📥 Carico checkpoint: {checkpoint_path}")
        # Fix per weights_only=False
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Gestione sia se c'è 'model' key o solo state_dict
        state_dict = ckpt.get('model', ckpt)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"⚠️ Checkpoint {checkpoint_path} non trovato! Visualizzo modello non allenato.")

    model.eval()

    # 4. Dataset (Validation)
    data_cfg = config["DATA"]
    val_tf = build_transforms(data_cfg, is_train=False)
    DatasetClass = get_dataset(config["DATASET"])
    
    val_set = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=val_tf,
    )
    
    print(f"🖼️ Estraggo immagini casuali dal validation set ({len(val_set)} immagini)...")

    # 5. Visualizzazione
    num_samples = 4
    indices = random.sample(range(len(val_set)), num_samples)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = val_set[idx]
            img_tensor = sample['image'].unsqueeze(0).to(device) # [1, 3, H, W]
            gt_density = sample['density'].unsqueeze(0).to(device) # [1, 1, H, W]
            
            # Forward solo Pi
            preds = model.forward_pi_only(img_tensor)
            pi_prob = preds["pi_prob"] # [1, 1, H_grid, W_grid]
            
            # Upsample di Pi alla dimensione immagine per visualizzazione sovrapposta
            pi_map_up = torch.nn.functional.interpolate(
                pi_prob, size=img_tensor.shape[-2:], mode='bilinear'
            ).squeeze().cpu().numpy()
            
            # Immagine originale
            img_np = denormalize(sample['image'])
            
            # Colonna 1: Immagine Originale
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title(f"Input Image (idx {idx})")
            axes[i, 0].axis('off')
            
            # Colonna 2: Ground Truth (dove sono le persone)
            axes[i, 1].imshow(gt_density.squeeze().cpu().numpy(), cmap='jet')
            axes[i, 1].set_title("Ground Truth Density")
            axes[i, 1].axis('off')
            
            # Colonna 3: La tua Pi-Head (Probabilità "Pieno")
            # Sovrapponiamo la mappa Pi (giallo=pieno, viola=vuoto) sull'immagine
            axes[i, 2].imshow(img_np)
            im = axes[i, 2].imshow(pi_map_up, cmap='inferno', alpha=0.6, vmin=0, vmax=1)
            axes[i, 2].set_title(f"Predicted $\pi$ (Prob. Crowd)")
            axes[i, 2].axis('off')
            
            if i == 0:
                fig.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

    out_path = "check_stage1.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"✅ Visualizzazione salvata in: {out_path}")
    print("Controlla l'immagine per vedere se la maschera 'accende' solo le folle!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_sha.yaml")
    # Punta al checkpoint che si sta salvando ora (last_stage1_model.pth)
    parser.add_argument("--ckpt", default="experiments/sha_zip_clip_ebc/stage1/last_stage1_model.pth")
    args = parser.parse_args()
    
    main(args.config, args.ckpt)

    # python visualize_stage1.py --config config_sha.yaml --ckpt exp/sha_zip_clip_ebc/stage1/last_stage1_model.pth