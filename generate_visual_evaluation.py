import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
import os
from PIL import Image
from torchvision import transforms

from models import ZIPCLIPEBCModel
from datasets.utils import denormalize # Assumendo che tu l'abbia in utils

def save_visual_report(image_path, model, device, config, output_name="report.png"):
    model.eval()
    block_size = config['DATA']['ZIP_BLOCK_SIZE']
    
    # 1. Preprocessing Immagine
    original_img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((448, 448)), # Dimensione standard dai tuoi train
        transforms.ToTensor(),
        transforms.Normalize(mean=config['DATA']['NORM_MEAN'], std=config['DATA']['NORM_STD'])
    ])
    img_tensor = transform(original_img).unsqueeze(0).to(device)

    # 2. Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        pi = outputs['pi'][0, 0].cpu().numpy()            # Prob. Vuoto [H/16, W/16]
        ebc_density = outputs['ebc_density'][0, 0].cpu().numpy() # Densità EBC
    
    # 3. Preparazione Stage 1: Filtro Strutturale
    # Creiamo una maschera di overlay (Rosso per zone "vuote" secondo il modello)
    prob_occupied = 1.0 - pi
    mask_occupied = (prob_occupied > 0.5).astype(float)
    
    # Resize della maschera alla dimensione immagine per l'overlay
    mask_full = Image.fromarray(mask_occupied).resize((448, 448), resample=Image.NEAREST)
    mask_full = np.array(mask_full)

    # 4. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # Pannello 0: Immagine Originale
    axes[0].imshow(original_img.resize((448, 448)))
    axes[0].set_title("Original Image (448x448)")
    axes[0].axis('off')

    # Pannello 1: Stage 1 - Structural Gating (pi-head)
    # Mostriamo l'immagine con una griglia e blocchi colorati
    axes[1].imshow(original_img.resize((448, 448)))
    # Sovrapponiamo la maschera pi: zone trasparenti = piene, zone scure = vuote
    axes[1].imshow(1 - mask_full, cmap='Reds', alpha=0.4) 
    axes[1].set_title(f"Stage 1: Structural Filter\n(Red = Masked as Empty)")
    axes[1].axis('off')

    # Pannello 2: Stage 2 - EBC Semantic Counting
    # Mostriamo la density map generata dal VLM
    im2 = axes[2].imshow(ebc_density, cmap='jet')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title(f"Stage 2: Semantic Density\nPred Count: {ebc_density.sum():.2f}")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    print(f"✅ Visual evaluation saved to {output_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True, help='Path immagine')
    parser.add_argument('--config', type=str, default='config/config_sha.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ZIPCLIPEBCModel(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])

    save_visual_report(args.img, model, device, config)