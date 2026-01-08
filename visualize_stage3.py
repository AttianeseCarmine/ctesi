import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import yaml
import argparse
from torchvision import transforms

# Imports
from models.zip_model import ZIPModel

def visualize_prediction(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Config & Caricamento solo della parte ZIP (il filtro)
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    
    zip_model = ZIPModel(config).to(device)
    
    # Caricamento Checkpoint Stage 3 (usando la mappatura corretta)
    print(f"Loading weights from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    
    zip_dict = {}
    for key, value in state_dict.items():
        if key.startswith('stage1.'): # Chiave usata in train_stage3.py
            new_key = key.replace('stage1.', '')
            zip_dict[new_key] = value
            
    zip_model.load_state_dict(zip_dict, strict=False)
    zip_model.eval()

    # 2. Preparazione Immagine
    original_img = Image.open(args.image).convert('RGB')
    W, H = original_img.size
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(original_img).unsqueeze(0).to(device)

    # 3. Logica a Griglia (Simulazione Hard Gating)
    tile_size = 224
    threshold = 0.5 # Stessa soglia dell'eval
    
    # Padding
    pad_h = (tile_size - H % tile_size) % tile_size
    pad_w = (tile_size - W % tile_size) % tile_size
    x_padded = F.pad(img_tensor, (0, pad_w, 0, pad_h))
    
    # Unfold per ottenere le patch
    patches = x_padded.unfold(2, tile_size, tile_size).unfold(3, tile_size, tile_size)
    # (B, C, Rows, Cols, Tile, Tile)
    _, _, rows, cols, _, _ = patches.shape
    
    # Flatten per il modello
    patches_flat = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, 3, tile_size, tile_size)
    
    # Predizione ZIP
    with torch.no_grad():
        zip_out = zip_model(patches_flat)
        # Score per blocco (max probability)
        scores = F.adaptive_max_pool2d(zip_out, (1, 1)).view(rows, cols).cpu().numpy()

    # 4. Disegno
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(original_img)
    
    print(f"🧩 Grid Analysis: {rows}x{cols} blocks")
    
    for r in range(rows):
        for c in range(cols):
            score = scores[r, c]
            x_pos = c * tile_size
            y_pos = r * tile_size
            
            # Se siamo fuori dall'immagine originale (zona di padding), non disegniamo
            if x_pos >= W or y_pos >= H:
                continue

            # Logica colori
            if score > threshold:
                # ACCEPTED (Verde - CLIP verrà eseguito qui)
                color = 'lime'
                alpha = 0.2
                style = 'solid'
                txt = f"ON\n{score:.2f}"
            else:
                # REJECTED (Rosso - CLIP saltato)
                color = 'red'
                alpha = 0.35
                style = 'dotted'
                txt = f"OFF\n{score:.2f}"
            
            # Rettangolo
            rect = patches.Rectangle((x_pos, y_pos), tile_size, tile_size, 
                                     linewidth=2, edgecolor=color, facecolor=color, alpha=alpha, linestyle=style)
            ax.add_patch(rect)
            
            # Testo Score
            ax.text(x_pos + 10, y_pos + 30, txt, color='white', fontsize=9, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.6, pad=2))

    plt.title(f"Stage 3 Hard Gating Logic (Threshold > {threshold})\nGreen = CLIP Counts | Red = Background Filtered")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to stage3 checkpoint')
    args = parser.parse_args()
    
    visualize_prediction(args)