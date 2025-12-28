#!/usr/bin/env python3
"""
ZIP-CLIP-EBC: Visualizzazione Valutazione Stage 1
Genera immagini con griglia che mostra la classificazione dei blocchi.

Legenda colori:
- 🟢 Verde: True Positive  (predetto PIENO, è PIENO) ✓
- 🔴 Rosso: False Negative (predetto VUOTO, è PIENO) ✗ GRAVE!
- 🔵 Blu:   False Positive (predetto PIENO, è VUOTO) ✗
- ⬜ Nessun colore: True Negative (predetto VUOTO, è VUOTO) ✓
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import yaml
import argparse
import os
from PIL import Image

from models.zip_clip_ebc_model import ZIPCLIPEBCModel
from datasets.sha import SHA
from datasets.transforms import build_transforms


def crowd_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'density': torch.stack([item['density'] for item in batch]),
        'points': [item['points'] for item in batch],
        'img_path': [item['img_path'] for item in batch]
    }


def denormalize_image(img_tensor, mean, std):
    """Denormalizza l'immagine per visualizzazione."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def visualize_single_image(
    img_tensor, 
    gt_density, 
    pi_prob, 
    block_size,
    threshold,
    mean, 
    std,
    img_name="image",
    save_path=None,
    show_grid_lines=True
):
    """
    Visualizza una singola immagine con griglia e classificazione blocchi.
    
    Args:
        img_tensor: Immagine normalizzata [3, H, W]
        gt_density: Density map GT [1, H, W]
        pi_prob: Probabilità di VUOTO [1, Hb, Wb]
        block_size: Dimensione del blocco (es. 16)
        threshold: Soglia per classificazione
        mean, std: Per denormalizzazione
        img_name: Nome immagine
        save_path: Dove salvare
        show_grid_lines: Se mostrare le linee della griglia
    """
    # Denormalizza immagine
    img = denormalize_image(img_tensor, mean, std)
    H, W = img.shape[:2]
    
    # Calcola GT mask (blocchi pieni)
    gt_counts = F.avg_pool2d(gt_density.unsqueeze(0), block_size) * (block_size**2)
    gt_mask_occupied = (gt_counts > 0).float().squeeze()  # [Hb, Wb]
    
    # Predizione
    prob_occupied = 1.0 - pi_prob.squeeze()  # [Hb, Wb]
    
    # Allinea dimensioni se necessario
    Hb, Wb = gt_mask_occupied.shape
    if prob_occupied.shape != gt_mask_occupied.shape:
        prob_occupied = F.interpolate(
            prob_occupied.unsqueeze(0).unsqueeze(0), 
            size=(Hb, Wb), 
            mode='bilinear', 
            align_corners=False
        ).squeeze()
    
    pred_occupied = (prob_occupied > threshold).float()
    
    # Calcola TP, FN, FP, TN per ogni blocco
    gt_np = gt_mask_occupied.cpu().numpy()
    pred_np = pred_occupied.cpu().numpy()
    prob_np = prob_occupied.cpu().numpy()
    
    # Crea figura
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # === PLOT 1: Immagine con classificazione blocchi ===
    ax1 = axes[0]
    ax1.imshow(img)
    
    # Disegna blocchi colorati
    for i in range(Hb):
        for j in range(Wb):
            x = j * block_size
            y = i * block_size
            
            gt_occ = gt_np[i, j] > 0.5
            pred_occ = pred_np[i, j] > 0.5
            
            # Determina colore
            if gt_occ and pred_occ:
                # True Positive - Verde
                color = (0, 1, 0, 0.3)
                edge_color = 'green'
            elif gt_occ and not pred_occ:
                # False Negative - Rosso (GRAVE: perso persone!)
                color = (1, 0, 0, 0.5)
                edge_color = 'red'
            elif not gt_occ and pred_occ:
                # False Positive - Blu
                color = (0, 0, 1, 0.3)
                edge_color = 'blue'
            else:
                # True Negative - Nessun colore
                color = None
                edge_color = 'gray' if show_grid_lines else None
            
            # Disegna rettangolo
            if color is not None:
                rect = patches.Rectangle(
                    (x, y), block_size, block_size,
                    linewidth=1, edgecolor=edge_color,
                    facecolor=color
                )
                ax1.add_patch(rect)
            elif show_grid_lines:
                rect = patches.Rectangle(
                    (x, y), block_size, block_size,
                    linewidth=0.5, edgecolor='gray',
                    facecolor='none', alpha=0.3
                )
                ax1.add_patch(rect)
    
    # Legenda
    legend_elements = [
        patches.Patch(facecolor=(0, 1, 0, 0.5), edgecolor='green', label='TP (Pieno ✓)'),
        patches.Patch(facecolor=(1, 0, 0, 0.5), edgecolor='red', label='FN (Perso! ✗)'),
        patches.Patch(facecolor=(0, 0, 1, 0.5), edgecolor='blue', label='FP (Falso allarme)'),
        patches.Patch(facecolor='white', edgecolor='gray', label='TN (Vuoto ✓)'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)
    ax1.set_title(f'Classificazione Blocchi\n{img_name}')
    ax1.axis('off')
    
    # === PLOT 2: Heatmap probabilità π (occupato) ===
    ax2 = axes[1]
    ax2.imshow(img)
    
    # Upscale prob_occupied per overlay
    prob_upscaled = F.interpolate(
        torch.tensor(prob_np).unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode='nearest'
    ).squeeze().numpy()
    
    im = ax2.imshow(prob_upscaled, cmap='RdYlGn', alpha=0.6, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax2, fraction=0.046, label='P(Occupato)')
    ax2.set_title('Probabilità Occupato (1-π)\nVerde=Pieno, Rosso=Vuoto')
    ax2.axis('off')
    
    # === PLOT 3: Ground Truth con punti ===
    ax3 = axes[2]
    ax3.imshow(img)
    
    # Mostra density map come overlay
    gt_den = gt_density.squeeze().cpu().numpy()
    if gt_den.max() > 0:
        ax3.imshow(gt_den, cmap='hot', alpha=0.5)
    
    # Griglia GT
    for i in range(Hb):
        for j in range(Wb):
            x = j * block_size
            y = i * block_size
            gt_occ = gt_np[i, j] > 0.5
            
            if gt_occ:
                rect = patches.Rectangle(
                    (x, y), block_size, block_size,
                    linewidth=2, edgecolor='lime',
                    facecolor='none'
                )
                ax3.add_patch(rect)
    
    gt_count = gt_den.sum()
    ax3.set_title(f'Ground Truth (Count: {gt_count:.0f})\nVerde = Blocchi con persone')
    ax3.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  💾 Salvata: {save_path}")
    else:
        plt.show()


def visualize_comparison(
    img_tensor,
    gt_density,
    pi_prob,
    block_size,
    threshold,
    mean,
    std,
    img_name="image",
    save_path=None
):
    """
    Crea un confronto 2x2 più compatto.
    """
    img = denormalize_image(img_tensor, mean, std)
    H, W = img.shape[:2]
    
    # Calcoli
    gt_counts = F.avg_pool2d(gt_density.unsqueeze(0), block_size) * (block_size**2)
    gt_mask = (gt_counts > 0).float().squeeze()
    prob_occ = 1.0 - pi_prob.squeeze()
    
    Hb, Wb = gt_mask.shape
    if prob_occ.shape != gt_mask.shape:
        prob_occ = F.interpolate(
            prob_occ.unsqueeze(0).unsqueeze(0),
            size=(Hb, Wb),
            mode='bilinear',
            align_corners=False
        ).squeeze()
    
    pred_mask = (prob_occ > threshold).float()
    
    gt_np = gt_mask.cpu().numpy()
    pred_np = pred_mask.cpu().numpy()
    
    # Statistiche
    tp = ((pred_np > 0.5) & (gt_np > 0.5)).sum()
    fn = ((pred_np <= 0.5) & (gt_np > 0.5)).sum()
    fp = ((pred_np > 0.5) & (gt_np <= 0.5)).sum()
    tn = ((pred_np <= 0.5) & (gt_np <= 0.5)).sum()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Immagine originale con GT
    ax = axes[0, 0]
    ax.imshow(img)
    gt_den = gt_density.squeeze().cpu().numpy()
    if gt_den.max() > 0:
        ax.imshow(gt_den, cmap='hot', alpha=0.4)
    ax.set_title(f'Immagine + GT Density\nCount GT: {gt_den.sum():.0f}')
    ax.axis('off')
    
    # 2. Heatmap π
    ax = axes[0, 1]
    ax.imshow(img)
    prob_up = F.interpolate(
        prob_occ.unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode='nearest'
    ).squeeze().cpu().numpy()
    im = ax.imshow(prob_up, cmap='RdYlGn', alpha=0.6, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title('P(Occupato) = 1 - π\nVerde=Pieno, Rosso=Vuoto')
    ax.axis('off')
    
    # 3. Classificazione con griglia
    ax = axes[1, 0]
    ax.imshow(img)
    
    for i in range(Hb):
        for j in range(Wb):
            x, y = j * block_size, i * block_size
            gt_occ = gt_np[i, j] > 0.5
            pred_occ = pred_np[i, j] > 0.5
            
            if gt_occ and pred_occ:
                color, ec = (0, 1, 0, 0.4), 'green'
            elif gt_occ and not pred_occ:
                color, ec = (1, 0, 0, 0.6), 'red'
            elif not gt_occ and pred_occ:
                color, ec = (0, 0, 1, 0.4), 'blue'
            else:
                continue
            
            rect = patches.Rectangle((x, y), block_size, block_size,
                                     linewidth=1, edgecolor=ec, facecolor=color)
            ax.add_patch(rect)
    
    legend_elements = [
        patches.Patch(facecolor=(0, 1, 0, 0.6), label=f'TP: {tp}'),
        patches.Patch(facecolor=(1, 0, 0, 0.6), label=f'FN: {fn} (PERSI!)'),
        patches.Patch(facecolor=(0, 0, 1, 0.6), label=f'FP: {fp}'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_title(f'Classificazione Blocchi ({block_size}x{block_size})')
    ax.axis('off')
    
    # 4. Error Map
    ax = axes[1, 1]
    error_map = np.zeros((Hb, Wb, 3))
    
    # TP = Verde, FN = Rosso, FP = Blu, TN = Grigio chiaro
    error_map[(pred_np > 0.5) & (gt_np > 0.5)] = [0, 1, 0]      # TP
    error_map[(pred_np <= 0.5) & (gt_np > 0.5)] = [1, 0, 0]     # FN
    error_map[(pred_np > 0.5) & (gt_np <= 0.5)] = [0, 0, 1]     # FP
    error_map[(pred_np <= 0.5) & (gt_np <= 0.5)] = [0.9, 0.9, 0.9]  # TN
    
    ax.imshow(error_map, interpolation='nearest')
    ax.set_title(f'Error Map (blocchi)\nTP={tp} FN={fn} FP={fp} TN={tn}')
    ax.axis('off')
    
    # Titolo generale
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0
    fig.suptitle(f'{img_name}\nAccuracy blocchi: {acc:.1%} | Threshold: {threshold}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  💾 Salvata: {save_path}")
    else:
        plt.show()


def generate_visualizations(
    model,
    dataloader,
    device,
    config,
    output_dir,
    num_samples=10,
    threshold=0.5
):
    """Genera visualizzazioni per un subset di immagini."""
    
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    block_size = config['DATA'].get('ZIP_BLOCK_SIZE', 16)
    mean = config['DATA']['NORM_MEAN']
    std = config['DATA']['NORM_STD']
    
    print(f"\n🎨 Generazione visualizzazioni...")
    print(f"   Output dir: {output_dir}")
    print(f"   Block size: {block_size}")
    print(f"   Threshold: {threshold}")
    print(f"   Num samples: {num_samples}\n")
    
    count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            if count >= num_samples:
                break
            
            imgs = batch['image'].to(device)
            gt_density = batch['density'].to(device)
            img_paths = batch['img_path']
            
            outputs = model(imgs)
            pi_prob = outputs['pi']
            
            for i in range(imgs.size(0)):
                if count >= num_samples:
                    break
                
                img_name = os.path.basename(img_paths[i]).replace('.jpg', '').replace('.png', '')
                
                # Genera visualizzazione dettagliata
                save_path = os.path.join(output_dir, f'{img_name}_analysis.png')
                
                visualize_comparison(
                    img_tensor=imgs[i].cpu(),
                    gt_density=gt_density[i].cpu(),
                    pi_prob=pi_prob[i].cpu(),
                    block_size=block_size,
                    threshold=threshold,
                    mean=mean,
                    std=std,
                    img_name=img_name,
                    save_path=save_path
                )
                
                count += 1
    
    print(f"\n✅ Generate {count} visualizzazioni in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualizza risultati Stage 1')
    parser.add_argument('--config', type=str, default='configs/config_sha.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./visualizations/stage1')
    parser.add_argument('--num_samples', type=int, default=10, 
                        help='Numero di immagini da visualizzare')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    # Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Modello
    model = ZIPCLIPEBCModel(config).to(device)
    
    # Checkpoint
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        dataset_name = config.get('DATASET', 'sha')
        ckpt_path = f"./checkpoints/{dataset_name}/stage1/best_model.pth"
    
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"✅ Checkpoint caricato: {ckpt_path}")
    else:
        print(f"⚠️  Checkpoint non trovato: {ckpt_path}")
    
    # Dataset
    data_cfg = config['DATA']
    val_dataset = SHA(
        root=data_cfg['ROOT'],
        split='val',
        transforms=build_transforms(data_cfg, is_train=False)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=crowd_collate
    )
    
    print(f"📂 Validation set: {len(val_dataset)} immagini")
    
    # Genera visualizzazioni
    generate_visualizations(
        model=model,
        dataloader=val_loader,
        device=device,
        config=config,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        threshold=args.threshold
    )


if __name__ == '__main__':
    main()
