#!/usr/bin/env python3
"""
ZIP-CLIP-EBC: Visualizzazione Stage 2 (CLIP-EBC Head)
Genera visualizzazioni stile paper CLIP-EBC con heatmap density.

Output per ogni immagine:
- Riga 1: Immagine originale
- Riga 2: GT Density Map (heatmap) + GT Count
- Riga 3: Predicted Density Map (heatmap) + Pred Count
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import yaml
import argparse
import os
from scipy.ndimage import gaussian_filter

from models import ZIPCLIPEBCModel
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
    # Clip extra per evitare warning matplotlib (errori precisione float32)
    img = np.clip(img, 0, 1)
    return img


def create_density_overlay(image, density_map, alpha=0.6, sigma=2):
    """
    Crea overlay della density map sull'immagine.
    Stile paper CLIP-EBC: blu scuro → ciano → verde → giallo → rosso
    
    Args:
        image: Immagine RGB [H, W, 3]
        density_map: Density map [H, W]
        alpha: Trasparenza dell'overlay
        sigma: Sigma per smoothing gaussiano
        
    Returns:
        Immagine con overlay
    """
    H, W = image.shape[:2]
    
    # Resize density map se necessario
    if density_map.shape != (H, W):
        density_map = F.interpolate(
            torch.tensor(density_map).unsqueeze(0).unsqueeze(0).float(),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
    
    # Smooth per visualizzazione migliore
    if sigma > 0:
        density_smooth = gaussian_filter(density_map, sigma=sigma)
    else:
        density_smooth = density_map
    
    # Normalizza per visualizzazione
    if density_smooth.max() > 0:
        density_norm = density_smooth / density_smooth.max()
    else:
        density_norm = density_smooth
    
    # Colormap personalizzata (stile paper)
    # Blu scuro → Ciano → Verde → Giallo → Rosso
    colors = [
        (0.0, 0.0, 0.3),    # Blu molto scuro
        (0.0, 0.0, 0.8),    # Blu
        (0.0, 0.8, 1.0),    # Ciano
        (0.0, 1.0, 0.5),    # Verde-ciano
        (0.5, 1.0, 0.0),    # Verde-giallo
        (1.0, 1.0, 0.0),    # Giallo
        (1.0, 0.5, 0.0),    # Arancione
        (1.0, 0.0, 0.0),    # Rosso
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list('density', colors, N=256)
    
    # Applica colormap
    density_colored = cmap(density_norm)[:, :, :3]
    
    # Crea maschera per l'overlay (solo dove c'è densità)
    mask = density_norm > 0.01
    
    # Blend usando np.where per evitare problemi di broadcasting
    # Espandi la maschera a 3 canali per il broadcasting
    mask_3d = mask[:, :, np.newaxis]
    
    # Blend: dove c'è densità, mischia; altrimenti mantieni l'originale
    overlay = np.where(mask_3d, 
                       (1 - alpha) * image + alpha * density_colored, 
                       image)
    
    # Clip per evitare warning di matplotlib (errori di precisione float)
    overlay = np.clip(overlay, 0, 1)
    
    return overlay, density_colored


def visualize_single_image_paper_style(
    img_tensor,
    gt_density,
    pred_density,
    mean,
    std,
    img_name="image",
    save_path=None,
    figsize=(15, 12)
):
    """
    Visualizza una singola immagine nello stile del paper CLIP-EBC.
    
    Layout:
    [Immagine Originale]
    [GT Density + Count]
    [Pred Density + Count]
    """
    # Denormalizza immagine
    img = denormalize_image(img_tensor, mean, std)
    H, W = img.shape[:2]
    
    # Prepara density maps
    gt_den = gt_density.squeeze().cpu().numpy()
    pred_den = pred_density.squeeze().cpu().numpy()
    
    # Resize pred_density se necessario
    if pred_den.shape != gt_den.shape:
        pred_den = F.interpolate(
            torch.tensor(pred_den).unsqueeze(0).unsqueeze(0).float(),
            size=gt_den.shape,
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
    
    # Conteggi
    gt_count = gt_den.sum()
    pred_count = pred_den.sum()
    error = abs(pred_count - gt_count)
    
    # Crea overlay
    gt_overlay, gt_heatmap = create_density_overlay(img, gt_den, alpha=0.7, sigma=3)
    pred_overlay, pred_heatmap = create_density_overlay(img, pred_den, alpha=0.7, sigma=3)
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    # 1. Immagine originale
    axes[0].imshow(img)
    axes[0].set_title(f'{img_name}', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 2. GT Density
    axes[1].imshow(gt_overlay)
    axes[1].set_title(f'GT: {gt_count:.0f}', fontsize=14, fontweight='bold', color='green')
    axes[1].axis('off')
    
    # 3. Predicted Density
    color = 'green' if error < gt_count * 0.1 else ('orange' if error < gt_count * 0.2 else 'red')
    axes[2].imshow(pred_overlay)
    axes[2].set_title(f'Pred: {pred_count:.1f} (Error: {error:.1f})', 
                      fontsize=14, fontweight='bold', color=color)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()


def visualize_grid_paper_style(
    images_data,
    mean,
    std,
    save_path=None,
    max_cols=6
):
    """
    Visualizza multiple immagini in una griglia stile paper.
    
    Layout (come nell'immagine del paper):
    Riga 1: Immagini originali
    Riga 2: GT Density maps
    Riga 3: Predicted Density maps
    
    Args:
        images_data: Lista di dict con 'img', 'gt_density', 'pred_density', 'name'
        mean, std: Per denormalizzazione
        save_path: Dove salvare
        max_cols: Numero massimo di colonne
    """
    n_images = min(len(images_data), max_cols)
    
    fig, axes = plt.subplots(3, n_images, figsize=(4 * n_images, 12))
    
    if n_images == 1:
        axes = axes.reshape(3, 1)
    
    for col, data in enumerate(images_data[:n_images]):
        img = denormalize_image(data['img'], mean, std)
        gt_den = data['gt_density'].squeeze().cpu().numpy()
        pred_den = data['pred_density'].squeeze().cpu().numpy()
        
        # Resize se necessario
        if pred_den.shape != gt_den.shape:
            pred_den = F.interpolate(
                torch.tensor(pred_den).unsqueeze(0).unsqueeze(0).float(),
                size=gt_den.shape,
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()
        
        gt_count = gt_den.sum()
        pred_count = pred_den.sum()
        
        # Overlay
        gt_overlay, _ = create_density_overlay(img, gt_den, alpha=0.7, sigma=3)
        pred_overlay, _ = create_density_overlay(img, pred_den, alpha=0.7, sigma=3)
        
        # Riga 1: Immagine originale
        axes[0, col].imshow(img)
        axes[0, col].axis('off')
        
        # Riga 2: GT Density
        axes[1, col].imshow(gt_overlay)
        axes[1, col].set_title(f'GT: {gt_count:.0f}', fontsize=11, color='green')
        axes[1, col].axis('off')
        
        # Riga 3: Pred Density
        axes[2, col].imshow(pred_overlay)
        axes[2, col].set_title(f'Pred: {pred_count:.1f}', fontsize=11, color='blue')
        axes[2, col].axis('off')
    
    # Titoli righe
    fig.text(0.02, 0.83, 'Original', fontsize=12, fontweight='bold', rotation=90, va='center')
    fig.text(0.02, 0.5, 'GT Density', fontsize=12, fontweight='bold', rotation=90, va='center')
    fig.text(0.02, 0.17, 'Predicted', fontsize=12, fontweight='bold', rotation=90, va='center')
    
    plt.tight_layout(rect=[0.03, 0, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"💾 Griglia salvata: {save_path}")
    else:
        plt.show()


def visualize_comparison_detailed(
    img_tensor,
    gt_density,
    pred_density,
    mean,
    std,
    img_name="image",
    save_path=None
):
    """
    Visualizzazione dettagliata con confronto side-by-side.
    
    Layout 2x2:
    [Originale]        [Pred overlay]
    [GT overlay]       [Differenza]
    """
    img = denormalize_image(img_tensor, mean, std)
    H, W = img.shape[:2]
    
    gt_den = gt_density.squeeze().cpu().numpy()
    pred_den = pred_density.squeeze().cpu().numpy()
    
    # Resize
    if pred_den.shape != gt_den.shape:
        pred_den = F.interpolate(
            torch.tensor(pred_den).unsqueeze(0).unsqueeze(0).float(),
            size=gt_den.shape,
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
    
    # Upscale per visualizzazione
    gt_den_full = F.interpolate(
        torch.tensor(gt_den).unsqueeze(0).unsqueeze(0).float(),
        size=(H, W),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()
    
    pred_den_full = F.interpolate(
        torch.tensor(pred_den).unsqueeze(0).unsqueeze(0).float(),
        size=(H, W),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()
    
    gt_count = gt_den.sum()
    pred_count = pred_den.sum()
    error = abs(pred_count - gt_count)
    error_pct = (error / gt_count * 100) if gt_count > 0 else 0
    
    # Overlay
    gt_overlay, _ = create_density_overlay(img, gt_den_full, alpha=0.7, sigma=2)
    pred_overlay, _ = create_density_overlay(img, pred_den_full, alpha=0.7, sigma=2)
    
    # Differenza (pred - gt)
    diff = pred_den_full - gt_den_full
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Originale
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Immagine Originale', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. Pred overlay
    axes[0, 1].imshow(pred_overlay)
    color = 'green' if error_pct < 10 else ('orange' if error_pct < 20 else 'red')
    axes[0, 1].set_title(f'Predicted: {pred_count:.1f}', fontsize=12, fontweight='bold', color=color)
    axes[0, 1].axis('off')
    
    # 3. GT overlay
    axes[1, 0].imshow(gt_overlay)
    axes[1, 0].set_title(f'Ground Truth: {gt_count:.0f}', fontsize=12, fontweight='bold', color='green')
    axes[1, 0].axis('off')
    
    # 4. Differenza
    max_diff = max(abs(diff.min()), abs(diff.max()), 0.01)
    im = axes[1, 1].imshow(diff, cmap='RdBu_r', vmin=-max_diff, vmax=max_diff)
    axes[1, 1].set_title(f'Differenza (Pred - GT)\nError: {error:.1f} ({error_pct:.1f}%)', 
                         fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, label='Sovrastima ← → Sottostima')
    
    fig.suptitle(f'{img_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
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
    use_pi_gating=False,
    generate_grid=True
):
    """Genera visualizzazioni per un subset di immagini."""
    
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    mean = config['DATA']['NORM_MEAN']
    std = config['DATA']['NORM_STD']
    
    print(f"\n🎨 Generazione visualizzazioni Stage 2...")
    print(f"   Output dir: {output_dir}")
    print(f"   Num samples: {num_samples}")
    print(f"   Use π-gating: {use_pi_gating}\n")
    
    images_data = []
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
            
            # Forward
            if use_pi_gating:
                outputs = model(imgs)
                pred_density = outputs['final_density']
            else:
                outputs = model.forward_stage2(imgs)
                pred_density = outputs['ebc_density']
            
            for i in range(imgs.size(0)):
                if count >= num_samples:
                    break
                
                img_name = os.path.basename(img_paths[i]).replace('.jpg', '').replace('.png', '')
                
                # Salva dati per griglia
                images_data.append({
                    'img': imgs[i].cpu(),
                    'gt_density': gt_density[i].cpu(),
                    'pred_density': pred_density[i].cpu(),
                    'name': img_name
                })
                
                # Visualizzazione singola dettagliata
                save_path = os.path.join(output_dir, f'{img_name}_detailed.png')
                visualize_comparison_detailed(
                    img_tensor=imgs[i].cpu(),
                    gt_density=gt_density[i].cpu(),
                    pred_density=pred_density[i].cpu(),
                    mean=mean,
                    std=std,
                    img_name=img_name,
                    save_path=save_path
                )
                
                # Visualizzazione stile paper
                save_path_paper = os.path.join(output_dir, f'{img_name}_paper_style.png')
                visualize_single_image_paper_style(
                    img_tensor=imgs[i].cpu(),
                    gt_density=gt_density[i].cpu(),
                    pred_density=pred_density[i].cpu(),
                    mean=mean,
                    std=std,
                    img_name=img_name,
                    save_path=save_path_paper
                )
                
                count += 1
    
    # Genera griglia stile paper
    if generate_grid and len(images_data) > 0:
        grid_path = os.path.join(output_dir, 'grid_paper_style.png')
        visualize_grid_paper_style(
            images_data=images_data[:6],  # Max 6 per la griglia
            mean=mean,
            std=std,
            save_path=grid_path
        )
    
    print(f"\n✅ Generate {count} visualizzazioni in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualizza risultati Stage 2')
    parser.add_argument('--config', type=str, default='configs/config_sha.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./visualizations/stage2')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--use_pi_gating', action='store_true')
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
        ckpt_path = f"./checkpoints/{dataset_name}/stage2/best_model.pth"
    
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
        use_pi_gating=args.use_pi_gating
    )


if __name__ == '__main__':
    main()