#!/usr/bin/env python3
"""
ZIP-CLIP-EBC: Visualizzazione Stage 1 (Overlay Style)
Genera 3 immagini di analisi con maschere colorate:
🟢 TP (Verde) | 🔴 FN (Rosso - Persi) | 🔵 FP (Blu - Rumore)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
import os
from PIL import Image

from models.zip_model import ZIPModel
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
    """Denormalizza l'immagine per visualizzazione (Restituisce numpy [H, W, 3])."""
    mean = torch.tensor(mean).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(img_tensor.device)
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img.cpu().permute(1, 2, 0).numpy()


def generate_overlay_mask(gt_mask, pred_mask, img_shape):
    """
    Crea una maschera RGB per l'overlay.
    """
    H, W = img_shape[:2]
    
    # Resize delle maschere alla dimensione dell'immagine originale
    gt_full = F.interpolate(gt_mask, size=(H, W), mode='nearest').squeeze().cpu().numpy()
    pred_full = F.interpolate(pred_mask, size=(H, W), mode='nearest').squeeze().cpu().numpy()
    
    # Inizializza overlay nero
    overlay = np.zeros((H, W, 3), dtype=np.float32)
    
    # --- LOGICA COLORI ---
    # TP: Pred=1 & GT=1 -> Verde (Successo)
    overlay[(pred_full == 1) & (gt_full == 1)] = [0, 1, 0] 
    
    # FN: Pred=0 & GT=1 -> Rosso (Persone perse - GRAVE)
    overlay[(pred_full == 0) & (gt_full == 1)] = [1, 0, 0]
    
    # FP: Pred=1 & GT=0 -> Blu (Rumore/Sfondo - Accettabile)
    overlay[(pred_full == 1) & (gt_full == 0)] = [0, 0.5, 1] 
    
    # Maschera booleana dove c'è colore
    mask_indices = np.any(overlay > 0, axis=-1)
    
    return overlay, mask_indices


def load_checkpoint_safe(model, ckpt_path):
    """
    Carica i pesi ignorando quelli con shape mismatch (utile se la config è cambiata).
    """
    checkpoint = torch.load(ckpt_path, map_location=next(model.parameters()).device)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    model_state = model.state_dict()
    filtered_state = {}
    ignored_keys = []

    for k, v in state_dict.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                filtered_state[k] = v
            else:
                ignored_keys.append(k)
        # Ignora chiavi non presenti nel modello attuale
        
    if ignored_keys:
        print(f"⚠️  Attenzione: {len(ignored_keys)} layer ignorati per mismatch di shape (es. CLIP head cambiata).")
        print(f"    Esempio ignorato: {ignored_keys[0]}")
    
    model.load_state_dict(filtered_state, strict=False)
    print(f"✅ Checkpoint caricato con successo (Safe Mode): {ckpt_path}")


def generate_visualizations(model, dataloader, device, config, threshold=0.15, num_images=3, output_dir='visualizations'):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    block_size = config['DATA'].get('ZIP_BLOCK_SIZE', 16)
    mean = config['DATA']['NORM_MEAN']
    std = config['DATA']['NORM_STD']
    
    print(f"🎨 Generazione di {num_images} immagini overlay in '{output_dir}' (Threshold={threshold})...")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_images: break
            
            img_tensor = batch['image'].to(device)
            gt_density = batch['density'].to(device)
            img_path = batch['img_path'][0]
            img_name = os.path.basename(img_path).split('.')[0]
            
            # --- 1. Ground Truth ---
            gt_counts = F.avg_pool2d(gt_density, block_size) * (block_size**2)
            gt_mask = (gt_counts > 0).float()
            
            # --- 2. Predizione ---
            outputs = model(img_tensor)
            pi_logits = outputs['pi_logits']
            prob_occupied = torch.sigmoid(pi_logits)
            
            # Allinea dimensioni
            if prob_occupied.shape[-2:] != gt_mask.shape[-2:]:
                prob_occupied = F.interpolate(prob_occupied, size=gt_mask.shape[-2:], mode='bilinear')
            
            # Binarizzazione
            pred_mask = (prob_occupied > threshold).float()
            
            # --- 3. Preparazione Immagine Base ---
            img_np = denormalize_image(img_tensor[0], mean, std)
            
            # --- 4. Creazione Overlay ---
            overlay_rgb, mask_bool = generate_overlay_mask(gt_mask, pred_mask, img_np.shape)
            
            # Fonde l'immagine: 60% Originale + 40% Colore
            alpha = 0.4
            final_img = img_np.copy()
            final_img[mask_bool] = (1 - alpha) * img_np[mask_bool] + alpha * overlay_rgb[mask_bool]
            
            # --- 5. Plotting ---
            plt.figure(figsize=(12, 8))
            plt.imshow(final_img)
            
            # Statistiche
            tp = ((pred_mask == 1) & (gt_mask == 1)).sum().item()
            fn = ((pred_mask == 0) & (gt_mask == 1)).sum().item()
            fp = ((pred_mask == 1) & (gt_mask == 0)).sum().item()
            
            title_text = (f"Image: {img_name}\n"
                          f"Threshold: {threshold} | "
                          f"TP (Verde): {int(tp)} | FN (Rosso): {int(fn)} | FP (Blu): {int(fp)}")
            
            plt.title(title_text, fontsize=14, pad=10, backgroundcolor='white')
            plt.axis('off')
            
            # Salva
            save_path = os.path.join(output_dir, f"viz_{i}_{img_name}.jpg")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
            plt.close()
            print(f"✅ Salvata: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_sha.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--num_images', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default='visualizations')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Threshold logic
    if args.threshold is not None:
        final_threshold = args.threshold
    else:
        eval_cfg = config.get('EVAL_STAGE1', {})
        final_threshold = eval_cfg.get('THRESHOLD', 0.15)
    
    print(f"🔧 Using Threshold: {final_threshold}")

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    model = ZIPModel(config).to(device)
    
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        dataset_name = config.get('DATASET', 'sha')
        ckpt_path = f"./checkpoints/{dataset_name}/stage1/best_model.pth"
    
    if os.path.exists(ckpt_path):
        # Usa la nuova funzione di caricamento sicuro
        load_checkpoint_safe(model, ckpt_path)
    else:
        print(f"⚠️  Checkpoint non trovato: {ckpt_path}")

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
    
    generate_visualizations(
        model, 
        val_loader, 
        device, 
        config, 
        threshold=final_threshold, 
        num_images=args.num_images,
        output_dir=args.output_dir
    )