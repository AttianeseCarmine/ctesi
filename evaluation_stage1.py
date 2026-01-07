#!/usr/bin/env python3
"""
ZIP-CLIP-EBC: Evaluation Script for Stage 1 (Binary ZIP Head)
Valuta la capacità della π-head di classificare blocchi vuoti/pieni.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CORREZIONE: Usa lo stesso modello del training ---
from models.zip_model import ZIPModel
from datasets.sha import SHA
from datasets.transforms import build_transforms

def crowd_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'density': torch.stack([item['density'] for item in batch]),
        'img_path': [item['img_path'] for item in batch]
    }

def plot_confusion_matrix(tp, fp, tn, fn, save_path="confusion_matrix_stage1.png"):
    cm_data = np.array([[tp, fn], [fp, tn]])
    row_sums = cm_data.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 
    cm_perc = cm_data / row_sums

    labels = np.array([
        [f"TP\n{int(tp)}\n({cm_perc[0,0]:.1%})", f"FN\n{int(fn)}\n({cm_perc[0,1]:.1%})"],
        [f"FP\n{int(fp)}\n({cm_perc[1,0]:.1%})", f"TN\n{int(tn)}\n({cm_perc[1,1]:.1%})"]
    ])

    plt.figure(figsize=(8, 6))
    sns.set_style("white")
    ax = sns.heatmap(cm_data, annot=labels, fmt='', cmap='Blues', cbar=True,
                     linewidths=1, linecolor='black', annot_kws={"size": 14, "weight": "bold"})
    ax.set_xticklabels(['Pred: PIENO', 'Pred: VUOTO'], fontsize=12)
    ax.set_yticklabels(['GT: PIENO', 'GT: VUOTO'], fontsize=12, va='center')
    plt.title('Stage 1 Confusion Matrix (ResNet50 ZIP-Head)', fontsize=16, pad=20)
    plt.ylabel('Ground Truth', fontsize=14)
    plt.xlabel('Prediction', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Grafico Matrice di Confusione salvato in: {save_path}")

def evaluate_stage1(model, dataloader, device, threshold=0.5):
    model.eval()
    tp, fp, tn, fn = 0, 0, 0, 0
    num_images = 0

    print(f"Running Stage 1 Evaluation with Threshold={threshold}...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if batch is None: continue
            imgs = batch['image'].to(device)
            gt_density = batch['density'].to(device)

            # 1. Forward
            outputs = model(imgs)
            pi_logits = outputs['pi_logits']
            prob_occupied = torch.sigmoid(pi_logits)
            
            # 2. Ground Truth Binaria (Adattiva)
            # Calcoliamo la GT alla stessa risoluzione dell'output del modello
            h_out, w_out = pi_logits.shape[2:]
            scale = (imgs.shape[2] * imgs.shape[3]) / (h_out * w_out)
            gt_down = F.adaptive_avg_pool2d(gt_density, (h_out, w_out)) * scale
            gt_mask = (gt_down > 0.001).float() # Soglia bassa per definire "presenza"

            # 3. Predizione Binaria
            preds_mask = (prob_occupied > threshold).float()

            # 4. Metriche Pixel-wise (Block-wise)
            tp += ((preds_mask == 1) & (gt_mask == 1)).sum().item()
            fp += ((preds_mask == 1) & (gt_mask == 0)).sum().item()
            tn += ((preds_mask == 0) & (gt_mask == 0)).sum().item()
            fn += ((preds_mask == 0) & (gt_mask == 1)).sum().item()
            
            num_images += imgs.size(0)

    # Calcolo Metriche
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    print("\n" + "="*50)
    print("📊 STAGE 1 RESULTS (Binary Classification)")
    print("="*50)
    print(f"Threshold: {threshold}")
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1-Score:  {f1:.2%}")
    print("="*50)
    
    plot_confusion_matrix(tp, fp, tn, fn)
    return f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_shb.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    with open(args.config, 'r') as f: config = yaml.safe_load(f)

    # Threshold selection
    if args.threshold is not None:
        final_threshold = args.threshold
    else:
        final_threshold = config.get('EVAL_STAGE1', {}).get('THRESHOLD', 0.5)

    device = torch.device(f'cuda:{args.gpu}')
    
    # --- MODIFICA: Istanzia ZIPModel ---
    model = ZIPModel(config).to(device)
    
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        dataset_name = config.get('DATASET', 'shb')
        ckpt_path = f"./checkpoints/{dataset_name}/stage1/best_model.pth"
    
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        # Gestione robusta delle chiavi
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"⚠️ Checkpoint not found: {ckpt_path}")
    
    # Validation Loader
    val_dataset = SHA(config['DATA']['ROOT'], 'val', build_transforms(config['DATA'], False))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=crowd_collate)

    evaluate_stage1(model, val_loader, device, threshold=final_threshold)