#!/usr/bin/env python3
"""
ZIP-CLIP-EBC: Evaluation Script for Stage 1 (π-Head)
Valuta la capacità della π-head di classificare blocchi vuoti/pieni.
Genera metriche testuali e una Confusion Matrix grafica.
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

from models.zip_clip_ebc_model import ZIPCLIPEBCModel
from datasets.sha import SHA
from datasets.transforms import build_transforms


def crowd_collate(batch):
    """Collate function per gestire batch con dimensioni variabili."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'density': torch.stack([item['density'] for item in batch]),
        'points': [item['points'] for item in batch],
        'img_path': [item['img_path'] for item in batch]
    }


def plot_confusion_matrix(tp, fp, tn, fn, save_path="confusion_matrix_stage1.png"):
    """
    Genera e salva una matrice di confusione in stile heatmap.
    """
    cm_data = np.array([
        [tp, fn],  # GT: Pieno
        [fp, tn]   # GT: Vuoto
    ])
    
    # Calcolo percentuali per riga
    row_sums = cm_data.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 
    cm_perc = cm_data / row_sums

    labels = np.array([
        [f"TP\n{int(tp)}\n({cm_perc[0,0]:.1%})", f"FN\n{int(fn)}\n({cm_perc[0,1]:.1%})"],
        [f"FP\n{int(fp)}\n({cm_perc[1,0]:.1%})", f"TN\n{int(tn)}\n({cm_perc[1,1]:.1%})"]
    ])

    plt.figure(figsize=(8, 6))
    sns.set_style("white")
    
    ax = sns.heatmap(
        cm_data, 
        annot=labels, 
        fmt='', 
        cmap='Blues', 
        cbar=True,
        linewidths=1,
        linecolor='black',
        annot_kws={"size": 14, "weight": "bold"}
    )

    ax.set_xticklabels(['Pred: PIENO', 'Pred: VUOTO'], fontsize=12)
    ax.set_yticklabels(['GT: PIENO', 'GT: VUOTO'], fontsize=12, va='center')
    
    plt.title('Stage 1 Confusion Matrix (π-head filter)', fontsize=16, pad=20)
    plt.ylabel('Ground Truth (Realtà)', fontsize=14)
    plt.xlabel('Prediction (Modello)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Grafico Matrice di Confusione salvato in: {save_path}")


def evaluate_stage1(model, dataloader, device, block_size=16, threshold=0.5):
    model.eval()
    
    tp, fp, tn, fn = 0, 0, 0, 0
    total_mae = 0.0
    num_images = 0

    print(f"Running Stage 1 Evaluation (π-head) with Threshold={threshold}...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if batch is None:
                continue
                
            imgs = batch['image'].to(device)
            gt_density = batch['density'].to(device)

            gt_counts = F.avg_pool2d(gt_density, block_size) * (block_size**2)
            gt_mask_occupied = (gt_counts > 0).float()

            outputs = model(imgs)
            pi_logits = outputs['pi_logits']
            prob_occupied = torch.sigmoid(pi_logits)
            
            if prob_occupied.shape[-2:] != gt_mask_occupied.shape[-2:]:
                prob_occupied = F.interpolate(prob_occupied, size=gt_mask_occupied.shape[-2:], mode='bilinear', align_corners=False)
            
            preds_occupied = (prob_occupied > threshold).float()

            tp += ((preds_occupied == 1) & (gt_mask_occupied == 1)).sum().item()
            fp += ((preds_occupied == 1) & (gt_mask_occupied == 0)).sum().item()
            tn += ((preds_occupied == 0) & (gt_mask_occupied == 0)).sum().item()
            fn += ((preds_occupied == 0) & (gt_mask_occupied == 1)).sum().item()
            
            if 'lambda_' in outputs and outputs['lambda_'] is not None:
                lambda_ = outputs['lambda_']
                if lambda_.shape[-2:] != prob_occupied.shape[-2:]:
                    lambda_ = F.interpolate(lambda_, size=prob_occupied.shape[-2:], mode='bilinear', align_corners=False)
                pred_count = (lambda_ * prob_occupied).sum(dim=[1,2,3])
            else:
                pred_count = prob_occupied.sum(dim=[1,2,3])
            
            gt_total = gt_density.sum(dim=[1,2,3])
            total_mae += torch.abs(pred_count - gt_total).sum().item()
            num_images += imgs.size(0)

    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    mae = total_mae / max(num_images, 1)

    print("\n" + "="*50)
    print("📊 STAGE 1 EVALUATION RESULTS")
    print("="*50)
    print(f"Threshold: {threshold}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy:.2%})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Count MAE: {mae:.2f}")
    print("="*50)
    
    print(f"\n📋 CONFUSION MATRIX (Text)")
    print(f"                    | Pred:PIENO | Pred:VUOTO |")
    print(f"   GT:PIENO (crowd) | TP={int(tp):>7,} | FN={int(fn):>7,} |")
    print(f"   GT:VUOTO (empty) | FP={int(fp):>7,} | TN={int(tn):>7,} |")
    print("="*50)
    
    plot_confusion_matrix(tp, fp, tn, fn)
    
    return f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_sha.yaml')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path al checkpoint (default: auto-detect)')
    # Impostiamo default=None per distinguere se l'utente l'ha passato o no
    parser.add_argument('--threshold', type=float, default=None, 
                        help='Override threshold (se None usa config)')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    # Carica Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # === LOGICA SELEZIONE SOGLIA ===
    # 1. Priorità: Argomento da riga di comando
    # 2. Secondaria: Configurazione YAML
    # 3. Fallback: 0.5 default
    if args.threshold is not None:
        final_threshold = args.threshold
        print(f"🔧 Threshold da riga di comando: {final_threshold}")
    else:
        # Cerca nel config, se non c'è usa 0.5
        eval_cfg = config.get('EVAL_STAGE1', {})
        final_threshold = eval_cfg.get('THRESHOLD', 0.5)
        print(f"🔧 Threshold da config: {final_threshold}")

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")

    # Inizializza Modello
    model = ZIPCLIPEBCModel(config).to(device)
    
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
        # exit(1) 
    
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

    evaluate_stage1(
        model, 
        val_loader, 
        device, 
        block_size=data_cfg.get('ZIP_BLOCK_SIZE', 16),
        threshold=final_threshold  # Usiamo la soglia decisa
    )