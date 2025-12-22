#!/usr/bin/env python3
"""
ZIP-CLIP-EBC: Evaluation Script for Stage 1 (π-Head)
Valuta la capacità della π-head di classificare blocchi vuoti/pieni.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse

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


def evaluate_stage1(model, dataloader, device, block_size=16, threshold=0.5):
    model.eval()
    
    # Inizializzazione contatori per metriche
    tp, fp, tn, fn = 0, 0, 0, 0
    total_mae = 0.0
    num_images = 0

    print(f"Running Stage 1 Evaluation (π-head)...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if batch is None:
                continue
                
            imgs = batch['image'].to(device)
            gt_density = batch['density'].to(device)

            # 1. Calcolo Ground Truth: se un blocco ha > 0 persone, è "Occupato"
            gt_counts = F.avg_pool2d(gt_density, block_size) * (block_size**2)
            gt_mask_occupied = (gt_counts > 0).float()

            # 2. Forward del modello
            outputs = model(imgs)
            
            # π è la probabilità di "VUOTO", quindi prob OCCUPATO = (1 - pi)
            pi_vuoto = outputs['pi']
            prob_occupied = 1.0 - pi_vuoto
            
            # Allinea dimensioni se necessario
            if prob_occupied.shape[-2:] != gt_mask_occupied.shape[-2:]:
                prob_occupied = F.interpolate(prob_occupied, size=gt_mask_occupied.shape[-2:], mode='bilinear', align_corners=False)
            
            # 3. Predizione binaria basata sulla soglia
            preds_occupied = (prob_occupied > threshold).float()

            # 4. Aggiornamento metriche di classificazione
            tp += ((preds_occupied == 1) & (gt_mask_occupied == 1)).sum().item()
            fp += ((preds_occupied == 1) & (gt_mask_occupied == 0)).sum().item()
            tn += ((preds_occupied == 0) & (gt_mask_occupied == 0)).sum().item()
            fn += ((preds_occupied == 0) & (gt_mask_occupied == 1)).sum().item()
            
            # 5. MAE sulla stima del conteggio
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

    # Calcolo metriche finali
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    mae = total_mae / max(num_images, 1)

    print("\n" + "="*50)
    print("📊 STAGE 1 EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy:.2%})")
    print(f"Precision: {precision:.4f} (Capacità di non dare falsi positivi)")
    print(f"Recall:    {recall:.4f} (Capacità di trovare tutte le persone)")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Count MAE: {mae:.2f}")
    print("="*50)
    
    print(f"\n📋 CONFUSION MATRIX")
    print(f"                    | Pred:PIENO | Pred:VUOTO |")
    print(f"   GT:PIENO (crowd) | TP={tp:>7,} | FN={fn:>7,} |")
    print(f"   GT:VUOTO (empty) | FP={fp:>7,} | TN={tn:>7,} |")
    print("="*50)
    
    return f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_sha.yaml')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path al checkpoint (default: auto-detect)')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    # Carica Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")

    # Inizializza Modello
    model = ZIPCLIPEBCModel(config).to(device)
    
    # Carica checkpoint
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        dataset_name = config.get('DATASET', 'sha')
        ckpt_path = f"./checkpoints/{dataset_name}/stage1/best_model.pth"
    
    import os
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Gestisce diversi formati di salvataggio
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"✅ Checkpoint caricato: {ckpt_path}")
    else:
        print(f"⚠️  Checkpoint non trovato: {ckpt_path}")
    
    # Dataset di Validazione
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

    # Valutazione
    evaluate_stage1(
        model, 
        val_loader, 
        device, 
        block_size=data_cfg.get('ZIP_BLOCK_SIZE', 16),
        threshold=args.threshold
    )