#!/usr/bin/env python3
"""
ZIP-CLIP-EBC: Evaluation Script for Stage 2 (CLIP-EBC Head)
Valuta MAE e RMSE sul conteggio.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse
import os

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


def evaluate_stage2(model, val_loader, device):
    model.eval()
    mae, mse, total = 0, 0, 0
    total_gt, total_pred = 0, 0

    print(f"\nRunning Stage 2 Evaluation (CLIP-EBC Head)...")

    with torch.no_grad():
        for batch in tqdm(val_loader):
            if batch is None:
                continue
                
            imgs = batch['image'].to(device)
            gt_density = batch['density'].to(device)
            
            # Ground truth count
            gt_count = gt_density.sum().item()
            
            # Predizione EBC: usa forward_stage2 per avere solo l'output EBC
            outputs = model.forward_stage2(imgs)
            pred_count = outputs['ebc_density'].sum().item()
            
            # Accumula metriche
            mae += abs(pred_count - gt_count)
            mse += (pred_count - gt_count) ** 2
            total += 1
            
            total_gt += gt_count
            total_pred += pred_count

    # Calcola metriche finali
    mae = mae / total
    rmse = (mse / total) ** 0.5

    print("\n" + "="*50)
    print("📊 STAGE 2 EVALUATION RESULTS")
    print("="*50)
    print(f"   Total GT Count:   {total_gt:.0f}")
    print(f"   Total Pred Count: {total_pred:.0f}")
    print(f"   MAE:  {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   Num Images: {total}")
    print("="*50)

    return {"mae": mae, "rmse": rmse}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_sha.yaml')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path al checkpoint (default: auto-detect)')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    # Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")

    # Modello
    model = ZIPCLIPEBCModel(config).to(device)

    # Carica checkpoint (auto-detect basato su dataset)
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        dataset_name = config.get('DATASET', 'sha')
        ckpt_path = f"./checkpoints/{dataset_name}/stage2/best_model.pth"

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
    evaluate_stage2(model, val_loader, device)