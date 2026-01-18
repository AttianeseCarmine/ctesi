#!/usr/bin/env python3
"""
Train Stage 2: CLIP-EBC Official Replica (Fixed)
"""
import os
import yaml
import json
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast # Import corretto per Mixed Precision
from tqdm import tqdm
import shutil
# --- IMPORTS AGGIORNATI ---
# Assicurati che i file siano in models/clip/model.py e losses/clip_ebc_loss.py
from models.clip.model import CLIP_EBC
from losses.clip_ebc_loss import DACELoss

# Assumi che tu abbia un dataset loader funzionante
# Se non hai datasets/sha.py, devi usare il tuo loader custom
try:
    from datasets.sha import SHA 
    from datasets.transforms import build_transforms
except ImportError:
    print("⚠️ Dataset SHA non trovato, assicurati di avere il file dataset corretto.")

def adjust_learning_rate(optimizer, epoch, args):
    """Cosine schedule con Warmup"""
    lr_max = args['TRAIN_STAGE2']['LR_HEAD']
    warmup_epochs = args['TRAIN_STAGE2']['WARMUP_EPOCHS']
    max_epochs = args['TRAIN_STAGE2']['EPOCHS']

    if epoch < warmup_epochs:
        lr = lr_max * (epoch + 1) / (warmup_epochs + 1e-8)
    else:
        # Cosine annealing
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        lr = lr_max * 0.5 * (1. + math.cos(math.pi * progress))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main(config_path):
    # 1. Load Config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg['DEVICE'])

    # --- MODIFICA: Creazione cartella e salvataggio Configurazione ---
    output_dir = os.path.join("checkpoints", cfg['RUN_NAME'], "stage2")
    os.makedirs(output_dir, exist_ok=True)
    # Copia il file .yaml originale nella cartella di output
    config_filename = os.path.basename(config_path)
    saved_config_path = os.path.join(output_dir, config_filename)
    shutil.copy(config_path, saved_config_path)
    print(f"📄 Configuration saved to: {saved_config_path}")
    # ---------------------------------------------------------------

    # 2. Load Bins & Anchors (DAL CONFIGURATORE)
    print(f"⚙️ Loading bins from YAML config...")
    
    # Correzione chiavi e variabili
    raw_bins = cfg['BINS']
    raw_anchors = cfg['BIN_CENTERS'] # Corretto per matchare il YAML
    
    # 3. Pulizia dei bin e degli anchor
    cleaned_bins = []
    for b in raw_bins: # Usa raw_bins, non bins_list
        start = float(b[0])
        val_end = b[1]
        
        # Gestione flessibile per infinito (accetta "inf", 9999 o float('inf'))
        if str(val_end).lower() == "inf" or val_end >= 9999:
            end = float('inf')
        else:
            end = float(val_end)
        cleaned_bins.append((start, end))

    # Definisci cleaned_anchors (prima mancava)
    cleaned_anchors = [float(a) for a in raw_anchors]

    print(f"✅ Loaded {len(cleaned_bins)} bins: {cleaned_bins}")
    print(f"✅ Loaded anchors: {cleaned_anchors}")

    # 3. Initialize Model
    model = CLIP_EBC(
        backbone=cfg['CLIP_EBC_MODEL']['BACKBONE'],
        bins=cleaned_bins,
        anchor_points=cleaned_anchors, # Ora la variabile esiste
        reduction=cfg['CLIP_EBC_MODEL']['REDUCTION'],
        input_size=cfg['CLIP_EBC_MODEL']['INPUT_SIZE'],
        num_vpt=cfg['CLIP_EBC_MODEL']['NUM_VPT'],
        deep_vpt=cfg['CLIP_EBC_MODEL']['DEEP_VPT'],
        vpt_drop=cfg['CLIP_EBC_MODEL']['VPT_DROP'],
        # Se PROMPT_TYPE non è nel yaml, usa un default "number"
        prompt_type=cfg.get('PROMPT_TYPE', "number") 
    ).to(device)

    print(f" -- MODELLO CON REDUCTION  -> {cfg['CLIP_EBC_MODEL']['REDUCTION']} -- ")
    print(f" -- MODELLO CON WEIGHT_COUNT  -> {cfg['LOSS_STAGE2']['WEIGHT_COUNT']} -- ")

    # 4. Dataset & Dataloader
    # Nota: Adatta questo pezzo al tuo dataset loader specifico
    # SHA deve ritornare: image, density_map, point_list
    train_transform = build_transforms(cfg['DATA'], is_train=True)
    val_transform = build_transforms(cfg['DATA'], is_train=False)
    
    train_dataset = SHA(cfg['DATA']['ROOT'], 'train', build_transforms(cfg['DATA'], True))
    val_dataset = SHA(cfg['DATA']['ROOT'], 'val', build_transforms(cfg['DATA'], False))
    
    # Collate function custom per gestire liste di punti di lunghezza variabile
    # --- COLLATE FUNCTION CORRETTA (Per Dizionari) ---
    def collate_fn(batch):
        """
        Gestisce il batch quando il dataset restituisce un dizionario.
        """
        # 1. Estrazione Immagini (chiave 'image')
        # Se fallisce qui, stampa le chiavi: print(batch[0].keys())
        imgs = torch.stack([item['image'] for item in batch])
        
        # 2. Estrazione Density Map (chiave 'density' o 'label')
        # Alcuni dataset usano 'density', altri 'label'. Li gestiamo entrambi.
        if 'density' in batch[0]:
            densities = torch.stack([item['density'] for item in batch])
        elif 'label' in batch[0]:
            densities = torch.stack([item['label'] for item in batch])
        else:
            raise KeyError(f"Chiave densità mancante. Chiavi trovate: {list(batch[0].keys())}")

        # 3. Estrazione Punti (chiave 'points' o 'keypoints')
        if 'points' in batch[0]:
            points = [item['points'] for item in batch]
        elif 'keypoints' in batch[0]:
            points = [item['keypoints'] for item in batch]
        else:
            # Se non ci sono punti, proviamo a restituire una lista vuota o fallire
            # Per SHA solitamente 'points' esiste.
            raise KeyError(f"Chiave punti mancante. Chiavi trovate: {list(batch[0].keys())}")
        
        return imgs, densities, points

    train_loader = DataLoader(train_dataset, batch_size=cfg['TRAIN_STAGE2']['BATCH_SIZE'], shuffle=True, num_workers=cfg['TRAIN_STAGE2']['NUM_WORKERS'], collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=cfg['TRAIN_STAGE2']['NUM_WORKERS'], collate_fn=collate_fn)

    # 5. Optimizer & Loss
    # Filtriamo i parametri: Backone ViT è congelato, alleniamo VPT e decoder
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params_to_optimize, lr=cfg['TRAIN_STAGE2']['LR_HEAD'], weight_decay=cfg['TRAIN_STAGE2']['WEIGHT_DECAY'])
    
    criterion = DACELoss(
        bins=cleaned_bins,
        reduction=cfg['CLIP_EBC_MODEL']['REDUCTION'],
        weight_count_loss=cfg['LOSS_STAGE2']['WEIGHT_COUNT'],
        count_loss=cfg['LOSS_STAGE2']['COUNT_LOSS'], # "dmcount"
        weight_ot=cfg['LOSS_STAGE2']['WEIGHT_OT'],
        weight_tv=cfg['LOSS_STAGE2']['WEIGHT_TV'],
        input_size=cfg['CLIP_EBC_MODEL']['INPUT_SIZE']
    ).to(device)

    scaler = GradScaler() # Per Mixed Precision

    # 6. Training Loop
    best_mae = float('inf')
    
    for epoch in range(cfg['TRAIN_STAGE2']['EPOCHS']):
        adjust_learning_rate(optimizer, epoch, cfg)
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{cfg['TRAIN_STAGE2']['EPOCHS']}")
        for imgs, gt_density, gt_points in pbar:
            imgs = imgs.to(device)
            gt_density = gt_density.to(device)
            # gt_points è una lista di tensori, li spostiamo su device dentro la loss o qui se serve
            
            optimizer.zero_grad()
            
            with autocast():
                # CLIP_EBC in training ritorna (logits, density_map)
                pred_logits, pred_density = model(imgs)
                
                # Calcolo Loss
                loss, loss_dict = criterion(pred_logits, pred_density, gt_density, gt_points)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'ce': loss_dict.get('ce_loss', 0).item()})

        # 7. Validation Loop
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for imgs, gt_density, gt_points in val_loader:
                imgs = imgs.to(device)
                gt_count = len(gt_points[0]) # Batch size 1 in validation
                
                # In eval, CLIP_EBC ritorna solo density_map (o expected count map)
                pred_map = model(imgs)
                pred_count = pred_map.sum().item()
                
                val_mae += abs(pred_count - gt_count)
        
        val_mae /= len(val_dataset)
        print(f"📊 Epoch {epoch+1} Result: Val MAE: {val_mae:.2f} (Best: {best_mae:.2f})")
        
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), f"checkpoints/{cfg['RUN_NAME']}/stage2/best_model.pth")
            print("💾 Model Saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_vit_sha.yaml')
    args = parser.parse_args()
    main(args.config)