#!/usr/bin/env python3
"""
Evaluate: Divide et Impera Strategy (Strict Conditional)
========================================================
Logica:
1. Divide l'immagine in blocchi.
2. Passa ogni blocco in ZIP (Stage 1).
3. Se Prob(Persone) < Threshold -> Scarta il blocco (Conteggio = 0).
4. Se Prob(Persone) > Threshold -> Passa a CLIP (Stage 2) -> Somma il conteggio.
"""

import os
import yaml
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

# --- IMPORTS ---
from models.zip_model import ZIPModel
from models.clip_ebc_model import CLIPEBCModel
from datasets.sha import SHA
from datasets.transforms import build_transforms

def divide_and_conquer_predict(stage1, stage2, image, patch_size=448, stride=448, threshold=0.6, device='cuda'):
    """
    Esegue la logica condizionale a blocchi.
    """
    stage1.eval()
    stage2.eval()
    
    B, C, H, W = image.shape
    total_count = 0.0
    
    # Statistiche per curiosità
    blocks_total = 0
    blocks_processed = 0
    
    with torch.no_grad():
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                blocks_total += 1
                
                # Calcolo coordinate crop
                y_end = min(y + patch_size, H)
                x_end = min(x + patch_size, W)
                y_start = max(y_end - patch_size, 0)
                x_start = max(x_end - patch_size, 0)
                
                # Estrai Patch
                patch = image[:, :, y_start:y_end, x_start:x_end].to(device)
                
                # --- PASSO 1: IL FILTRO (ZIP) ---
                zip_out = stage1(patch)
                
                # Ottieni probabilità (Gestisce sia output dict che tensore)
                if isinstance(zip_out, dict):
                    logits = zip_out.get('pi_logits', zip_out.get('logit_pi'))
                else:
                    logits = zip_out
                
                # Calcola probabilità massima nel blocco (o media, a scelta)
                # Se anche solo un pezzetto del blocco ha alta probabilità, lo teniamo.
                prob_presence = torch.sigmoid(logits).max().item()
                
                # --- LA DECISIONE (Thresholding) ---
                if prob_presence < threshold:
                    # SCARTA: Non chiamare CLIP, risparmia tempo e riduci FP
                    continue
                
                # --- PASSO 2: IL CONTATORE (CLIP) ---
                blocks_processed += 1
                clip_out = stage2(patch)
                
                # Ottieni densità
                density = clip_out['ebc_density'] # [1, 1, h, w]
                
                # Somma solo la parte non sovrapposta (logica sliding window base)
                # Nota: Per semplicità qui sommiamo tutto il patch processato.
                # Per precisione estrema sui bordi servirebbe un canvas, ma questo rende l'idea.
                patch_count = density.sum().item()
                
                # (Opzionale) Possiamo ri-applicare la maschera locale per pulire i bordi del patch
                # mask_local = torch.sigmoid(logits)
                # patch_count = (density * mask_local).sum().item()
                
                total_count += patch_count

    return total_count, blocks_processed, blocks_total

def evaluate(args):
    # 1. Configurazione
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device} | Dataset: {config['DATASET']}")

    # 2. Inizializza i Modelli "Specialisti"
    print("🏗️  Building Specialist Models...")
    zip_model = ZIPModel(config).to(device)
    clip_model = CLIPEBCModel(config).to(device)
    
    # 3. Carica i Pesi dal "Joint Model" addestrato
    # Il file .pth di Stage 3 contiene le chiavi 'stage1.xxx' e 'stage2.xxx'
    print(f"📥 Loading Checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # Separa i pesi
    zip_dict = {k.replace('stage1.', ''): v for k, v in state_dict.items() if k.startswith('stage1.')}
    clip_dict = {k.replace('stage2.', ''): v for k, v in state_dict.items() if k.startswith('stage2.')}
    
    zip_model.load_state_dict(zip_dict, strict=False)
    clip_model.load_state_dict(clip_dict, strict=False)
    print("✅ Weights Split & Loaded into ZIP and CLIP models.")

    # 4. Dataset
    val_transforms = build_transforms(config['DATA'], is_train=False) 
    dataset = SHA(config['DATA']['ROOT'], 'val', val_transforms)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # 5. Valutazione
    mae_accum = 0.0
    mse_accum = 0.0
    total_skipped = 0
    total_blocks = 0
    
    # Threshold definisce quanto severo è il filtro ZIP
    # 0.5 è neutro. >0.5 è severo (scarta dubbi). <0.5 è permissivo.
    CONFIDENCE_THRESHOLD = 0.6
    
    print(f"🚀 Starting Divide & Conquer Eval (Threshold: {CONFIDENCE_THRESHOLD})...")
    
    pbar = tqdm(loader)
    for batch in pbar:
        img = batch['image'] # Non spostare su GPU qui, lo fa la funzione patchwise
        gt_count = len(batch['points'][0]) 
        
        # Esegui la logica a blocchi
        pred_count, n_proc, n_tot = divide_and_conquer_predict(
            zip_model, 
            clip_model, 
            img, 
            patch_size=448, 
            stride=448, # Nessuna sovrapposizione per massima velocità
            threshold=CONFIDENCE_THRESHOLD, 
            device=device
        )
        
        # Metriche
        error = abs(pred_count - gt_count)
        mae_accum += error
        mse_accum += error ** 2
        
        total_skipped += (n_tot - n_proc)
        total_blocks += n_tot
        
        pbar.set_postfix({
            'GT': gt_count, 
            'Pred': f"{pred_count:.1f}", 
            'Kept': f"{n_proc}/{n_tot}"
        })

    # 6. Risultati Finali
    final_mae = mae_accum / len(dataset)
    final_mse = (mse_accum / len(dataset)) ** 0.5
    skip_rate = (total_skipped / total_blocks) * 100 if total_blocks > 0 else 0
    
    print("\n" + "="*50)
    print(f"🏆 FINAL RESULTS: {config['DATASET'].upper()}")
    print(f"   Strategy: Divide et Impera (Skip empty blocks)")
    print(f"   MAE: {final_mae:.2f}")
    print(f"   MSE: {final_mse:.2f}")
    print(f"   Efficiency: {skip_rate:.1f}% of blocks were SKIPPED (Pure Background)")
    print("="*50 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/config_shb.yaml")
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    
    evaluate(args)