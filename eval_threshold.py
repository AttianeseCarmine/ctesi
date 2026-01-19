import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Import dal tuo progetto
from utils import get_dataloader, setup, init_seeds
from models.zip_model import ZIPModel
from models import get_model 

# =============================================================================
# WRAPPER CLASS
# =============================================================================
class InferencePipeline(nn.Module):
    def __init__(self, config, stage1_path, stage2_path):
        super().__init__()
        
        # --- 1. Carica Modello Stage 1 (ZIP / Mask) ---
        print(f"🔹 Loading Stage 1 (Mask) from: {stage1_path}")
        self.stage1 = ZIPModel(config)
        
        # Carica pesi Stage 1
        ckpt1 = torch.load(stage1_path, map_location='cpu')
        state1 = ckpt1['model_state_dict'] if 'model_state_dict' in ckpt1 else ckpt1
        self.stage1.load_state_dict(state1, strict=False)
        
        # --- 2. Carica Modello Stage 2 (Density / CLIP-EBC) ---
        print(f"🔹 Loading Stage 2 (Density) from: {stage2_path}")
        
        backbone_type = config.get('model', 'vit_b_16')
        inp_size = config.get('input_size', 224)
        reduction = config.get('reduction', 8)
        
        # --- FIX DEFINITIVO PER BINS E ANCHOR POINTS ---
        truncation = config.get('truncation', 4) 
        
        # Il modello vuole una LISTA DI TUPLE per i bin, non un tensore.
        # Creiamo dei bin "puntiformi" [(0,0), (1,1), (2,2)...] che rappresentano i conteggi interi.
        # Questo soddisfa il requisito "len(b) == 2".
        bins = [(float(i), float(i)) for i in range(truncation + 1)]
        
        # Gli anchor points devono corrispondere. Usiamo un tensore semplice.
        anchor_points = torch.tensor([float(i) for i in range(truncation + 1)])
        
        self.stage2 = get_model(
            backbone=backbone_type,
            input_size=inp_size,
            reduction=reduction,
            bins=bins,           # <--- Ora è una lista di tuple [(0,0), (1,1)...]
            anchor_points=anchor_points, # <--- Tensore lunghezza 5
            prompt_type=config.get('prompt_type', 'word'),
            num_vpt=config.get('num_vpt', 32),
            vpt_drop=config.get('vpt_drop', 0.0),
            deep_vpt=not config.get('shallow_vpt', False)
        )
        
        # Carica pesi Stage 2
        ckpt2 = torch.load(stage2_path, map_location='cpu')
        state2 = ckpt2['model_state_dict'] if 'model_state_dict' in ckpt2 else ckpt2
        state2 = {k.replace('module.', ''): v for k, v in state2.items()}
        
        self.stage2.load_state_dict(state2, strict=False)

    def forward(self, x):
        # Forward Stage 1
        out1 = self.stage1(x)
        logits_mask = out1['pi_logits'] if isinstance(out1, dict) else out1
        
        # Forward Stage 2
        out2 = self.stage2(x)
        # Gestione output CLIP-EBC (può essere tupla o tensore)
        if isinstance(out2, tuple):
            _, density_map = out2
        elif isinstance(out2, dict):
            density_map = out2.get('density', out2.get('ebc_density'))
        else:
            density_map = out2
            
        return logits_mask, density_map

# =============================================================================
# EVALUATION LOGIC
# =============================================================================
def evaluate_thresholds(model, loader, device, thresholds):
    model.eval()
    
    mae_results = {th: 0.0 for th in thresholds}
    mse_results = {th: 0.0 for th in thresholds}
    num_samples = 0
    
    print("🚀 Running Inference...")
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            # Gestione input eterogenei (dict o tuple)
            if isinstance(batch, dict):
                images = batch['image'].to(device)
                gt_points = batch['points']
            else:
                images, gt_points, _ = batch
                images = images.to(device)
            
            # Conta persone reali (Ground Truth)
            gt_counts = [len(p) for p in gt_points]

            # Forward pass
            logits_mask, density_map = model(images)
            
            # Calcola probabilità (Sigmoide sui logits)
            prob_mask = torch.sigmoid(logits_mask)

            # --- ALLINEAMENTO DIMENSIONI ---
            # Interpolazione: Porta la maschera (es. 32x32) alla dimensione della densità (es. 56x56)
            if prob_mask.shape[-2:] != density_map.shape[-2:]:
                prob_mask = F.interpolate(
                    prob_mask,
                    size=density_map.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            # --- LOOP SULLE SOGLIE ---
            # Calcoliamo i risultati per tutte le soglie in una sola passata
            for th in thresholds:
                # Logica ZIP: Se prob > th consideriamo "sfondo" o "folla"?
                # DIPENDE da come è stato addestrato Stage 1.
                # Caso A: Output è probabilità di "Zero/Sfondo". -> Mask = (prob < th)
                # Caso B: Output è probabilità di "Folla".       -> Mask = (prob > th)
                # ZIP solitamente predice p(Zero), quindi usiamo (prob < th) per tenere la folla.
                # Se i risultati sono terribili, prova a invertire la disuguaglianza.
                
                # Assumiamo Caso A (Standard ZIP): p = probabilità di essere vuoto.
                # Vogliamo mantenere i pixel dove la probabilità di essere vuoto è BASSA.
                # Esempio th=0.8 -> Teniamo tutto ciò che ha prob_vuoto < 0.2 (molto severo) o < 0.8 (lasco)?
                # Solitamente si usa: mask = (prob_sfondo < threshold)
                
                mask = (prob_mask < (1.0 - th)).float() 

                final_density = density_map * mask
                pred_counts = final_density.sum(dim=(1, 2, 3))
                
                for pred, gt in zip(pred_counts, gt_counts):
                    err = abs(pred.item() - gt)
                    mae_results[th] += err
                    mse_results[th] += err ** 2
            
            num_samples += len(images)

    # Aggregazione finale
    final_results = []
    for th in thresholds:
        mae = mae_results[th] / num_samples
        mse = (mse_results[th] / num_samples) ** 0.5
        final_results.append((th, mae, mse))
        
    return final_results

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    # Argomenti principali
    parser.add_argument('--config', type=str, required=True, help="Path al config YAML")
    parser.add_argument('--dataset', type=str, required=True, help="Nome dataset (es: sha)")
    
    # Path modelli
    parser.add_argument('--ckpt_stage1', type=str, required=True)
    parser.add_argument('--ckpt_stage2', type=str, required=True)
    
    # Parametri tecnici
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    
    # Setup Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️  Using device: {device}")

    # 1. Caricamento Configurazione
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # 2. Creazione Namespace per get_dataloader
    # Convertiamo il dict in Namespace e sovrascriviamo con argomenti da terminale se necessario
    args_ds = argparse.Namespace(**config_dict)
    
    # Forziamo parametri specifici per la validazione
    args_ds.dataset = args.dataset
    args_ds.batch_size = 1        # Batch size 1 è più sicura per eval (immagini dimensioni diverse)
    args_ds.sliding_window = False 
    args_ds.local_rank = -1       # Disabilita DDP
    args_ds.distributed = False
    
    # Assicuriamo che esistano chiavi che get_dataloader potrebbe cercare
    if not hasattr(args_ds, 'num_workers'): args_ds.num_workers = 4
    if not hasattr(args_ds, 'data_dir'): args_ds.data_dir = './data'

    # 3. Dataloader
    # IMPORTANTE: Se non hai la cartella 'val', cambia 'val' in 'test' qui sotto
    split_name = 'val' 
    print(f"📊 Loading Data split: {split_name} for {args.dataset}...")
    
    try:
        val_loader = get_dataloader(args_ds, split=split_name, ddp=False)
    except Exception as e:
        print(f"⚠️  Errore caricamento 'val': {e}")
        print("🔄 Provo con split 'test'...")
        val_loader = get_dataloader(args_ds, split='test', ddp=False)

    # 4. Inizializzazione Pipeline
    model = InferencePipeline(config_dict, args.ckpt_stage1, args.ckpt_stage2).to(device)

    # 5. Esecuzione Test
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"\n🔎 Testing Thresholds: {thresholds}")
    
    results = evaluate_thresholds(model, val_loader, device, thresholds)
    
    # 6. Report
    print("\n" + "="*45)
    print(f"{'Threshold':<10} | {'MAE':<10} | {'MSE':<10}")
    print("-" * 45)
    
    best_res = None
    best_mae = float('inf')
    
    for th, mae, mse in results:
        print(f"{th:<10.2f} | {mae:<10.4f} | {mse:<10.4f}")
        if mae < best_mae:
            best_mae = mae
            best_res = (th, mae, mse)
            
    print("="*45)
    print(f"🏆 BEST RESULT:")
    print(f"   Threshold: {best_res[0]}")
    print(f"   MAE:       {best_res[1]:.4f}")
    print(f"   MSE:       {best_res[2]:.4f}")
    print("="*45)

if __name__ == '__main__':
    main()