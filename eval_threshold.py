#!/usr/bin/env python3
import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F  # Assicurati di avere questo import all'inizio
# Importiamo direttamente le classi dei modelli base
from models.zip_model import ZIPModel
from models.clip_ebc_model import CLIPEBCModel
from datasets.sha import SHA
from datasets.transforms import build_transforms

# --- DEFINIAMO LA PIPELINE LOCALMENTE (Per evitare errori di import) ---
class Stage3Pipeline(nn.Module):
    def __init__(self, model_s1, model_s2, bin_centers, threshold=0.5):
        super().__init__()
        self.model_s1 = model_s1
        self.model_s2 = model_s2
        self.register_buffer('bin_centers', torch.tensor(bin_centers, dtype=torch.float32))
        self.threshold = threshold

    def forward(self, x):
        # Forward Stage 1 (ZIP)
        out1 = self.model_s1(x)
        
        # Forward Stage 2 (CLIP)
        out2 = self.model_s2(x)
        
        return {**out1, **out2}

def crowd_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'points': [item['points'] for item in batch]
    }
def evaluate_with_threshold(model, loader, device, threshold):
    model.eval()
    total_mae = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            if batch is None: continue
            
            img = batch['image'].to(device)
            # Gestione batch size > 1
            gt_counts = [len(p) for p in batch['points']]
            
            # Forward
            out = model(img)
            
            # 1. Recupera Probabilità Maschera (Stage 1 - Bassa Risoluzione 32x32)
            if 'pi' in out:
                prob_map = out['pi'] 
            else:
                prob_map = torch.sigmoid(out['pi_logits'])

            # 2. Recupera Densità (Stage 2 - Alta Risoluzione 64x64)
            density_map = out['ebc_density']
            
            # --- FIX CRITICA: Interpolazione per allineare le risoluzioni ---
            # Portiamo la maschera da 32x32 alla stessa dimensione della densità (64x64)
            if prob_map.shape[-2:] != density_map.shape[-2:]:
                prob_map = F.interpolate(
                    prob_map,
                    size=density_map.shape[-2:], # Target: dimensione della densità
                    mode='bilinear',
                    align_corners=False
                )
            # ----------------------------------------------------------------

            # 3. Applica Soglia (Logica Inversa Standard ZIP)
            # Se prob_map è la probabilità di "zero" (sfondo), allora:
            # - Valori alti (>0.5) = Sfondo sicuro
            # - Valori bassi (<0.5) = Possibile Persona
            # Noi teniamo i pixel dove la probabilità di essere sfondo è BASSA.
            # Esempio: Threshold 0.6 -> Teniamo tutto ciò che ha prob_sfondo < 0.4
            
            # Nota: Regola questo (1.0 - threshold) o solo (threshold) in base 
            # a come il tuo ZIPHead è stato addestrato.
            # Prova prima questa logica standard:
            mask = (prob_map < (1.0 - threshold)).float()

            # 4. Calcola MAE Finale
            final_density = density_map * mask
            pred_counts = final_density.sum(dim=(1, 2, 3))
            
            for pred, gt in zip(pred_counts, gt_counts):
                total_mae += abs(pred.item() - gt)
                num_samples += 1
            
    return total_mae / num_samples if num_samples > 0 else 0

def main():
    parser = argparse.ArgumentParser()
    # Default paths - modificali se necessario
    parser.add_argument('--config', type=str, default="configs/config_shb.yaml")
    parser.add_argument('--checkpoint', type=str, default="checkpoints/shb/stage3/best_model.pth")
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}')
    
    print(f"📂 Config: {args.config}")
    print(f"📂 Model: {args.checkpoint}")
    
    # 1. Carica Configurazione
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    
    # 2. Inizializza Modello
    print("🏗️  Costruzione modello...")
    model_s1 = ZIPModel(config)
    model_s2 = CLIPEBCModel(config)
    
    # Wrapper
    model = Stage3Pipeline(model_s1, model_s2, config['BIN_CENTERS']).to(device)
    
    # 3. Carica Pesi
    if os.path.exists(args.checkpoint):
        print("🔄 Caricamento pesi...")
        ckpt = torch.load(args.checkpoint, map_location=device)
        # Gestione caso in cui il checkpoint contiene 'model' o è flat
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        
        # Carica ignorando eventuali mismatch minori (strict=False)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"⚠️ Warning caricamento (probabile mismatch nomi): {e}")
            print("Tentativo di caricamento con strict=False...")
            model.load_state_dict(state_dict, strict=False)
            
        print("✅ Pesi caricati.")
    else:
        print(f"❌ Errore: Checkpoint non trovato in {args.checkpoint}")
        return

    # 4. Dataset
    print("📊 Caricamento Dataset Val...")
    val_ds = SHA(config['DATA']['ROOT'], 'val', build_transforms(config['DATA'], False))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=crowd_collate)
    
    # 5. Loop di Valutazione Soglie
    print("\n🔎 --- INIZIO RICERCA SOGLIA OTTIMALE ---")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    best_th = 0
    best_mae = float('inf')
    
    print(f"{'Threshold':<10} | {'MAE':<10}")
    print("-" * 25)
    
    for th in thresholds:
        mae = evaluate_with_threshold(model, val_loader, device, th)
        print(f"{th:<10.2f} | {mae:<10.4f}")
        
        if mae < best_mae:
            best_mae = mae
            best_th = th
            
    print("-" * 25)
    print(f"🏆 BEST RESULT: MAE = {best_mae:.4f} @ Threshold = {best_th}")
    print(f"   (Confronta con il tuo precedente best: 16.31)")

if __name__ == '__main__':
    main()