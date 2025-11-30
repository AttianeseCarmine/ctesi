import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import yaml
import numpy as np

# --- I TUOI IMPORT ---
from models.zip_clip_ebc_model import build_model
from losses import build_stage1_loss
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

@torch.no_grad()
def validate_checkpoint(model, criterion, dataloader, device, config, checkpoint_path):
    """
    Validazione dettagliata Stage 1 (ZIP-CLIP) con diagnostica su Pi e Lambda.
    """
    model.eval()
    total_loss = 0.0
    mae = 0.0
    mse = 0.0
    
    # Per calcolare GT count dai blocchi se necessario
    block_size = config["DATA"]["ZIP_BLOCK_SIZE"]

    print("\n===== DEBUG STAGE 1 (Pi-Head Diagnostics) =====")
    print("Analisi distribuzione di Pi (Probabilità 'Blocco Pieno') e Lambda (Conteggio)")
    print("----------------------------------------------------------------")

    progress_bar = tqdm(dataloader, desc="Validating Stage 1")

    for idx, batch in enumerate(progress_bar):
        # Gestione batch (dict o list)
        if isinstance(batch, dict):
            images = batch['image']
            gt_density = batch['density']
        else:
            images, gt_density = batch[0], batch[1]

        images = images.to(device)
        gt_density = gt_density.to(device)

        # Forward pass completo
        preds = model(images)
        
        # Loss (specifica per Stage 1)
        loss, loss_dict = criterion(preds, gt_density)
        total_loss += loss.item()

        # --- ESTRAZIONE METRICHE DIAGNOSTICHE ---
        
        # 1. Analisi Pi (Probabilità che il blocco NON sia vuoto)
        # Nel tuo modello: pi_prob è già [B, 1, H, W] con valori tra 0 e 1
        pi_prob = preds["pi_prob"] 
        
        pi_mean = pi_prob.mean().item()
        pi_min, pi_max = pi_prob.min().item(), pi_prob.max().item()
        
        # Quanti blocchi sono accesi sopra la soglia 0.1 e 0.5?
        pct_over_01 = (pi_prob > 0.1).float().mean().item() * 100
        pct_over_05 = (pi_prob > 0.5).float().mean().item() * 100

        # 2. Analisi Lambda (Conteggio stimato da CLIP EBC)
        # Nota: In Stage 1 questo potrebbe essere casuale se EBC è congelato/non trainato
        lam_maps = preds["lambda_maps"]
        lam_mean = lam_maps.mean().item()
        lam_min, lam_max = lam_maps.min().item(), lam_maps.max().item()

        # 3. Conteggi totali
        # pred_count del modello usa già (pi * lambda)
        pred_count = preds["pred_count"].item() if preds["pred_count"].numel() == 1 else preds["pred_count"][0].item()
        
        # Calcolo GT count
        gt_count = gt_density.sum().item()

        mae += abs(pred_count - gt_count)
        mse += (pred_count - gt_count) ** 2

        # Log ogni 10 immagini per vedere cosa succede dentro
        if idx % 10 == 0:
            print(f"[IMG {idx:03d}] Pi: [{pi_min:.3f}, {pi_max:.3f}] μ={pi_mean:.3f} (>{pct_over_05:.1f}%) "
                  f"| λ: [{lam_min:.1f}, {lam_max:.1f}] μ={lam_mean:.1f} "
                  f"| Count: Pred={pred_count:.1f} GT={gt_count:.1f}")

        # Aggiorna barra
        postfix = {
            'loss': f"{loss.item():.4f}",
            'pi_avg': f"{pi_mean:.3f}"
        }
        progress_bar.set_postfix(postfix)

    # Statistiche finali
    N = len(dataloader.dataset)
    avg_loss = total_loss / len(dataloader)
    avg_mae = mae / N
    avg_rmse = (mse / N) ** 0.5

    print("\n--- RISULTATI VALIDAZIONE STAGE 1 ---")
    print(f"Checkpoint:   {checkpoint_path}")
    print(f"Dataset:      {config['DATASET']} (split: {config['DATA']['VAL_SPLIT']})")
    print("-------------------------------------")
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Avg Pi (Active): {pi_mean:.3f} (Target ideale ~0.2-0.3)")
    print(f"MAE:             {avg_mae:.2f}")
    print(f"RMSE:            {avg_rmse:.2f}")
    print("-------------------------------------\n")


def main(config_path, checkpoint_path):
    # 1. Carica Config
    config = load_config(config_path)
    device = torch.device(config['DEVICE'])
    init_seeds(config['SEED'])

    print(f"🚀 Avvio Valutazione Stage 1")
    print(f"   Config: {config_path}")

    # 2. Costruisci Modello (ZIP-CLIP-EBC)
    model = build_model(config).to(device)

    # 3. Carica Checkpoint
    if not os.path.isfile(checkpoint_path):
        # Prova a cercare nella cartella di output standard se il path non è assoluto
        out_dir = os.path.join(config['EXP']['OUT_DIR'], config['RUN_NAME'], "stage1")
        alt_path = os.path.join(out_dir, checkpoint_path)
        if os.path.isfile(alt_path):
            checkpoint_path = alt_path
        else:
            print(f"❌ Errore: Checkpoint non trovato in {checkpoint_path}")
            print(f"   Cercato anche in: {alt_path}")
            return

    print(f"✅ Caricamento checkpoint da: {checkpoint_path}")
    
    # Fix per PyTorch 2.6+ (weights_only=False)
    raw_state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Gestione dizionario (se è un checkpoint completo o solo pesi)
    if isinstance(raw_state, dict) and 'model' in raw_state:
        state_dict = raw_state['model']
    else:
        state_dict = raw_state

    # Carica pesi con gestione errori morbida (strict=False utile se cambi architettura leggermente)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"ℹ️ Parametri mancanti (ok se sono solo head inutilizzate): {len(missing)}")
    if unexpected:
        print(f"ℹ️ Parametri inaspettati nel checkpoint: {len(unexpected)}")
    
    print("✅ Modello caricato.")

    # 4. Loss (Factory Function)
    criterion = build_stage1_loss(config).to(device)

    # 5. Dataset e DataLoader
    DatasetClass = get_dataset(config['DATASET'])
    data_cfg = config['DATA']
    val_tf = build_transforms(data_cfg, is_train=False)
    
    val_dataset = DatasetClass(
        root=data_cfg['ROOT'],
        split=data_cfg['VAL_SPLIT'],
        block_size=data_cfg['ZIP_BLOCK_SIZE'],
        transforms=val_tf,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1, # Batch 1 per valutazione accurata
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # 6. Esegui validazione
    validate_checkpoint(model, criterion, val_loader, device, config, checkpoint_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 (ZIP-CLIP-EBC)")
    parser.add_argument("--config", default="config_sha.yaml", help="Path al config file.")
    # Default cerca l'ultimo salvato
    parser.add_argument("--checkpoint", default="experiments/sha_zip_clip_ebc/stage1/last_stage1_model.pth", 
                        help="Path al checkpoint (.pth).")
    
    args = parser.parse_args()
    main(args.config, args.checkpoint)

    #python evaluate_stage1.py --config config_sha.yaml --checkpoint experiments/sha_zip_clip_ebc/stage1/last_stage1_model.pth