import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import yaml
import numpy as np
import math

# --- I TUOI IMPORT ---
from models.zip_clip_ebc_model import build_model
from losses import build_stage3_loss
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def sliding_window_predict(model, image, window_size=448, stride=336, device="cuda"):
    """
    Esegue la predizione usando una finestra scorrevole per mantenere l'alta risoluzione.
    Gestisce automaticamente il padding e la media nelle sovrapposizioni.
    """
    B, _, H, W = image.shape
    assert B == 1, "Sliding window supporta solo batch_size=1 per ora."

    # Fattore di downsampling del modello (CLIP ViT-B/16 = 16)
    downsample_ratio = 16 
    
    # Se l'immagine è più piccola della finestra, pad
    pad_h = 0
    pad_w = 0
    if H < window_size or W < window_size:
        pad_h = max(0, window_size - H)
        pad_w = max(0, window_size - W)
        # Pad: (left, right, top, bottom)
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
        H, W = image.shape[2], image.shape[3]

    # Dimensioni output attese
    h_out = H // downsample_ratio
    w_out = W // downsample_ratio
    
    # Canvas per accumulare densità e conteggi (per la media)
    density_map = torch.zeros((1, 1, h_out, w_out), device=device)
    count_map = torch.zeros((1, 1, h_out, w_out), device=device)
    
    # Calcola griglia
    h_steps = math.ceil((H - window_size) / stride) + 1
    w_steps = math.ceil((W - window_size) / stride) + 1
    
    # print(f"   -> Sliding Window: {H}x{W} -> {h_steps}x{w_steps} patches")

    for h in range(h_steps):
        for w in range(w_steps):
            # Calcola coordinate crop (gestione bordo destro/inferiore)
            # Se l'ultimo step sfora, torniamo indietro per allinearci al bordo
            y1 = h * stride
            x1 = w * stride
            
            if y1 + window_size > H:
                y1 = H - window_size
            if x1 + window_size > W:
                x1 = W - window_size
                
            y2 = y1 + window_size
            x2 = x1 + window_size
            
            # Estrai crop
            crop = image[:, :, y1:y2, x1:x2].to(device)
            
            # Forward pass
            with torch.no_grad():
                # Modalità standard (non restituisce intermedi per risparmiare memoria)
                preds = model(crop)
                # Output density è [1, 1, window/16, window/16] -> [1, 1, 28, 28]
                crop_pred = preds["density_map"]
            
            # Calcola coordinate nel canvas di output (scalate)
            y1_out = y1 // downsample_ratio
            x1_out = x1 // downsample_ratio
            y2_out = y1_out + crop_pred.shape[2]
            x2_out = x1_out + crop_pred.shape[3]
            
            # Accumula
            density_map[:, :, y1_out:y2_out, x1_out:x2_out] += crop_pred
            count_map[:, :, y1_out:y2_out, x1_out:x2_out] += 1.0
            
    # Media nelle sovrapposizioni
    final_density = density_map / torch.clamp(count_map, min=1.0)
    
    # Rimuovi padding se aggiunto
    if pad_h > 0 or pad_w > 0:
        orig_h_out = (H - pad_h) // downsample_ratio
        orig_w_out = (W - pad_w) // downsample_ratio
        final_density = final_density[:, :, :orig_h_out, :orig_w_out]

    # Restituisce somma, mappa densità completa, e media attivazione pi (opzionale, qui non calcolata per patch)
    return final_density.sum(), final_density

@torch.no_grad()
def validate_stage3(model, criterion, dataloader, device, config, checkpoint_path):
    model.eval()
    
    # Sovrascrivi la soglia Pi per l'inferenza (più aggressiva per recuperare folla)
    if hasattr(model, 'pi_thresh'):
        model.pi_thresh = 0.90 # Valore suggerito per recuperare folla persa
        print(f"⚡ Override Inference: PI_THRESH set to {model.pi_thresh}")

    total_mae = 0.0
    total_mse = 0.0
    
    # Metriche differenziate
    zero_mae = 0.0
    zero_samples = 0
    crowd_mae = 0.0
    crowd_samples = 0
    
    # Configurazione Sliding Window
    # Usa la crop size del training come window size
    win_size = config['DATA'].get('CROP_SIZE', 448)
    # Stride: 50% o 75% della window size (es. 336 è 75% di 448)
    stride = int(win_size * 0.50)  
    
    # Override Soglia Ottimale (Sperimentale)
    if hasattr(model, 'pi_thresh'):
        model.pi_thresh = 0.22
    print("\n===== DIAGNOSTICA STAGE 3 (Joint + Sliding Window) =====")
    print(f"Window Size: {win_size}x{win_size} | Stride: {stride}")
    print("Monitoraggio: Performance Globale + Capacità di soppressione background (Zero-MAE)")
    
    progress_bar = tqdm(dataloader, desc="Eval Stage 3")

    for idx, batch in enumerate(progress_bar):
        if isinstance(batch, dict):
            images = batch['image']
            gt_density = batch['density']
        else:
            images, gt_density = batch[0], batch[1]

        # Nota: immagini non spostate su device subito, lo fa sliding_window a pezzi
        # gt_density serve su device per il confronto
        gt_density = gt_density.to(device)

        # --- SLIDING WINDOW PREDICTION ---
        pred_count, _ = sliding_window_predict(model, images, window_size=win_size, stride=stride, device=device)
        pred_count = pred_count.item()
        
        # GT Count
        gt_count = gt_density.sum().item()
        err = abs(pred_count - gt_count)
        
        total_mae += err
        total_mse += (pred_count - gt_count) ** 2
        
        # Analisi differenziata
        if gt_count < 1.0:
            zero_mae += err
            zero_samples += 1
            type_str = "🟢 EMPTY"
        else:
            crowd_mae += err
            crowd_samples += 1
            type_str = "🔴 CROWD"

        if idx % 20 == 0:
            # Nota: Pi-Act non è facilmente calcolabile in sliding window senza salvare tutto, 
            # quindi lo omettiamo o stampiamo NaN per velocità
            print(f"[IMG {idx:03d}] {type_str} | GT: {gt_count:6.1f} | Pred: {pred_count:6.1f} | Err: {err:5.1f}")

    # Aggregazione
    N = len(dataloader.dataset)
    avg_mae = total_mae / N
    avg_rmse = (total_mse / N) ** 0.5
    
    avg_zero_mae = zero_mae / max(zero_samples, 1)
    avg_crowd_mae = crowd_mae / max(crowd_samples, 1)

    print("\n" + "="*50)
    print(f"🚀 RISULTATI FINALI STAGE 3 (SLIDING WINDOW) - {os.path.basename(checkpoint_path)}")
    print("="*50)
    print(f"Global MAE:        {avg_mae:.3f}")
    print(f"Global RMSE:       {avg_rmse:.3f}")
    print("-" * 30)
    print(f"Analisi:")
    print(f"   MAE su Vuoti:    {avg_zero_mae:.3f}")
    print(f"   MAE su Folla:    {avg_crowd_mae:.3f}")
    print("="*50 + "\n")

def main(config_path, checkpoint_path):
    config = load_config(config_path)
    device = torch.device(config['DEVICE'])
    init_seeds(config['SEED'])

    print(f"🚀 Avvio Valutazione Stage 3 (Joint)")
    model = build_model(config).to(device)

    # Cerca il checkpoint stage 3
    if not checkpoint_path:
        base_dir = os.path.join(config['EXP']['OUT_DIR'], config['RUN_NAME'], "stage3")
        checkpoint_path = os.path.join(base_dir, "best_stage3_model.pth")
    
    print(f"📂 Loading: {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
    else:
        print("⚠️ Checkpoint non trovato. Valutazione modello non inizializzato.")
        return

    # Loss non serve per validazione pura MAE/RMSE, ma la carichiamo per compatibilità
    criterion = build_stage3_loss(config).to(device)
    
    DatasetClass = get_dataset(config['DATASET'])
    
    # IMPORTANTE: Per sliding window, vogliamo l'immagine originale o resize minimo, non crop
    # Modifichiamo le trasformazioni di validazione per non fare resize drastici se possibile
    # Tuttavia, il tuo build_transforms attuale probabilmente fa resize. 
    # L'ideale è avere un dataloader che restituisce l'immagine intera.
    val_tf = build_transforms(config['DATA'], is_train=False)
    
    val_dataset = DatasetClass(
        root=config['DATA']['ROOT'],
        split=config['DATA']['VAL_SPLIT'],
        block_size=config['DATA']['ZIP_BLOCK_SIZE'],
        transforms=val_tf,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    validate_stage3(model, criterion, val_loader, device, config, checkpoint_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_sha.yaml")
    parser.add_argument("--checkpoint", default="", help="Override path checkpoint")
    args = parser.parse_args()
    main(args.config, args.checkpoint)