import argparse
import torch
import os
import numpy as np
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional

# --- Import specifici del tuo repository ---
# Assicurati che questi file esistano nelle cartelle indicate
from utils import calculate_errors, sliding_window_predict
from models import get_model
from datasets import Crowd  

def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    sliding_window: bool = False,
    window_size: Optional[int] = None,
    stride: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    pred_counts, target_counts = [], []
    
    if sliding_window:
        if window_size is None or stride is None:
            raise ValueError("Window size e stride devono essere specificati per sliding_window=True")

    print(f"Inizio valutazione su {len(data_loader)} immagini...")
    
    # tqdm per barra di progresso
    for i, batch in enumerate(tqdm(data_loader)):
        # Gestione robusta del batch unpack
        image = batch[0]
        target_points = batch[1]
        
        image = image.to(device)
        
        # Ground Truth: conta il numero di punti per ogni immagine nel batch
        target_counts.append([len(p) for p in target_points])

        with torch.set_grad_enabled(False):
            if sliding_window:
                # Predizione con finestra scorrevole per immagini grandi (es. ShanghaiTech)
                pred_density = sliding_window_predict(model, image, window_size, stride)
            else:
                # Predizione diretta
                pred_density = model(image)

            # Somma della mappa di densità per ottenere il conteggio totale
            # clip_ebc ritorna spesso [Batch, Bins, H, W], dobbiamo sommare o prendere l'output corretto
            # Solitamente per CLIP-EBC l'output finale è già una mappa di densità o logit.
            # Se il modello ritorna la densità prevista:
            pred_counts.append(pred_density.sum(dim=(1, 2, 3)).cpu().numpy().tolist())

    # Appiattiamo le liste (handling per batch_size > 1)
    pred_counts = np.array([item for sublist in pred_counts for item in sublist])
    target_counts = np.array([item for sublist in target_counts for item in sublist])
    
    assert len(pred_counts) == len(target_counts), f"Mismatch: {len(pred_counts)} preds vs {len(target_counts)} targets"
    
    metrics = calculate_errors(pred_counts, target_counts)
    return metrics
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Valutazione CLIP-EBC')
    
    # Dati
    parser.add_argument('--dataset', default='sha', help='dataset name: sha, shb, qnrf, jhu')
    parser.add_argument('--data-dir', required=True, help='percorso root del dataset')
    
    # Modello
    parser.add_argument('--model', default='clip_vit_b_16', help='backbone type')
    parser.add_argument('--reduction', type=int, default=8, help='reduction factor (8, 16, 32)')
    
    # TRUNCATION: Aggiunto perché serve per navigare il JSON (es. "4" per SHA)
    parser.add_argument('--truncation', type=int, default=4, help='Truncation level (top key in json)')
    
    parser.add_argument('--config-dir', default='./configs', help='cartella dove sono i json reduction_X.json')
    parser.add_argument('--prompt-type', default='word')
    
    # Input
    parser.add_argument('--input-size', type=int, default=224, help='dimensione crop input network')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Checkpoint
    parser.add_argument('--resume', required=True, help='percorso checkpoint .pth')
    parser.add_argument('--gpu', default='0', help='id gpu')
    
    # Sliding Window
    parser.add_argument('--sliding_window', action='store_true')
    parser.add_argument('--window-size', type=int, default=224)
    parser.add_argument('--stride', type=int, default=224)

    args = parser.parse_args()

    # 1. Setup Device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando device: {device}")

    # 2. CARICAMENTO CONFIGURAZIONE JSON (BINS & ANCHOR POINTS)
    config_file = os.path.join(args.config_dir, f"reduction_{args.reduction}.json")
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"File config non trovato: {config_file}")
    
    print(f"Caricamento configurazione da: {config_file}")
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    ds_key = args.dataset.lower() # es. 'sha'
    trunc_key = str(args.truncation) # es. '4'

    # --- LOGICA DI RICERCA NEL JSON AGGIORNATA ---
    found_config = None

    # Caso 1: Struttura annidata (Truncation -> Dataset) [Quella che hai tu]
    if trunc_key in config_data and ds_key in config_data[trunc_key]:
        print(f"Configurazione trovata sotto truncation '{trunc_key}'")
        found_config = config_data[trunc_key][ds_key]
    
    # Caso 2: Struttura piatta (Solo Dataset)
    elif ds_key in config_data:
        found_config = config_data[ds_key]
        
    # Caso 3: Fallback (Cerca il dataset in qualsiasi chiave numerica)
    else:
        print(f"Warning: Chiave truncation '{trunc_key}' non trovata. Cerco ovunque...")
        for k in config_data:
            if isinstance(config_data[k], dict) and ds_key in config_data[k]:
                print(f"Configurazione trovata sotto chiave '{k}'")
                found_config = config_data[k][ds_key]
                break
    
    if found_config is None:
        raise KeyError(f"Impossibile trovare i parametri per '{ds_key}' nel JSON. Controlla il nome del dataset o la struttura del file.")

    # Estrazione Bins e Anchors
    # Solitamente dentro c'è un'altra chiave tipo "fine", "coarse", o "dynamic". CLIP-EBC usa "fine" di default.
    raw_bins = found_config['bins']
    raw_anchors = found_config['anchor_points']

    # Se 'bins' è un dizionario, prendi la chiave 'fine' (o la prima disponibile)
    if isinstance(raw_bins, dict):
        key_type = 'fine' if 'fine' in raw_bins else list(raw_bins.keys())[0]
        bins_list = raw_bins[key_type]
        anchor_points_list = raw_anchors[key_type]
        if isinstance(anchor_points_list, dict): # A volte anchor è ulteriormente annidato (middle/average)
             anchor_points_list = anchor_points_list.get('average', anchor_points_list.get('middle'))
    else:
        bins_list = raw_bins
        anchor_points_list = raw_anchors

    print(f"Bins caricati: {len(bins_list)} intervalli")

    # 3. Inizializzazione Modello
    print(f"Costruzione modello {args.model}...")
    
    model = get_model(
        backbone=args.model,
        input_size=args.input_size,
        reduction=args.reduction,
        bins=bins_list,
        anchor_points=anchor_points_list,
        prompt_type=args.prompt_type,
        # Parametri CLIP specifici
        num_vpt=32,
        vpt_drop=0.0,
        deep_vpt=True
    )

    # 4. Caricamento Pesi
    print(f"Caricamento pesi da {args.resume}...")
    checkpoint = torch.load(args.resume, map_location=device)
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Pulizia prefisso 'module.'
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(device)

    # 5. Dataset
    print("Caricamento Dati...")
    val_dataset = Crowd(
        dataset=args.dataset, 
        root=args.data_dir, 
        split='val', 
        crop_size=args.input_size,
        reduction=args.reduction,
        method='val'
    )
    
    data_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )

    # 6. Esecuzione
    results = evaluate(
        model=model,
        data_loader=data_loader,
        device=device,
        sliding_window=args.sliding_window,
        window_size=args.window_size,
        stride=args.stride
    )

    print("\n" + "="*40)
    print(f"RISULTATI FINALI:")
    print(f"MAE: {results['mae']:.2f}")
    print(f"MSE: {results['mse']:.2f}")
    print("="*40 + "\n")