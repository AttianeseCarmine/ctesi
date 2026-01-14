import argparse
import os
import torch
import yaml
from torch.utils.data import DataLoader
import sys
import numpy as np

# Import interni
from datasets.builder import build_dataset
from datasets.transforms import build_transforms
from eval import evaluate

# Import esterni (CLIP-EBC)
sys.path.append("external_libs/CLIP-EBC") 

try:
    from models import get_model 
except ImportError:
    print("⚠️ CLIP-EBC non trovato nel path specificato.")
    get_model = None

def load_config(path):
    with open(path, 'r') as f: return yaml.safe_load(f)

def eval_collate(batch):
    """Gestisce il batching per la valutazione."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    images = torch.stack([item['image'] for item in batch])
    points = [item['points'] for item in batch]
    img_paths = [item['img_path'] for item in batch]
    return images, points, img_paths

def test_clip_standalone(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Avvio Test Stage 2 (Solo CLIP-EBC) su {device}")

    # 1. Carica Configurazione
    cfg = load_config(args.config)
    
    # 2. Estrai Parametri Critici dal Config
    # CLIP-EBC ha bisogno di sapere quali bin usare per costruire la testa di classificazione
    if 'BINS' in cfg and 'BIN_CENTERS' in cfg:
        bins = cfg['BINS']
        anchor_points = cfg['BIN_CENTERS']
        print(f"✅ Trovati {len(bins)} bin nel config.")
    else:
        raise ValueError("❌ Errore: BINS o BIN_CENTERS mancanti nel config.yaml!")

    # Leggi parametri architetturali
    clip_head_cfg = cfg.get('CLIP_EBC_HEAD', {})
    reduction = clip_head_cfg.get('REDUCTION', 8) # Default 8 se non specificato
    
    # 3. Dataset
    # Usa le trasformazioni di validation (Resize + Norm CLIP)
    # Assicurati che NORM_MEAN/STD siano quelli di CLIP nel yaml
    val_trans = build_transforms(cfg['DATA'], is_train=False)
    dataset = build_dataset(cfg, split=args.split, transforms=val_trans)
    
    loader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4,
        collate_fn=eval_collate
    )
    
    print(f"📊 Dataset: {len(dataset)} immagini (Split: {args.split})")

    # 4. Costruisci Modello
    print(f"🔹 Building Model: CLIP-{args.backbone} | Reduction: {reduction}")
    
    model = get_model(
        backbone=f"clip_{args.backbone}", 
        input_size=224,       # CLIP standard input size
        reduction=reduction,  # Deve matchare il training (es. 16 o 8)
        bins=bins,            # <--- ORA USIAMO I BIN CORRETTI
        anchor_points=anchor_points # <--- E GLI ANCHOR POINTS CORRETTI
    )

    # 5. Carica Checkpoint
    if args.ckpt:
        print(f"📥 Loading checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)
        
        # Gestione chiavi state_dict
        state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Carica pesi (strict=True per essere sicuri che la testa sia giusta)
        # Se fallisce qui, vuol dire che i BINS nel config non corrispondono a quelli del checkpoint
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if len(missing) > 0:
            print(f"⚠️  Missing keys: {len(missing)}")
            # Se mancano keys della testa (classifier), i risultati saranno sbagliati
    else:
        print("⚠️  ATTENZIONE: Nessun checkpoint specificato! Risultati casuali.")
    
    model.to(device)
    model.eval()

    # 6. Valutazione
    print("running evaluation...")
    
    # Sliding window è fondamentale per immagini grandi (SHA)
    results = evaluate(
        model=model,
        data_loader=loader,
        device=device,
        sliding_window=args.sliding_window,
        window_size=args.window_size,
        stride=args.stride
    )

    # 7. Stampa
    print("\n" + "="*40)
    print(f"🎯 RISULTATI STAGE 2")
    print("-" * 40)
    print(f"   MAE  : {results['mae']:.4f}")
    print(f"   RMSE : {results['rmse']:.4f}")
    print("="*40 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path al config .yaml")
    parser.add_argument('--ckpt', type=str, required=True, help="Path al checkpoint .pth")
    parser.add_argument('--backbone', type=str, default='resnet50', help="resnet50, vit_b_16, etc.")
    parser.add_argument('--split', type=str, default='test', help="val o test")
    
    # Parametri Sliding Window
    parser.add_argument('--sliding_window', action='store_true', help="Attiva sliding window")
    parser.add_argument('--window_size', type=int, default=224, help="Size della patch (224 per CLIP)")
    parser.add_argument('--stride', type=int, default=224, help="Stride (uguale a window_size per non sovrapporre)")
    
    args = parser.parse_args()
    
    if args.sliding_window and args.stride is None:
        args.stride = args.window_size
        
    test_clip_standalone(args)