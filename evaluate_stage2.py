import argparse
import os
import torch
import yaml
import json
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
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    images = torch.stack([item['image'] for item in batch])
    points = [item['points'] for item in batch]
    img_paths = [item['img_path'] for item in batch]
    return images, points, img_paths

def get_fallback_bins():
    """
    Genera bins standard (0-100) per quando il JSON è incompleto.
    Questo serve per i pesi 'Full' (best_mae) che usano ~100 classi.
    """
    print("⚠️  ATTENZIONE: JSON incompleto. Generazione Bins di FALLBACK (Range 0-100)...")
    # Genera bin: [0,0], [1,1], ..., [99,99], [100, inf]
    bins = [[i, i] for i in range(100)] + [[100, float("inf")]]
    
    # Anchor points semplici (valore del bin)
    anchors = [float(b[0]) for b in bins]
    
    return bins, anchors

def load_bins_from_json(reduction, dataset_name):
    """
    Carica i bin dal JSON. Se manca la config FULL, usa il fallback.
    """
    json_path = f"configs/reduction_{reduction}.json"
    
    if not os.path.exists(json_path):
        print(f"⚠️  File {json_path} non trovato. Uso FALLBACK.")
        return get_fallback_bins()
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Mapping nomi
    dataset_mapping = {
        'sha': ['part_a', 'shanghaitech_part_a', 'sha'],
        'shb': ['part_b', 'shanghaitech_part_b', 'shb'],
        'qnrf': ['qnrf', 'ucf_qnrf'],
        'nwpu': ['nwpu', 'nwpu_crowd']
    }
    
    possible_names = dataset_mapping.get(dataset_name, [dataset_name])
    target_cfg = None
    
    # Cerca SOLO configurazioni FULL ("0", "None", "null")
    # Ignora le chiavi "2", "4", "11" presenti nel file, perché sono troncate.
    priority_keys = ["0", "None", "null"] 
    
    for key in priority_keys:
        if key in data:
            for name in possible_names:
                if name in data[key]:
                    target_cfg = data[key][name]
                    print(f"✅ Configurazione JSON trovata: Key='{key}' | Dataset='{name}'")
                    break
        if target_cfg: break
            
    # Cerca alla radice (se non annidato)
    if target_cfg is None:
        for name in possible_names:
            if name in data:
                target_cfg = data[name]
                print(f"✅ Configurazione JSON trovata (Root): Dataset='{name}'")
                break

    # Se non troviamo la config FULL, attiviamo il fallback invece di crashare
    if target_cfg is None:
        print(f"❌ Nessuna configurazione FULL trovata nel JSON per {possible_names}.")
        print("   (Le configurazioni '2', '4', '8', '11' sono troncate e non adatte ai pesi 'best_mae').")
        return get_fallback_bins()

    granularity = "fine"
    bins = target_cfg["bins"][granularity]
    anchor_points = target_cfg["anchor_points"][granularity]["average"]
    
    return bins, anchor_points

def test_clip_standalone(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Avvio Test Stage 2 (Solo CLIP-EBC) su {device}")

    # 1. Config
    cfg = load_config(args.config)
    dataset_name = cfg.get('DATASET', 'sha')
    print(f"📄 Dataset letto dal config: {dataset_name}")
    
    # 2. Bins (JSON o Fallback)
    print(f"📥 Caricamento configurazione per Reduction {args.reduction}...")
    try:
        bins, anchor_points = load_bins_from_json(args.reduction, dataset_name)
        print(f"✅ Bin caricati: {len(bins)}")
    except Exception as e:
        print(f"❌ Errore critico generazione bin: {e}")
        return

    # 3. Dataset
    val_trans = build_transforms(cfg['DATA'], is_train=False)
    dataset = build_dataset(cfg, split=args.split, transforms=val_trans)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=eval_collate)
    print(f"📊 Dataset: {len(dataset)} immagini")

    # 4. Modello
    print(f"🔹 Building Model: CLIP-{args.backbone} | Reduction: {args.reduction}")
    model = get_model(
        backbone=f"clip_{args.backbone}", 
        input_size=224,       
        reduction=args.reduction, 
        bins=bins,            
        anchor_points=anchor_points, 
        prompt_type="word"    
    )

    # 5. Checkpoint
    if args.ckpt:
        print(f"📥 Loading checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)
        
        state_dict = ckpt.get('model', ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt)))
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Load weights
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        
        # Verifica se la testa è stata caricata
        head_missing = [k for k in missing if 'classifier' in k or 'regressor' in k]
        if len(head_missing) > 0:
            print("\n⛔️ WARNING: Pesi della testa NON caricati!")
            print("   Il checkpoint ha un numero di bin diverso da quello generato.")
            print(f"   Bins usati: {len(bins)}")
        else:
            print("✅ Pesi caricati correttamente (Head inclusa).")

    model.to(device)
    model.eval()

    # 6. Eval
    print("running evaluation...")
    results = evaluate(
        model=model,
        data_loader=loader,
        device=device,
        sliding_window=args.sliding_window,
        window_size=args.window_size,
        stride=args.stride
    )

    print("\n" + "="*40)
    print(f"🎯 RISULTATI STAGE 2 - {dataset_name.upper()}")
    print("-" * 40)
    print(f"   MAE  : {results['mae']:.4f}")
    print(f"   RMSE : {results['rmse']:.4f}")
    print("="*40 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--reduction', type=int, default=8)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--sliding_window', action='store_true')
    parser.add_argument('--window_size', type=int, default=224)
    parser.add_argument('--stride', type=int, default=224)
    
    args = parser.parse_args()
    if args.sliding_window and args.stride is None: args.stride = args.window_size
    test_clip_standalone(args)