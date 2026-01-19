import argparse
import yaml
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- IMPORTS CORRETTI ---
from models.zip_model import ZIPModel
from datasets.transforms import build_transforms
# Sostituiamo datasets.builder con l'import diretto della classe
from datasets.crowd import Crowd 

# --- FUNZIONE BUILD_DATASET LOCALE (Per fixare l'errore) ---
def build_dataset(config, split, transforms):
    """
    Costruisce il dataset in base alla configurazione.
    Adattato per funzionare con la classe Crowd generica.
    """
    dataset_name = config.get('dataset', 'sha').lower()
    data_dir = config.get('data_dir', './data')
    
    # Determina la root corretta in base al dataset
    if dataset_name == 'sha':
        root = os.path.join(data_dir, 'ShanghaiTech/part_A')
    elif dataset_name == 'shb':
        root = os.path.join(data_dir, 'ShanghaiTech/part_B')
    elif dataset_name == 'qnrf':
        root = os.path.join(data_dir, 'UCF-QNRF_ECCV18')
    else:
        # Fallback generico
        root = os.path.join(data_dir, dataset_name)

    # Istanzia la classe Crowd (che gestisce SHA, SHB, QNRF)
    # Nota: Assumiamo che Crowd prenda (root, split, transforms, ...)
    # Se il costruttore ha parametri diversi, adattali qui.
    dataset = Crowd(
        root=root,
        split=split,
        transforms=transforms,
        return_filename=True # Utile per debug, se supportato
    )
    return dataset

# --- COLLAGE FUNCTION ---
def crowd_collate(batch):
    """Gestisce batch con immagini di dimensioni diverse"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    res = {}
    if "image" in batch[0]:
        res["image"] = torch.stack([item["image"] for item in batch])
    if "density" in batch[0]:
        res["density"] = torch.stack([item["density"] for item in batch])
    if "points" in batch[0]:
        res["points"] = [item["points"] for item in batch]
    if "img_path" in batch[0]: # Gestione filename se presente
        res["img_path"] = [item["img_path"] for item in batch]
        
    return res

# --- EVALUATION LOGIC ---
@torch.no_grad()
def evaluate(model, loader, device, threshold=0.2):
    model.eval()

    tp, tn, fp, fn = 0, 0, 0, 0
    total_samples = 0

    for batch in tqdm(loader, desc=f"Evaluating (thr={threshold})", leave=False):
        if batch is None: continue

        images = batch["image"].to(device)
        gt_density = batch["density"].to(device)

        # 1. Forward
        outputs = model(images)
        pi_logits = outputs["pi_logits"]
        probs = torch.sigmoid(pi_logits)

        # 2. Ground Truth Binaria
        h_out, w_out = pi_logits.shape[2:]
        scale_factor = (images.shape[2] * images.shape[3]) / (h_out * w_out)
        gt_down = F.adaptive_avg_pool2d(gt_density, (h_out, w_out)) * scale_factor
        
        # Binary Mask: >0.001 è folla
        gt_binary = (gt_down > 0.001).float()

        # 3. Predizione
        pred_binary = (probs > threshold).float()

        # 4. Statistiche
        tp += ((pred_binary == 1) & (gt_binary == 1)).sum().item()
        tn += ((pred_binary == 0) & (gt_binary == 0)).sum().item()
        fp += ((pred_binary == 1) & (gt_binary == 0)).sum().item()
        fn += ((pred_binary == 0) & (gt_binary == 1)).sum().item()
        
    # Metriche
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {"threshold": threshold, "f1": f1, "accuracy": accuracy, 
            "precision": precision, "recall": recall}

def print_results(results):
    print("\n" + "=" * 60)
    print(f"{'Thr':<5} | {'F1':<8} | {'Acc':<8} | {'Prec':<8} | {'Rec':<8}")
    print("-" * 60)
    for r in results:
        print(f"{r['threshold']:<5.2f} | {r['f1']:.2%} | {r['accuracy']:.2%} | {r['precision']:.2%} | {r['recall']:.2%}")
    print("=" * 60)
    
    best = max(results, key=lambda x: x["f1"])
    print(f"🏆 BEST F1: {best['f1']:.2%} @ Threshold {best['threshold']}")

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    
    args = parser.parse_args()
    device = torch.device(args.device)

    # 1. Config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"🏗️  Loading Model: {config.get('model', 'unknown')}")
    model = ZIPModel(config).to(device)

    # 2. Load Weights
    print(f"📥 Loading Checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    # 3. Dataset
    # Mock config per transforms (vecchia struttura)
    mean_val = config.get('norm_mean') or config.get('NORM_MEAN') or [0.485, 0.456, 0.406]
    std_val = config.get('norm_std') or config.get('NORM_STD') or [0.229, 0.224, 0.225]
    input_size = config.get('input_size') or config.get('INPUT_SIZE') or 224

    # COSTRUZIONE DIZIONARIO CORRETTO
    data_cfg = {
        'INPUT_SIZE': input_size,
        'NORM_MEAN': mean_val,  # <--- La chiave corretta è questa!
        'NORM_STD': std_val     # <--- La chiave corretta è questa!
    }
    
    val_transforms = build_transforms(data_cfg, is_train=False)
    
    # Try loading 'val', fallback to 'test'
    split = 'val'
    try:
        dataset = build_dataset(config, split, val_transforms)
        # Test rapido per vedere se il path esiste davvero
        if len(dataset) == 0: raise FileNotFoundError("Dataset vuoto")
    except Exception:
        print(f"⚠️  Split '{split}' non trovato/vuoto. Provo 'test'...")
        split = 'test'
        dataset = build_dataset(config, split, val_transforms)

    print(f"📊 Dataset: {split.upper()} - {len(dataset)} immagini")
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                        num_workers=4, collate_fn=crowd_collate)

    # 4. Sweep
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    
    print("🚀 Starting Eval...")
    for th in thresholds:
        res = evaluate(model, loader, device, th)
        results.append(res)
        
    print_results(results)

if __name__ == "__main__":
    main()