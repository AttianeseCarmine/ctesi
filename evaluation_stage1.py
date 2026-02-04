import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
from tqdm import tqdm

# --- IMPORT MODELLI E UTILS ---
try:
    from utils.data_utils import get_dataloader
except ImportError:
    try:
        sys.path.append(os.getcwd())
        from utils.data_utils import get_dataloader
    except ImportError:
        from utils import get_dataloader

from models.zip_model import ZIPModel

# --- CLASSE MOCK ARGOMENTI ---
class ConfigArgs:
    """
    Classe necessaria per convertire il dizionario di config in un oggetto
    compatibile con get_dataloader (che si aspetta args.parametro).
    """
    def __init__(self, config, cli_args):
        self.dataset = 'sha'
        self.data_dir = './data'
        self.batch_size = 1
        self.num_workers = 4
        self.sliding_window = False
        self.resize_to_multiple = False
        self.zero_pad_to_multiple = False
        self.regression = False
        self.prompt_type = None
        self.device = cli_args.device
        
        # Sovrascrive i default con i valori del config file
        for key, value in config.items():
            setattr(self, key, value)
            
        # Forzature specifiche per la valutazione Stage 1
        self.sliding_window = False
        self.regression = False

# --- LOGICA DI VALUTAZIONE ---
@torch.no_grad()
def evaluate_patch_level(model, dataloader, device, thresholds):
    model.eval()
    
    # Inizializza un dizionario di statistiche per ogni soglia testata
    stats = [{'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 
              'retained_density': 0.0, 'total_density': 0.0} for _ in thresholds]
    
    print(f"[*] Avvio valutazione PATCH-LEVEL su {len(dataloader)} batch...")
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        img = None
        gt_density = None

        # 1. Recupero Dati Robusto
        if isinstance(batch, dict):
            img = batch.get('image') or batch.get('img')
            gt_density = batch.get('density') or batch.get('labels')
        elif isinstance(batch, (list, tuple)):
            img = batch[0]
            if len(batch) >= 3: gt_density = batch[2]
            elif len(batch) == 2: gt_density = batch[1]

        if img is None or gt_density is None: continue

        img = img.to(device)
        gt_density = gt_density.to(device)

        # 2. Forward Pass
        output = model(img)
        logits = output['pi_logits'] if isinstance(output, dict) else output
        probs = torch.sigmoid(logits)

        # 3. Ground Truth a livello di Patch
        # Determiniamo se un patch (es. 16x16) contiene folla guardando la GT originale.
        h_in, w_in = img.shape[2], img.shape[3]
        h_out, w_out = logits.shape[2], logits.shape[3]
        stride_h, stride_w = h_in // h_out, w_in // w_out
        
        # Calcolo densità media nel patch * area del patch
        patch_area = stride_h * stride_w
        gt_patch_density = F.avg_pool2d(gt_density, kernel_size=(stride_h, stride_w), stride=(stride_h, stride_w)) * patch_area
        
        # Binary GT: Se la somma della densità nel patch è > 0.001, è folla (1).
        gt_binary_patch = (gt_patch_density > 0.001).float()

        # 4. Aggiornamento statistiche per ogni soglia
        for i, th in enumerate(thresholds):
            pred_binary = (probs > th).float()
            
            # --- CALCOLO MATRICE DI CONFUSIONE ---
            tp = ((pred_binary == 1) & (gt_binary_patch == 1)).sum().item()
            tn = ((pred_binary == 0) & (gt_binary_patch == 0)).sum().item()
            fp = ((pred_binary == 1) & (gt_binary_patch == 0)).sum().item() # False Positive (Allarme)
            fn = ((pred_binary == 0) & (gt_binary_patch == 1)).sum().item() # False Negative (Perso)
            
            stats[i]['tp'] += tp
            stats[i]['tn'] += tn
            stats[i]['fp'] += fp
            stats[i]['fn'] += fn

            # Metrica ausiliaria: Densità preservata
            mask_up = F.interpolate(pred_binary, size=gt_density.shape[-2:], mode='nearest')
            stats[i]['retained_density'] += (gt_density * mask_up).sum().item()
            stats[i]['total_density'] += gt_density.sum().item()

    return stats

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    
    args_cli = parser.parse_args()

    # Setup Device
    if args_cli.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args_cli.device)
    print(f"⚙️  Device: {device}")

    # Load Config
    with open(args_cli.config, "r") as f: config_dict = yaml.safe_load(f)
    args = ConfigArgs(config_dict, args_cli)
    args.batch_size = args_cli.batch_size

    # Load Model
    print(f"🏗️  Loading ZIPModel...")
    model = ZIPModel(config_dict).to(device)

    # Load Checkpoint
    if not os.path.exists(args_cli.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args_cli.checkpoint}")
    
    print(f"📥 Loading Weights: {args_cli.checkpoint}")
    ckpt = torch.load(args_cli.checkpoint, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=False)

    # Load Dataset
    print(f"📂 Loading Dataset: {args.dataset} (split='val')")
    try:
        loader = get_dataloader(args, split='val', ddp=False)
        if len(loader) == 0: raise ValueError("Empty val loader")
    except Exception:
        print("⚠️  Loader 'val' failed/empty. Trying 'test'...")
        loader = get_dataloader(args, split='test', ddp=False)

    print(f"📊 Images to evaluate: {len(loader.dataset)}")

    # Run Eval
    thresholds = np.arange(0.05, 0.96, 0.05)
    stats = evaluate_patch_level(model, loader, device, thresholds)
    
    # --- STAMPA RISULTATI ---
    print("\n" + "="*95)
    print(f"{'Thr':<5} | {'F1 (Patch)':<10} | {'Acc (Patch)':<10} | {'Prec':<8} | {'Rec':<8} | {'Dens. Kept%':<12}")
    print("-" * 95)
    
    best_f1 = -1.0
    best_th = -1.0
    best_stats = None # Variabile per salvare la matrice migliore
    
    epsilon = 1e-8
    for i, th in enumerate(thresholds):
        s = stats[i]
        
        tp, tn, fp, fn = s['tp'], s['tn'], s['fp'], s['fn']
        
        acc = (tp + tn) / (tp + tn + fp + fn + epsilon)
        prec = tp / (tp + fp + epsilon)
        rec = tp / (tp + fn + epsilon)
        f1 = 2 * (prec * rec) / (prec + rec + epsilon)
        dens_kept_pct = (s['retained_density'] / (s['total_density'] + epsilon)) * 100
        
        # Salviamo il risultato migliore
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_stats = s # <-- Qui salviamo i dati per la matrice
            
        print(f"{th:<5.2f} | {f1:<10.2%} | {acc:<10.2%} | {prec:<8.2%} | {rec:<8.2%} | {dens_kept_pct:<10.2f}%")
        
    print("-" * 95)
    print(f"🏆 BEST F1: {best_f1:.2%} @ Threshold {best_th:.2f}")
    
    # --- STAMPA MATRICE DI CONFUSIONE DEL MIGLIOR RISULTATO ---
    if best_stats:
        tp = best_stats['tp']
        tn = best_stats['tn']
        fp = best_stats['fp']
        fn = best_stats['fn']
        total = tp + tn + fp + fn
        
        print("\n" + "="*40)
        print(f"📊 MATRICE DI CONFUSIONE (Best Thr: {best_th:.2f})")
        print("="*40)
        print(f"{'':>12} {'PREDETTO':^25}")
        print(f"{'':>12} {'CROWD (1)':^12} | {'NO-CROWD (0)':^12}")
        print("-" * 40)
        print(f"{'REALE':<8} {'1':>3} | {tp:^12d} | {fn:^12d}")
        print(f"{'(GT)':<8} {'0':>3} | {fp:^12d} | {tn:^12d}")
        print("-" * 40)
        print(f"TP (Veri Positivi): {tp}")
        print(f"TN (Veri Negativi): {tn}")
        print(f"FP (Falsi Allarmi): {fp}")
        print(f"FN (Folla Persa)  : {fn}")
        print(f"Patches Totali    : {total}")
        print("="*40 + "\n")

if __name__ == "__main__":
    main()