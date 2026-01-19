import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm

from models.zip_model import ZIPModel
from utils import get_dataloader

def get_args():
    parser = argparse.ArgumentParser(description='Valutazione Stage 1 - Patch Level (Coerente con Training)')
    parser.add_argument('--config', type=str, default='config_stage1.yaml', help='Path al file di config')
    parser.add_argument('--gpu', default='0', type=str, help='ID della GPU')
    parser.add_argument('--ckpt', type=str, default=None, help='Path al checkpoint')
    return parser.parse_args()

@torch.no_grad()
def evaluate_patch_level(model, dataloader, device, thresholds):
    model.eval()
    
    # Statistiche per ogni soglia
    stats = [{'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'retained_density': 0.0, 'total_density': 0.0} for _ in thresholds]
    
    print("[*] Avvio valutazione PATCH-LEVEL (simulazione logica di training)...")
    
    for batch in tqdm(dataloader, desc="Valutazione"):
        img = None
        gt_density = None

        # --- 1. Recupero Dati ---
        if isinstance(batch, dict):
            img = batch['image'].to(device)
            if 'density' in batch: gt_density = batch['density'].to(device)
            elif 'labels' in batch: gt_density = batch['labels'].to(device)
        elif isinstance(batch, (list, tuple)):
            img = batch[0].to(device)
            # Fix per il tuo dataset che ritorna (img, points, density)
            if len(batch) >= 3:
                gt_density = batch[2].to(device)
            elif len(batch) == 2 and isinstance(batch[1], torch.Tensor):
                gt_density = batch[1].to(device)

        if img is None or gt_density is None: continue

        # --- 2. Forward Pass ---
        output = model(img)
        # Il modello restituisce logits di dimensione [B, 1, H/16, W/16] (es. 14x14)
        logits = output['pi_logits'] if isinstance(output, dict) else output
        probs = torch.sigmoid(logits)

        # --- 3. Generazione Ground Truth a Patch (Logica Training) ---
        # Per confrontare con l'output 14x14, dobbiamo sapere se nel blocco originale 16x16 c'era folla.
        # Usiamo MaxPool sulla densità: se nel blocco 16x16 c'è un picco > 0, il patch è POSITIVO.
        # Calcoliamo il kernel size in base al rapporto di riduzione
        h_in, w_in = img.shape[2], img.shape[3]
        h_out, w_out = logits.shape[2], logits.shape[3]
        stride_h, stride_w = h_in // h_out, w_in // w_out
        
        # GT Binaria a livello di Patch (14x14)
        # Se la somma della densità nel patch è > epsilon, allora è folla.
        gt_patch_density = F.avg_pool2d(gt_density, kernel_size=(stride_h, stride_w), stride=(stride_h, stride_w)) * (stride_h * stride_w)
        gt_binary_patch = (gt_patch_density > 0.001).float()

        # --- 4. Calcolo Metriche per ogni Soglia ---
        for i, th in enumerate(thresholds):
            # A. Metriche di Classificazione (F1, Accuracy) su PATCH
            pred_binary = (probs > th).float()
            
            tp = ((pred_binary == 1) & (gt_binary_patch == 1)).sum().item()
            tn = ((pred_binary == 0) & (gt_binary_patch == 0)).sum().item()
            fp = ((pred_binary == 1) & (gt_binary_patch == 0)).sum().item()
            fn = ((pred_binary == 0) & (gt_binary_patch == 1)).sum().item()
            
            stats[i]['tp'] += tp
            stats[i]['tn'] += tn
            stats[i]['fp'] += fp
            stats[i]['fn'] += fn

            # B. Metrica "Simil-MAE" (Quanto densità reale perdiamo mascherando?)
            # Upsample della maschera per applicarla alla densità originale
            mask_up = F.interpolate(pred_binary, size=gt_density.shape[-2:], mode='nearest')
            
            # Densità che "sopravvive" al filtro
            retained = (gt_density * mask_up).sum().item()
            total = gt_density.sum().item()
            
            stats[i]['retained_density'] += retained
            stats[i]['total_density'] += total

    return stats

def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Configurazione
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    for key, value in config.items(): setattr(args, key, value)
    
    # Defaults
    if not hasattr(args, 'dataset'): args.dataset = 'sha'
    args.sliding_window = False
    args.resize_to_multiple = False
    args.zero_pad_to_multiple = False
    args.regression = False
    args.prompt_type = None
    
    # Loader & Modello
    print(f"[*] Loading Dataset & Model...")
    loader = get_dataloader(args, split='val', ddp=False)
    model = ZIPModel(config).to(device)
    
    # Pesi
    ckpt_path = args.ckpt
    if ckpt_path is None: ckpt_path = os.path.join(config.get('ckpt_dir', ''), 'best_model.pth')
    
    if os.path.exists(ckpt_path):
        print(f"[*] Loading weights: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)
    else:
        print(f"[!] Errore: Checkpoint {ckpt_path} non trovato.")
        return

    # Range Soglie
    thresholds = np.arange(0.0, 1.05, 0.05)
    stats = evaluate_patch_level(model, loader, device, thresholds)
    
    # Stampa Tabella
    print("\n" + "="*95)
    print(f"{'Thr':<5} | {'F1 (Patch)':<10} | {'Acc (Patch)':<10} | {'Prec':<8} | {'Rec':<8} | {'Dens. Kept%':<12}")
    print("-" * 95)
    
    best_f1 = -1.0
    best_th = -1.0
    
    for i, th in enumerate(thresholds):
        s = stats[i]
        epsilon = 1e-7
        
        # Metriche Classificazione
        tp, tn, fp, fn = s['tp'], s['tn'], s['fp'], s['fn']
        acc = (tp + tn) / (tp + tn + fp + fn + epsilon)
        prec = tp / (tp + fp + epsilon)
        rec = tp / (tp + fn + epsilon)
        f1 = 2 * (prec * rec) / (prec + rec + epsilon)
        
        # Metrica Ritenzione Densità (Simile al tuo vecchio script)
        # Percentuale di folla reale che non viene cancellata dalla maschera
        dens_kept_pct = (s['retained_density'] / (s['total_density'] + epsilon)) * 100
        
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            
        print(f"{th:<5.2f} | {f1:<10.4f} | {acc:<10.4f} | {prec:<8.4f} | {rec:<8.4f} | {dens_kept_pct:<10.2f}%")
        
    print("-" * 95)
    print(f"🏆 BEST F1 (Patch-Level): {best_f1:.4f} @ Threshold: {best_th:.2f}")
    print("   Nota: 'Dens. Kept%' indica quanta folla reale viene preservata dalla maschera.")
    print("="*95 + "\n")

if __name__ == '__main__':
    main()