import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse

from models import ZIPCLIPEBCModel
from datasets import Crowd

def evaluate_stage1(model, dataloader, device, block_size=16, threshold=0.5):
    model.eval()
    
    # Inizializzazione contatori per metriche manuali (Stile Chri)
    tp, fp, tn, fn = 0, 0, 0, 0
    total_mae = 0.0

    print(f"Running Stage 1 Evaluation (π-head)...")
    
    with torch.no_grad():
        for imgs, gt_density, _ in tqdm(dataloader):
            imgs = imgs.to(device)
            gt_density = gt_density.to(device)

            # 1. Calcolo Ground Truth: se un blocco ha > 0 persone, è "Occupato"
            # Usiamo avg_pool per sommare la densità nel blocco
            gt_counts = F.avg_pool2d(gt_density, block_size) * (block_size**2)
            gt_mask_occupied = (gt_counts > 0).float() # 1 se c'è gente, 0 se vuoto

            # 2. Forward del modello
            outputs = model(imgs)
            # ATTENZIONE: Nel tuo progetto, pi è la probabilità di "VUOTO" (Zero-Inflation)
            # Quindi la probabilità di essere OCCUPATO è (1 - pi)
            pi_vuoto = outputs['pi']
            prob_occupied = 1.0 - pi_vuoto
            
            # 3. Predizione binaria basata sulla soglia
            preds_occupied = (prob_occupied > threshold).float()

            # 4. Aggiornamento metriche di classificazione
            tp += ((preds_occupied == 1) & (gt_mask_occupied == 1)).sum().item()
            fp += ((preds_occupied == 1) & (gt_mask_occupied == 0)).sum().item()
            tn += ((preds_occupied == 0) & (gt_mask_occupied == 0)).sum().item()
            fn += ((preds_occupied == 0) & (gt_mask_occupied == 1)).sum().item()
            
            # 5. MAE sulla stima del conteggio tramite ZIP rate (opzionale nello stage 1)
            # Se la pi_head ha anche il ramo lambda, lo valutiamo
            if 'lambda_' in outputs:
                pred_count = (outputs['lambda_'] * prob_occupied).sum(dim=[1,2,3])
                gt_total = gt_density.sum(dim=[1,2,3])
                total_mae += torch.abs(pred_count - gt_total).sum().item()

    # Calcolo metriche finali
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    mae = total_mae / len(dataloader.dataset)

    print("\n" + "="*30)
    print("STAGE 1 EVALUATION RESULTS")
    print("="*30)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (Capacità di non dare falsi positivi)")
    print(f"Recall:    {recall:.4f} (Capacità di trovare tutte le persone)")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Count MAE: {mae:.2f}")
    print("="*30)
    
    return f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config_sha.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path al file .pth')
    args = parser.parse_args()

    # Carica Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Inizializza Modello e carica pesi
    model = ZIPCLIPEBCModel(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    
    # Dataset di Validazione (usa i parametri del tuo config)
    val_dataset = Crowd(
        root=config['DATA']['ROOT'],
        split=config['DATA']['VAL_SPLIT'],
        is_train=False,
        config=config
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    evaluate_stage1(model, val_loader, device, block_size=config['DATA']['ZIP_BLOCK_SIZE'])