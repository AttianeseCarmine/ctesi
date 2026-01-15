import argparse
import yaml
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import necessari dal tuo progetto
from models.zip_model import ZIPModel
from datasets.builder import build_dataset  
from datasets.transforms import build_transforms

def crowd_collate(batch):
    """Gestisce batch con immagini di dimensioni diverse (se necessario)"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'density': torch.stack([item['density'] for item in batch]),
        'img_path': [item['img_path'] for item in batch]
    }

@torch.no_grad()
def evaluate(model, loader, device, threshold=0.2):
    """
    Valuta il modello Stage 1 (Binary Segmentation).
    Logica allineata al 100% con train_stage1.py.
    """
    model.eval()
    
    tp, tn, fp, fn = 0, 0, 0, 0
    
    print(f"⚙️  Evaluation Threshold: {threshold}")
    
    for batch in tqdm(loader, desc="Calculating Metrics"):
        if batch is None: continue
        
        images = batch['image'].to(device)
        gt_density = batch['density'].to(device)
        
        # 1. Forward
        outputs = model(images)
        pi_logits = outputs['pi_logits'] # [B, 1, H_out, W_out]
        probs = torch.sigmoid(pi_logits)
        
        # 2. Prepara Ground Truth Binaria (Allineamento dimensioni)
        h_out, w_out = pi_logits.shape[2:]
        
        # Scaling factor per mantenere la somma della densità corretta dopo il pooling
        scale_factor = (images.shape[2] * images.shape[3]) / (h_out * w_out)
        
        # Downsample della densità GT alla risoluzione dell'output del modello
        gt_down = F.adaptive_avg_pool2d(gt_density, (h_out, w_out)) * scale_factor
        
        # Definizione Target Binario:
        # Se nel blocco c'è anche una minima presenza (> 0.001), è considerato "Folla" (1)
        gt_binary = (gt_down > 0.001).float()
        
        # 3. Predizione Binaria
        pred_binary = (probs > threshold).float()
        
        # 4. Aggiornamento Statistiche (Vettorizzato per velocità)
        tp += ((pred_binary == 1) & (gt_binary == 1)).sum().item()
        tn += ((pred_binary == 0) & (gt_binary == 0)).sum().item()
        fp += ((pred_binary == 1) & (gt_binary == 0)).sum().item()
        fn += ((pred_binary == 0) & (gt_binary == 1)).sum().item()
        
    # Calcolo Metriche
    # Aggiungiamo 1e-8 per evitare divisioni per zero
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    return {
        'f1': f1, 
        'accuracy': accuracy, 
        'precision': precision, 
        'recall': recall,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 Model (ZIP)")
    parser.add_argument('--config', type=str, required=True, help="Path to config file (yaml)")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to trained .pth model")
    parser.add_argument('--threshold', type=float, default=0.2, help="Probability threshold (default: 0.2 like training)")
    parser.add_argument('--device', type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    
    # 1. Carica Configurazione
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"🏗️  Building Dataset & Model...")
    
    # 2. Carica Dataset (Validation)
    val_transforms = build_transforms(config['DATA'], is_train=False)
    val_dataset = build_dataset(config, 'val', val_transforms)
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=crowd_collate
    )
    
    # 3. Carica Modello
    model = ZIPModel(config).to(device)
    
    if os.path.isfile(args.checkpoint):
        print(f"📥 Loading weights from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Gestione dizionario checkpoint vs state_dict diretto
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # Rimuovi prefisso 'module.' se presente (Training parallelo)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return

    # 4. Esegui Valutazione
    print(f"🚀 Starting Evaluation on {len(val_dataset)} images...")
    metrics = evaluate(model, val_loader, device, threshold=args.threshold)
    
    # 5. Stampa Risultati
    print("\n" + "="*40)
    print(f"📊 EVALUATION RESULTS ({config['BACKBONE']['TYPE']})")
    print("="*40)
    print(f"🎯 F1-Score:   {metrics['f1']:.2%}")
    print(f"🎯 Accuracy:   {metrics['accuracy']:.2%}")
    print(f"🎯 Precision:  {metrics['precision']:.2%}")
    print(f"🎯 Recall:     {metrics['recall']:.2%}")
    print("-" * 40)
    print(f"🔢 Raw Counts:")
    print(f"   TP (Correct Crowd): {metrics['TP']}")
    print(f"   TN (Correct Empty): {metrics['TN']}")
    print(f"   FP (False Alarm):   {metrics['FP']}")
    print(f"   FN (Missed Crowd):  {metrics['FN']}")
    print("="*40)

if __name__ == "__main__":
    main()