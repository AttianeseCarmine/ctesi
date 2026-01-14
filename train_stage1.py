#!/usr/bin/env python3
"""
============================================================
STAGE 1: ZIP HEAD TRAINING (Binary Segmentation)
============================================================
Obiettivo: Addestrare SOLO la 'zip_head' (pi) a distinguere 
background (vuoto) da foreground (folla).

Approccio Semplificato (Official Style):
- Input: Immagine (Crop 448x448 o simili)
- Output: Mappa di probabilità (pi)
- Target: Maschera binaria (1 se c'è gente, 0 se vuoto)
- Loss: BCEWithLogitsLoss (pesata per sbilanciamento)
============================================================
"""
import shutil
import argparse
import yaml
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# Import Moduli Custom
from models.zip_model import ZIPModel
from datasets.builder import build_dataset  
from datasets.transforms import build_transforms
from utils.train_utils import seed_everything

# =============================================================================
# UTILS
# =============================================================================
def crowd_collate(batch):
    """
    Gestisce il batching per il nuovo BaseCrowdDataset.
    Filtra eventuali elementi None e stacka i tensori.
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    
    images = torch.stack([item['image'] for item in batch])
    densities = torch.stack([item['density'] for item in batch])
    
    # Points possono avere dimensioni diverse (numero di persone variabile),
    # quindi li lasciamo in una lista, non usiamo torch.stack
    points = [item['points'] for item in batch]
    img_paths = [item['img_path'] for item in batch]
    
    return {
        'image': images,
        'density': densities,
        'points': points,
        'img_path': img_paths
    }

def save_checkpoint(state, is_best, save_dir, filename='last_model.pth'):
    last_path = os.path.join(save_dir, filename)
    torch.save(state, last_path)
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        shutil.copyfile(last_path, best_path)

# =============================================================================
# EVALUATION FUNCTION (Binary Metrics)
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, device, config):
    model.eval()
    
    tp, tn, fp, fn = 0, 0, 0, 0
    
    # Legge la soglia dal config (es. 0.3), default 0.2 se manca
    threshold = config.get('EVAL_STAGE1', {}).get('THRESHOLD', 0.2)
    
    for batch in tqdm(loader, desc=f"Eval (Thr={threshold})"):
        if batch is None: continue
        
        images = batch['image'].to(device)
        gt_density = batch['density'].to(device)
        
        # 1. Forward
        outputs = model(images)
        pi_logits = outputs['pi_logits'] # [B, 1, H/16, W/16]
        probs = torch.sigmoid(pi_logits)
        
        # 2. Prepara Ground Truth Binaria (match dimensions)
        h_out, w_out = pi_logits.shape[2:]
        
        # Downsample della densità alla risoluzione dell'output (H/16)
        # gt_density è [B, 1, H, W] con 1 dove c'è una testa.
        # Facciamo Average Pooling per vedere la densità media nel blocco
        scale_factor = (images.shape[2] * images.shape[3]) / (h_out * w_out)
        gt_down = F.adaptive_avg_pool2d(gt_density, (h_out, w_out)) * scale_factor
        
        # Se nel blocco c'è > 0.001 persone (praticamente > 0), è "Folla" (1)
        gt_binary = (gt_down > 0.001).float()
        
        # 3. Predizione Binaria
        pred_binary = (probs > threshold).float()
        
        # 4. Metriche
        tp += ((pred_binary == 1) & (gt_binary == 1)).sum().item()
        tn += ((pred_binary == 0) & (gt_binary == 0)).sum().item()
        fp += ((pred_binary == 1) & (gt_binary == 0)).sum().item()
        fn += ((pred_binary == 0) & (gt_binary == 1)).sum().item()
        
    # Calcolo F1-Score
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    return {'f1': f1, 'acc': acc, 'prec': precision, 'rec': recall}

# =============================================================================
# TRAINING LOOP
# =============================================================================
def train_stage1_simple():
    parser = argparse.ArgumentParser()
    # Default aggiornato per puntare al file config SHB corretto
    parser.add_argument('--config', type=str, default='configs/config_shb_vit.yaml')
    parser.add_argument("--out", type=str, default="checkpoints/shb_vit/stage1", help="Output directory")
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)

    # --- SALVATAGGIO CONFIG ---
    saved_config_path = os.path.join(args.out, "config.yaml")
    shutil.copy(args.config, saved_config_path)
    print(f"📄 Configuration saved to: {saved_config_path}")

    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    seed_everything(config.get('SEED', 42))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print("Using device:", device)
    
    # Setup Paths
    dataset_name = config.get('RUN_NAME', 'experiment')
    save_dir = Path(args.out)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Stage 1: Binary Classifier Training (ZIP)")
    print(f"   Run Name: {dataset_name} | Save to: {save_dir}")
    
    # Dataset
    # Usiamo build_transforms che ora legge NORM_MEAN/STD di CLIP dal config
    train_dataset = build_dataset(config, 'train', build_transforms(config['DATA'], True))
    val_dataset = build_dataset(config, 'val', build_transforms(config['DATA'], False))
    
    bs = config['TRAIN_STAGE1'].get('BATCH_SIZE', 16)
    
    # Train Loader: drop_last=True per evitare batch incompleti che possono dare problemi con BatchNorm
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, 
                              num_workers=config['TRAIN_STAGE1'].get('NUM_WORKERS', 8), 
                              collate_fn=crowd_collate, drop_last=True)
    
    # Val Loader: batch_size=1 è sicuro perché le immagini di val possono avere size diverse (Resize2Multiple)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                            num_workers=4, collate_fn=crowd_collate)
    
    # Modello
    model = ZIPModel(config).to(device)
    
    # Resume
    ckpt_path = save_dir / 'best_model.pth'
    if ckpt_path.exists():
        print(f"🔄 Riprendo il training dal checkpoint: {ckpt_path}")
        # Gestione caricamento sicuro (rimozione prefisso module. se presente)
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt if isinstance(ckpt, dict) else ckpt.state_dict()
        if 'model' in state_dict: state_dict = state_dict['model']
        model.load_state_dict(state_dict, strict=False)
    else:
        print("🚀 Nessun checkpoint trovato, inizio da zero.")

    # --- FREEZE & UNFREEZE ---
    # Congela tutto
    for p in model.parameters(): p.requires_grad = False
    
    # Sblocca Backbone
    for p in model.backbone.parameters(): p.requires_grad = True
    
    # Sblocca ZIP Head
    if hasattr(model, 'zip_head'):
        for p in model.zip_head.parameters(): p.requires_grad = True
        head_params = model.zip_head.parameters()
    elif hasattr(model, 'pi_head'):
        for p in model.pi_head.parameters(): p.requires_grad = True
        head_params = model.pi_head.parameters()
    else:
        # Fallback se i nomi sono diversi
        print("⚠️ Warning: Head specifica non trovata, sblocco tutto ciò che ha 'head' nel nome.")
        head_params = []
        for name, p in model.named_parameters():
            if 'head' in name or 'pi_' in name:
                p.requires_grad = True
                head_params.append(p)

    # Optimizer
    lr = float(config['TRAIN_STAGE1']['LR_HEAD'])
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': lr * 0.1}, 
        {'params': head_params, 'lr': lr}
    ], weight_decay=1e-4)
    
    scaler = GradScaler(enabled=use_cuda)
    
    # Loss: Pesata (POS_WEIGHT)
    pos_weight_val = float(config['TRAIN_STAGE1'].get('POS_WEIGHT', 15.0))
    print(f"⚖️  Loss Pos Weight: {pos_weight_val}")
    pos_weight_tensor = torch.tensor([pos_weight_val]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    best_f1 = 0.0
    epochs = config['TRAIN_STAGE1']['EPOCHS']
    val_interval = config['TRAIN_STAGE1'].get('VAL_INTERVAL', 1)
    
    print("🔧 Training Start...")
    
    for epoch in range(epochs):
        model.train()
        avg_loss = 0
        steps = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{epochs}")
        for batch in pbar:
            if batch is None: continue
            
            # Immagini (norm CLIP) e Densità (binaria sparsa)
            images = batch['image'].to(device)
            gt_density = batch['density'].to(device)
            
            optimizer.zero_grad()
            
            with autocast(device_type="cuda", enabled=use_cuda):
                # 1. Forward
                outputs = model(images)
                pi_logits = outputs['pi_logits']
                
                # 2. Target Binario (Downsample Density)
                h_out, w_out = pi_logits.shape[2:]
                scale = (images.shape[2] * images.shape[3]) / (h_out * w_out)
                gt_down = F.adaptive_avg_pool2d(gt_density, (h_out, w_out)) * scale
                target_binary = (gt_down > 0.001).float()
                
                # 3. Loss
                loss = criterion(pi_logits, target_binary)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            avg_loss += loss.item()
            steps += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # Validation
        if (epoch + 1) % val_interval == 0:
            # Passiamo config a evaluate per leggere la THRESHOLD corretta
            metrics = evaluate(model, val_loader, device, config)
            print(f"\n📊 Val Ep {epoch+1}: F1={metrics['f1']:.2%} | Acc={metrics['acc']:.2%} | Prec={metrics['prec']:.2%} | Rec={metrics['rec']:.2%}")
            
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                save_checkpoint(model.state_dict(), True, str(save_dir), 'last_model.pth')
                print(f"🌟 New Best Saved! (F1: {best_f1:.2%})")
            else:
                save_checkpoint(model.state_dict(), False, str(save_dir), 'last_model.pth')

if __name__ == '__main__':
    train_stage1_simple()