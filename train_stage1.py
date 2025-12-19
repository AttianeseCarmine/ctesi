import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from models import ZIPCLIPEBCModel
from datasets import Crowd
# Se non hai una collate_fn specifica in datasets/__init__.py, usa quella di default o definiscila qui
# from datasets import collate_fn 

def train_stage1(config_path):
    # 1. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config.get('DEVICE', 'cuda'))
    run_name = config['RUN_NAME']
    output_dir = os.path.join("outputs", run_name, "stage1")
    os.makedirs(output_dir, exist_ok=True)

    print(f"🚀 Starting Stage 1 Training: {run_name}")

    # 2. Dataset & Dataloader
    # Nota: Crowd usa internamente i path definiti in 'DATA' nel config
    train_dataset = Crowd(
        dataset=config['DATASET'], 
        split=config['DATA']['TRAIN_SPLIT'],
        transforms=None, # Le trasformazioni sono gestite internamente se config è passato o hardcoded
        return_filename=False
    )
    val_dataset = Crowd(
        dataset=config['DATASET'], 
        split=config['DATA']['VAL_SPLIT'],
        return_filename=False
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['TRAIN_STAGE1']['BATCH_SIZE'], 
        shuffle=True, 
        num_workers=config['TRAIN_STAGE1']['NUM_WORKERS'],
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['TRAIN_STAGE1']['BATCH_SIZE'], 
        shuffle=False, 
        num_workers=config['TRAIN_STAGE1']['NUM_WORKERS']
    )

    # 3. Model
    model = ZIPCLIPEBCModel(config).to(device)
    
    # Congela tutto tranne la pi-head e (opzionalmente) il backbone parziale
    for name, param in model.named_parameters():
        if "pi_head" in name:
            param.requires_grad = True
        elif "backbone" in name:
            param.requires_grad = True # Fine-tuning leggero del backbone
        else:
            param.requires_grad = False # Congela EBC head e CLIP

    # 4. Optimizer & Scheduler
    optimizer = AdamW([
        {'params': model.pi_head.parameters(), 'lr': float(config['TRAIN_STAGE1']['LR_PI_HEAD'])},
        {'params': model.backbone.parameters(), 'lr': float(config['TRAIN_STAGE1']['LR_BACKBONE'])}
    ])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Loss weight per bilanciare classi (pieni sono rari)
    pos_weight = torch.tensor([config['LOSS_STAGE1']['POS_WEIGHT']]).to(device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 5. Training Loop
    best_f1 = 0.0
    block_size = config['DATA']['ZIP_BLOCK_SIZE']

    for epoch in range(config['TRAIN_STAGE1']['EPOCHS']):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Ep {epoch+1} [Train]"):
            # Gestione batch (se Crowd restituisce 3 o 4 valori)
            if len(batch) == 3:
                imgs, gt_points, gt_density = batch
            else:
                imgs, gt_points, gt_density, _ = batch
                
            imgs = imgs.to(device)
            gt_density = gt_density.to(device)

            # Genera Ground Truth per la pi-head (Occupancy Grid)
            # AvgPool somma la densità nel blocco (dato che density map è normalizzata per somma=count)
            # Moltiplichiamo per block_size^2 se la mappa era densità pura, ma spesso è count map.
            # Assumiamo gt_density sia count map pixel-wise.
            with torch.no_grad():
                gt_counts = F.avg_pool2d(gt_density, kernel_size=block_size, stride=block_size, divisor_override=1)
                gt_occupancy = (gt_counts > 0).float() # 1 se c'è gente, 0 se vuoto

            # Forward
            outputs = model(imgs) # Ritorna dict
            pi_logits = outputs['pi'] # Nota: controlla se il tuo modello ritorna logits o sigmoid
            # Se il modello applica già sigmoid, usa BCELoss, altrimenti BCEWithLogitsLoss
            # Assumiamo output raw (logits) per stabilità numerica con BCEWithLogitsLoss
            
            # ATTENZIONE: Se la tua classe ZIPHead ritorna p(vuoto), 
            # e gt_occupancy è 1=pieno, devi invertire o la label o l'output.
            # Se pi_head predice prob. VUOTO:
            # target_vuoto = 1 - gt_occupancy
            # loss = bce_loss(pi_logits, target_vuoto) 
            
            # Assumiamo logica standard: pi = probabilità di PRESENZA (come molti framework OD)
            # Se invece nel tuo ZIPHead pi = prob. VUOTO, usa: target = (gt_counts == 0).float()
            
            # Verificando il tuo pi_head.py, sembra standard. Usiamo target presenza.
            loss = bce_loss(pi_logits, gt_occupancy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        if (epoch + 1) % config['TRAIN_STAGE1']['VAL_INTERVAL'] == 0:
            f1 = validate_stage1(model, val_loader, device, block_size)
            print(f"Epoch {epoch+1} - Loss: {train_loss/len(train_loader):.4f} - Val F1: {f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                torch.save({'model': model.state_dict()}, os.path.join(output_dir, "best_stage1_model.pth"))
                print("✅ Saved Best Model")

def validate_stage1(model, loader, device, block_size):
    model.eval()
    tp, fp, fn = 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3: imgs, _, gt_density = batch
            else: imgs, _, gt_density, _ = batch
            
            imgs = imgs.to(device)
            gt_density = gt_density.to(device)
            
            gt_counts = F.avg_pool2d(gt_density, block_size, stride=block_size, divisor_override=1)
            gt_mask = (gt_counts > 0).float()
            
            outputs = model(imgs)
            preds = (torch.sigmoid(outputs['pi']) > 0.5).float() # Assumendo output sia logits
            
            tp += torch.sum((preds == 1) & (gt_mask == 1)).item()
            fp += torch.sum((preds == 1) & (gt_mask == 0)).item()
            fn += torch.sum((preds == 0) & (gt_mask == 1)).item()
            
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_sha.yaml')
    args = parser.parse_args()
    train_stage1(args.config)