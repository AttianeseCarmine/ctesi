import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from models import ZIPCLIPEBCModel
from datasets import Crowd

def train_stage2(config_path):
    with open(config_path, 'r') as f: config = yaml.safe_load(f)
    device = torch.device(config.get('DEVICE', 'cuda'))
    
    # Load Paths
    run_name = config['RUN_NAME']
    ckpt_stage1 = os.path.join("outputs", run_name, "stage1", "best_stage1_model.pth")
    output_dir = os.path.join("outputs", run_name, "stage2")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🚀 Starting Stage 2 Training (EBC Only)")

    # Data
    train_dataset = Crowd(config['DATASET'], config['DATA']['TRAIN_SPLIT'])
    val_dataset = Crowd(config['DATASET'], config['DATA']['VAL_SPLIT'])
    train_loader = DataLoader(train_dataset, batch_size=config['TRAIN_STAGE2']['BATCH_SIZE'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) # Batch 1 per eval count preciso

    # Model
    model = ZIPCLIPEBCModel(config).to(device)
    
    # Load Stage 1 weights
    if os.path.exists(ckpt_stage1):
        print(f"📥 Loading weights from {ckpt_stage1}")
        checkpoint = torch.load(ckpt_stage1)
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        print("⚠️ Warning: Stage 1 checkpoint not found. Training from scratch.")

    # Freeze pi_head and backbone
    for param in model.parameters(): param.requires_grad = False
    for param in model.clip_ebc_head.parameters(): param.requires_grad = True # Solo EBC trainable
    
    optimizer = AdamW(model.clip_ebc_head.parameters(), lr=float(config['TRAIN_STAGE2']['LR_EBC_HEAD']))
    
    # Bins definition (semplificato, prendili dal tuo config o utils)
    # Assumiamo che model.clip_ebc_head.bin_centers sia definito
    
    best_mae = float('inf')
    
    for epoch in range(config['TRAIN_STAGE2']['EPOCHS']):
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Ep {epoch+1} [Stage 2]"):
            if len(batch) == 3: imgs, _, gt_density = batch
            else: imgs, _, gt_density, _ = batch
            
            imgs = imgs.to(device)
            gt_density = gt_density.to(device)
            
            # Forward: model deve avere metodo per usare solo EBC o ignorare maschera pi
            # Modifica il tuo modello per accettare un flag 'use_pi_mask=False' in forward
            # Oppure usa model.predict_ebc_only() se l'hai implementato
            outputs = model(imgs) 
            
            # Qui ci servono i logits EBC e la densità predetta
            # Assicurati che outputs contenga 'ebc_logits' o simile
            # Calcolo loss conteggio (L1) + Classification Loss (se implementata)
            
            pred_count = outputs['ebc_density'].sum(dim=(1,2,3))
            gt_count = gt_density.sum(dim=(1,2,3))
            
            loss = F.l1_loss(pred_count, gt_count)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        if (epoch + 1) % config['TRAIN_STAGE2']['VAL_INTERVAL'] == 0:
            mae = validate_stage2(model, val_loader, device)
            print(f"Epoch {epoch+1} - Loss: {train_loss:.4f} - Val MAE: {mae:.2f}")
            
            if mae < best_mae:
                best_mae = mae
                torch.save({'model': model.state_dict()}, os.path.join(output_dir, "best_stage2_model.pth"))

def validate_stage2(model, loader, device):
    model.eval()
    mae = 0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3: imgs, _, gt_density = batch
            else: imgs, _, gt_density, _ = batch
            imgs = imgs.to(device)
            gt_cnt = gt_density.sum().item()
            
            out = model(imgs)
            # Nello stage 2 valutiamo quanto è buono il conteggio puro
            # Possiamo ignorare la pi_map o usarla, dipende dalla strategia
            pred_cnt = out['ebc_density'].sum().item()
            
            mae += abs(pred_cnt - gt_cnt)
    return mae / len(loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_sha.yaml')
    args = parser.parse_args()
    train_stage2(args.config)