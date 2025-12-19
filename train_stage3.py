import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from models import ZIPCLIPEBCModel
from datasets import Crowd

def train_stage3(config_path):
    with open(config_path, 'r') as f: config = yaml.safe_load(f)
    device = torch.device(config.get('DEVICE', 'cuda'))
    run_name = config['RUN_NAME']
    ckpt_stage2 = os.path.join("outputs", run_name, "stage2", "best_stage2_model.pth")
    output_dir = os.path.join("outputs", run_name, "stage3")
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Starting Stage 3 (End-to-End Fine-tuning)")
    
    # Dataset
    train_loader = DataLoader(Crowd(config['DATASET'], config['DATA']['TRAIN_SPLIT']), 
                              batch_size=4, shuffle=True, num_workers=4) # Batch size piccolo per GPU
    val_loader = DataLoader(Crowd(config['DATASET'], config['DATA']['VAL_SPLIT']), 
                            batch_size=1, shuffle=False)

    model = ZIPCLIPEBCModel(config).to(device)
    
    # Load Stage 2
    if os.path.exists(ckpt_stage2):
        print(f"📥 Loading Stage 2 weights: {ckpt_stage2}")
        model.load_state_dict(torch.load(ckpt_stage2)['model'])
    
    # Unfreeze all (con LR differenziati)
    for param in model.parameters(): param.requires_grad = True
    
    # LR bassissimo per backbone per non distruggere le feature
    optimizer = AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.pi_head.parameters(), 'lr': 1e-4},
        {'params': model.clip_ebc_head.parameters(), 'lr': 1e-4}
    ])
    
    best_mae = float('inf')
    
    for epoch in range(100): # Meno epoche per fine-tuning
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Ep {epoch+1} [Joint]"):
            if len(batch) == 3: imgs, _, gt_density = batch
            else: imgs, _, gt_density, _ = batch
            imgs, gt_density = imgs.to(device), gt_density.to(device)
            
            outputs = model(imgs)
            
            # Loss congiunta:
            # 1. Loss Conteggio Finale (che include implicitamente la maschera pi)
            final_pred = outputs['final_density'].sum(dim=(1,2,3))
            gt_count = gt_density.sum(dim=(1,2,3))
            l1_loss = F.l1_loss(final_pred, gt_count)
            
            # 2. Loss Supervisione Intermedia (opzionale ma consigliata)
            #    Manteniamo la pi-head on track confrontandola ancora con la GT binaria
            #    Manteniamo l'EBC head on track nelle zone piene
            
            loss = l1_loss # Semplificato, puoi aggiungere termini ausiliari
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Eval
        mae = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch[0].to(device)
                gt = batch[2].sum().item()
                pred = model(imgs)['final_density'].sum().item()
                mae += abs(pred - gt)
        mae /= len(val_loader)
        
        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f} - Val MAE: {mae:.2f}")
        
        if mae < best_mae:
            best_mae = mae
            torch.save({'model': model.state_dict()}, os.path.join(output_dir, "best_stage3_model.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_sha.yaml')
    args = parser.parse_args()
    train_stage3(args.config)