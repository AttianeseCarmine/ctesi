import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- IMPORTS ---
from models.zip_clip_ebc_model import ZIPCLIPEBCModel
from datasets.shha import SHHA
from datasets.transforms import build_transforms
from losses.clip_ebc_loss import CLIPEBCLoss 
from utils.train_utils import AverageMeter, seed_everything

# Funzione per gestire batch di dimensioni variabili
def crowd_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    images = torch.stack([item['image'] for item in batch])
    densities = torch.stack([item['density'] for item in batch])
    points = [item['points'] for item in batch]
    paths = [item['img_path'] for item in batch]
    return {'image': images, 'density': densities, 'points': points, 'img_path': paths}

def save_checkpoint(state, is_best, save_dir, filename='last_model.pth'):
    last_path = os.path.join(save_dir, filename)
    torch.save(state, last_path)
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        shutil.copyfile(last_path, best_path)
        print(f"⭐ Salvato nuovo Best Model in: {best_path}")

def train_stage2():
    parser = argparse.ArgumentParser(description='Stage 2: Train CLIP-EBC')
    parser.add_argument('--config', type=str, default='configs/config_sha.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    # 1. Carica Configurazione
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    seed_everything(config.get('SEED', 42))
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    dataset_name = config.get('DATASET', 'dataset')
    
    # Setup Directory
    base_save_dir = config.get('save_dir', './checkpoints')
    stage2_dir = os.path.join(base_save_dir, dataset_name, 'stage2')
    os.makedirs(stage2_dir, exist_ok=True)
    print(f"🚀 [Stage 2] Start training VLM on {dataset_name}")
    print(f"📂 Checkpoint dir: {stage2_dir}")

    # 2. Setup Dataset
    data_cfg = config['DATA']
    t_cfg = config['TRAIN_STAGE2']
    
    train_dataset = SHHA(root=data_cfg['ROOT'], split='train', transforms=build_transforms(data_cfg, True)) 
    val_dataset = SHHA(root=data_cfg['ROOT'], split='val', transforms=build_transforms(data_cfg, False))

    train_loader = DataLoader(train_dataset, batch_size=t_cfg['BATCH_SIZE'], shuffle=True, 
                              num_workers=data_cfg.get('WORKERS', 4), collate_fn=crowd_collate, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                            num_workers=data_cfg.get('WORKERS', 4), collate_fn=crowd_collate)

    # 3. Setup Model
    model = ZIPCLIPEBCModel(config).to(device)
    
    # Parametri: Congela Pi-Head, Sblocca Backbone & EBC
    for p in model.pi_head.parameters(): p.requires_grad = False
    for p in model.backbone.parameters(): p.requires_grad = True
    for p in model.clip_ebc_head.parameters(): p.requires_grad = True

    # 4. Optimizer
    optimizer = optim.AdamW([
        {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': float(t_cfg['LR_BACKBONE'])},
        {'params': [p for p in model.clip_ebc_head.parameters() if p.requires_grad], 'lr': float(t_cfg['LR_HEAD'])}
    ], weight_decay=float(t_cfg['WEIGHT_DECAY']))

    # =========================================================================
    # ✅ FIX CRUCIALE: SETUP LOSS DAL CONFIG
    # =========================================================================
    # Leggiamo i bin diretti dal file YAML per evitare errori di conversione interna
    if 'BIN_CENTERS' not in config:
        raise ValueError("Errore: BIN_CENTERS mancante in config_sha.yaml")
    if 'BINS' not in config:
        raise ValueError("Errore: BINS mancante in config_sha.yaml")

    # Creiamo il tensore dei centri (per calcoli numerici della loss)
    bin_centers_tensor = torch.tensor(config['BIN_CENTERS'], dtype=torch.float32).to(device)
    
    # Recuperiamo la lista pura dei Bins (per l'unpacking (lo, hi))
    bins_list = config['BINS']

    print(f"⚙️ Configurazione Loss:")
    print(f"   - Bins: {bins_list}")
    print(f"   - Centers: {bin_centers_tensor.cpu().numpy()}")

    # Inizializziamo la Loss
    criterion = CLIPEBCLoss(config, bin_centers_tensor).to(device)
    
    # FORZIAMO la lista 'bins' dentro la criterion per assicurarci che sia una lista Python pura
    # Questo previene l'errore "too many values to unpack" se la classe avesse convertito in Tensore
    criterion.bins = bins_list 
    # =========================================================================
    
    best_mae = float('inf')
    val_interval = t_cfg.get('VAL_INTERVAL', 1)

    for epoch in range(t_cfg['EPOCHS']):
        model.train()
        model.pi_head.eval() # Pi-Head sempre in eval
        losses = AverageMeter()
        
        loader_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{t_cfg['EPOCHS']}", leave=False)
        
        for i, batch in enumerate(loader_bar):
            if batch is None: continue
            images = batch['image'].to(device)
            gt_density = batch['density'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            
            # Recupero logits con gestione chiavi robusta
            if 'ebc_logits' in outputs: vlm_logits = outputs['ebc_logits']
            elif 'clip_ebc_logits' in outputs: vlm_logits = outputs['clip_ebc_logits']
            elif 'logits' in outputs: vlm_logits = outputs['logits']
            else:
                # Debug se fallisce tutto
                print(f"Keys available: {outputs.keys()}")
                raise KeyError("Output EBC non trovato")

            loss = criterion(vlm_logits, gt_density)
            
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), images.size(0))
            loader_bar.set_postfix(loss=f"{losses.avg:.4f}")

        # Validation
        if (epoch + 1) % val_interval == 0 or (epoch + 1) == t_cfg['EPOCHS']:
            mae = validate(val_loader, model, bin_centers_tensor, device)
            print(f"\n📊 Epoch {epoch+1}: Loss {losses.avg:.4f} | Val MAE {mae:.2f}")
            
            is_best = mae < best_mae
            if is_best: best_mae = mae
            
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_mae': best_mae,
                'optimizer': optimizer.state_dict(),
            }, is_best, stage2_dir)
        else:
            save_checkpoint({'epoch': epoch+1, 'state_dict': model.state_dict()}, False, stage2_dir)

def validate(loader, model, bin_centers, device):
    model.eval()
    mae_meter = AverageMeter()
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            if batch is None: continue
            images = batch['image'].to(device)
            gt_density = batch['density'].to(device)
            
            outputs = model(images)
            if 'ebc_logits' in outputs: vlm_logits = outputs['ebc_logits']
            else: vlm_logits = outputs.get('clip_ebc_logits', list(outputs.values())[-1])
            
            # Decoding: Expected Value = Sum( Prob_i * Value_i )
            probs = torch.softmax(vlm_logits, dim=1) 
            
            # Reshape bin_centers per broadcasting [1, N_bins, 1, 1]
            bins_view = bin_centers.view(1, -1, 1, 1)
            
            pred_map = (probs * bins_view).sum(dim=1)
            
            pred_count = pred_map.sum().item()
            gt_count = gt_density.sum().item()
            mae_meter.update(abs(gt_count - pred_count), images.size(0))
            
    return mae_meter.avg

if __name__ == '__main__':
    train_stage2()