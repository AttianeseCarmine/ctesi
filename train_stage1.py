import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.zip_clip_ebc_model import ZIPCLIPEBCModel
from datasets.sha import SHHA
from datasets.transforms import build_transforms
from utils.train_utils import AverageMeter, seed_everything

# --- UTILS ---
def compute_binary_metrics(preds, targets):
    preds = preds.view(-1)
    targets = targets.view(-1)
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    return tp.item(), fp.item(), fn.item()

def crowd_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    images = torch.stack([item['image'] for item in batch])
    densities = torch.stack([item['density'] for item in batch])
    points = [item['points'] for item in batch] # List, variable length
    paths = [item['img_path'] for item in batch]
    return {'image': images, 'density': densities, 'points': points, 'img_path': paths}

def save_checkpoint(state, is_best, save_dir, filename='last_model.pth'):
    last_path = os.path.join(save_dir, filename)
    torch.save(state, last_path)
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        shutil.copyfile(last_path, best_path)
        print(f"⭐ Salvato nuovo Best Model in: {best_path}")

# --- MAIN ---
def train_stage1():
    parser = argparse.ArgumentParser(description='Stage 1: Train Pi-Head')
    parser.add_argument('--config', type=str, default='configs/config_sha.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    seed_everything(config.get('SEED', 42))
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    dataset_name = config.get('DATASET', 'dataset')
    
    # 1. Cartelle Salvataggio
    base_save_dir = config.get('save_dir', './checkpoints')
    stage1_dir = os.path.join(base_save_dir, dataset_name, 'stage1')
    os.makedirs(stage1_dir, exist_ok=True)
    print(f"🚀 [Stage 1] Start training on {dataset_name}. Saving to: {stage1_dir}")

    # 2. Dataset
    data_cfg = config['DATA']
    train_dataset = SHHA(root=data_cfg['ROOT'], split='train', transforms=build_transforms(data_cfg, True)) 
    val_dataset = SHHA(root=data_cfg['ROOT'], split='val', transforms=build_transforms(data_cfg, False))

    # Parametri dal Config
    t_cfg = config['TRAIN_STAGE1']
    train_loader = DataLoader(train_dataset, batch_size=t_cfg['BATCH_SIZE'], shuffle=True, 
                              num_workers=data_cfg.get('WORKERS', 4), collate_fn=crowd_collate, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                            num_workers=data_cfg.get('WORKERS', 4), collate_fn=crowd_collate)

    # 3. Model
    model = ZIPCLIPEBCModel(config).to(device)
    
    # Freeze CLIP-EBC, Unlock Pi-Head & Backbone
    for p in model.clip_ebc_head.parameters(): p.requires_grad = False
    for p in model.pi_head.parameters(): p.requires_grad = True
    for p in model.backbone.parameters(): p.requires_grad = True

    # 4. Optimizer & Loss
    optimizer = optim.AdamW([
        {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': float(t_cfg['LR_BACKBONE'])},
        {'params': [p for p in model.pi_head.parameters() if p.requires_grad], 'lr': float(t_cfg['LR_HEAD'])}
    ], weight_decay=float(t_cfg['WEIGHT_DECAY']))

    # Pos Weight per bilanciare la loss (importante per il tuo F1 score!)
    pos_weight = torch.tensor([float(t_cfg.get('POS_WEIGHT', 1.0))]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = 0.0
    val_interval = t_cfg.get('VAL_INTERVAL', 1)

    for epoch in range(t_cfg['EPOCHS']):
        model.train()
        losses = AverageMeter()
        
        loader_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{t_cfg['EPOCHS']}", leave=False)
        
        for batch in loader_bar:
            if batch is None: continue
            images = batch['image'].to(device)
            gt_density = batch['density'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            pi_logits = outputs['pi_logits']

            # Adaptive Pooling per creare target 0/1 dai density maps
            h_out, w_out = pi_logits.shape[2:]
            scale_h = images.shape[2] / h_out
            scale_w = images.shape[3] / w_out
            ds_count = F.adaptive_avg_pool2d(gt_density, (h_out, w_out)) * (scale_h * scale_w)
            
            # Target: 1 se Vuoto (< 0.001), 0 se Pieno
            target_mask = (ds_count < 1e-3).float()

            loss = criterion(pi_logits, target_mask)
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), images.size(0))
            loader_bar.set_postfix(loss=f"{losses.avg:.4f}")

        # Validation
        if (epoch + 1) % val_interval == 0 or (epoch + 1) == t_cfg['EPOCHS']:
            metrics = validate(val_loader, model, device)
            print(f"\n📊 Epoch {epoch+1}: Loss {losses.avg:.4f} | Val F1 {metrics['f1']:.2%} | Acc {metrics['acc']:.2%}")
            
            is_best = metrics['f1'] > best_f1
            if is_best: best_f1 = metrics['f1']
            
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_f1': best_f1,
                'optimizer': optimizer.state_dict(),
            }, is_best, stage1_dir)
        else:
            save_checkpoint({'epoch': epoch+1, 'state_dict': model.state_dict()}, False, stage1_dir)

def validate(loader, model, device):
    model.eval()
    tp, fp, fn, correct, total = 0, 0, 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            if batch is None: continue
            images = batch['image'].to(device)
            gt_density = batch['density'].to(device)
            
            pi_logits = model(images)['pi_logits']
            h_out, w_out = pi_logits.shape[2:]
            scale_h, scale_w = images.shape[2]/h_out, images.shape[3]/w_out
            ds_count = F.adaptive_avg_pool2d(gt_density, (h_out, w_out)) * (scale_h * scale_w)
            
            target = (ds_count < 1e-3).float()
            preds = (pi_logits > 0.0).float()
            
            b_tp, b_fp, b_fn = compute_binary_metrics(preds, target)
            tp += b_tp; fp += b_fp; fn += b_fn
            correct += (preds == target).sum().item()
            total += preds.numel()
            
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return {'acc': correct/(total+1e-8), 'f1': f1}

if __name__ == '__main__':
    train_stage1()