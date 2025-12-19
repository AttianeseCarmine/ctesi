import torch
from torch.utils.data import DataLoader
# evaluation_stage2.py snippet
def evaluate_stage2(model, val_loader, device):
    model.eval()
    mae, mse, total = 0, 0, 0

    with torch.no_grad():
        for imgs, gt_density, _ in val_loader:
            imgs = imgs.to(device)
            gt_count = gt_density.sum().item()
            
            # Predizione EBC: calcola il valore atteso dai bin
            # (prob_bin_i * centro_bin_i)
            outputs = model(imgs)
            # Nota: qui valutiamo solo la capacità di conteggio dell'EBC head
            pred_count = outputs['ebc_count'].sum().item() 
            
            mae += abs(pred_count - gt_count)
            mse += (pred_count - gt_count)**2
            total += 1

    return {"mae": mae/total, "rmse": (mse/total)**0.5}