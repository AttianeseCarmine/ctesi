import torch
from torch.utils.data import DataLoader
# evaluation_stage3.py snippet
def evaluate_stage3(model, val_loader, device, pi_threshold=0.7):
    model.eval()
    mae, mse = 0, 0

    with torch.no_grad():
        for imgs, gt_density, _ in val_loader:
            imgs = imgs.to(device)
            gt_count = gt_density.sum().item()

            outputs = model(imgs)
            pi = outputs['pi'] # Prob. di vuoto
            ebc_density = outputs['ebc_density'] # Densità dai bin
            
            # Mascheratura: se pi è alto, forziamo a zero
            mask = (pi < pi_threshold).float() 
            final_density = ebc_density * mask
            pred_count = final_density.sum().item()

            mae += abs(pred_count - gt_count)
            mse += (pred_count - gt_count)**2

    return {"final_mae": mae/len(val_loader), "final_rmse": (mse/len(val_loader))**0.5}