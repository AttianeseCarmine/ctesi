import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple
from tqdm import tqdm
import numpy as np
import logging

logger = logging.getLogger(__name__)

def pad_to_multiple(x: torch.Tensor, k: int = 16):
    """Padding per ViT (multipli di 16)."""
    h, w = x.shape[-2:]
    h_pad = (k - h % k) % k
    w_pad = (k - w % k) % k
    if h_pad > 0 or w_pad > 0:
        x = F.pad(x, (0, w_pad, 0, h_pad))
    return x, h, w

def calculate_errors(pred_counts, gt_counts):
    pred_counts = np.array(pred_counts)
    gt_counts = np.array(gt_counts)
    
    # Filtra NaN e Inf
    valid_mask = np.isfinite(pred_counts) & np.isfinite(gt_counts)
    if not valid_mask.all():
        # logger.warning(f"⚠️ Ignorati {len(pred_counts) - valid_mask.sum()} campioni NaN/Inf.")
        pred_counts = pred_counts[valid_mask]
        gt_counts = gt_counts[valid_mask]

    if len(pred_counts) == 0:
        return 0.0, 0.0

    mae = np.mean(np.abs(pred_counts - gt_counts))
    rmse = np.sqrt(np.mean((pred_counts - gt_counts) ** 2))
    return mae, rmse

def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    stage: int = 3,
    desc: str = "Evaluating"
) -> Tuple[float, float]:
    
    model.eval()
    pred_counts = []
    gt_counts = []
    
    # Usa tqdm solo se non è disabilitato
    pbar = tqdm(data_loader, desc=desc, leave=False)
    
    with torch.no_grad():
        for batch in pbar:
            # Gestione flessibile del batch (dict o tupla)
            if isinstance(batch, dict):
                images = batch['image'].to(device)
                gt_points = batch['points']
            else:
                # Fallback per dataset standard (img, points, density)
                images, gt_points_raw, _ = batch
                images = images.to(device)
                gt_points = gt_points_raw

            # Calcola Ground Truth dai punti
            batch_gt = [len(p) for p in gt_points]
            gt_counts.extend(batch_gt)

            # Padding per ViT
            images, _, _ = pad_to_multiple(images, k=16)

            # --- LOGICA DI PREDIZIONE PER STADIO ---
            if stage == 1:
                # STAGE 1: Valuta solo ZIP (Backbone + ConvZIPHead)
                # 1. Estrai features
                features = model.backbone(images)
                # 2. Passa alla zip_head
                zip_out = model.zip_head(features.float())
                
                # 3. Calcola densità: pi * lambda
                pi_logits = zip_out["logit_pi_maps"] # [B, 2, H, W]
                lambda_map = zip_out["lambda_maps"]  # [B, 1, H, W]
                
                pi_prob = F.softmax(pi_logits, dim=1)[:, 1:2, :, :] # Probabilità classe "non-vuoto"
                
                # Predizione = probabilità * intensità
                pred_map = pi_prob * lambda_map
                
            else:
                # STAGE 2/3: Valuta output completo (EBC filtrato da ZIP)
                pred_map = model(images)

            # Somma per ottenere il conteggio
            batch_preds = pred_map.sum(dim=(1, 2, 3)).cpu().numpy()
            
            # Sostituisci eventuali NaN residui con 0 per non rompere il training
            batch_preds = np.nan_to_num(batch_preds, nan=0.0, posinf=0.0, neginf=0.0)
            pred_counts.extend(batch_preds)

    mae, rmse = calculate_errors(pred_counts, gt_counts)
    return mae, rmse