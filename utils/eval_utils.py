# utils/eval_utils.py

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Union, List

# ==================================================================
# === CLASSE AverageMeter ===
# ==================================================================
class AverageMeter:
    """
    Calcola e memorizza la media e il valore corrente.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

# ==================================================================
# === FUNZIONI DI VALUTAZIONE ===
# ==================================================================

def calculate_errors(pred_counts: np.ndarray, gt_counts: np.ndarray) -> Dict[str, float]:
    assert isinstance(pred_counts, np.ndarray), f"Expected numpy.ndarray, got {type(pred_counts)}"
    assert isinstance(gt_counts, np.ndarray), f"Expected numpy.ndarray, got {type(gt_counts)}"
    assert len(pred_counts) == len(gt_counts), f"Length of predictions and ground truths should be equal, but got {len(pred_counts)} and {len(gt_counts)}"
    
    indices = gt_counts > 0
    
    errors = {
        "mae": np.mean(np.abs(pred_counts - gt_counts)),
        "rmse": np.sqrt(np.mean((pred_counts - gt_counts) ** 2)),
    }
    
    if np.any(indices):
        errors["nae"] = np.mean(np.abs(pred_counts[indices] - gt_counts[indices]) / gt_counts[indices])
    else:
        errors["nae"] = 0.0
        
    return errors


def resize_density_map(x: Tensor, size: Tuple[int, int]) -> Tensor:
    x_sum = torch.sum(x, dim=(-1, -2), keepdim=True)
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    current_sum = torch.sum(x, dim=(-1, -2), keepdim=True).float()
    scale_factor = torch.nan_to_num(x_sum.float() / (current_sum + 1e-8), nan=0.0, posinf=0.0, neginf=0.0)
    return x * scale_factor


def sliding_window_predict(
    model: nn.Module,
    image: Tensor,
    window_size: int,
    stride: int,
    block_size: int,
    max_num_windows: int = 20
) -> np.ndarray:
    
    assert image.ndim == 4 and image.shape[0] == 1, "Input image must be [1, C, H, W]"
    
    model.eval()
    B, C, image_height, image_width = image.shape
    
    window_height = window_size
    window_width = window_size
    stride_height = stride
    stride_width = stride
    
    pad_h = (stride_height - (image_height - window_height) % stride_height) % stride_height
    pad_w = (stride_width - (image_width - window_width) % stride_width) % stride_width

    padded_image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
    
    windows = padded_image.unfold(2, window_height, stride_height).unfold(3, window_width, stride_width)
    num_rows = windows.shape[2]
    num_cols = windows.shape[3]
    
    windows = windows.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, window_height, window_width)
    
    preds = []
    for i in range(0, len(windows), max_num_windows):
        with torch.no_grad():
            preds_ = model(windows[i: min(i + max_num_windows, len(windows))])
        preds.append(preds_.cpu().numpy())
        
    preds = np.concatenate(preds, axis=0)
    
    out_h = image_height // block_size
    out_w = image_width // block_size
    
    pred_map = np.zeros((preds.shape[1], out_h, out_w), dtype=np.float32)
    count_map = np.zeros_like(pred_map)

    idx = 0
    for i in range(num_rows):
        for j in range(num_cols):
            x_start_out = i * stride_height // block_size
            y_start_out = j * stride_width // block_size
            
            pred_patch = preds[idx]
            patch_h, patch_w = pred_patch.shape[1], pred_patch.shape[2]
            
            x_end_out = min(x_start_out + patch_h, out_h)
            y_end_out = min(y_start_out + patch_w, out_w)
            
            clip_h = x_end_out - x_start_out
            clip_w = y_end_out - y_start_out
            
            if clip_h > 0 and clip_w > 0:
                pred_map[:, x_start_out:x_end_out, y_start_out:y_end_out] += pred_patch[:, :clip_h, :clip_w]
                count_map[:, x_start_out:x_end_out, y_start_out:y_end_out] += 1
            
            idx += 1

    pred_map /= (count_map + 1e-8)
    return pred_map


def evaluate_mae_rmse(
    pred_den_map: torch.Tensor,
    gt_points: Union[List[torch.Tensor], torch.Tensor],
    sliding_window: bool = False,
    **kwargs
) -> Tuple[float, float]:
    
    # 1. Calcola i conteggi predetti per OGNI immagine nel batch separatamente
    # pred_den_map: [B, 1, H, W] -> Sum spatial dims -> [B, 1] -> flatten -> [B]
    pred_counts = pred_den_map.detach().cpu().flatten(start_dim=1).sum(dim=1)
    
    batch_mae = 0.0
    batch_mse = 0.0
    batch_size = pred_counts.shape[0]
    
    # 2. Itera sul batch per confrontare predizione[i] con target[i]
    for i in range(batch_size):
        pred_c = pred_counts[i].item()
        gt_c = 0.0
        
        curr_gt = None
        
        # Gestione input flessibile: gt_points può essere lista o tensore
        if isinstance(gt_points, list):
            if i < len(gt_points):
                curr_gt = gt_points[i]
        elif isinstance(gt_points, torch.Tensor):
            # Se è un tensore [B, N, 2], prendiamo l'elemento i-esimo
            curr_gt = gt_points[i]
            
        # Calcola il conteggio reale
        if curr_gt is not None:
            # len() su un tensore [N, 2] restituisce N (numero di punti)
            gt_c = len(curr_gt)

        batch_mae += abs(pred_c - gt_c)
        batch_mse += (pred_c - gt_c) ** 2
    
    # Restituisce la media dell'errore sul batch
    return batch_mae / batch_size, batch_mse / batch_size