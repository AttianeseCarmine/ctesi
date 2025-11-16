# utils/eval_utils.py

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Union, List # Aggiunto List

# ==================================================================
# === INIZIO MODIFICA: CLASSE 'AverageMeter' MANCANTE ===
# ==================================================================
# Aggiungi questa classe, richiesta da trainer.py

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
# === FINE MODIFICA ===
# ==================================================================


# --- Il tuo codice originale (eval_utils.py) inizia qui ---

def calculate_errors(pred_counts: np.ndarray, gt_counts: np.ndarray) -> Dict[str, float]:
    assert isinstance(pred_counts, np.ndarray), f"Expected numpy.ndarray, got {type(pred_counts)}"
    assert isinstance(gt_counts, np.ndarray), f"Expected numpy.ndarray, got {type(gt_counts)}"
    assert len(pred_counts) == len(gt_counts), f"Length of predictions and ground truths should be equal, but got {len(pred_counts)} and {len(gt_counts)}"
    
    # Filtra per evitare divisioni per zero in NAE
    indices = gt_counts > 0
    
    errors = {
        "mae": np.mean(np.abs(pred_counts - gt_counts)),
        "rmse": np.sqrt(np.mean((pred_counts - gt_counts) ** 2)),
    }
    
    # Calcola NAE solo se ci sono campioni > 0
    if np.any(indices):
        errors["nae"] = np.mean(np.abs(pred_counts[indices] - gt_counts[indices]) / gt_counts[indices])
    else:
        errors["nae"] = 0.0
        
    return errors


def resize_density_map(x: Tensor, size: Tuple[int, int]) -> Tensor:
    # Calcola la somma originale per preservare il conteggio
    x_sum = torch.sum(x, dim=(-1, -2), keepdim=True)
    
    # Interpola
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    
    # Riscala per mantenere il conteggio originale
    current_sum = torch.sum(x, dim=(-1, -2), keepdim=True).float()
    
    # Evita divisione per zero
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
    
    # Calcola il padding necessario
    pad_h = (stride_height - (image_height - window_height) % stride_height) % stride_height
    pad_w = (stride_width - (image_width - window_width) % stride_width) % stride_width

    padded_image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
    padded_height = padded_image.shape[2]
    padded_width = padded_image.shape[3]

    # Estrai le patch
    windows = padded_image.unfold(2, window_height, stride_height).unfold(3, window_width, stride_width)
    num_rows = windows.shape[2]
    num_cols = windows.shape[3]
    
    windows = windows.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, window_height, window_width)
    
    # Esegui l'inferenza in batch
    preds = []
    for i in range(0, len(windows), max_num_windows):
        with torch.no_grad():
            preds_ = model(windows[i: min(i + max_num_windows, len(windows))])
        preds.append(preds_.cpu().numpy())
        
    preds = np.concatenate(preds, axis=0)
    
    # Dimensioni della mappa di output
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
    gt_points: List[torch.Tensor],
    sliding_window: bool = False,
    **kwargs
) -> Tuple[float, float]:
    
    pred_cnt = pred_den_map.float().sum().item()
    
    gt_cnt = 0
    if gt_points and gt_points[0] is not None:
        gt_cnt = len(gt_points[0])

    mae = abs(pred_cnt - gt_cnt)
    rmse = (pred_cnt - gt_cnt)**2
    
    return mae, rmse