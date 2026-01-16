import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Union

def calculate_errors(pred_counts: np.ndarray, gt_counts: np.ndarray) -> Dict[str, float]:
    assert isinstance(pred_counts, np.ndarray), f"Expected numpy.ndarray, got {type(pred_counts)}"
    assert isinstance(gt_counts, np.ndarray), f"Expected numpy.ndarray, got {type(gt_counts)}"
    assert len(pred_counts) == len(gt_counts), f"Length of predictions and ground truths should be equal, but got {len(pred_counts)} and {len(gt_counts)}"
    errors = {
        "mae": np.mean(np.abs(pred_counts - gt_counts)),
        "rmse": np.sqrt(np.mean((pred_counts - gt_counts) ** 2)),
    }
    return errors


def resize_density_map(x: Tensor, size: Tuple[int, int]) -> Tensor:
    x_sum = torch.sum(x, dim=(-1, -2))
    x = F.interpolate(x, size=size, mode="bilinear")
    scale_factor = torch.nan_to_num(torch.sum(x, dim=(-1, -2)) / x_sum, nan=0.0, posinf=0.0, neginf=0.0)
    return x * scale_factor


def sliding_window_predict(
    model: nn.Module,
    image: torch.Tensor,
    window_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    device: torch.device = None  # Aggiunto per risolvere il TypeError
) -> torch.Tensor:
    """
    Versione corretta per CLIP-EBC con supporto a dizionario e parametro device.
    """
    assert len(image.shape) == 4
    window_size = (int(window_size), int(window_size)) if isinstance(window_size, (int, float)) else tuple(window_size)
    stride = (int(stride), int(stride)) if isinstance(stride, (int, float)) else tuple(stride)

    image_height, image_width = image.shape[-2:]
    window_height, window_width = window_size
    stride_height, stride_width = stride

    num_rows = int(np.ceil((image_height - window_height) / stride_height) + 1)
    num_cols = int(np.ceil((image_width - window_width) / stride_width) + 1)

    # Identifica il fattore di riduzione (di default 16 per CLIP-EBC)
    if hasattr(model, "reduction"):
        reduction = model.reduction
    elif hasattr(model, "visual_encoder") and hasattr(model.visual_encoder, "reduction"):
        reduction = model.visual_encoder.reduction
    else:
        reduction = 16

    windows = []
    for i in range(num_rows):
        for j in range(num_cols):
            x_start, y_start = i * stride_height, j * stride_width
            x_end, y_end = x_start + window_height, y_start + window_width
            
            if x_end > image_height:
                x_start, x_end = image_height - window_height, image_height
            if y_end > image_width:
                y_start, y_end = image_width - window_width, image_width

            window = image[:, :, x_start:x_end, y_start:y_end]
            windows.append(window)

    windows = torch.cat(windows, dim=0).to(image.device)

    model.eval()
    with torch.no_grad():
        # FIX: Gestione output dizionario del modello CLIPEBCModel
        output = model(windows)
        preds = output['ebc_density'] if isinstance(output, dict) else output
    
    preds = preds.cpu().numpy()

    # Ricomposizione della mappa di densità
    pred_map = np.zeros((preds.shape[1], image_height // reduction, image_width // reduction), dtype=np.float32)
    count_map = np.zeros((preds.shape[1], image_height // reduction, image_width // reduction), dtype=np.float32)
    
    idx = 0
    for i in range(num_rows):
        for j in range(num_cols):
            x_start, y_start = i * stride_height, j * stride_width
            x_end, y_end = x_start + window_height, y_start + window_width
            
            if x_end > image_height:
                x_start, x_end = image_height - window_height, image_height
            if y_end > image_width:
                y_start, y_end = image_width - window_width, image_width

            # Mapping spaziale sulla mappa ridotta
            r_x_start, r_x_end = x_start // reduction, x_end // reduction
            r_y_start, r_y_end = y_start // reduction, y_end // reduction
            
            pred_map[:, r_x_start:r_x_end, r_y_start:r_y_end] += preds[idx]
            count_map[:, r_x_start:r_x_end, r_y_start:r_y_end] += 1.
            idx += 1

    pred_map /= (count_map + 1e-8)
    return torch.from_numpy(pred_map).unsqueeze(0)