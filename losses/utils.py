# attianesecarmine/ctesi/ctesi-add-CLIP_EBC/losses/utils.py
import torch
from torch import Tensor
from typing import List, Tuple

def _reshape_density(gt_den_map: Tensor, block_size: int) -> Tensor:
    """Raggruppa la density map in blocchi."""
    B, C, H, W = gt_den_map.shape
    assert H % block_size == 0 and W % block_size == 0, \
        f"Le dimensioni H e W ({H}x{W}) non sono divisibili per block_size ({block_size})"
    
    gt_den_map = gt_den_map.view(
        B, C, H // block_size, block_size, W // block_size, block_size
    )
    gt_den_map = gt_den_map.permute(0, 1, 2, 4, 3, 5).contiguous()
    gt_den_map = gt_den_map.view(
        B, C, (H // block_size) * (W // block_size), block_size, block_size
    )
    gt_den_map = gt_den_map.sum(dim=(-1, -2))
    gt_den_map = gt_den_map.view(
        B, C, H // block_size, W // block_size
    )
    return gt_den_map

def _bin_count(gt_den_map: Tensor, bins: List[Tuple[float, float]]) -> Tensor:
    """Converte la density map in una mappa di classi (bin)."""
    B, C, H, W = gt_den_map.shape
    num_bins = len(bins)
    
    # Inizializza la mappa delle classi (B, H, W)
    gt_class_map = torch.full(
        (B, H, W), fill_value=num_bins - 1, dtype=torch.long, device=gt_den_map.device
    )
    
    gt_den_map_flat = gt_den_map.squeeze(1) # Rimuovi canale C (B, H, W)
    
    # Assegna ogni blocco al bin corretto
    for i, (low, high) in enumerate(bins):
        mask = (gt_den_map_flat >= low) & (gt_den_map_flat <= high)
        gt_class_map[mask] = i
        
    return gt_class_map