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
    
    # --- MODIFICA: Converti i bin in float PRIMA di usarli ---
    float_bins = [(float(low), float(high)) for low, high in bins]
    
    B, C, H, W = gt_den_map.shape
    num_bins = len(bins)
    
    # Inizializza la mappa delle classi (B, H, W)
    gt_class_map = torch.full(
        (B, H, W), fill_value=-1, dtype=torch.long, device=gt_den_map.device # Inizializza con -1 (invalido)
    )
    
    gt_den_map_flat = gt_den_map.squeeze(1) # Rimuovi canale C (B, H, W)
    
    # Assegna ogni blocco al bin corretto
    # NOTA: I bin si sovrappongono (es. [0,0] e [1,1]), usiamo la logica dei bin di EBC
    # che si basa sui *centri* dei bin.
    
    # Crea i bordi dei bin
    bin_edges = [float_bins[0][0]] # Inizia con 0
    for i in range(len(float_bins) - 1):
        # Il bordo è a metà tra i centri
        center_curr = (float_bins[i][0] + float_bins[i][1]) / 2
        center_next = (float_bins[i+1][0] + float_bins[i+1][1]) / 2
        border = (center_curr + center_next) / 2
        bin_edges.append(border)
    bin_edges.append(float('inf')) # Bordo finale

    # Esempio: bin_edges = [0.0, 0.5, 1.5, 2.5, ..., 14.5, inf]

    for i in range(num_bins):
        low = bin_edges[i]
        high = bin_edges[i+1]
        
        if i == 0:
             # Il primo bin (classe 0) include lo 0
            mask = (gt_den_map_flat >= low) & (gt_den_map_flat < high)
        else:
            mask = (gt_den_map_flat >= low) & (gt_den_map_flat < high)
            
        gt_class_map[mask] = i

    # Controlla se qualche valore non è stato assegnato (non dovrebbe succedere)
    if (gt_class_map == -1).any():
        print("ATTENZIONE: Alcuni valori di densità non sono stati assegnati a nessun bin.")
        # Assegna i non assegnati al bin più vicino (o all'ultimo bin)
        gt_class_map[gt_class_map == -1] = num_bins - 1 
        
    return gt_class_map