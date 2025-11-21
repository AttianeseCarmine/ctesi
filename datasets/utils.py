import torch
from torch import Tensor
from scipy.ndimage import gaussian_filter
from typing import Optional, List, Tuple


def get_id(x: str) -> int:
    filename_no_ext = x.split(".")[0]
    id_str = filename_no_ext.replace("IMG_", "") # Rimuove il prefisso
    return int(id_str)

def generate_density_map(label: Tensor, height: int, width: int, sigma: Optional[float] = None) -> Tensor:
    """
    Generate the density map based on the dot annotations provided by the label.
    """
    density_map = torch.zeros((1, height, width), dtype=torch.float32)

    if len(label) > 0:
        assert len(label.shape) == 2 and label.shape[1] == 2, f"label should be a Nx2 tensor, got {label.shape}."
        label_ = label.long()
        label_[:, 0] = label_[:, 0].clamp(min=0, max=width - 1)
        label_[:, 1] = label_[:, 1].clamp(min=0, max=height - 1)
        density_map[0, label_[:, 1], label_[:, 0]] = 1.0

    if sigma is not None:
        assert sigma > 0, f"sigma should be positive if not None, got {sigma}."
        density_map = torch.from_numpy(gaussian_filter(density_map, sigma=sigma))

    return density_map

def collate_fn(batch):
    # 1. Decomprimi i 3 valori restituiti da Crowd.__getitem__
    #    Ogni elemento è (image_crops, label_crops, density_crops)
    #    image_crops shape: [num_crops, 3, H, W]
    images, labels, density_maps = zip(*batch)
    
    # 2. Usa torch.cat invece di torch.stack.
    #    torch.cat concatena lungo la dimensione 0, fondendo la dimensione del batch
    #    con quella dei crop.
    #    Da lista di [num_crops, 3, H, W] -> Tensore [Batch*num_crops, 3, H, W]
    images = torch.cat(images, 0)
    density_maps = torch.cat(density_maps, 0)
    
    # 3. Appiattisci la lista dei punti.
    #    'labels' è una tupla di liste (una lista di tensori per ogni immagine nel batch).
    #    Dobbiamo ottenere una singola lista lunga quanto il batch effettivo (Batch * num_crops).
    points = [p for sublist in labels for p in sublist]
    
    # 4. Restituisci la tupla nell'ordine corretto per il Trainer
    return images, points, density_maps