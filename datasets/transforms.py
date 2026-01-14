import random
import math
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

# ==========================================================
# TRANSFORM PIPELINE (MERGED ZIP + CLIP-EBC)
# ==========================================================

class Compose(object):
    """Applica una lista di trasformazioni in sequenza."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, pts=None, den=None):
        for t in self.transforms:
            img, pts, den = t(img, pts, den)
        return img, pts, den

# --- TRASFORMAZIONI BASE ---

class ToTensor(object):
    """Converte PIL/Numpy in Tensor [0,1] e gestisce la densità."""
    def __call__(self, img, pts=None, den=None):
        # 1. Immagine
        img = F.to_tensor(img)
        
        # 2. Densità (se presente, converte in tensore)
        if den is not None:
            if not isinstance(den, torch.Tensor):
                den = torch.from_numpy(den).float()
            if den.dim() == 2:
                den = den.unsqueeze(0) # [H, W] -> [1, H, W]
                
        return img, pts, den

class Normalize(object):
    """Normalizza con Mean/Std (default CLIP)."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, pts=None, den=None):
        img = F.normalize(img, self.mean, self.std)
        return img, pts, den

# --- TRASFORMAZIONI GEOMETRICHE (LOGICA CLIP-EBC) ---

class RandomCrop(object):
    """
    Esegue un ritaglio casuale di dimensione fissa.
    Se l'immagine è più piccola del crop, esegue padding con 0.
    Allinea Immagine, Punti e Mappa di Densità.
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, img, pts, den=None):
        w, h = img.size
        
        # 1. Padding (se l'immagine è più piccola del crop size)
        pad_w = max(0, self.size - w)
        pad_h = max(0, self.size - h)
        
        if pad_w > 0 or pad_h > 0:
            # Pad immagine (destra, basso)
            img = F.pad(img, (0, 0, pad_w, pad_h), fill=0)
            # Pad densità (se presente)
            if den is not None:
                den = np.pad(den, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        
        # Aggiorna dimensioni dopo padding
        w_new, h_new = img.size
        
        # 2. Coordinate casuali per il crop
        i = random.randint(0, h_new - self.size)
        j = random.randint(0, w_new - self.size)
        
        # 3. Crop Immagine
        img = F.crop(img, i, j, self.size, self.size)
        
        # 4. Crop Densità
        if den is not None:
            den = den[i:i+self.size, j:j+self.size]
            
        # 5. Shift e Filtro Punti
        if pts is not None and len(pts) > 0:
            pts = pts.copy()
            pts[:, 0] -= j # Shift X
            pts[:, 1] -= i # Shift Y
            
            # Mantieni solo i punti che cadono nel nuovo crop
            mask = (pts[:, 0] >= 0) & (pts[:, 0] < self.size) & \
                   (pts[:, 1] >= 0) & (pts[:, 1] < self.size)
            pts = pts[mask]
            
        return img, pts, den

class RandomHorizontalFlip(object):
    """Flip orizzontale coerente per Img, Punti e Densità."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, pts, den=None):
        if random.random() < self.p:
            w, h = img.size
            
            # 1. Flip Immagine
            img = F.hflip(img)
            
            # 2. Flip Punti
            if pts is not None and len(pts) > 0:
                pts = pts.copy()
                pts[:, 0] = w - pts[:, 0] # Inverti coordinata X
                # Clip per sicurezza (punti sul bordo esatto)
                mask = (pts[:, 0] >= 0) & (pts[:, 0] < w)
                pts = pts[mask]
                
            # 3. Flip Densità
            if den is not None:
                den = np.fliplr(den).copy() # Numpy flip è su asse 1 (W)
                
        return img, pts, den

class Resize2Multiple(object):
    """
    Ridimensiona l'immagine affinché altezza e larghezza siano multipli di 'base'.
    Fondamentale per ViT (patch size 16) e CLIP.
    """
    def __init__(self, base=16):
        self.base = base

    def __call__(self, img, pts, den=None):
        w, h = img.size
        # Calcola nuove dimensioni (arrotondamento per eccesso)
        new_h = int(math.ceil(h / self.base) * self.base)
        new_w = int(math.ceil(w / self.base) * self.base)
        
        if (new_w, new_h) == (w, h):
            return img, pts, den
            
        # Resize Immagine
        img = img.resize((new_w, new_h), Image.BICUBIC)
        
        # Scala Punti
        scale_w = new_w / w
        scale_h = new_h / h
        
        if pts is not None and len(pts) > 0:
            pts = pts.copy()
            pts[:, 0] *= scale_w
            pts[:, 1] *= scale_h
            
        # Nota: La density map solitamente viene rigenerata dai punti 
        # o non usata nel validation standard, quindi qui non la scaliamo 
        # (interpolare una density map sparsa è rischioso).
        # Se serve, ZIP la rigenererà dai punti scalati.
            
        return img, pts, den

# ==========================================================
# BUILDER
# ==========================================================

def build_transforms(cfg_data, is_train=True):
    # Default: Normalizzazione CLIP (OpenAI)
    # Se nel config non c'è, usa questi valori standard di CLIP
    mean = cfg_data.get('NORM_MEAN', [0.48145466, 0.4578275, 0.40821073])
    std = cfg_data.get('NORM_STD', [0.26862954, 0.26130258, 0.27577711])
    
    transforms_list = []
    
    if is_train:
        # TRAINING:
        # 1. Random Horizontal Flip
        transforms_list.append(RandomHorizontalFlip(p=0.5))
        
        # 2. Random Crop (Fondamentale per patch counting e CLIP)
        # Usa CROP_SIZE dal config (es. 448 o 384 per CLIP)
        crop_size = cfg_data.get('CROP_SIZE', 448) 
        transforms_list.append(RandomCrop(crop_size))
        
    else:
        # VALIDATION:
        # 1. Resize intelligente per ViT (multipli di 16)
        # Non si fa crop in validation per contare tutta l'immagine
        transforms_list.append(Resize2Multiple(base=16))

    # COMUNI:
    # 3. Conversione a Tensor e Normalizzazione
    transforms_list.append(ToTensor())
    transforms_list.append(Normalize(mean, std))
    
    return Compose(transforms_list)