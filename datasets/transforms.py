import random
import math
import numbers
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, pts=None, den=None):
        for t in self.transforms:
            img, pts, den = t(img, pts, den)
        return img, pts, den

# --- Trasformazioni Base ---

class ToTensor(object):
    def __call__(self, img, pts=None, den=None):
        if isinstance(img, Image.Image):
            img = F.to_tensor(img)
        return img, pts, den

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, pts=None, den=None):
        img = F.normalize(img, self.mean, self.std)
        return img, pts, den

# --- Trasformazioni Geometriche (Simil-CLIP-EBC) ---

class RandomCrop(object):
    """Random Crop allineato a CLIP-EBC."""
    def __init__(self, size):
        self.size = size

    def __call__(self, img, pts, den=None):
        w, h = img.size
        # Se l'immagine è più piccola del crop, pad con 0
        if w < self.size or h < self.size:
            pad_w = max(0, self.size - w)
            pad_h = max(0, self.size - h)
            img = F.pad(img, (0, 0, pad_w, pad_h), fill=0) # Pad destra/basso
            # Se avessi densità, padderesti anche quella
            w, h = img.size # Nuove dimensioni

        i = random.randint(0, h - self.size)
        j = random.randint(0, w - self.size)
        
        img = F.crop(img, i, j, self.size, self.size)
        
        # Aggiusta i punti
        if pts is not None and len(pts) > 0:
            pts = pts.copy() # Non modificare l'originale
            pts[:, 0] -= j # x
            pts[:, 1] -= i # y
            # Filtra punti fuori dal crop
            mask = (pts[:, 0] >= 0) & (pts[:, 0] < self.size) & \
                   (pts[:, 1] >= 0) & (pts[:, 1] < self.size)
            pts = pts[mask]
            
        return img, pts, den

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, pts, den=None):
        if random.random() < self.p:
            w, h = img.size
            img = F.hflip(img)
            if pts is not None and len(pts) > 0:
                pts[:, 0] = w - pts[:, 0] # Inverti X
                # Nota: i punti esattamente sul bordo potrebbero uscire, ma ok
                mask = (pts[:, 0] >= 0) & (pts[:, 0] < w)
                pts = pts[mask]
        return img, pts, den

class Resize2Multiple(object):
    """
    Ridimensiona l'immagine affinché i lati siano multipli di 'base'.
    Fondamentale per ViT e CLIP che lavorano a patch (es. 16 o 14).
    """
    def __init__(self, base=16):
        self.base = base

    def __call__(self, img, pts, den=None):
        w, h = img.size
        # Calcola nuove dimensioni (arrotonda per eccesso o difetto, qui eccesso standard)
        new_h = int(math.ceil(h / self.base) * self.base)
        new_w = int(math.ceil(w / self.base) * self.base)
        
        if (new_w, new_h) == (w, h):
            return img, pts, den
            
        img = img.resize((new_w, new_h), Image.BICUBIC)
        
        # Scala i punti
        if pts is not None and len(pts) > 0:
            scale_w = new_w / w
            scale_h = new_h / h
            pts[:, 0] *= scale_w
            pts[:, 1] *= scale_h
            
        return img, pts, den

# --- Builder ---

def build_transforms(cfg_data, is_train=True):
    mean = cfg_data.get('NORM_MEAN', [0.48145466, 0.4578275, 0.40821073])
    std = cfg_data.get('NORM_STD', [0.26862954, 0.26130258, 0.27577711])
    
    transforms_list = []
    
    if is_train:
        # 1. Random Crop (Standard per training)
        crop_size = cfg_data.get('CROP_SIZE', 256)
        transforms_list.append(RandomCrop(crop_size))
        
        # 2. Flip
        transforms_list.append(RandomHorizontalFlip(p=0.5))
    else:
        # 1. Validation: Nessun crop, solo resize intelligente per ViT
        # CLIP ViT usa patch size 16 o 14. 
        # ZIP (il tuo modello) usa patch size 16.
        # Quindi forziamo multipli di 16.
        transforms_list.append(Resize2Multiple(base=16))

    # 3. ToTensor e Normalize (Sempre alla fine)
    transforms_list.append(ToTensor())
    transforms_list.append(Normalize(mean, std))
    
    return Compose(transforms_list)