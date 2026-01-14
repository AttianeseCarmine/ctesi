import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image 

class BaseCrowdDataset(Dataset):
    def __init__(self, root, split, transforms=None):
        self.root = root
        self.split = split
        self.transforms = transforms 
        self.image_list = self.get_image_list(split)
        
        if not self.image_list:
             raise FileNotFoundError(f"Nessuna immagine trovata per split '{split}' in root '{root}'")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        img_path = self.image_list[i]
        
        # 1. Carica Immagine e Punti
        img = Image.open(img_path).convert("RGB")
        pts = self.load_points(img_path) 
        
        # 2. Applica Trasformazioni (Simil-CLIP)
        # Nota: Qui non passiamo densità alle trasformazioni, la generiamo DOPO 
        # per essere sicuri che corrisponda ai punti trasformati.
        if self.transforms:
            img_tensor, pts_transformed, _ = self.transforms(img, pts, None)
        else:
            img_tensor = img
            pts_transformed = pts

        # 3. Genera Densità Sparsa (Target per ZIP) dai punti trasformati
        # ZIP richiede una mappa con 1 dove c'è la testa.
        h, w = img_tensor.shape[1], img_tensor.shape[2]
        den_tensor = self.points_to_sparse_density(pts_transformed, h, w)

        return {
            "image": img_tensor,
            "points": torch.from_numpy(pts_transformed).float() if pts_transformed is not None else torch.zeros((0, 2)),
            "density": den_tensor,
            "img_path": img_path,
        }

    def points_to_sparse_density(self, points, h, w):
        """Crea mappa densità sparsa (ZIP style)"""
        den = torch.zeros((1, h, w), dtype=torch.float32)
        if points is not None and len(points) > 0:
            # Arrotonda coordinate
            pts_long = np.round(points).astype(int)
            for pt in pts_long:
                x, y = pt[0], pt[1]
                if 0 <= y < h and 0 <= x < w:
                    den[0, y, x] = 1.0
        return den

    def get_image_list(self, split):
        raise NotImplementedError
    
    def load_points(self, img_path):
        raise NotImplementedError