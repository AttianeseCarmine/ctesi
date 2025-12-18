import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class JHUDataset(Dataset):
    def __init__(self, root, split, block_size=16, transforms=None):
        self.root = root
        
        # Cartelle create dallo script preprocess
        self.img_dir = os.path.join(root, split, "images")
        self.den_dir = os.path.join(root, split, "labels")
        self.transforms = transforms
        
        self.img_files = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        
        # Check
        if len(self.img_files) > 0:
            test_npy = os.path.join(self.den_dir, os.path.splitext(self.img_files[0])[0] + '.npy')
            if not os.path.exists(test_npy):
                raise RuntimeError(f"❌ JHU Dataset: Non trovo i file .npy in {self.den_dir}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        base_name = os.path.splitext(img_name)[0]
        
        # Immagine
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except:
             return {'image': torch.zeros(3, 256, 256), 'density': torch.zeros(1, 256, 256)}

        # Density
        den_path = os.path.join(self.den_dir, base_name + '.npy')
        try:
            density = np.load(den_path).astype(np.float32)
        except:
            w, h = image.size
            density = np.zeros((h, w), dtype=np.float32)

        # Transforms
        if self.transforms:
            outputs = self.transforms(image, density)
            image = outputs[0]
            density = outputs[1]
            
        # Converti density in Tensor se è numpy (Fix per collate_fn)
        if isinstance(density, np.ndarray):
            density = torch.from_numpy(np.ascontiguousarray(density))
        
        # Assicurati (1, H, W)
        if density.dim() == 2:
            density = density.unsqueeze(0)

        return {
            'image': image,
            'density': density,
            'points': torch.tensor([]) 
        }