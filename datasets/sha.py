# datasets/shha.py
# Versione aggiornata per supportare diverse strutture di directory
#
# Strutture supportate:
# 1. Originale ShanghaiTech: train_data/images, test_data/images
# 2. Alternativa: train/images, val/images
# 3. Con labels invece di ground_truth

import os
import glob
import numpy as np
import scipy.io as sio
from .base_dataset import BaseCrowdDataset


class SHHA(BaseCrowdDataset):
    """
    ShanghaiTech Part A Dataset Loader.
    
    Supporta multiple strutture di directory:
    - train_data/images + ground_truth/
    - train/images + labels/
    - val/images (come alias per test)
    """
    
    def get_image_list(self, split):
        """
        Trova le immagini per lo split specificato.
        
        Supporta:
        - split="train" o "train_data" → cerca train/ o train_data/
        - split="val" o "test" o "test_data" → cerca val/, test/, test_data/
        """
        # Mappa split a possibili directory
        if split in ["train", "train_data"]:
            split_dirs = ["train_data", "train"]
        elif split in ["val", "test", "test_data"]:
            split_dirs = ["val", "test_data", "test"]
        else:
            split_dirs = [split]
        
        # Cerca in ogni possibile directory
        for split_dir in split_dirs:
            # Possibili sottocartelle per le immagini
            candidates = [
                os.path.join(self.root, split_dir, "images"),
                os.path.join(self.root, split_dir, "img"),
                os.path.join(self.root, split_dir),  # Immagini direttamente nella cartella
            ]
            
            for img_dir in candidates:
                if os.path.isdir(img_dir):
                    # Cerca jpg e png
                    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
                    imgs += sorted(glob.glob(os.path.join(img_dir, "*.png")))
                    imgs += sorted(glob.glob(os.path.join(img_dir, "*.JPG")))
                    imgs += sorted(glob.glob(os.path.join(img_dir, "*.PNG")))
                    
                    if len(imgs) > 0:
                        print(f"[SHHA] Trovate {len(imgs)} immagini in {img_dir}")
                        return sorted(set(imgs))  # Rimuovi duplicati
        
        # Nessuna immagine trovata
        searched = []
        for split_dir in split_dirs:
            searched.append(os.path.join(self.root, split_dir, "images"))
            searched.append(os.path.join(self.root, split_dir))
        
        raise FileNotFoundError(
            f"Nessuna immagine trovata per split '{split}' in {self.root}\n"
            f"Cercato in: {searched}"
        )

    def load_points(self, img_path):
        """
        Carica i punti associati a un'immagine.
        
        Supporta:
        - .mat (ShanghaiTech originale): ground_truth/GT_IMG_xxx.mat
        - .npy (preprocessato): labels/IMG_xxx.npy o new-anno/GT_IMG_xxx.npy
        - .txt (formato semplice): labels/IMG_xxx.txt
        """
        # Directory base (es. data/sha/train)
        base_dir = os.path.dirname(os.path.dirname(img_path))
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        
        # === PROVA 1: File .npy in labels/ ===
        npy_paths = [
            os.path.join(base_dir, "labels", f"{base_name}.npy"),
            os.path.join(base_dir, "labels", f"GT_{base_name}.npy"),
            os.path.join(base_dir, "new-anno", f"GT_{base_name}.npy"),
        ]
        
        for npy_path in npy_paths:
            if os.path.isfile(npy_path):
                pts = np.load(npy_path)
                # Assicurati che sia Nx2
                if pts.ndim == 1:
                    pts = pts.reshape(-1, 2)
                return np.array(pts[:, :2], dtype=np.float32)
        
        # === PROVA 2: File .mat in ground_truth/ o ground-truth/ ===
        mat_paths = [
            os.path.join(base_dir, "ground_truth", f"GT_{base_name}.mat"),
            os.path.join(base_dir, "ground-truth", f"GT_{base_name}.mat"),
            os.path.join(base_dir, "gt", f"GT_{base_name}.mat"),
            os.path.join(base_dir, "gt", f"{base_name}.mat"),
        ]
        
        for mat_path in mat_paths:
            if os.path.isfile(mat_path):
                try:
                    mat = sio.loadmat(mat_path)
                    # Struttura ShanghaiTech: image_info[0,0][0,0][0]
                    if "image_info" in mat:
                        pts = mat["image_info"][0, 0][0, 0][0]
                    elif "annPoints" in mat:
                        pts = mat["annPoints"]
                    elif "points" in mat:
                        pts = mat["points"]
                    else:
                        # Prova la prima chiave che non inizia con __
                        for key in mat.keys():
                            if not key.startswith("__"):
                                pts = mat[key]
                                break
                    return np.array(pts, dtype=np.float32)
                except Exception as e:
                    print(f"[SHHA] Warning: errore lettura {mat_path}: {e}")
                    continue
        
        # === PROVA 3: File .txt in labels/ ===
        txt_paths = [
            os.path.join(base_dir, "labels", f"{base_name}.txt"),
            os.path.join(base_dir, "gt", f"{base_name}.txt"),
        ]
        
        for txt_path in txt_paths:
            if os.path.isfile(txt_path):
                pts = []
                with open(txt_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.replace(",", " ").split()
                        if len(parts) >= 2:
                            try:
                                x, y = float(parts[0]), float(parts[1])
                                pts.append([x, y])
                            except ValueError:
                                continue
                return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 2), dtype=np.float32)
        
        # Nessun file GT trovato
        raise FileNotFoundError(
            f"Nessun file ground truth trovato per {img_path}\n"
            f"Cercato:\n"
            f"  NPY: {npy_paths}\n"
            f"  MAT: {mat_paths}\n"
            f"  TXT: {txt_paths}"
        )