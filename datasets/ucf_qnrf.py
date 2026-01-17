import os
import glob
import numpy as np
import scipy.io as sio
import torch
from .base_dataset import BaseCrowdDataset

class UCF_QNRF(BaseCrowdDataset):
    """
    UCF-QNRF Dataset Loader (NPY Support Added).
    Supporta:
    - .npy (Priorità 1: Formato veloce)
    - .mat (Priorità 2: Formato originale)
    - .txt (Priorità 3: Formato semplice)
    """

    def get_image_list(self, split):
        """Trova tutte le immagini nella cartella split."""
        if split.lower() == 'train':
            split_names = ['train', 'Train', 'train_data']
        else:
            split_names = ['test', 'Test', 'val', 'Val', 'test_data']

        img_list = []
        
        # Cerca le immagini ricorsivamente
        for s_name in split_names:
            # Paths comuni
            paths_to_check = [
                os.path.join(self.root, s_name),             # data/qnrf/train
                os.path.join(self.root, s_name, 'images'),   # data/qnrf/train/images
            ]
            
            for p in paths_to_check:
                if os.path.isdir(p):
                    print(f"[UCF-QNRF] Scansiono cartella immagini: {p}")
                    for ext in ['*.jpg', '*.JPG', '*.png', '*.jpeg']:
                        found = glob.glob(os.path.join(p, ext))
                        img_list.extend(found)

        img_list = sorted(list(set(img_list)))
        print(f"[UCF-QNRF] Trovate {len(img_list)} immagini per lo split '{split}'")
        return img_list

    def load_points(self, img_path):
        """
        Carica i punti GT cercando .npy, .mat o .txt.
        """
        base_dir = os.path.dirname(img_path) # es. data/qnrf/train/images
        filename = os.path.basename(img_path) # es. 1107.jpg
        name_no_ext = os.path.splitext(filename)[0] # es. 1107
        
        parent_dir = os.path.dirname(base_dir) # es. data/qnrf/train
        
        candidates = []
        
        # Generiamo le varianti del nome (con e senza prefisso 'img_')
        names_to_check = [name_no_ext]
        if name_no_ext.startswith('img_'):
            names_to_check.append(name_no_ext.replace('img_', ''))
        
        # Cartelle dove cercare i GT
        # Nota: Aggiunto base_dir per il caso "flat" (immagini e npy insieme)
        search_dirs = [base_dir] 
        # Aggiungi cartelle parallele comuni
        for d in ['gt', 'ground_truth', 'labels', 'annotations', 'maps', 'npy_gt']:
            search_dirs.append(os.path.join(parent_dir, d))

        # COSTRUISCI LISTA CANDIDATI
        for directory in search_dirs:
            if not os.path.isdir(directory): continue
            
            for name in names_to_check:
                # Priorità 1: .npy (Quello che hai tu)
                candidates.append(os.path.join(directory, name + ".npy"))
                # Priorità 2: .mat
                candidates.append(os.path.join(directory, name + "_ann.mat"))
                candidates.append(os.path.join(directory, name + ".mat"))
                # Priorità 3: .txt
                candidates.append(os.path.join(directory, name + ".txt"))

        # CERCA IL FILE
        for path in candidates:
            if os.path.exists(path):
                if path.endswith('.npy'):
                    return self._load_npy(path)
                elif path.endswith('.mat'):
                    return self._load_mat(path)
                elif path.endswith('.txt'):
                    return self._load_txt(path)
        
        # DEBUG: Stampa solo se fallisce veramente
        # print(f"⚠️ GT mancante per {filename}. Cercato npy/mat/txt in {search_dirs}") 
        return np.zeros((0, 2), dtype=np.float32)

    def _load_npy(self, path):
        try:
            # Carica il file numpy
            pts = np.load(path)
            # Assicurati che sia float32 e shape (N, 2)
            return pts.astype(np.float32)
        except Exception as e:
            print(f"Errore lettura NPY {path}: {e}")
            return np.zeros((0, 2), dtype=np.float32)

    def _load_mat(self, path):
        try:
            mat = sio.loadmat(path)
            if 'annPoints' in mat:
                pts = mat['annPoints']
            elif 'image_info' in mat:
                pts = mat['image_info'][0, 0][0, 0][0]
            elif 'points' in mat:
                pts = mat['points']
            else:
                return np.zeros((0, 2), dtype=np.float32)
            return pts.astype(np.float32)
        except Exception:
            return np.zeros((0, 2), dtype=np.float32)

    def _load_txt(self, path):
        pts = []
        try:
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().replace(',', ' ').split()
                    if len(parts) >= 2:
                        pts.append([float(parts[0]), float(parts[1])])
        except Exception:
            pass
        return np.array(pts, dtype=np.float32)