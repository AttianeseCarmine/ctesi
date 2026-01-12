import os
import glob
import numpy as np
from .base_dataset import BaseCrowdDataset


class UCF_QNRF(BaseCrowdDataset):
    """
    UCF-QNRF Dataset Loader.

    Supporta:
    - preprocess: split/images + split/labels (txt)
    - flat: txt accanto alle immagini
    - originale: .mat in split/gt
    """

    IMG_GLOBS = ("*.jpg", "*.jpeg", "*.png")

    def _glob_images(self, directory: str):
        """Ritorna lista ordinata immagini (jpg/jpeg/png) in directory."""
        imgs = []
        for pat in self.IMG_GLOBS:
            imgs.extend(glob.glob(os.path.join(directory, pat)))
        return sorted(imgs)

    def get_image_list(self, split):
        """
        Trova tutte le immagini per lo split specificato.

        Cerca in ordine:
        A) preprocess: ROOT/split/images/*
        B) originale:  ROOT/Split/images/* (Split=Train/val)
        C) flat:       ROOT/split/scene*/*
        D) flat:       ROOT/split/*
        """
        split_mapping = {
            "train": ["train", "Train"],
            "val":  ["val", "Val"],  # UCF-QNRF non ha val, usa test
        }
        split_candidates = split_mapping.get(split.lower(), [split])

        # A) preprocess e B) originale (stesso pattern "split/images", cambia solo split_name)
        for split_name in split_candidates:
            img_dir = os.path.join(self.root, split_name, "images")
            if os.path.isdir(img_dir):
                found = self._glob_images(img_dir)
                if found:
                    print(f"[UCF-QNRF] Trovate {len(found)} immagini in {img_dir}")
                    return found

        # C) flat: ROOT/split/scene*/*
        for split_name in split_candidates:
            scene_pattern = os.path.join(self.root, split_name, "scene*")
            scene_dirs = sorted(glob.glob(scene_pattern))
            imgs = []
            for scene_dir in scene_dirs:
                if os.path.isdir(scene_dir):
                    imgs.extend(self._glob_images(scene_dir))
            if imgs:
                print(f"[UCF-QNRF] Trovate {len(imgs)} immagini in {self.root}/{split_name}/scene*/")
                return sorted(imgs)

        # D) flat diretto: ROOT/split/*
        for split_name in split_candidates:
            flat_dir = os.path.join(self.root, split_name)
            if os.path.isdir(flat_dir):
                found = self._glob_images(flat_dir)
                if found:
                    print(f"[UCF-QNRF] Trovate {len(found)} immagini in {flat_dir}")
                    return found

        raise FileNotFoundError(
            f"Nessuna immagine trovata per split '{split}' in {self.root}\n"
            f"Cercato in: {split_candidates}\n"
            f"Pattern: split/images | split/scene* | split/"
        )

    def load_points(self, img_path):
        """
        Carica i punti GT per un'immagine.

        Cerca in ordine:
        1) preprocess labels/: ROOT/split/labels/base.txt
           (img_path = ROOT/split/images/base.jpg)
        2) flat: txt accanto all'immagine (stesso nome base)
        3) originale: .mat in ROOT/Split/gt/base_ann.mat o base.mat

        Ritorna: np.ndarray [N,2] float32
        """
        # Nome base immagine
        base = os.path.splitext(os.path.basename(img_path))[0]

        # 1) preprocess: .../split/images/img.ext -> .../split/labels/img.txt
        # split_dir = .../split
        split_dir = os.path.dirname(os.path.dirname(img_path))
        labels_dir = os.path.join(split_dir, "labels")
        txt_path = os.path.join(labels_dir, base + ".txt")
        if os.path.isfile(txt_path):
            pts = self._load_txt_points(txt_path)
            return np.array(pts, dtype=np.float32)

        # 2) flat: txt accanto all'immagine
        txt_path2 = os.path.splitext(img_path)[0] + ".txt"
        if os.path.isfile(txt_path2):
            pts = self._load_txt_points(txt_path2)
            return np.array(pts, dtype=np.float32)

        # 3) originale: .mat in gt/
        # img_path: ROOT/Train/images/img_0001.jpg
        # gt_path:  ROOT/Train/gt/img_0001_ann.mat
        parent_dir = os.path.dirname(os.path.dirname(img_path))  # ROOT/Train o ROOT/Test
        mat_path = os.path.join(parent_dir, "gt", f"{base}_ann.mat")
        if os.path.isfile(mat_path):
            pts = self._load_mat_points(mat_path)
            return np.array(pts, dtype=np.float32)

        mat_path2 = os.path.join(parent_dir, "gt", f"{base}.mat")
        if os.path.isfile(mat_path2):
            pts = self._load_mat_points(mat_path2)
            return np.array(pts, dtype=np.float32)

        print(f"[UCF-QNRF] Warning: GT non trovato per {img_path}")
        return np.zeros((0, 2), dtype=np.float32)

    def _load_txt_points(self, txt_path):
        """
        Carica punti da file .txt.
        Formato atteso per riga: x y
        (tollerante a virgole e spazi multipli)
        """
        pts = []
        try:
            with open(txt_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.replace(",", " ").split()
                    if len(parts) < 2:
                        continue
                    try:
                        x, y = float(parts[0]), float(parts[1])
                        # filtro minimo (evita valori NaN/negativi)
                        if np.isfinite(x) and np.isfinite(y) and x >= 0 and y >= 0:
                            pts.append([x, y])
                    except ValueError:
                        continue
        except Exception as e:
            print(f"[UCF-QNRF] Errore lettura TXT {txt_path}: {e}")
        return pts

    def _load_mat_points(self, mat_path):
        """Carica punti da file .mat (formato UCF-QNRF)."""
        import scipy.io as sio

        try:
            mat = sio.loadmat(mat_path)

            # Prova diverse chiavi comuni
            for key in ["annPoints", "image_info", "points", "gt"]:
                if key not in mat:
                    continue

                data = mat[key]

                # Gestisci struttura nested di image_info
                if key == "image_info":
                    try:
                        data = data[0][0][0][0][0]
                    except (IndexError, TypeError):
                        continue

                if hasattr(data, "shape") and len(data.shape) >= 2 and data.shape[1] >= 2:
                    arr = data[:, :2]
                    # filtra non-finiti
                    arr = arr[np.isfinite(arr).all(axis=1)]
                    return arr.tolist()

            print(f"[UCF-QNRF] Warning: chiavi disponibili in {mat_path}: {list(mat.keys())}")
            return []

        except Exception as e:
            print(f"[UCF-QNRF] Errore caricamento {mat_path}: {e}")
            return []
