import os
import glob
import cv2
import numpy as np
import scipy.io
import scipy.ndimage
from tqdm import tqdm
import shutil

# --- CONFIGURAZIONE ---
SOURCE_ROOT = "./data/ucf-qnrf"      # Dove sono le cartelle Train/Test originali
DEST_ROOT = "./data_npy/ucf-qnrf"       # Dove metteremo i dati processati
MAX_SIZE = 2048                 # Dimensione massima (lato lungo) per evitare OOM

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_density_map(shape, points, sigma=4.0):
    """
    Genera una density map con kernel Gaussiani fissi.
    Per QNRF, sigma=4 è un buon compromesso tra precisione e smoothness.
    """
    density = np.zeros(shape, dtype=np.float32)
    gt_count = len(points)
    
    if gt_count == 0:
        return density

    # Crea una mappa di punti
    for p in points:
        x, y = int(p[0]), int(p[1])
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            density[y, x] = 1

    # Applica filtro gaussiano
    density = scipy.ndimage.gaussian_filter(density, sigma=sigma, mode='constant')
    return density

def process_set(split_name, source_folder):
    print(f"🚀 Processing {split_name} set...")
    
    # Cartelle di destinazione
    img_dest_dir = os.path.join(DEST_ROOT, split_name, "images")
    den_dest_dir = os.path.join(DEST_ROOT, split_name, "density")
    ensure_dir(img_dest_dir)
    ensure_dir(den_dest_dir)
    
    # Trova tutte le immagini jpg
    img_paths = glob.glob(os.path.join(source_folder, "*.jpg"))
    
    for img_path in tqdm(img_paths):
        # 1. Carica Immagine
        basename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(basename)[0]
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Errore caricamento: {img_path}")
            continue
            
        h, w = img.shape[:2]
        
        # 2. Carica Annotazioni (.mat)
        mat_path = os.path.join(source_folder, f"{name_no_ext}_ann.mat")
        if not os.path.exists(mat_path):
            print(f"⚠️ Annotazione mancante per {basename}")
            continue
            
        mat = scipy.io.loadmat(mat_path)
        # In QNRF le coordinate sono solitamente sotto la chiave 'annPoints'
        points = mat['annPoints'].astype(float)
        
        # 3. Resize Intelligente (Se l'immagine è enorme)
        scale_ratio = 1.0
        if max(h, w) > MAX_SIZE:
            scale_ratio = MAX_SIZE / max(h, w)
            new_w = int(w * scale_ratio)
            new_h = int(h * scale_ratio)
            
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            points = points * scale_ratio # Scaliamo anche i punti!
            h, w = new_h, new_w
            
        # 4. Genera Density Map
        density = generate_density_map((h, w), points, sigma=4.0)
        
        # 5. Salva
        # Salva immagine (ridimensionata o originale)
        cv2.imwrite(os.path.join(img_dest_dir, basename), img)
        # Salva density map (.npy)
        np.save(os.path.join(den_dest_dir, f"{name_no_ext}.npy"), density)

if __name__ == "__main__":
    # QNRF ha cartelle "Train" e "Test" (con la maiuscola)
    process_set("train", os.path.join(SOURCE_ROOT, "Train"))
    process_set("val", os.path.join(SOURCE_ROOT, "Test"))
    
    print("\n✅ Preprocessing QNRF completato!")
    print(f"📂 Dati salvati in: {DEST_ROOT}")