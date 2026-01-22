import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
import random

# ==========================================
# CONFIGURAZIONE STANDARD (Nomi 00001.jpg)
# ==========================================
SRC_ROOT = "data/1_NWPU_original" 
DST_ROOT = "data/nwpu"

MIN_SIZE = 448
MAX_SIZE = 3072 

def parse_txt_file(txt_path):
    """Legge le coordinate dal file .txt originale"""
    points = []
    if not os.path.exists(txt_path):
        return np.array([])
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().replace(',', ' ').split()
            try:
                if len(parts) >= 2:
                    x = float(parts[0]) 
                    y = float(parts[1])
                    points.append([x, y])
            except ValueError:
                continue
    return np.array(points, dtype=np.float32)

def resize_and_save(img_path, txt_path, save_img_dir, save_lbl_dir, new_name_idx):
    # 1. Leggi immagine e label
    img = cv2.imread(img_path)
    if img is None: return 0
    coords = parse_txt_file(txt_path)
    
    h, w = img.shape[:2]
    
    # 2. Logica Resize (Standard CLIP-EBC)
    ratio = 1.0
    if min(h, w) < MIN_SIZE:
        ratio = MIN_SIZE / min(h, w)
    elif max(h, w) > MAX_SIZE:
        ratio = MAX_SIZE / max(h, w)
    
    new_w = int(round(w * ratio / 32) * 32)
    new_h = int(round(h * ratio / 32) * 32)
    
    img_res = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # 3. Scalatura coordinate
    scale_x = new_w / w
    scale_y = new_h / h
    if len(coords) > 0:
        coords[:, 0] *= scale_x
        coords[:, 1] *= scale_y
    
    # 4. Generazione Nome Standard (00001.jpg)
    # Usa 5 cifre per sicurezza (fino a 99.999 immagini)
    std_name = f"{new_name_idx:05d}" 
    
    # 5. Salvataggio
    cv2.imwrite(os.path.join(save_img_dir, f"{std_name}.jpg"), img_res)
    
    # Salva label solo se necessario (la cartella labels potrebbe essere None per il test)
    if save_lbl_dir is not None:
        np.save(os.path.join(save_lbl_dir, f"{std_name}.npy"), coords)
    
    return len(coords)

def process_dataset():
    print(f"🚀 Preprocessing NWPU -> Standard Format (00001.jpg)")
    
    # --- RACCOLTA FILE ---
    # Raccogliamo TUTTI i file di train originale
    all_train_files = glob.glob(os.path.join(SRC_ROOT, "train", "**", "*.jpg"), recursive=True)
    
    # Raccogliamo TUTTI i file di test originale (che useremo come Val)
    all_val_files = glob.glob(os.path.join(SRC_ROOT, "test", "**", "*.jpg"), recursive=True)
    
    # Se vuoi creare un set di Test "vero" (senza label) per il futuro, potresti dividere ulteriormente
    # ma per ora manteniamo la logica "Paper Mode": Train -> Train, Test -> Val
    
    splits = {
        'train': all_train_files,
        'val':   all_val_files
    }
    
    print(f"📸 Trovati: Train={len(all_train_files)}, Val={len(all_val_files)}")

    total_people = 0
    
    for split_name, img_list in splits.items():
        print(f"\n📂 Elaborazione Split: {split_name.upper()}...")
        
        # Cartelle Destinazione
        dst_img_dir = os.path.join(DST_ROOT, split_name, "images")
        dst_lbl_dir = os.path.join(DST_ROOT, split_name, "labels")
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)
        
        # Mischiamo la lista per non avere le scene tutte vicine (opzionale ma consigliato)
        # random.shuffle(img_list) 
        # (Commentato: meglio mantenere l'ordine per riproducibilità, o se preferisci mischiare scommentalo)
        
        count_people_split = 0
        
        # Loop con Enumerator per generare ID progressivi (1, 2, 3...)
        for idx, img_path in enumerate(tqdm(img_list)):
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            
            # idx + 1 perché vogliamo partire da 00001, non 00000
            n_people = resize_and_save(img_path, txt_path, dst_img_dir, dst_lbl_dir, idx + 1)
            count_people_split += n_people
            
        print(f"   Done {split_name}: {idx+1} immagini generate.")
        print(f"   Persone in {split_name}: {int(count_people_split)}")
        total_people += count_people_split

    print("\n✅ Finito! Dataset NWPU standardizzato.")
    print(f"   Totale Persone: {int(total_people)}")
    print(f"   I file ora si chiamano 00001.jpg, 00002.jpg, ecc.")

if __name__ == "__main__":
    process_dataset()