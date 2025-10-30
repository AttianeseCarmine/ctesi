import os
import numpy as np
import scipy.io as sio
from glob import glob
from tqdm import tqdm

# --- Configurazione ---

# Percorso della tua cartella di progetto (dove si trova questo script)
PROJECT_DIR = "/user/cattianese/ctesi" 

# Cartella di input (dove si trovano i .mat originali)
SRC_DIR = "/mnt/localstorage/cattianese/ShanghaiTech"

# Cartella di output (dove salveremo i .npy)
# Verrà creata DENTRO LA CARTELLA DEL PROGETTO
DEST_DIR = os.path.join(PROJECT_DIR, "data_npy")
# --------------------

# Mappatura tra i nomi delle cartelle originali e quelli attesi dallo script
# (Originale) -> (Atteso dallo script)
dataset_map = {
    "part_A": "sha",
    "part_B": "shb"
}

# Mappatura degli split
split_map = {
    "train_data": "train",
    "test_data": "val"  # Lo script 'test.py' usa 'val' per i test
}

def convert_files(src_path, dest_path):
    print(f"Conversione da {src_path} a {dest_path}...")
    
    # 1. Trova tutti i file .mat
    # Nota: lo script si aspetta GT_IMG_X.mat -> IMG_X.npy
    mat_files = glob(os.path.join(src_path, "ground-truth", "GT_*.mat"))
    if not mat_files:
        print(f"Attenzione: Nessun file 'GT_*.mat' trovato in {src_path}")
        return

    # 2. Crea la cartella di destinazione
    os.makedirs(dest_path, exist_ok=True)

    # 3. Converti ogni file
    for mat_file_path in tqdm(mat_files, desc="Convertendo"):
        try:
            # Carica il .mat
            mat_data = sio.loadmat(mat_file_path)
            
            # Estrai le coordinate (questa è la struttura standard di ShanghaiTech)
            # Corrisponde a image_info[0, 0][0, 0][0]
            coordinates = mat_data['image_info'][0, 0][0, 0][0]
            
            # Crea il nome del file di output
            base_name = os.path.basename(mat_file_path) # Es: GT_IMG_100.mat
            out_name = base_name.replace("GT_", "").replace(".mat", ".npy") # Es: IMG_100.npy
            out_file_path = os.path.join(dest_path, out_name)

            # Salva come .npy
            np.save(out_file_path, coordinates)

        except Exception as e:
            print(f"\nErrore durante la conversione di {mat_file_path}: {e}")

# --- Esecuzione ---
print("Avvio pre-processing dei file .mat in .npy...")

for part_name, dataset_name in dataset_map.items():
    for split_orig, split_new in split_map.items():
        
        # /mnt/localstorage/cattianese/ShanghaiTech/part_A/train_data
        src_folder = os.path.join(SRC_DIR, part_name, split_orig)
        
        # /user/cattianese/ctesi/data_npy/sha/train/labels
        dest_folder = os.path.join(DEST_DIR, dataset_name, split_new, "labels")
        
        if os.path.isdir(src_folder):
            convert_files(src_folder, dest_folder)
        else:
            print(f"Cartella sorgente non trovata: {src_folder}, salto.")

print("Conversione completata.")
print(f"I file .npy sono stati salvati in: {DEST_DIR}")