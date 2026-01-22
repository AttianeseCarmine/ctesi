import os
import numpy as np
import glob
from tqdm import tqdm
import sys

# CONFIGURAZIONE
# Assicurati che questo percorso punti alla tua cartella dati radice
DATA_ROOT = "data/nwpu" 

def check_dataset_integrity(root_dir):
    print(f"🔍 Avvio controllo integrità su: {root_dir}")
    
    if not os.path.exists(root_dir):
        print(f"❌ ERRORE: La cartella {root_dir} non esiste!")
        return

    for split in ['train', 'val']:
        print(f"\n" + "="*50)
        print(f"📂 Analisi split: {split.upper()}")
        print("="*50)
        
        # Percorso etichette (Ground Truth Points)
        labels_dir = os.path.join(root_dir, split, 'labels')
        
        if not os.path.exists(labels_dir):
            print(f"⚠️  Cartella labels non trovata: {labels_dir}")
            continue
            
        npy_files = glob.glob(os.path.join(labels_dir, "*.npy"))
        print(f"📄 Trovati {len(npy_files)} file .npy")
        
        if len(npy_files) == 0:
            print("❌ NESSUN FILE .npy TROVATO! Hai eseguito preprocess.py?")
            continue

        valid_count = 0
        empty_count = 0  # File leggibile ma con 0 persone
        corrupt_count = 0 # File non leggibile
        total_people = 0
        
        # Barra di progresso
        pbar = tqdm(npy_files, desc=f"Checking {split}")
        
        for fpath in pbar:
            try:
                # Prova a caricare il file
                points = np.load(fpath)
                
                # Controllo integrità contenuto
                if points.size == 0:
                    empty_count += 1
                else:
                    # Verifica forma (N, 2)
                    if len(points.shape) != 2 or points.shape[1] != 2:
                        print(f"\n⚠️ Forma anomala in {os.path.basename(fpath)}: {points.shape}")
                        corrupt_count += 1
                        continue
                        
                    # Verifica valori negativi (coordinate impossibili)
                    if points.min() < 0:
                        print(f"\n⚠️ Coordinate negative in {os.path.basename(fpath)}")
                    
                    valid_count += 1
                    total_people += len(points)
                    
            except Exception as e:
                print(f"\n❌ ERRORE lettura file {os.path.basename(fpath)}: {e}")
                corrupt_count += 1

        # RIEPILOGO
        print(f"\n📊 RIEPILOGO {split.upper()}:")
        print(f"   ✅ File Validi (con persone): {valid_count}")
        print(f"   ☁️  File Vuoti (0 persone):   {empty_count} (Normale se basso, sospetto se 100%)")
        print(f"   ❌ File Corrotti (Errori):    {corrupt_count}")
        print(f"   👥 Totale Persone Contate:    {total_people}")
        
        if valid_count > 0:
            avg = total_people / (valid_count + empty_count)
            print(f"   📈 Media persone per img:     {avg:.2f}")
        else:
            print("   ⚠️  ATTENZIONE: Non sono state trovate persone!")

if __name__ == "__main__":
    check_dataset_integrity(DATA_ROOT)