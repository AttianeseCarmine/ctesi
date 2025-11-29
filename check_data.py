import torch
from torch.utils.data import DataLoader
from datasets.crowd import Crowd
import numpy as np
import os

# Configurazione (come nel tuo yaml)
dataset_name = "sha"
split = "train"
root_check = "./data/sha" # Percorso dove dovrebbero essere i dati

print(f"🔍 Verifica preliminare percorsi...")
if not os.path.exists(root_check):
    print(f"❌ ATTENZIONE: La cartella {root_check} non esiste! Controlla di aver spostato i dati.")
else:
    print(f"✅ Cartella dati trovata: {root_check}")

try:
    # 1. Inizializza il Dataset (senza trasformazioni complesse per ora)
    print("\n📦 Caricamento Dataset...")
    train_set = Crowd(
        dataset=dataset_name, 
        split=split, 
        transforms=None, # Carichiamo i dati grezzi per controllo
        return_filename=True 
    )
    print(f"✅ Dataset inizializzato. Trovati {len(train_set)} campioni.")

    # 2. Test del DataLoader
    loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    
    print("\n🔄 Avvio scansione batch di prova...")
    for i, batch in enumerate(loader):
        # Il dataset restituisce: images, points (labels), density_maps, image_names
        images, labels, den_maps, filenames = batch
        
        # Check dimensioni
        print(f"\nBatch {i+1}:")
        print(f"  - Immagini: {images.shape} (Range: {images.min():.2f} - {images.max():.2f})")
        
        # Check Label (.npy)
        # Nota: le label qui sono tensori di coordinate o density map a seconda di come hai fatto il dataset.
        # Nello script crowd.py originale, 'label' sono coordinate dei punti.
        # Verifichiamo se ci sono punti per ogni immagine
        for j, fname in enumerate(filenames):
            n_points = labels[j].shape[0] # Se labels contiene le coordinate
            # Oppure se labels è una lista di tensori a causa del collate_fn personalizzato, adattare qui.
            # Assumiamo output standard del tuo script Crowd
            
            print(f"  - {fname}: {n_points} persone annotate (gt)")

        # Check Density Maps (generata al volo)
        print(f"  - Density Map: {den_maps.shape} (Somma conteggio: {den_maps.sum().item():.2f})")

        # Check NaN (Cruciale per il tuo errore precedente)
        if torch.isnan(images).any():
            print(f"💀 ERRORE: NaN rilevati nelle IMMAGINI del file {filenames}")
            break
        if torch.isnan(den_maps).any():
            print(f"💀 ERRORE: NaN rilevati nelle DENSITY MAPS del file {filenames}")
            break
            
        if i >= 2: # Controlliamo solo i primi 3 batch
            print("\n✅ Check completato con successo sui primi batch!")
            break

except Exception as e:
    print(f"\n❌ ERRORE CRITICO durante il caricamento: {e}")
    import traceback
    traceback.print_exc()