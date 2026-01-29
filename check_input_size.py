# test_geometry.py
import torch
import torch.nn as nn
from models.zip_model import ZIPModel
from models import get_model

print("--- Inizio Test Geometria ---")

# 1. Configurazione ZIP (Simulata)
config_vit = {
    'BACKBONE': {'TYPE': 'vit_b_16', 'PRETRAINED': False},
    'ZIP_HEAD': {'HIDDEN_DIM': 256},
    'REDUCTION': 8,   # <-- AGGIUNGI QUESTA RIGA
}


print("1. Inizializzazione ZIP...")
zip_model = ZIPModel(config_vit)
zip_model.eval()

# 2. Inizializzazione CLIP
print("2. Inizializzazione CLIP...")
clip_model = get_model(
    backbone='clip_vit_b_16',
    input_size=448,
    reduction=8,  # <--- IMPORTANTE: Qui stai testando la riduzione 16
    bins=[(0,0), (1,1), (2,2), (3,3), (4,float('inf'))],
    anchor_points=[0, 1, 2, 3, 5],
    # --- PARAMETRI MANCANTI CHE CAUSAVANO L'ERRORE ---
    num_vpt=32,          # Default standard
    prompt_type='word',  # Default standard
    vpt_drop=0.0,
    deep_vpt=True
)
clip_model.eval()

# 3. Creazione Input Dummy (Batch=1, RGB, 448x448)
x = torch.randn(1, 3, 448, 448)
print(f"3. Input Shape: {x.shape}")

# 4. Forward Pass
print("4. Esecuzione Forward...")
with torch.no_grad():
    zip_out = zip_model(x)
    clip_out = clip_model(x)

# 5. Analisi Output ZIP
print("\n--- RISULTATI ---")
if isinstance(zip_out, dict):
    # Cerca la chiave giusta
    key = 'logits' if 'logits' in zip_out else list(zip_out.keys())[0]
    zip_shape = zip_out[key].shape
    print(f"ZIP output ({key}): {zip_shape}")
else:
    print(f"ZIP output (tensor): {zip_out.shape}")

# 6. Analisi Output CLIP
if isinstance(clip_out, dict):
    if 'density' in clip_out:
        print(f"CLIP output (density): {clip_out['density'].shape}")
    if 'ebc_logits' in clip_out:
        print(f"CLIP output (ebc_logits): {clip_out['ebc_logits'].shape}")
elif isinstance(clip_out, (tuple, list)):
    print(f"CLIP output (tuple[0]): {clip_out[0].shape}")
    if len(clip_out) > 1:
        print(f"CLIP output (tuple[1]): {clip_out[1].shape}")
else:
    print(f"CLIP output (tensor): {clip_out.shape}")

print("\n--- Fine Test ---")