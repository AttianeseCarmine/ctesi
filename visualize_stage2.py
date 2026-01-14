import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import yaml

# Import del tuo progetto (Stage 1)
from models.zip_model import ZIPModel
from datasets.transforms import build_transforms
from datasets.utils import generate_density_map

# --- CONFIGURAZIONE IMPORT CLIP ESTERNO ---
import sys
# Aggiungi il path della libreria esterna CLIP-EBC
sys.path.append("external_libs/CLIP-EBC") 
# Nota: Adatta questo import in base a come si chiama la classe nel repo ufficiale
# Supponiamo si chiami CLIP_EBC o get_model
try:
    from models import get_model as get_clip_model # Esempio tipico
except ImportError:
    print("⚠️ Attenzione: Non riesco a importare CLIP-EBC da external_libs.")
    print("   Assicurati di aver clonato il repo e aggiustato il sys.path.")

def load_config(path):
    with open(path, 'r') as f: return yaml.safe_load(f)

def denormalize(tensor, mean, std):
    """Denormalizza l'immagine per la visualizzazione (Tensor -> Numpy RGB)"""
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    img = tensor * std + mean
    return img.permute(1, 2, 0).cpu().numpy().clip(0, 1)

def run_visualization(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Carica Configurazioni
    zip_cfg = load_config(args.zip_config)
    
    # 2. Carica Modello ZIP (Stage 1)
    print(f"🔹 Loading ZIP Model (Stage 1)...")
    zip_model = ZIPModel(zip_cfg).to(device)
    zip_ckpt = torch.load(args.zip_ckpt, map_location=device)
    # Gestione state_dict (rimuove 'module.' se presente)
    state = zip_ckpt.get('model', zip_ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    zip_model.load_state_dict(state)
    zip_model.eval()
    
    # 3. Carica Modello CLIP (Stage 2 - Black Box)
    print(f"🔹 Loading CLIP Model (Stage 2)...")
    # Esempio di caricamento (adatta in base alla API del repo ufficiale)
    # clip_model = get_clip_model(backbone="resnet50", input_size=224, reduction=8)
    # clip_model.load_state_dict(torch.load(args.clip_ckpt))
    # clip_model.eval().to(device)
    
    # 4. Preparazione Dati
    # Usiamo le trasformazioni di Validation (Resize2Multiple, Norm CLIP)
    val_trans = build_transforms(zip_cfg['DATA'], is_train=False)
    mean = zip_cfg['DATA'].get('NORM_MEAN', [0.481, 0.457, 0.408])
    std = zip_cfg['DATA'].get('NORM_STD', [0.268, 0.261, 0.275])

    # Trova immagini
    img_files = sorted([f for f in os.listdir(args.img_dir) if f.endswith(('.jpg', '.png'))])
    
    # Crea cartella output
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"📸 Inizio Visualizzazione su {len(img_files)} immagini...")
    
    for fname in tqdm(img_files):
        img_path = os.path.join(args.img_dir, fname)
        img_pil = Image.open(img_path).convert('RGB')
        
        # --- A. Inferenza ZIP ---
        img_tensor, _, _ = val_trans(img_pil, None, None)
        img_tensor = img_tensor.unsqueeze(0).to(device) # [1, 3, H, W]
        
        with torch.no_grad():
            out_zip = zip_model(img_tensor)
            pi_logits = out_zip['pi_logits'] # [1, 1, H/16, W/16]
            prob_map = torch.sigmoid(pi_logits)
            
            # Upsample della mappa di probabilità alle dimensioni originali per la maschera
            mask_low = (prob_map > args.threshold).float()
            mask_high = F.interpolate(mask_low, size=img_tensor.shape[2:], mode='nearest')
            
            # --- B. Inferenza CLIP (Simulata/Reale) ---
            # Qui applicheresti la maschera all'immagine o estrarresti le patch
            # Esempio logico:
            # masked_img = img_tensor * mask_high
            # pred_count = clip_model(masked_img).item() 
            
            # Placeholder per il conteggio (dato che non ho il modello CLIP caricato)
            pred_count = 0.0 # Sostituisci con la chiamata reale a CLIP
            
        # --- C. Visualizzazione ---
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # 1. Immagine Originale
        img_np = denormalize(img_tensor[0], mean, std)
        axes[0].imshow(img_np)
        axes[0].set_title(f"Input: {fname}")
        axes[0].axis('off')
        
        # 2. ZIP Probability Map
        prob_map_np = prob_map[0, 0].cpu().numpy()
        axes[1].imshow(prob_map_np, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title("ZIP Probability Map")
        axes[1].axis('off')
        
        # 3. Maschera Binaria (Threshold)
        mask_np = mask_high[0, 0].cpu().numpy()
        axes[2].imshow(mask_np, cmap='gray')
        axes[2].set_title(f"ZIP Mask (Thr={args.threshold})")
        axes[2].axis('off')
        
        # 4. Risultato Finale (Overlay)
        axes[3].imshow(img_np)
        axes[3].imshow(mask_np, cmap='jet', alpha=0.4) # Overlay semitrasparente
        axes[3].set_title(f"Stage 2 Prediction\nCount: {pred_count:.1f}")
        axes[3].axis('off')
        
        # Salva
        save_path = os.path.join(args.out_dir, f"vis_stage2_{fname}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_config", type=str, required=True, help="Path al config dello Stage 1")
    parser.add_argument("--zip_ckpt", type=str, required=True, help="Path al checkpoint .pth di ZIP")
    parser.add_argument("--clip_ckpt", type=str, required=True, help="Path al checkpoint .pth di CLIP")
    parser.add_argument("--img_dir", type=str, required=True, help="Cartella immagini di test")
    parser.add_argument("--out_dir", type=str, default="visualize/stage2_results")
    parser.add_argument("--threshold", type=float, default=0.3, help="Soglia per la maschera ZIP")
    
    args = parser.parse_args()
    run_visualization(args)