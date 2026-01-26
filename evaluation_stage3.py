import sys
import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

# Aggiunge la directory corrente al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import specifici del tuo progetto
from models import get_model
from models.zip_model import ZIPModel
# Se SHA non è disponibile direttamente, usa il fallback
try:
    from datasets.sha import SHA
except ImportError:
    from datasets.crowd import Crowd as SHA
    
import torchvision.transforms as T

# --- IMPOSTAZIONI ---
WINDOW_SIZE = 224  # Dimensione patch (allineata al training di CLIP/ViT)
STRIDE = 224       # Passo (uguale a size = no sovrapposizione, più veloce)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model_weights(model, checkpoint_path, device):
    """Carica i pesi gestendo prefissi e formati diversi."""
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Warning: Checkpoint not found at {checkpoint_path}")
        return model
        
    print(f"🔄 Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Gestione 'model', 'model_state_dict', 'state_dict'
    if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
        state_dict = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Rimuove prefisso 'module.'
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Caricamento flessibile
    msg = model.load_state_dict(new_state_dict, strict=False)
    # print(f"   Missing keys: {len(msg.missing_keys)}") 
    model.to(device)
    model.eval()
    return model

def predict_patch_joint(zip_model, clip_model, patch, threshold):
    """
    Esegue inferenza congiunta su una singola patch.
    """
    # 1. ZIP (Filtro)
    out_zip = zip_model(patch)
    if isinstance(out_zip, dict):
        pi_logits = out_zip.get('pi_logits', out_zip.get('logit_pi'))
    else:
        pi_logits = out_zip
    
    pi_prob = torch.sigmoid(pi_logits)

    # 2. CLIP (Conteggio)
    out_clip = clip_model(patch)
    if isinstance(out_clip, (tuple, list)):
        raw_density = out_clip[1]
    elif isinstance(out_clip, dict):
        raw_density = out_clip['density']
    else:
        raw_density = out_clip

    # 3. Hard Masking
    # Resize della maschera ZIP per matchare la densità CLIP
    if pi_prob.shape[-2:] != raw_density.shape[-2:]:
        pi_prob = F.interpolate(pi_prob, size=raw_density.shape[-2:], mode='bilinear', align_corners=False)

    # Crea la maschera (1 se prob > th, 0 altrimenti)
    mask = (pi_prob > threshold).float()
    
    # Applica filtro
    final_density = raw_density * mask
    
    return final_density.sum().item()

def run_evaluation(loader, zip_model, clip_model, device, thresholds):
    """
    Esegue la sliding window sull'intero dataset per diverse soglie.
    """
    stats = {t: {'mae': 0.0, 'mse': 0.0} for t in thresholds}
    total_samples = 0
    
    print(f"\n🚀 Avvio Valutazione Sliding Window (Size: {WINDOW_SIZE}, Stride: {STRIDE})")
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            images = batch['image'] # Tensore [B, C, H, W]
            
            # Ground truth count
            if isinstance(batch['points'], list):
                gt_counts = [len(p) for p in batch['points']]
            else:
                gt_counts = batch['counts'].tolist()
            
            # Loop sulle immagini del batch (generalmente batch_size=1 per eval)
            for idx, img in enumerate(images):
                gt = gt_counts[idx]
                img = img.unsqueeze(0) # [1, C, H, W]
                
                _, _, h_img, w_img = img.shape
                
                # Calcola padding necessario per rendere l'immagine divisibile per la finestra
                pad_h = math.ceil(h_img / WINDOW_SIZE) * WINDOW_SIZE - h_img
                pad_w = math.ceil(w_img / WINDOW_SIZE) * WINDOW_SIZE - w_img
                
                # Padding (Left, Right, Top, Bottom)
                if pad_h > 0 or pad_w > 0:
                    img_padded = F.pad(img, (0, pad_w, 0, pad_h))
                else:
                    img_padded = img
                
                ph, pw = img_padded.shape[2:]
                
                # --- SLIDING WINDOW ---
                # Raccogliamo i conti per ogni threshold per questa immagine
                img_counts = {t: 0.0 for t in thresholds}
                
                for i in range(0, ph, STRIDE):
                    for j in range(0, pw, STRIDE):
                        # Estrai patch
                        patch = img_padded[:, :, i:i+WINDOW_SIZE, j:j+WINDOW_SIZE].to(device)
                        
                        # Ottimizzazione: forward una volta sola per patch
                        out_zip = zip_model(patch)
                        zip_logits = out_zip['pi_logits'] if isinstance(out_zip, dict) else out_zip
                        zip_prob = torch.sigmoid(zip_logits)
                        
                        out_clip = clip_model(patch)
                        clip_dens = out_clip[1] if isinstance(out_clip, (tuple, list)) else out_clip['density'] if isinstance(out_clip, dict) else out_clip
                        
                        # Resize
                        if zip_prob.shape[-2:] != clip_dens.shape[-2:]:
                            zip_prob = F.interpolate(zip_prob, size=clip_dens.shape[-2:], mode='bilinear')
                            
                        # Calcola count per ogni threshold
                        for t in thresholds:
                            mask = (zip_prob > t).float()
                            filtered_dens = clip_dens * mask
                            img_counts[t] += filtered_dens.sum().item()
                
                # Aggiorna metriche globali
                for t in thresholds:
                    err = abs(img_counts[t] - gt)
                    stats[t]['mae'] += err
                    stats[t]['mse'] += err ** 2
                
                total_samples += 1

    # --- STAMPA RISULTATI ---
    print("\n" + "="*50)
    print(f"📊 RISULTATI ({total_samples} immagini analizzate)")
    print(f"{'Threshold':<10} | {'MAE':<10} | {'MSE':<10}")
    print("-" * 38)
    
    best_mae = float('inf')
    best_t = -1.0

    for t in thresholds:
        avg_mae = stats[t]['mae'] / total_samples
        avg_mse = (stats[t]['mse'] / total_samples) ** 0.5
        print(f"{t:<10.1f} | {avg_mae:<10.4f} | {avg_mse:<10.4f}")
        
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_t = t
            
    print("-" * 38)
    print(f"🌟 Best Threshold: {best_t} (MAE: {best_mae:.4f})")
    print("="*50 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_zip', type=str, required=True)
    parser.add_argument('--config_clip', type=str, required=True)
    parser.add_argument('--ckpt_zip', type=str, required=True)
    parser.add_argument('--ckpt_clip', type=str, required=True)
    parser.add_argument('--dataset_root', type=str, default='./data_npy/shb')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Dataset senza Resize aggressivo
    # Usiamo una trasformazione che converte solo in tensore e normalizza
    val_trans = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"📂 Loading Dataset: {args.dataset_root}")
    # Nota: transform=val_trans impedisce il resize a 224x224 di default
    val_dataset = SHA(root=args.dataset_root, split='val', transform=val_trans)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # 2. Configs
    cfg_zip = load_config(args.config_zip)
    cfg_clip = load_config(args.config_clip)
    
    # 3. Models
    print("\n🏗️  Loading ZIP Model...")
    # Parametri ZIP dal config
    zip_params = cfg_zip['model'] if 'model' in cfg_zip else cfg_zip
    zip_model = ZIPModel(**zip_params).to(device)
    zip_model = load_model_weights(zip_model, args.ckpt_zip, device)
    
    print("\n🏗️  Loading CLIP Model...")
    # Parametri CLIP
    # Trucco: creiamo un oggetto Namespace finto per get_model se necessario
    class ConfigObj:
        def __init__(self, d): self.__dict__.update(d)
    
    clip_dict = cfg_clip['model'] if 'model' in cfg_clip else cfg_clip
    # Aggiungi default se mancano (allineamento con trainer.py)
    if 'prompt_type' not in clip_dict: clip_dict['prompt_type'] = 'word'
    
    # Tentativo robusto di caricamento CLIP
    try:
        # Prova con get_model e wrapper argomenti
        clip_args = ConfigObj(clip_dict)
        # get_model richiede campi specifici, li estraiamo dal dict
        clip_model = get_model(clip_args, device)
    except Exception as e:
        print(f"⚠️ get_model fallito ({e}). Provo istanziazione diretta...")
        # Fallback: assume che sia CLIPEBCModel o simile
        from models.model import CLIPEBCModel # Adatta se il nome è diverso
        clip_model = CLIPEBCModel(**clip_dict).to(device)
        
    clip_model = load_model_weights(clip_model, args.ckpt_clip, device)
    
    # 4. Esecuzione
    # Testiamo range di threshold
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    run_evaluation(val_loader, zip_model, clip_model, device, thresholds)