import argparse
import os
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import yaml
from PIL import Image
from torchvision import transforms

# Import del tuo modello
from models.zip_model import ZIPModel

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tensor = tensor.clone().detach().cpu().squeeze().permute(1, 2, 0).numpy()
    img = (tensor * std) + mean
    return np.clip(img, 0, 1)

def get_transform(h, w):
    # Ridimensiona a multipli di 16 per il ViT
    new_h = math.ceil(h / 16) * 16
    new_w = math.ceil(w / 16) * 16
    return transforms.Compose([
        transforms.Resize((new_h, new_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Avvio Debug Stage 1 su: {args.image_path}")
    
    # 1. Configurazione: Carica e analizza
    dataset_name = "unknown"
    backbone_name = "unknown"

    if os.path.exists(args.config):
        print(f"📖 Leggo configurazione da: {args.config}")
        config = load_config(args.config)
        
        # Recupera Dataset
        dataset_name = config.get('dataset', 'sha')
        
        # Recupera Backbone (Gestione Robusta)
        # Prima prova la chiave piatta 'model'
        backbone_name = config.get('model')
        
        # Se 'model' non c'è, prova a scavare in BACKBONE -> TYPE
        if not backbone_name:
            backbone_cfg = config.get('BACKBONE', {})
            if isinstance(backbone_cfg, dict):
                backbone_name = backbone_cfg.get('TYPE', 'vit_b_16')
            else:
                backbone_name = 'vit_b_16' # Fallback estremo
    else:
        print("⚠️ Config file non trovato, uso valori default.")
        config = {} # Config vuoto, il modello potrebbe crashare se non gestisce default
        # In questo caso estremo, potresti voler aggiungere i parametri minimi a 'config' qui
    
    print(f"ℹ️  Dataset rilevato: {dataset_name}")
    print(f"ℹ️  Backbone rilevata: {backbone_name}")

    # Inizializza Modello
    model = ZIPModel(config).to(device)
    
    # 2. Caricamento Pesi (con controllo rigoroso)
    if os.path.isfile(args.checkpoint):
        print(f"📥 Caricamento checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        # Rimuove prefisso module. se presente
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        try:
            model.load_state_dict(new_state_dict, strict=True)
            print("✅ Pesi caricati perfettamente (Strict Mode)")
        except Exception as e:
            print(f"⚠️ Warning: Caricamento strict fallito, provo loose: {e}")
            model.load_state_dict(new_state_dict, strict=False)
    else:
        print("❌ Checkpoint non trovato!")
        return
    model.eval()

    # 3. Preparazione Immagine
    raw_img = Image.open(args.image_path).convert('RGB')
    orig_w, orig_h = raw_img.size
    transform = get_transform(orig_h, orig_w)
    img_tensor = transform(raw_img).unsqueeze(0).to(device)

    # 4. Inferenza
    with torch.no_grad():
        out = model(img_tensor)
        logits = out['pi_logits'] 
        probs = torch.sigmoid(logits)

    # 5. Analisi Statistica
    l_min, l_max, l_mean = logits.min().item(), logits.max().item(), logits.mean().item()
    p_min, p_max, p_mean = probs.min().item(), probs.max().item(), probs.mean().item()

    print("\n📊 STATISTICHE OUTPUT MODELLO:")
    print(f"   LOGITS (Raw)  -> Min: {l_min:+.4f} | Max: {l_max:+.4f} | Mean: {l_mean:+.4f}")
    print(f"   PROBS (0-1)   -> Min: {p_min:.4f}  | Max: {p_max:.4f}  | Mean: {p_mean:.4f}")
    print("-" * 60)
    

    # 6. Visualizzazione
    img_vis = denormalize(img_tensor)
    H, W = img_vis.shape[:2]
    
    heatmap = F.interpolate(probs, size=(H, W), mode='bilinear', align_corners=False).squeeze().cpu().numpy()

    # PLOT (solo 2 colonne: immagine input + heatmap)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(img_vis)
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    im = axes[1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title("Probability Heatmap (0.0 - 1.0)")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)


    # SALVATAGGIO
    img_name_clean = os.path.splitext(os.path.basename(args.image_path))[0] # DEFINITA QUI ORA!
    output_dir = "visualize"
    os.makedirs(output_dir, exist_ok=True)
    
    out_file = f"{output_dir}/stage1_heatmap_{dataset_name}_{backbone_name}_{img_name_clean}_thr{args.threshold}.png"
    
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    
    print(f"\n✅ Risultato salvato in: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_stage1.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=0.5, help="Soglia per probabilità (User Logic)")
    
    args = parser.parse_args()
    main(args)