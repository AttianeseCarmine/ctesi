# visualize_stage1.py
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import yaml
import argparse
import os
from torchvision import transforms
from models.zip_model import ZIPModel

# === DEFAULTS ===
DEFAULT_IMG = "./data/shb/val/images/IMG_1.jpg"
# Nota: Il config di default serve solo se non ne troviamo uno salvato col modello
DEFAULT_CFG = "./configs/config_shb.yaml" 
DEFAULT_CKPT = "./checkpoints/shb_resnet50/stage1/best_model.pth"
def get_smart_config_path(ckpt_path, arg_config_path):
    """Cerca il config nella cartella del checkpoint."""
    if not ckpt_path: return arg_config_path
    ckpt_dir = os.path.dirname(ckpt_path)
    saved_cfg = os.path.join(ckpt_dir, "config.yaml")
    if os.path.exists(saved_cfg):
        print(f"🔄 Smart Load: Config trovato nel checkpoint -> {saved_cfg}")
        return saved_cfg
    print(f"⚠️  Config non trovato nel checkpoint. Uso: {arg_config_path}")
    return arg_config_path

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Stage 1 (ZIP Filter)")
    parser.add_argument('--image', type=str, default=DEFAULT_IMG)
    parser.add_argument('--config', type=str, default=DEFAULT_CFG)
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CKPT)
    parser.add_argument('--threshold', type=float, default=0.5)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️  Stage 1 Viz | Img: {args.image}")

    # 1. Configurazione Intelligente
    final_config_path = get_smart_config_path(args.checkpoint, args.config)
    with open(final_config_path, 'r') as f: config = yaml.safe_load(f)
    
    # 2. Caricamento Modello
    try:
        model = ZIPModel(config).to(device)
    except Exception as e:
        print(f"❌ Errore costruzione modello: {e}")
        return
    
    if os.path.exists(args.checkpoint):
        print(f"📥 Loading Checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        try:
            model.load_state_dict(state_dict, strict=False)
        except RuntimeError as e:
            print(f"❌ ERRORE MISMATCH ARCHITETTURA: {e}")
            print("💡 Soluzione: Il checkpoint usa un backbone diverso (es. ResNet vs VGG) rispetto al config.")
            return
    else:
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return

    model.eval()

    # 3. Caricamento Immagine
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        return

    orig_img_bgr = cv2.imread(args.image)
    if orig_img_bgr is None: return
    orig_img_rgb = cv2.cvtColor(orig_img_bgr, cv2.COLOR_BGR2RGB)
    h, w = orig_img_rgb.shape[:2]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config['DATA']['NORM_MEAN'], std=config['DATA']['NORM_STD'])
    ])
    img_tensor = transform(orig_img_rgb).unsqueeze(0).to(device)

    # 4. Inferenza
    with torch.no_grad():
        out = model(img_tensor)
        pi_logits = out['pi_logits'] if isinstance(out, dict) else out
        probs = torch.sigmoid(pi_logits) # Output: 1.0 = Persona (Keep), 0.0 = Vuoto (Kill)

    # 5. Visualizzazione (Logica Invertita Corretta)
    output_dir = "visualize"
    os.makedirs(output_dir, exist_ok=True)
    
    dset_name = config.get('DATASET', 'dataset').lower()
    backbone_cfg = config.get('BACKBONE', {})
    backbone_name = backbone_cfg.get('TYPE', 'default') if isinstance(backbone_cfg, dict) else 'default'
    save_path = os.path.join(output_dir, f"stage1_{dset_name}_{backbone_name}.png")

    probs_np = probs.squeeze().cpu().numpy()
    probs_resized = cv2.resize(probs_np, (w, h), interpolation=cv2.INTER_NEAREST)

    # --- LOGICA CORRETTA ---
    # Nel tuo training: 1 = Persona, 0 = Vuoto.
    # Quindi: Se prob > threshold (es. 0.5) => È UNA PERSONA.
    is_people = probs_resized > args.threshold  # <--- INVERTITO RISPETTO A PRIMA

    # Creiamo l'immagine finale partendo dall'originale (così i vuoti sono già ok)
    final_vis = orig_img_rgb.copy()

    # Creiamo un overlay verde SOLO dove c'è gente
    overlay = orig_img_rgb.copy()
    overlay[is_people] = [0, 255, 0] # Imposta Verde RGB

    # Applichiamo il blending solo sui pixel delle persone
    # Formula: alpha * verde + (1-alpha) * originale
    alpha = 0.4
    blended_people = cv2.addWeighted(overlay, alpha, orig_img_rgb, 1 - alpha, 0)
    
    # Sovrascriviamo nell'immagine finale SOLO i pixel dove is_people è True
    final_vis[is_people] = blended_people[is_people]
    # I pixel dove is_people è False restano quelli di orig_img_rgb (Invariati)

    # Plot e Salvataggio
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(orig_img_rgb)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(final_vis)
    plt.title(f"ZIP Detection ({backbone_name})\nGreen = People Detected (p > {args.threshold})")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"💾 Saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()