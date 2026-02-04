import argparse
import os
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import yaml
from PIL import Image, ImageDraw
from torchvision import transforms

# Import del tuo modello
from models.zip_model import ZIPModel

# ==============================================================================
# 1. FUNZIONI DI UTILITÀ
# ==============================================================================
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def denormalize(tensor):
    """Converte il tensore (normalizzato per la rete) in immagine visibile."""
    # Valori standard ImageNet usati in datasets/crowd.py
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    tensor = tensor.clone().detach().cpu()
    img = tensor.permute(1, 2, 0).numpy()
    
    # Denormalizza: input = (output * std) + mean
    img = (img * std) + mean
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def get_smart_transform(h, w):
    """Ridimensiona l'immagine per essere digeribile dal ViT (multiplo di 16)."""
    patch_size = 16
    new_h = math.ceil(h / patch_size) * patch_size
    new_w = math.ceil(w / patch_size) * patch_size
    
    # Valori standard ImageNet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize((new_h, new_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def load_gt_points(image_path):
    """
    Legge la VERITÀ dal file .npy delle etichette.
    Cerca in .../labels/IMG_X.npy basandosi su .../images/IMG_X.jpg
    """
    img_dir = os.path.dirname(image_path)    # es. .../val/images
    base_dir = os.path.dirname(img_dir)      # es. .../val
    fname = os.path.basename(image_path)     # IMG_1.jpg
    name_no_ext = os.path.splitext(fname)[0] # IMG_1
    
    # Costruisci il path atteso per la label
    label_path = os.path.join(base_dir, 'labels', f'{name_no_ext}.npy')
    
    print(f"🔍 [Data] Cerco labels in: {label_path}")
    
    if os.path.exists(label_path):
        try:
            points = np.load(label_path) # Array [N, 2] -> (x, y)
            print(f"✅ [Data] Trovati {len(points)} punti.")
            return points
        except Exception as e:
            print(f"⚠️ Errore lettura .npy: {e}")
            return None
    else:
        print(f"❌ File labels non trovato.")
        return None

def draw_zip_prediction(image_rgb, pi_prob, threshold=0.3):
    """
    Disegna la PREDIZIONE DEL MODELLO.
    - Background: Oscurato
    - Foreground: Normale + Griglia Rossa
    """
    H, W, _ = image_rgb.shape
    h_grid, w_grid = pi_prob.shape[2], pi_prob.shape[3]
    patch_h, patch_w = H / h_grid, W / w_grid

    # 1. Identifica Background (Upsampling della mappa di probabilità)
    pi_map_up = F.interpolate(pi_prob, size=(H, W), mode='nearest').squeeze().cpu().numpy()
    is_background = pi_map_up < threshold
    
    # 2. Oscura le zone di background
    vis_img = image_rgb.copy()
    # Riduci luminosità background del 70%
    vis_img[is_background] = vis_img[is_background] * 0.3 
    
    pil_img = Image.fromarray(vis_img.astype(np.uint8))
    draw = ImageDraw.Draw(pil_img, 'RGBA')
    
    prob_small = pi_prob.squeeze().cpu().numpy()
    
    empty_blocks = (prob_small < threshold).sum()
    total_blocks = h_grid * w_grid
    
    # 3. Disegna GRIGLIA ROSSA solo su FOREGROUND (dove il modello è confidente)
    for r in range(h_grid):
        for c in range(w_grid):
            if prob_small[r, c] >= threshold:
                x1 = int(c * patch_w)
                y1 = int(r * patch_h)
                x2 = int((c + 1) * patch_w)
                y2 = int((r + 1) * patch_h)
                
                # Rettangolo rosso semi-trasparente (bordo)
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 180), width=1)

    return np.array(pil_img), empty_blocks, total_blocks

# ==============================================================================
# MAIN
# ==============================================================================
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 1. Carica Configurazione
    print(f"📂 Caricamento config da: {args.config}")
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print("❌ Config file non trovato!")
        return

    # --- ESTRAZIONE INFO PER NOME FILE (Dataset & Backbone) ---
    dataset_name = config.get('dataset', 'unknown')
    
    # Recupera Backbone (Gestione Robusta)
    backbone_name = config.get('model')
    if not backbone_name:
        backbone_cfg = config.get('BACKBONE', {})
        if isinstance(backbone_cfg, dict):
            backbone_name = backbone_cfg.get('TYPE', 'unknown')
        else:
            backbone_name = 'unknown'

    print(f"\n🚀 AVVIO VISUALIZZAZIONE STAGE 1")
    print(f"   Dataset:  {dataset_name}")
    print(f"   Backbone: {backbone_name}")
    print(f"   Soglia:   {args.threshold}")

    # 2. Costruisci Modello
    try:
        model = ZIPModel(config).to(device)
    except Exception as e:
        print(f"❌ Errore creazione modello: {e}")
        return

    # 3. Carica Checkpoint
    if os.path.isfile(args.checkpoint):
        print(f"📥 Caricamento pesi da: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        
        # Gestione dizionario checkpoint
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        state_dict = ckpt['model'] if 'model' in ckpt else state_dict 
        
        # Rimuovi prefisso 'module.' se presente
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if len(missing) > 0:
            print(f"⚠️ Chiavi mancanti: {len(missing)} (normale per caricamento parziale/backbone)")
    else:
        print("❌ Checkpoint non trovato!")
        return
    model.eval()

    # 4. Carica Dati (Immagine)
    if not os.path.exists(args.image_path):
        print(f"❌ Immagine non trovata: {args.image_path}")
        return

    if args.image_path.endswith('.npy'):
        img_np = np.load(args.image_path)
        raw_img = Image.fromarray(img_np.astype('uint8')).convert('RGB')
    else:
        raw_img = Image.open(args.image_path).convert('RGB')
    
    orig_W, orig_H = raw_img.size
    
    # Trasforma
    trans = get_smart_transform(orig_H, orig_W)
    img_tensor = trans(raw_img).unsqueeze(0).to(device)
    target_H, target_W = img_tensor.shape[2], img_tensor.shape[3]
    
    # 5. Carica Ground Truth (Points)
    gt_points = load_gt_points(args.image_path)
    
    # Riscala Punti GT per il plot
    gt_points_scaled = []
    if gt_points is not None and len(gt_points) > 0:
        scale_x = target_W / orig_W
        scale_y = target_H / orig_H
        for pt in gt_points:
            gt_points_scaled.append([pt[0] * scale_x, pt[1] * scale_y])
        gt_points_scaled = np.array(gt_points_scaled)
    else:
        gt_points_scaled = np.empty((0, 2))

    # 6. Inferenza
    with torch.no_grad():
        out = model(img_tensor)
        pi_logits = out['pi_logits']
        pi_prob = torch.sigmoid(pi_logits)

    # 7. Visualizzazione
    img_vis = denormalize(img_tensor.squeeze())
    
    # Genera Overlay
    mask_vis, empty_blocks, total_blocks = draw_zip_prediction(img_vis, pi_prob, threshold=args.threshold)
    empty_pct = (empty_blocks / total_blocks) * 100

    # PLOT
    os.makedirs("visualize", exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # --- PANNELLO 1: INPUT ---
    axes[0].imshow(img_vis)
    axes[0].set_title(f"Input ({orig_W}x{orig_H})", fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    # --- PANNELLO 2: GROUND TRUTH ---
    axes[1].imshow(img_vis)
    blue_overlay = np.zeros_like(img_vis)
    blue_overlay[:, :, 2] = 255
    axes[1].imshow(blue_overlay, alpha=0.2)
    
    if len(gt_points_scaled) > 0:
        axes[1].scatter(
            gt_points_scaled[:, 0], 
            gt_points_scaled[:, 1], 
            c='white', 
            s=20, 
            marker='.', 
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5
        )
        axes[1].set_title(f"GT: {len(gt_points)} Persone", fontsize=16, fontweight='bold', color='darkblue')
    else:
        axes[1].set_title("GT: N/A", fontsize=16, fontweight='bold', color='gray')
    axes[1].axis('off')
    
    # --- PANNELLO 3: MODEL PREDICTION ---
    axes[2].imshow(mask_vis)
    title_str = (f"Stage 1 Output \n"
                 f"Filtered: {int(empty_blocks)}/{total_blocks} patches ({empty_pct:.1f}%)")
    axes[2].set_title(title_str, fontsize=16, fontweight='bold', color='darkred')
    axes[2].axis('off')

    # Salvataggio con Nome Parlante
    img_name_clean = os.path.splitext(os.path.basename(args.image_path))[0]
    out_file = f"visualize/stage1_{dataset_name}_{backbone_name}_{img_name_clean}_thr{args.threshold}.png"
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Risultato salvato in: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizza output Stage 1 (Filtro Background)")
    parser.add_argument('--config', type=str, default='config_stage1.yaml', help="Path al file .yaml")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path al file .pth")
    parser.add_argument('--image_path', type=str, required=True, help="Path all'immagine di input")
    parser.add_argument('--threshold', type=float, default=0.5, help="Soglia probabilità")
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()
    main(args)