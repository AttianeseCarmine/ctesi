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
        return yaml.safe_load(f)

def denormalize(tensor, mean, std):
    """Converte il tensore (normalizzato per la rete) in immagine visibile."""
    tensor = tensor.clone().detach().cpu()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    img = tensor.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def get_smart_transform(h, w, config):
    """Ridimensiona l'immagine per essere digeribile dal ViT (multiplo di 16)."""
    patch_size = 16
    new_h = math.ceil(h / patch_size) * patch_size
    new_w = math.ceil(w / patch_size) * patch_size
    
    mean = config['DATA'].get('NORM_MEAN', [0.485, 0.456, 0.406])
    std = config['DATA'].get('NORM_STD', [0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize((new_h, new_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def load_gt_points(image_path):
    """
    Legge la VERITÀ dal file .npy delle etichette.
    Struttura attesa: .../val/images/IMG_1.jpg -> .../val/labels/IMG_1.npy
    """
    img_dir = os.path.dirname(image_path)    # es. .../val/images
    base_dir = os.path.dirname(img_dir)      # es. .../val
    fname = os.path.basename(image_path)     # IMG_1.jpg
    name_no_ext = os.path.splitext(fname)[0] # IMG_1
    
    # Costruisci il path atteso
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
        # Fallback: prova a stampare il contenuto della cartella labels per debug
        labels_dir = os.path.join(base_dir, 'labels')
        if os.path.exists(labels_dir):
            print(f"❌ File non trovato. Contenuto di {labels_dir} (primi 5 file):")
            print(os.listdir(labels_dir)[:5])
        else:
            print(f"❌ Cartella labels non trovata: {labels_dir}")
        return None

def draw_zip_prediction(image_rgb, pi_prob, threshold=0.3):
    """
    Disegna la PREDIZIONE DEL MODELLO (Destra).
    - Background: Oscurato
    - Foreground: Normale + Griglia Rossa
    """
    H, W, _ = image_rgb.shape
    h_grid, w_grid = pi_prob.shape[2], pi_prob.shape[3]
    patch_h, patch_w = H / h_grid, W / w_grid

    # 1. Identifica Background
    pi_map_up = F.interpolate(pi_prob, size=(H, W), mode='nearest').squeeze().cpu().numpy()
    is_background = pi_map_up < threshold
    
    # 2. Oscura quelle zone (Background scuro)
    vis_img = image_rgb.copy()
    vis_img[is_background] = vis_img[is_background] * 0.3 
    
    pil_img = Image.fromarray(vis_img.astype(np.uint8))
    draw = ImageDraw.Draw(pil_img, 'RGBA')
    
    prob_small = pi_prob.squeeze().cpu().numpy()
    
    empty_blocks = (prob_small < threshold).sum()
    total_blocks = h_grid * w_grid
    
    # 3. Disegna GRIGLIA ROSSA su FOREGROUND
    for r in range(h_grid):
        for c in range(w_grid):
            if prob_small[r, c] >= threshold:
                x1 = int(c * patch_w)
                y1 = int(r * patch_h)
                x2 = int((c + 1) * patch_w)
                y2 = int((r + 1) * patch_h)
                
                # Rettangolo rosso
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=1)

    return np.array(pil_img), empty_blocks, total_blocks

# ==============================================================================
# MAIN
# ==============================================================================
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    
    # ESTRAZIONE RUN_NAME DAL CONFIG
    run_name = config.get('RUN_NAME', 'experiment')
    
    print(f"\n🚀 AVVIO VISUALIZZAZIONE CHECK")
    print(f"   Run Name: {run_name}")
    print(f"   Modello:  {config['BACKBONE']['TYPE']}")
    print(f"   Soglia:   {args.threshold}")

    # 1. Carica Modello
    model = ZIPModel(config).to(device)
    if os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt.get('model', ckpt)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        print("❌ Checkpoint non trovato!")
        return
    model.eval()

    # 2. Carica Dati
    if not os.path.exists(args.image_path):
        print(f"❌ Immagine non trovata: {args.image_path}")
        return

    raw_img = Image.open(args.image_path).convert('RGB')
    orig_W, orig_H = raw_img.size
    
    # Trasforma
    trans = get_smart_transform(orig_H, orig_W, config)
    img_tensor = trans(raw_img).unsqueeze(0).to(device)
    target_H, target_W = img_tensor.shape[2], img_tensor.shape[3]
    
    # Carica GT
    gt_points = load_gt_points(args.image_path)
    
    # Riscala Punti GT
    gt_points_scaled = []
    if gt_points is not None:
        scale_x = target_W / orig_W
        scale_y = target_H / orig_H
        for pt in gt_points:
            gt_points_scaled.append([pt[0] * scale_x, pt[1] * scale_y])
        gt_points_scaled = np.array(gt_points_scaled)

    # 3. Inferenza
    with torch.no_grad():
        out = model(img_tensor)
        pi_prob = torch.sigmoid(out['pi_logits']) 

    # 4. Visualizzazione
    mean = config['DATA'].get('NORM_MEAN', [0.485, 0.456, 0.406])
    std = config['DATA'].get('NORM_STD', [0.229, 0.224, 0.225])
    img_np = denormalize(img_tensor.squeeze(), mean, std)
    
    # Genera Overlay ZIP (Destra)
    mask_vis, empty_blocks, total_blocks = draw_zip_prediction(img_np, pi_prob, threshold=args.threshold)
    empty_pct = (empty_blocks / total_blocks) * 100

    # PLOT
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # --- PANNELLO 1: INPUT ---
    axes[0].imshow(img_np)
    axes[0].set_title(f"Input", fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    # --- PANNELLO 2: GROUND TRUTH (FILTRO BLU + PUNTI BIANCHI) ---
    axes[1].imshow(img_np) # Sfondo base
    
    # Crea Overlay Blu Chiaro
    blue_overlay = np.zeros_like(img_np)
    blue_overlay[:, :, 2] = 255 # Solo canale Blu al massimo
    # Sovrapponi con alpha 0.3 (30% blu, 70% immagine originale)
    axes[1].imshow(blue_overlay, alpha=0.3)
    
    if len(gt_points_scaled) > 0:
        # PUNTI BIANCHI
        axes[1].scatter(
            gt_points_scaled[:, 0], 
            gt_points_scaled[:, 1], 
            c='white',      # Colore Punti
            s=25,           # Dimensione
            marker='.', 
            alpha=0.9
        )
        axes[1].set_title(f"Ground Truth ({len(gt_points)} Persone)", fontsize=16, fontweight='bold', color='darkblue')
    else:
        axes[1].set_title("GT: Nessuna Etichetta Trovata", fontsize=16, fontweight='bold', color='red')
    axes[1].axis('off')
    
    # --- PANNELLO 3: MODEL PREDICTION ---
    axes[2].imshow(mask_vis)
    title_str = (f"ZIP Prediction (Thr={args.threshold})\n"
                 f"Ignored: {int(empty_blocks)}/{total_blocks} ({empty_pct:.1f}%)")
    axes[2].set_title(title_str, fontsize=16, fontweight='bold', color='darkred')
    axes[2].axis('off')

    # Salvataggio con Nome Specifico
    img_name_clean = os.path.splitext(os.path.basename(args.image_path))[0] # es. IMG_10
    out_file = f"visualize/visualize_{run_name}_stage1_{img_name_clean}.png"
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=100, bbox_inches='tight')
    print(f"\n✅ Salvato risultato in: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path al file .yaml")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path al file .pth")
    parser.add_argument('--image_path', type=str, required=True, help="Path all'immagine .jpg")
    parser.add_argument('--threshold', type=float, default=0.3, help="Soglia probabilità")
    parser.add_argument('--device', type=str, default="cuda")
    
    args = parser.parse_args()
    main(args)