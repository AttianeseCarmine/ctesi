import os
import sys
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Import dai tuoi moduli esistenti
current_dir = os.path.abspath(os.path.dirname(__file__))
from datasets import standardize_dataset_name
from models import get_model
from models.zip_model import ZIPModel
from models.joint_model import ZIPCLIPJointModel
from utils import get_dataloader, get_logger

# =============================================================================
# 🛠️ FIX AL VOLO (MONKEY PATCH)
# Risolve l'errore "ValueError: too many values to unpack" senza toccare i file originali
# =============================================================================
import models.clip.utils as clip_utils

def fixed_format_count(val, prompt_type):
    # Questa è la versione corretta della funzione che nel file originale è buggata
    left, right = val
    if prompt_type == "word":
        return clip_utils.num2word(left), clip_utils.num2word(right)
    return left, right

# Sostituiamo la funzione rotta in memoria
clip_utils.format_count = fixed_format_count
print("✅ Patch applicata: bug in models.clip.utils corretto in memoria.")
# =============================================================================


# ==========================================
# Argomenti
# ==========================================
parser = ArgumentParser("Debug Alignment Stage 3")
parser.add_argument("--config", type=str, default="configs/config.yaml")
parser.add_argument("--s1", type=str, required=True, help="Checkpoint Stage 1")
parser.add_argument("--s2", type=str, required=True, help="Checkpoint Stage 2")
parser.add_argument("--model", type=str, default="clip_vit_b_16")
parser.add_argument("--input_size", type=int, default=224)
parser.add_argument("--reduction", type=int, default=8)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_images", type=int, default=10) 
parser.add_argument("--out_dir", type=str, default="debug_vis")

# Parametri dummy
parser.add_argument("--regression", action="store_true")
parser.add_argument("--truncation", type=int, default=4)
parser.add_argument("--anchor_points", type=str, default="average")
parser.add_argument("--prompt_type", type=str, default="word")
parser.add_argument("--granularity", type=str, default="fine")
parser.add_argument("--num_vpt", type=int, default=32)
parser.add_argument("--vpt_drop", type=float, default=0.0)
parser.add_argument("--shallow_vpt", action="store_true")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--sliding_window", action="store_true")
parser.add_argument("--stride", type=int, default=None)
parser.add_argument("--window_size", type=int, default=None)
parser.add_argument("--resize_to_multiple", action="store_true")
parser.add_argument("--zero_pad_to_multiple", action="store_true")

# Mean/Std di CLIP per denormalizzare le immagini
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(1, 1, 3)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(1, 1, 3)

def load_weights_only(path, map_location="cpu"):
    # weights_only=False per evitare warning
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

def denormalize_image(tensor_img):
    img = tensor_img.permute(1, 2, 0).cpu().numpy()
    img = img * CLIP_STD + CLIP_MEAN
    img = np.clip(img, 0, 1)
    return img

# --- DEFINIZIONE HOOKS ---
def hook_s1_output(module, input, output):
    # Vogliamo vedere l'OUTPUT di ZIP (la maschera)
    if isinstance(output, dict):
        # ZIP di solito ritorna un dizionario con 'pi_logits' o 'logit_pi'
        out_tens = output.get('pi_logits', output.get('logit_pi', None))
    else:
        out_tens = output # Caso fallback
    
    if out_tens is not None:
        print(f"🔹 [DEBUG] ZIP (Stage1) Output: {out_tens.shape}")
    else:
        print(f"🔹 [DEBUG] ZIP (Stage1) Output: Struttura non riconosciuta (Type: {type(output)})")

def hook_s2_output(module, input, output):
    # Output CLIP-EBC
    if isinstance(output, (tuple, list)):
        density = output[1]
        print(f"🔸 [DEBUG] CLIP-EBC (Stage2) Output Density: {density.shape}")
    else:
        print(f"🔸 [DEBUG] CLIP-EBC (Stage2) Output Density: {output.shape}")

def main():
    args = parser.parse_args()
    args.dataset = standardize_dataset_name(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"🔬 Inizio Debug Geometrico per {args.model}...")

    # --- 1. CONFIGURAZIONE BINS ---
    import json
    try:
        config_path = os.path.join(current_dir, "configs", f"reduction_{args.reduction}.json")
        with open(config_path, "r") as f:
            full_cfg = json.load(f)
            if args.dataset in full_cfg[str(args.truncation)]:
                ds_cfg = full_cfg[str(args.truncation)][args.dataset]
            else:
                ds_cfg = list(full_cfg[str(args.truncation)].values())[0]
            
        bins_raw = ds_cfg["bins"][args.granularity]
        anchors_raw = ds_cfg["anchor_points"][args.granularity]["average"]
        args.bins = [(float(b[0]), float(b[1])) for b in bins_raw]
        args.anchor_points = [float(p) for p in anchors_raw]
    except Exception as e:
        print(f"⚠️ Warning: Uso dummy bins ({e})")
        args.bins = [(0,1), (1,2), (2,3), (3,4)]
        args.anchor_points = [0.5, 1.5, 2.5, 3.5]

    # --- 2. CARICAMENTO MODELLO ---
    print("🏗️  Costruzione Modello...")
    with open(args.config, "r") as f:
        yaml_cfg = yaml.safe_load(f)
    zip_cfg = {
        "BACKBONE": yaml_cfg.get("BACKBONE", "resnet50"),
        "ZIP_HEAD": yaml_cfg.get("ZIP_HEAD", {"HIDDEN_DIM": 256}),
        "REDUCTION": args.reduction,
    }
    
    stage1 = ZIPModel(zip_cfg).to(device)
    stage1.load_state_dict(load_weights_only(args.s1, device), strict=False)

    stage2 = get_model(
        backbone=args.model,
        input_size=args.input_size,
        reduction=args.reduction,
        bins=args.bins,
        anchor_points=args.anchor_points,
        prompt_type=args.prompt_type,
        num_vpt=args.num_vpt,
        vpt_drop=args.vpt_drop,
        deep_vpt=not args.shallow_vpt,
    ).to(device)
    stage2.load_state_dict(load_weights_only(args.s2, device), strict=False)

    model = ZIPCLIPJointModel(stage1, stage2).to(device)
    model.eval()

    # --- 3. REGISTRAZIONE HOOKS ---
    model.stage1.register_forward_hook(hook_s1_output)
    model.stage2.register_forward_hook(hook_s2_output)
    print("✅ Hooks di debug registrati con successo.")

    # --- 4. DATALOADER ---
    val_loader = get_dataloader(args, split="val", ddp=False)

    # --- 5. LOOP ---
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if count >= args.num_images:
                break

            if isinstance(batch, dict):
                imgs = batch["image"].to(device)
            else:
                imgs, _, _ = batch
                imgs = imgs.to(device)

            print(f"\n📸 [DEBUG] Image {count} - Original Batch Shape: {imgs.shape}")
            
            # Forward Pass
            out = model(imgs)

            # Salvataggio immagine (opzionale per questo test numerico, ma utile)
            batch_size_curr = imgs.shape[0]
            for b in range(batch_size_curr):
                if count >= args.num_images: break
                
                img_np = denormalize_image(imgs[b])
                H, W, _ = img_np.shape
                
                zip_mask = out['pi_prob'][b].squeeze().cpu().numpy()
                raw_dens = out['raw_density'][b].squeeze().cpu().numpy()
                fin_dens = out['final_density'][b].squeeze().cpu().numpy()
                
                def resize_to_img(m):
                    import cv2
                    return cv2.resize(m, (W, H))

                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                axes[0].imshow(img_np); axes[0].set_title("Input")
                axes[1].imshow(resize_to_img(zip_mask), cmap='gray'); axes[1].set_title("ZIP Mask")
                axes[2].imshow(resize_to_img(raw_dens), cmap='jet'); axes[2].set_title(f"CLIP Raw")
                axes[3].imshow(resize_to_img(fin_dens), cmap='jet'); axes[3].set_title(f"Final")
                
                save_path = os.path.join(args.out_dir, f"debug_img_{count:03d}.png")
                plt.savefig(save_path)
                plt.close(fig)
                count += 1
    
    print("\n✅ Finito.")

if __name__ == "__main__":
    main()