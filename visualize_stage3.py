import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
import random
from torchvision.transforms import functional as F

# I tuoi moduli
from models.zip_clip_ebc_model import build_model
from datasets import get_dataset
from datasets.transforms import build_transforms

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def denormalize(tensor):
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * std + mean)
    img = np.clip(img, 0, 1)
    return img

def main(config_path, checkpoint_path):
    # 1. Setup
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"="*60)
    print(f"🚀 VISUALIZE STAGE 3 (Final Model)")
    print(f"   Config: {config_path}")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"="*60)

    # 2. Modello
    model = build_model(config).to(device)
    
    # 3. Carica Pesi
    if os.path.exists(checkpoint_path):
        print(f"📥 Carico checkpoint...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('model', ckpt)
        model.load_state_dict(state_dict, strict=False)
        print("✅ Pesi caricati.")
    else:
        print(f"⚠️  ATTENZIONE: Checkpoint {checkpoint_path} non trovato!")

    model.eval()

    # 4. Dataset
    data_cfg = config["DATA"]
    val_tf = build_transforms(data_cfg, is_train=False)
    DatasetClass = get_dataset(config["DATASET"])
    val_set = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=val_tf,
    )
    
    # 5. Visualizzazione
    num_samples = 4
    indices = random.sample(range(len(val_set)), num_samples)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 5 * num_samples))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    print(f"📸 Generazione grafici per {num_samples} immagini...")
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = val_set[idx]
            img_tensor = sample['image'].unsqueeze(0).to(device)
            gt_density = sample['density'].unsqueeze(0).to(device)
            
            # Forward Finale
            outputs = model(img_tensor)
            
            pred_density = outputs['density_map']
            pred_count = outputs['pred_count'].item()
            gt_count = gt_density.sum().item()
            
            img_np = denormalize(sample['image'])
            
            # Originale
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title(f"Image {idx}", fontsize=10)
            axes[i, 0].axis('off')
            
            # GT
            axes[i, 1].imshow(gt_density.squeeze().cpu().numpy(), cmap='jet')
            axes[i, 1].set_title(f"GT: {gt_count:.1f}", fontsize=10)
            axes[i, 1].axis('off')
            
            # Predizione Finale
            axes[i, 2].imshow(pred_density.squeeze().cpu().numpy(), cmap='jet')
            err = abs(pred_count - gt_count)
            # Verde se errore < 10%, Rosso altrimenti
            color = "green" if (gt_count > 0 and err/gt_count < 0.1) else "red"
            axes[i, 2].set_title(f"Pred: {pred_count:.1f} (Err: {err:.1f})", color=color, fontsize=12, fontweight='bold')
            axes[i, 2].axis('off')

    out_path = "check_stage3.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"\n✅ Salvato: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_sha.yaml")
    # Default: best model stage 3
    parser.add_argument("--ckpt", default="experiments/sha_zip_clip_ebc/stage3/best_stage3_model.pth")
    args = parser.parse_args()
    
    main(args.config, args.ckpt)