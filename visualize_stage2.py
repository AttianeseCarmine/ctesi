import argparse
import os
import json
import yaml
import re
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms
from scipy.ndimage import gaussian_filter

# Import model factory from your project structure
from models import get_model
from datasets import standardize_dataset_name

# ==============================================================================
# 1. CORE UTILS (Re-implemented for standalone robustness)
# ==============================================================================

def resize_density_map(x: torch.Tensor, size: tuple) -> torch.Tensor:
    """
    Resizes density map while preserving the total count (sum).
    Crucial for correct visualization of upsampled predictions.
    """
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(0)
        
    x_sum = torch.sum(x, dim=(-1, -2))
    x_resized = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    
    # Calculate scale factor to preserve the sum (count)
    current_sum = torch.sum(x_resized, dim=(-1, -2))
    scale_factor = torch.nan_to_num(x_sum / current_sum, nan=0.0, posinf=0.0, neginf=0.0)
    
    return x_resized * scale_factor.unsqueeze(-1).unsqueeze(-1)

def generate_density_map(points: np.ndarray, height: int, width: int, sigma: float = 8.0) -> np.ndarray:
    """
    Generates a Gaussian density map from point annotations.
    """
    density_map = np.zeros((height, width), dtype=np.float32)
    if len(points) > 0:
        # Clamp points to be within image bounds
        x = np.clip(points[:, 0], 0, width - 1).astype(int)
        y = np.clip(points[:, 1], 0, height - 1).astype(int)
        
        # Accumulate points
        for i in range(len(x)):
            density_map[y[i], x[i]] = 1.0
            
    # Apply Gaussian smoothing
    if sigma > 0:
        density_map = gaussian_filter(density_map, sigma=sigma)
        
    return density_map

def load_config_safe(config_path):
    """Loads YAML config handling python/tuple tags safely."""
    with open(config_path, 'r') as f:
        try:
            # Loader=yaml.Loader handles !!python/tuple tags
            return yaml.load(f, Loader=yaml.Loader)
        except Exception as e:
            print(f"Standard load failed, trying safe_load: {e}")
            f.seek(0)
            return yaml.safe_load(f)

# ==============================================================================
# 2. VISUALIZATION LOGIC
# ==============================================================================

def normalize_for_vis(img_tensor):
    """Denormalizes ImageNet tensor to [0,1] RGB numpy array."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()

def create_heatmap_overlay(image, density_map, alpha=0.6, use_global_norm=False):
    """
    Creates a 'jet' heatmap overlay on the image.
    """
    H, W = image.shape[:2]
    
    # Normalize density map for visualization
    d_min, d_max = density_map.min(), density_map.max()
    
    if d_max > 0:
        d_norm = (density_map - d_min) / (d_max - d_min)
    else:
        d_norm = density_map

    # Apply colormap (Jet is standard for density maps)
    heatmap = cm.jet(d_norm)[:, :, :3]
    
    # Create a mask to only colorize non-zero regions (prevents blue tint on background)
    # Threshold can be adjusted (e.g., 0.05 or 0.0 for full coverage)
    mask = d_norm > 0.05
    mask = mask[..., None]

    # Blend: (1-alpha)*Image + alpha*Heatmap
    overlay = image.copy()
    overlay[mask[:,:,0]] = (1 - alpha) * image[mask[:,:,0]] + alpha * heatmap[mask[:,:,0]]
    
    return np.clip(overlay, 0, 1)

def get_config_args(config_path, ckpt_path):
    """Extracts necessary args from config file and paths."""
    config = load_config_safe(config_path)
    
    # Extract backbone name from config or infer defaults
    model_name = config.get('MODEL', 'clip_vit_b_16')
    dataset = config.get('DATASET', 'sha')
    
    # Load reduction JSON for bins/anchors (CRITICAL for EBC)
    reduction = config.get('REDUCTION', 8)
    truncation = config.get('TRUNCATION', 4)
    granularity = config.get('GRANULARITY', 'fine')
    anchor_type = config.get('ANCHOR_POINTS', 'average')
    
    json_path = os.path.join("configs", f"reduction_{reduction}.json")
    if not os.path.exists(json_path):
        # Fallback to current dir or parent
        if os.path.exists(f"reduction_{reduction}.json"):
            json_path = f"reduction_{reduction}.json"
        else:
            raise FileNotFoundError(f"Could not find reduction JSON for reduction={reduction}")

    with open(json_path, 'r') as f:
        red_data = json.load(f)
        
    dataset_std = standardize_dataset_name(dataset)
    trunc_str = str(truncation)
    
    if trunc_str not in red_data or dataset_std not in red_data[trunc_str]:
        raise ValueError(f"Config combination not found in {json_path}: {truncation}/{dataset_std}")
        
    cfg_bins = red_data[trunc_str][dataset_std]["bins"][granularity]
    cfg_anchors = red_data[trunc_str][dataset_std]["anchor_points"][granularity]
    
    # Select anchors
    anchors_list = cfg_anchors["average"] if anchor_type == "average" else cfg_anchors["middle"]
    
    return {
        "model_name": model_name,
        "input_size": config.get("INPUT_SIZE", 224),
        "reduction": reduction,
        "bins": [(float(b[0]), float(b[1])) for b in cfg_bins],
        "anchor_points": [float(p) for p in anchors_list],
        "prompt_type": config.get("PROMPT_TYPE", "word"),
        "num_vpt": config.get("num_vpt", 32),
        "vpt_drop": config.get("vpt_drop", 0.0),
        "deep_vpt": not config.get("shallow_vpt", False)
    }

# ==============================================================================
# 3. MAIN SCRIPT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize Stage 2 (EBC) Results")
    parser.add_argument('--config', type=str, required=True, help="Path to config yaml")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--image_path', type=str, required=True, help="Path to input image")
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Visualizing Stage 2 | Device: {device}")

    # 1. Setup Model
    cfg = get_config_args(args.config, args.checkpoint)
    
    print(f"[*] Building Model: {cfg['model_name']}")
    model = get_model(
        backbone=cfg['model_name'],
        input_size=cfg['input_size'],
        reduction=cfg['reduction'],
        bins=cfg['bins'],
        anchor_points=cfg['anchor_points'],
        prompt_type=cfg['prompt_type'],
        num_vpt=cfg['num_vpt'],
        vpt_drop=cfg['vpt_drop'],
        deep_vpt=cfg['deep_vpt']
    ).to(device)

    # 2. Load Weights
    print(f"[*] Loading weights from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    # Clean keys (DDP 'module.' prefix)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 3. Process Image
    img_pil = Image.open(args.image_path).convert('RGB')
    orig_w, orig_h = img_pil.size
    
    # Transform matching training (resize to multiple of 16/reduction)
    resize_h = math.ceil(orig_h / 16) * 16
    resize_w = math.ceil(orig_w / 16) * 16
    
    transform = transforms.Compose([
        transforms.Resize((resize_h, resize_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # 4. Inference
    with torch.no_grad():
        # EBC model returns expected density in eval mode
        pred_density_lowres = model(img_tensor)
        
        # Handle tuple output if any
        if isinstance(pred_density_lowres, tuple):
            pred_density_lowres = pred_density_lowres[-1]

    # 5. Upsample Prediction
    # Use resize_density_map to preserve the count accurately
    pred_density = resize_density_map(pred_density_lowres, (orig_h, orig_w))
    pred_count = pred_density.sum().item()
    
    pred_map_np = pred_density.squeeze().cpu().numpy()

    # 6. Load Ground Truth
    gt_count = 0
    gt_map_np = np.zeros((orig_h, orig_w))
    
    # Try finding labels file
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    # Assuming standard structure: data/dataset/split/images/IMG.jpg -> .../labels/IMG.npy
    label_path = args.image_path.replace("images", "labels").replace(".jpg", ".npy").replace(".png", ".npy")
    
    if os.path.exists(label_path):
        points = np.load(label_path)
        gt_count = len(points)
        print(f"[*] Found GT: {gt_count} points")
        gt_map_np = generate_density_map(points, orig_h, orig_w, sigma=8.0)
    else:
        print("[!] Warning: GT label file not found, GT panel will be empty.")

    # 7. Create Visualizations
    img_np = np.array(img_pil) / 255.0
    
    # Create overlays
    vis_pred = create_heatmap_overlay(img_np, pred_map_np)
    vis_gt = create_heatmap_overlay(img_np, gt_map_np) if gt_count > 0 else img_np

    # 8. Plot & Save
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Panel 1: Original
    axes[0].imshow(img_np)
    axes[0].set_title(f"Original Image\n{orig_w}x{orig_h}", fontsize=16)
    axes[0].axis('off')
    
    # Panel 2: GT
    axes[1].imshow(vis_gt)
    axes[1].set_title(f"Ground Truth\nCount: {gt_count}", fontsize=16, color='green', fontweight='bold')
    axes[1].axis('off')
    
    # Panel 3: Prediction
    err = pred_count - gt_count
    axes[2].imshow(vis_pred)
    axes[2].set_title(f"Prediction (Stage 2)\nCount: {pred_count:.2f} (Err: {err:+.2f})", 
                      fontsize=16, color='blue', fontweight='bold')
    axes[2].axis('off')

    # Construct filename
    # stage2_{numero_immagine}_{backbone}.png
    img_num_match = re.search(r'(\d+)', base_name)
    img_num = img_num_match.group(1) if img_num_match else base_name
    backbone_short = cfg['model_name'].replace("clip_", "")
    
    out_dir = "visualize"
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"stage2_{img_num}_{backbone_short}.png")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"✅ Saved visualization to: {save_path}")

if __name__ == '__main__':
    main()