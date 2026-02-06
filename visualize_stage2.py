#!/usr/bin/env python3
# visualize_stage2.py
import os
import argparse
import yaml
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from datasets import standardize_dataset_name
from models import get_model
from utils import get_dataloader, sliding_window_predict


# -------------------------
# YAML loader that supports !!python/tuple
# -------------------------
class SafeTupleLoader(yaml.SafeLoader):
    pass


def construct_python_tuple(loader, node):
    return tuple(loader.construct_sequence(node))


SafeTupleLoader.add_constructor("tag:yaml.org,2002:python/tuple", construct_python_tuple)


# -------------------------
# Config loader (JSON + YAML)
# -------------------------
def load_config(config_path):
    """Carica un file di configurazione .json o .yaml/.yml."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    ext = os.path.splitext(config_path)[1].lower()

    if ext == ".json":
        with open(config_path, "r") as f:
            cfg = json.load(f)
        print(f"✅ Loaded JSON config from: {config_path}")
    elif ext in [".yaml", ".yml"]:
        with open(config_path, "r") as f:
            cfg = yaml.load(f, Loader=SafeTupleLoader)
        print(f"✅ Loaded YAML config from: {config_path}")
    else:
        # fallback
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            print("✅ Loaded config as JSON (no extension detected)")
        except json.JSONDecodeError:
            with open(config_path, "r") as f:
                cfg = yaml.load(f, Loader=SafeTupleLoader)
            print("✅ Loaded config as YAML (no extension detected)")

    return cfg or {}


# -------------------------
# Helpers
# -------------------------
def _unwrap_model_outputs(model_out):
    """
    Stage2 può ritornare:
      - Tensor pred_density
      - (pred_class, pred_density)
      - dict con chiavi tipo 'density', 'ebc_density', 'pred_density', 'final_density'
    Qui estraiamo SOLO la density.
    """
    if isinstance(model_out, torch.Tensor):
        return model_out

    if isinstance(model_out, (tuple, list)):
        if len(model_out) >= 2 and isinstance(model_out[1], torch.Tensor):
            return model_out[1]
        for x in model_out:
            if isinstance(x, torch.Tensor) and x.dim() == 4:
                return x
        raise TypeError(f"Tuple/list output but no density tensor found. Types: {[type(x) for x in model_out]}")

    if isinstance(model_out, dict):
        for k in ["pred_density", "density", "ebc_density", "final_density"]:
            if k in model_out and isinstance(model_out[k], torch.Tensor):
                return model_out[k]
        raise KeyError(f"Dict output but no known density key found. Keys: {list(model_out.keys())}")

    raise TypeError(f"Unsupported model output type: {type(model_out)}")


def _denorm_image(img_chw, mean, std):
    """Se mean/std mancano nel config, usa ImageNet default."""
    if mean is None or std is None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    mean = torch.tensor(mean, device=img_chw.device).view(3, 1, 1)
    std = torch.tensor(std, device=img_chw.device).view(3, 1, 1)
    x = img_chw * std + mean
    return x.clamp(0, 1)


def _resize_density_preserve_sum(density_b1hw, out_h, out_w):
    """Resize bilinear + scaling per conservare la somma (conteggio)."""
    in_h, in_w = density_b1hw.shape[-2:]
    if (in_h, in_w) == (out_h, out_w):
        return density_b1hw
    resized = F.interpolate(density_b1hw, size=(out_h, out_w), mode="bilinear", align_corners=False)
    scale = (in_h * in_w) / float(out_h * out_w)
    return resized * scale


def _robust_vmax(m_2d, q=0.995):
    """vmax robusto (percentile) così la mappa pred non resta 'tutta blu'."""
    x = m_2d.detach().float().cpu()
    if x.numel() == 0:
        return 1.0
    vmax = torch.quantile(x.flatten(), q).item()
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(x.max().item()) if x.max().item() > 0 else 1.0
    return vmax


def _resolve_subset_index(ds, idx):
    """Se ds è Subset (torch.utils.data.Subset), ritorna (base_ds, real_idx)."""
    if hasattr(ds, "indices") and hasattr(ds, "dataset"):
        try:
            return ds.dataset, int(ds.indices[idx])
        except Exception:
            return ds, idx
    return ds, idx


def _dataset_get_path(ds, idx):
    """Recupera path/nome immagine dal dataset (o subset) provando vari campi comuni."""
    base_ds, real_idx = _resolve_subset_index(ds, idx)

    for attr in ["img_paths", "image_paths", "imgs", "images", "im_list", "paths", "files", "img_list"]:
        if hasattr(base_ds, attr):
            arr = getattr(base_ds, attr)
            try:
                return arr[real_idx]
            except Exception:
                pass

    if hasattr(base_ds, "samples"):
        try:
            s = base_ds.samples[real_idx]
            if isinstance(s, (list, tuple)) and len(s) > 0:
                return s[0]
        except Exception:
            pass

    return None


def _find_indices_by_names(ds, names):
    """Match per basename (case-insensitive) o substring (funziona solo se dataset espone i path)."""
    wanted = [n.strip() for n in names if n.strip()]
    if not wanted:
        return []
    wanted_low = [w.lower() for w in wanted]
    found = []

    N = len(ds)
    for i in range(N):
        p = _dataset_get_path(ds, i)
        if not p:
            continue
        base = os.path.basename(str(p)).lower()
        full = str(p).lower()
        for w in wanted_low:
            if w == base or w in base or w in full:
                found.append(i)
                break

    seen = set()
    out = []
    for i in found:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _count_points(target_points):
    """Conteggio robusto dei punti GT."""
    if target_points is None:
        return 0

    if isinstance(target_points, torch.Tensor):
        if target_points.ndim >= 2:
            return int(target_points.shape[0])
        return int(target_points.numel())

    if isinstance(target_points, np.ndarray):
        if target_points.ndim >= 2:
            return int(target_points.shape[0])
        return int(target_points.size)

    if isinstance(target_points, (list, tuple)):
        if len(target_points) == 0:
            return 0
        first = target_points[0]
        if isinstance(first, torch.Tensor) and getattr(first, "ndim", 0) == 2:
            return int(first.shape[0])
        if isinstance(first, np.ndarray) and getattr(first, "ndim", 0) == 2:
            return int(first.shape[0])
        return int(len(target_points))

    try:
        return int(len(target_points))
    except Exception:
        return 0


def _points_to_xy(target_points):
    """Converte target_points in (xs, ys) (coords immagine)."""
    if target_points is None:
        return None, None

    # unwrap batch wrapper [Tensor Nx2] / [ndarray Nx2]
    if isinstance(target_points, (list, tuple)) and len(target_points) > 0:
        if isinstance(target_points[0], (torch.Tensor, np.ndarray)) and getattr(target_points[0], "ndim", 0) == 2:
            target_points = target_points[0]

    if isinstance(target_points, torch.Tensor):
        pts = target_points.detach().cpu().float()
        if pts.numel() == 0:
            return None, None
        return pts[:, 0].numpy(), pts[:, 1].numpy()

    if isinstance(target_points, np.ndarray):
        pts = target_points.astype(np.float32)
        if pts.size == 0:
            return None, None
        return pts[:, 0], pts[:, 1]

    pts = np.array(target_points, dtype=np.float32)
    if pts.size == 0:
        return None, None
    return pts[:, 0], pts[:, 1]


def _make_args_for_dataloader(cfg, dataset, input_size):
    """Crea un Namespace compatibile con get_dataloader() del progetto."""
    args = argparse.Namespace()

    args.dataset = standardize_dataset_name(dataset)
    args.batch_size = 1
    args.num_workers = 0
    args.input_size = int(cfg.get("input_size", input_size)) if isinstance(cfg.get("input_size", input_size), (int, float)) else input_size

    args.sliding_window = bool(cfg.get("sliding_window", False))
    args.stride = cfg.get("stride", None)
    args.window_size = cfg.get("window_size", None)
    args.resize_to_multiple = bool(cfg.get("resize_to_multiple", False))
    args.zero_pad_to_multiple = bool(cfg.get("zero_pad_to_multiple", False))

    # augmentation knobs (safe defaults)
    args.num_crops = int(cfg.get("num_crops", 1)) if isinstance(cfg.get("num_crops", 1), (int, float)) else 1
    args.min_scale = float(cfg.get("min_scale", 1.0))
    args.max_scale = float(cfg.get("max_scale", 2.0))
    args.brightness = float(cfg.get("brightness", 0.0))
    args.contrast = float(cfg.get("contrast", 0.0))
    args.saturation = float(cfg.get("saturation", 0.0))
    args.hue = float(cfg.get("hue", 0.0))
    args.kernel_size = int(cfg.get("kernel_size", 5)) if isinstance(cfg.get("kernel_size", 5), (int, float)) else 5
    args.saltiness = float(cfg.get("saltiness", 0.0))
    args.spiciness = float(cfg.get("spiciness", 0.0))
    args.jitter_prob = float(cfg.get("jitter_prob", 0.0))
    args.blur_prob = float(cfg.get("blur_prob", 0.0))
    args.noise_prob = float(cfg.get("noise_prob", 0.0))

    return args


def _parse_indices(indices_str, num_rows):
    """Parse di --indices '1,2,3' -> [1,2,3]"""
    if not indices_str:
        return None
    parts = [p.strip() for p in indices_str.split(",") if p.strip()]
    if not parts:
        return None
    out = []
    for p in parts:
        out.append(int(p))
    return out[:num_rows]


def _safe_int(x, default):
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _safe_float(x, default):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


# -------------------------
# Main visualize
# -------------------------
def visualize_stage2(
    config_path: str,
    ckpt_path: str,
    out_png: str,
    dataset: str,
    model_name: str,
    input_size: int,
    reduction: int,
    num_rows: int = 3,
    names: str = None,
    indices: str = None,
    seed: int = 0,
    alpha_pred: float = 0.55,
    alpha_gt_bg: float = 0.97,
    point_size: float = 28.0,
    point_edge: float = 0.5,
    device: str = "cuda:0",
):
    # device
    if torch.cuda.is_available():
        device_t = torch.device(device)
    else:
        print("ℹ️ CUDA non disponibile: uso CPU.")
        device_t = torch.device("cpu")

    # ---- load config ----
    cfg = load_config(config_path)

    # IMPORTANT: allinea input_size/reduction a TRAIN (se presenti nel config)
    input_size = _safe_int(cfg.get("input_size", input_size), input_size)
    reduction = _safe_int(cfg.get("reduction", reduction), reduction)

    # bins/anchor_points
    if "bins" not in cfg or "anchor_points" not in cfg:
        raise ValueError("Nel config non trovo 'bins' e/o 'anchor_points'.")

    bins = []
    for a, b in cfg["bins"]:
        a = float(a)
        if isinstance(b, str) and b.strip().lower() in [".inf", "inf", "infinity"]:
            b = float("inf")
        elif isinstance(b, float) and not np.isfinite(b):
            b = float("inf")
        else:
            b = float(b)
        bins.append((a, b))

    anchor_points_list = [float(x) for x in cfg["anchor_points"]]

    mean = cfg.get("NORM_MEAN", None)
    std = cfg.get("NORM_STD", None)

    # Stage2 params robusti (evita crash se nel cfg ci sono None)
    num_vpt_val = cfg.get("num_vpt", cfg.get("NUM_VPT", 32))
    num_vpt_val = 32 if num_vpt_val is None else int(num_vpt_val)

    vpt_drop_val = cfg.get("vpt_drop", cfg.get("VPT_DROP", 0.0))
    vpt_drop_val = 0.0 if vpt_drop_val is None else float(vpt_drop_val)

    shallow_val = cfg.get("shallow_vpt", cfg.get("SHALLOW_VPT", False))
    shallow_val = bool(shallow_val) if shallow_val is not None else False

    prompt_type_val = cfg.get("prompt_type", cfg.get("PROMPT_TYPE", "word")) or "word"

    # dataloader args
    dl_args = _make_args_for_dataloader(cfg, dataset=dataset, input_size=input_size)

    # ---- build model ----
    model = get_model(
        backbone=model_name,
        input_size=input_size,
        reduction=reduction,
        bins=bins,
        anchor_points=anchor_points_list,
        prompt_type=prompt_type_val,
        num_vpt=num_vpt_val,
        vpt_drop=vpt_drop_val,
        deep_vpt=not shallow_val,
    ).to(device_t)

    # ---- load ckpt ----
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    # ---- dataloader ----
    val_loader = get_dataloader(dl_args, split="val", ddp=False)
    ds = val_loader.dataset

    # ---- choose indices ----
    chosen = _parse_indices(indices, num_rows) if indices else None

    if chosen is not None:
        N = len(ds)
        bad = [i for i in chosen if i < 0 or i >= N]
        if bad:
            raise IndexError(f"--indices fuori range (0..{N-1}): {bad}")

    if chosen is None and names:
        name_list = [x.strip() for x in names.split(",") if x.strip()]
        idxs = _find_indices_by_names(ds, name_list)
        if idxs:
            chosen = idxs[:num_rows]
        else:
            print("⚠️ Nessuna immagine trovata con quei nomi (o dataset non espone i path).")

    if chosen is None:
        rng = np.random.default_rng(seed)
        N = len(ds)
        chosen = list(range(N)) if N < num_rows else rng.choice(N, size=num_rows, replace=False).tolist()

    print(f"🧾 Selected indices: {chosen}")

    # ---- output ----
    if not out_png.lower().endswith(".png"):
        out_png = out_png + ".png"
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    # ---- plot: 3 columns ----
    num_cols = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4.5 * num_rows))
    if num_rows == 1:
        axes = axes.reshape(1, num_cols)

    # sliding window params (COERENTI con eval.py)
    use_sw = bool(cfg.get("sliding_window", False))
    ws = _safe_int(cfg.get("window_size", None), input_size)
    st = _safe_int(cfg.get("stride", None), input_size)

    print(f"🧩 VIS settings | model={model_name} input_size={input_size} reduction={reduction}")
    print(f"🪟 sliding_window={use_sw} window_size={ws} stride={st}")

    with torch.no_grad():
        for row, idx in enumerate(chosen):
            sample = ds[idx]
            if not (isinstance(sample, (tuple, list)) and len(sample) >= 2):
                raise TypeError(f"Dataset item non tuple/list. idx={idx} type={type(sample)}")

            image = sample[0]
            target_points = sample[1] if len(sample) >= 2 else None

            # batch dimension
            if isinstance(image, torch.Tensor) and image.dim() == 3:
                image_b = image.unsqueeze(0).to(device_t)
            else:
                image_b = image.to(device_t)

            # ---- PRED: COERENTE con eval.py ----
            if use_sw:
                pred_out = sliding_window_predict(model, image_b, ws, st)
            else:
                pred_out = model(image_b)

            pred_density = _unwrap_model_outputs(pred_out)

            # counts
            gt_count = _count_points(target_points)
            pred_count = float(pred_density.sum().item())

            # denorm image for display
            img = _denorm_image(image_b[0].detach().cpu(), mean, std)  # 3xHxW
            H, W = img.shape[-2], img.shape[-1]

            # pred map resized for overlay
            pr_map = _resize_density_preserve_sum(pred_density[0:1].detach().cpu(), H, W)[0, 0]
            vmax_pr = _robust_vmax(pr_map, q=0.995)

            # filename / idx
            pth = _dataset_get_path(ds, idx)
            name_show = os.path.basename(str(pth)) if pth else f"idx={idx}"

            # ---- COL 1 ----
            ax = axes[row, 0]
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.set_title(f"{name_show} | GT: {gt_count}")
            ax.axis("off")

            # ---- COL 2 ----
            ax = axes[row, 1]
            ax.imshow(img.permute(1, 2, 0).numpy(), alpha=alpha_gt_bg)

            xs, ys = _points_to_xy(target_points)
            if xs is not None and ys is not None:
                xs = np.clip(xs, 0, W - 1)
                ys = np.clip(ys, 0, H - 1)
                ax.scatter(
                    xs, ys,
                    s=point_size,
                    c="white",
                    edgecolors="black",
                    linewidths=point_edge,
                )
            ax.set_title("GT Points")
            ax.axis("off")

            # ---- COL 3 ----
            ax = axes[row, 2]
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.imshow(pr_map.numpy(), cmap="jet", vmin=0, vmax=vmax_pr, alpha=alpha_pred)
            ax.set_title(f"Density Map | Pred: {pred_count:.1f}")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"✅ Saved visualization to: {out_png}")


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to config file (.json or .yaml)")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)

    p.add_argument("--model", type=str, required=True)  # es: clip_resnet50 / clip_vit_b_16
    p.add_argument("--input_size", type=int, default=224)  # fallback, ma verrà override da cfg se presente
    p.add_argument("--reduction", type=int, default=8)     # fallback, ma verrà override da cfg se presente

    p.add_argument("--out_png", type=str, default="stage2_vis.png")
    p.add_argument("--num_rows", type=int, default=3)

    p.add_argument("--indices", type=str, default=None, help='Esempio: --indices "17,201,55"')
    p.add_argument("--names", type=str, default=None, help='Esempio: --names "IMG_4.jpg,IMG_43.jpg,IMG_86.jpg"')

    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--alpha_pred", type=float, default=0.55)
    p.add_argument("--alpha_gt_bg", type=float, default=0.97)
    p.add_argument("--point_size", type=float, default=28.0)
    p.add_argument("--point_edge", type=float, default=0.5)

    p.add_argument("--device", type=str, default="cuda:0")

    args = p.parse_args()

    visualize_stage2(
        config_path=args.config,
        ckpt_path=args.checkpoint,
        out_png=args.out_png,
        dataset=args.dataset,
        model_name=args.model,
        input_size=args.input_size,
        reduction=args.reduction,
        num_rows=args.num_rows,
        names=args.names,
        indices=args.indices,
        seed=args.seed,
        alpha_pred=args.alpha_pred,
        alpha_gt_bg=args.alpha_gt_bg,
        point_size=args.point_size,
        point_edge=args.point_edge,
        device=args.device,
    )
