import os
import json
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from datasets import standardize_dataset_name
from models import get_model
from models.zip_model import ZIPModel
from models.joint_model import ZIPCLIPJointModel
from utils import get_dataloader, sliding_window_predict as sw_predict_stage2  # <- USATO per Stage2


# -------------------------
# YAML loader that supports !!python/tuple
# -------------------------
class SafeTupleLoader(yaml.SafeLoader):
    pass


def construct_python_tuple(loader, node):
    return tuple(loader.construct_sequence(node))


SafeTupleLoader.add_constructor("tag:yaml.org,2002:python/tuple", construct_python_tuple)


def load_config(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r") as f:
            return json.load(f) or {}
    if ext in [".yaml", ".yml"]:
        with open(path, "r") as f:
            return yaml.load(f, Loader=SafeTupleLoader) or {}
    # fallback
    try:
        with open(path, "r") as f:
            return json.load(f) or {}
    except Exception:
        with open(path, "r") as f:
            return yaml.load(f, Loader=SafeTupleLoader) or {}


def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x, default=None):
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


# -------------------------
# Dataloader args
# -------------------------
def _make_args_for_dataloader(cfg, dataset, input_size):
    args = argparse.Namespace()
    args.dataset = standardize_dataset_name(dataset)
    args.batch_size = 1
    args.num_workers = 0
    args.input_size = int(cfg.get("input_size", input_size)) if isinstance(cfg.get("input_size", input_size), (int, float)) else int(input_size)

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


# -------------------------
# Dataset path helpers
# -------------------------
def _resolve_subset_index(ds, idx):
    if hasattr(ds, "indices") and hasattr(ds, "dataset"):
        try:
            return ds.dataset, int(ds.indices[idx])
        except Exception:
            return ds, idx
    return ds, idx


def _dataset_get_path(ds, idx):
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


def _parse_indices(indices_str, num_rows):
    if not indices_str:
        return None
    parts = [p.strip() for p in indices_str.split(",") if p.strip()]
    if not parts:
        return None
    out = []
    for p in parts:
        out.append(int(p))
    return out[:num_rows]


def _choose_indices(ds, num_rows, indices, seed):
    if indices:
        chosen = _parse_indices(indices, num_rows)
        N = len(ds)
        bad = [i for i in chosen if i < 0 or i >= N]
        if bad:
            raise IndexError(f"--indices fuori range (0..{N-1}): {bad}")
        return chosen

    rng = np.random.default_rng(seed)
    N = len(ds)
    if N <= num_rows:
        return list(range(N))
    return rng.choice(N, size=num_rows, replace=False).tolist()


# -------------------------
# Misc helpers
# -------------------------
def _denorm_image(img_chw, mean, std):
    # Solo per visualizzare. Se mean/std sono None: fallback ImageNet.
    if mean is None or std is None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    mean = torch.tensor(mean, device=img_chw.device).view(3, 1, 1)
    std = torch.tensor(std, device=img_chw.device).view(3, 1, 1)
    x = img_chw * std + mean
    return x.clamp(0, 1)


def _unwrap_density(stage2_out):
    # identico alla logica robusta che usavi
    if isinstance(stage2_out, torch.Tensor):
        return stage2_out
    if isinstance(stage2_out, (tuple, list)):
        if len(stage2_out) >= 2 and isinstance(stage2_out[1], torch.Tensor):
            return stage2_out[1]
        for x in stage2_out:
            if isinstance(x, torch.Tensor) and x.dim() == 4:
                return x
        raise TypeError("Stage2 tuple/list output but no density tensor found.")
    if isinstance(stage2_out, dict):
        for k in ["pred_density", "density", "ebc_density", "final_density", "raw_density"]:
            if k in stage2_out and torch.is_tensor(stage2_out[k]):
                return stage2_out[k]
        raise KeyError(f"Stage2 dict output but no density key found. Keys: {list(stage2_out.keys())}")
    raise TypeError(f"Unsupported Stage2 output type: {type(stage2_out)}")


def _resize_density_preserve_sum(density_b1hw, out_h, out_w):
    """Resize bilinear + scaling per conservare la somma (conteggio)."""
    in_h, in_w = density_b1hw.shape[-2:]
    if (in_h, in_w) == (out_h, out_w):
        return density_b1hw
    resized = F.interpolate(density_b1hw, size=(out_h, out_w), mode="bilinear", align_corners=False)
    scale = (in_h * in_w) / float(out_h * out_w)
    return resized * scale


def _robust_vmax(m_2d, q=0.995):
    x = m_2d.detach().float().cpu()
    if x.numel() == 0:
        return 1.0
    vmax = torch.quantile(x.flatten(), q).item()
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(x.max().item()) if x.max().item() > 0 else 1.0
    return vmax


def _count_points(target_points):
    # assume points Nx2
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


def load_state_dict_flexible(model, ckpt_path: str, name="model", verbose=True):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    incompatible = model.load_state_dict(sd, strict=False)

    if verbose:
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        print(f"[{name}] loaded: {ckpt_path}")
        print(f"[{name}] missing_keys: {len(missing)}")
        if len(missing) > 0:
            print("  e.g.", missing[:20])
        print(f"[{name}] unexpected_keys: {len(unexpected)}")
        if len(unexpected) > 0:
            print("  e.g.", unexpected[:20])


def _draw_patch_grid(ax, H, W, patch_px, lw=0.45):
    if patch_px is None or patch_px <= 1:
        return
    ax.set_xticks(np.arange(-0.5, W, patch_px), minor=True)
    ax.set_yticks(np.arange(-0.5, H, patch_px), minor=True)
    ax.grid(which="minor", color="gray", linewidth=lw)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)


def _get_stage1_logits(out1):
    if isinstance(out1, dict):
        for k in ["pi_logits", "logit_pi", "logits", "pred_class"]:
            if k in out1 and torch.is_tensor(out1[k]):
                return out1[k]
        raise KeyError(f"Stage1 output dict senza pi_logits. Keys: {list(out1.keys())}")
    if torch.is_tensor(out1):
        return out1
    raise TypeError(f"Stage1 output type non supportato: {type(out1)}")


def _read_stage3_steepness(cfg_s3):
    t = cfg_s3.get("TRAIN_STAGE3", {}) if isinstance(cfg_s3.get("TRAIN_STAGE3", {}), dict) else {}
    base_steep = (
        _safe_float(t.get("BASE_STEEPNESS", None), None)
        or _safe_float(cfg_s3.get("base_steepness", None), None)
        or 1.0
    )
    max_steep = (
        _safe_float(t.get("MAX_STEEPNESS", None), None)
        or _safe_float(cfg_s3.get("max_steepness", None), None)
        or 2.0
    )
    return float(base_steep), float(max_steep)


# -------------------------
# Stage3 sliding-window (training-like)
# -------------------------
@torch.no_grad()
def sliding_window_predict_stage3(joint_model, image_b, window_size, stride, reduction):
    """
    Sliding-window per JOINT model (Stage3), coerente con il tuo train_stage3:
    - crop a window_size
    - out = model(crop)['final_density']
    - media su overlap
    """
    joint_model.eval()
    B, C, H, W = image_b.shape
    assert B == 1, "Stage3 sliding-window richiede batch_size=1"

    # se immagine piccola -> diretto
    if H <= window_size and W <= window_size:
        out = joint_model(image_b)
        return out["final_density"]

    # pad a multiplo di window_size (come nel tuo training)
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        image_b = F.pad(image_b, (0, pad_w, 0, pad_h), mode="constant", value=0)

    _, _, H_pad, W_pad = image_b.shape
    out_h = H_pad // reduction
    out_w = W_pad // reduction

    output_density = torch.zeros((1, 1, out_h, out_w), device=image_b.device)
    count_map = torch.zeros((1, 1, out_h, out_w), device=image_b.device)

    for y in range(0, H_pad, stride):
        for x in range(0, W_pad, stride):
            crop = image_b[:, :, y:y + window_size, x:x + window_size]
            if crop.shape[2] != window_size or crop.shape[3] != window_size:
                continue

            out = joint_model(crop)
            pred_crop = out["final_density"]

            y_out = y // reduction
            x_out = x // reduction
            h_c = pred_crop.shape[2]
            w_c = pred_crop.shape[3]

            output_density[:, :, y_out:y_out + h_c, x_out:x_out + w_c] += pred_crop
            count_map[:, :, y_out:y_out + h_c, x_out:x_out + w_c] += 1

    output_density = output_density / (count_map + 1e-6)

    # remove pad
    final_h = H // reduction
    final_w = W // reduction
    return output_density[:, :, :final_h, :final_w]


def main():
    p = argparse.ArgumentParser("Visualize Stage1 mask + Stage2 standalone + Stage3 final")

    p.add_argument("--s1_config", type=str, required=True)
    p.add_argument("--s1_ckpt", type=str, required=True)
    p.add_argument("--s2_config", type=str, required=True)
    p.add_argument("--s2_ckpt", type=str, required=True)
    p.add_argument("--s3_config", type=str, required=True)
    p.add_argument("--s3_ckpt", type=str, required=True)

    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--model", type=str, required=True)  # clip_resnet50 / clip_vit_b_16

    p.add_argument("--num_rows", type=int, default=3)
    p.add_argument("--indices", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--alpha_density", type=float, default=0.55)
    p.add_argument("--out_png", type=str, default="stage123_grid.png")
    p.add_argument("--device", type=str, default="cuda:0")

    # override threshold Stage1 (se vuoi provare valori diversi)
    p.add_argument("--mask_threshold", type=float, default=None)

    args = p.parse_args()

    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cpu":
        print("ℹ️ CUDA non disponibile: uso CPU.")

    cfg_s1 = load_config(args.s1_config)
    cfg_s2 = load_config(args.s2_config)
    cfg_s3 = load_config(args.s3_config)

    # reductions: keep them separate
    reduction_s1 = _safe_int(cfg_s1.get("reduction", cfg_s1.get("REDUCTION", 8)), 8)
    reduction_s2 = _safe_int(cfg_s2.get("reduction", cfg_s2.get("REDUCTION", 8)), 8)

    # stage3 reduction: se nel cfg c'è, usala; altrimenti usa reduction_s2 (di solito coincide)
    reduction_s3 = _safe_int(cfg_s3.get("reduction", cfg_s3.get("REDUCTION", None)), None)
    if reduction_s3 is None:
        reduction_s3 = reduction_s2

    patch_px = reduction_s1
    print(f"🧩 reduction_s1={reduction_s1} | reduction_s2={reduction_s2} | reduction_s3={reduction_s3} | patch_px={patch_px}")

    # mask threshold
    cfg_thr = cfg_s1.get("eval_threshold", None)
    mask_threshold = args.mask_threshold if args.mask_threshold is not None else cfg_thr
    if mask_threshold is None:
        mask_threshold = 0.5
        print("ℹ️ Stage1 eval_threshold non presente e --mask_threshold non fornito -> uso 0.5")
    mask_threshold = float(mask_threshold)
    print(f"🎚️ mask_threshold={mask_threshold}")

    # stage3 steepness
    _, max_steep = _read_stage3_steepness(cfg_s3)
    print(f"📌 Stage3 eval steepness={max_steep}")

    # Stage2 params (coerenti con visualize_stage2.py)
    input_size = _safe_int(cfg_s2.get("input_size", cfg_s2.get("INPUT_SIZE", 224)), 224)

    if "bins" not in cfg_s2 or "anchor_points" not in cfg_s2:
        raise ValueError("Nel config Stage2 non trovo 'bins' e/o 'anchor_points'.")

    bins = []
    for a, b in cfg_s2["bins"]:
        a = float(a)
        if isinstance(b, str) and b.strip().lower() in [".inf", "inf", "infinity"]:
            b = float("inf")
        elif isinstance(b, float) and not np.isfinite(b):
            b = float("inf")
        else:
            b = float(b)
        bins.append((a, b))
    anchor_points = [float(x) for x in cfg_s2["anchor_points"]]

    mean = cfg_s2.get("NORM_MEAN", None)
    std = cfg_s2.get("NORM_STD", None)

    # Stage2 inference mode (IDENTICO a visualize_stage2.py)
    s2_use_sw = bool(cfg_s2.get("sliding_window", False))
    s2_ws = _safe_int(cfg_s2.get("window_size", None), input_size)
    s2_st = _safe_int(cfg_s2.get("stride", None), input_size)
    print(f"🪟 Stage2 SW: {s2_use_sw} | ws={s2_ws} st={s2_st}")

    # Stage3 inference mode: se nel config stage3 c'è sliding_window, seguilo
    s3_use_sw = bool(cfg_s3.get("sliding_window", False))
    s3_ws = _safe_int(cfg_s3.get("window_size", None), input_size)
    s3_st = _safe_int(cfg_s3.get("stride", None), input_size)
    print(f"🪟 Stage3 SW: {s3_use_sw} | ws={s3_ws} st={s3_st}")

    # dataloader (val) — usiamo cfg_s2 per coerenza con stage2 (come fai di solito)
    dl_args = _make_args_for_dataloader(cfg_s2, dataset=args.dataset, input_size=input_size)
    val_loader = get_dataloader(dl_args, split="val", ddp=False)
    ds = val_loader.dataset

    chosen = _choose_indices(ds, args.num_rows, args.indices, args.seed)
    print(f"🧾 Selected indices: {chosen}")

    # Stage1 build (da cfg_s3 BACKBONE ma con reduction_s1)
    zip_head_cfg = cfg_s3.get("ZIP_HEAD", {"HIDDEN_DIM": 256})
    if "BACKBONE" not in cfg_s3:
        raise KeyError("config Stage3 non contiene BACKBONE (necessario per costruire ZIPModel).")

    zip_cfg = {
        "BACKBONE": cfg_s3["BACKBONE"],
        "ZIP_HEAD": zip_head_cfg,
        "REDUCTION": int(reduction_s1),
    }
    stage1 = ZIPModel(zip_cfg).to(device)
    load_state_dict_flexible(stage1, args.s1_ckpt, name="stage1", verbose=True)
    stage1.eval()

    # Stage2 build
    num_vpt_val = cfg_s2.get("num_vpt", cfg_s2.get("NUM_VPT", 32))
    num_vpt_val = 32 if num_vpt_val is None else int(num_vpt_val)

    vpt_drop_val = cfg_s2.get("vpt_drop", cfg_s2.get("VPT_DROP", 0.0))
    vpt_drop_val = 0.0 if vpt_drop_val is None else float(vpt_drop_val)

    shallow_val = cfg_s2.get("shallow_vpt", cfg_s2.get("SHALLOW_VPT", False))
    shallow_val = bool(shallow_val) if shallow_val is not None else False

    prompt_type_val = cfg_s2.get("prompt_type", cfg_s2.get("PROMPT_TYPE", "word")) or "word"

    stage2 = get_model(
        backbone=args.model,
        input_size=input_size,
        reduction=reduction_s2,
        bins=bins,
        anchor_points=anchor_points,
        prompt_type=prompt_type_val,
        num_vpt=num_vpt_val,
        vpt_drop=vpt_drop_val,
        deep_vpt=not shallow_val,
    ).to(device)

    load_state_dict_flexible(stage2, args.s2_ckpt, name="stage2", verbose=True)
    stage2.eval()

    # Stage3 joint
    stage3 = ZIPCLIPJointModel(stage1, stage2, steepness=float(max_steep)).to(device)
    load_state_dict_flexible(stage3, args.s3_ckpt, name="stage3", verbose=True)
    stage3.eval()

    out_png = args.out_png
    if not out_png.lower().endswith(".png"):
        out_png += ".png"
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    # Plot grid
    num_cols = 4
    fig, axes = plt.subplots(args.num_rows, num_cols, figsize=(5 * num_cols, 4.8 * args.num_rows))
    if args.num_rows == 1:
        axes = axes.reshape(1, num_cols)

    with torch.no_grad():
        for row, idx in enumerate(chosen):
            sample = ds[idx]
            image = sample[0]
            target_points = sample[1]

            image_b = image.unsqueeze(0).to(device) if (isinstance(image, torch.Tensor) and image.dim() == 3) else image.to(device)

            img_dn = _denorm_image(image_b[0].detach().cpu(), mean, std)
            H, W = img_dn.shape[-2], img_dn.shape[-1]
            gt_count = _count_points(target_points)

            # ----------------
            # Stage1 mask (solo visual e per capire gating)
            # ----------------
            out1 = stage1(image_b)
            pi_logits = _get_stage1_logits(out1)
            pi_prob = torch.sigmoid(pi_logits)

            pi_up = F.interpolate(pi_prob.detach().cpu(), size=(H, W), mode="nearest")
            mask_bin = (pi_up[0, 0] >= mask_threshold).float()
            keep = float(mask_bin.mean().item())

            # ----------------
            # Stage2 STANDALONE (IDENTICO a visualize_stage2.py)
            # ----------------
            if s2_use_sw:
                out2 = sw_predict_stage2(stage2, image_b, s2_ws, s2_st)
            else:
                out2 = stage2(image_b)

            raw_density = _unwrap_density(out2)
            raw_map = _resize_density_preserve_sum(raw_density[0:1].detach().cpu(), H, W)[0, 0]
            vmax_s2 = _robust_vmax(raw_map)
            pred2 = float(raw_map.sum().item())
            abs2 = abs(pred2 - gt_count)

            # ----------------
            # Stage3 final (coerente con training: sliding-window se attivo)
            # ----------------
            stage3.steepness = float(max_steep)

            if s3_use_sw:
                final_density = sliding_window_predict_stage3(stage3, image_b, s3_ws, s3_st, reduction_s3)
            else:
                out3 = stage3(image_b)
                final_density = out3["final_density"]

            final_map = _resize_density_preserve_sum(final_density[0:1].detach().cpu(), H, W)[0, 0]
            vmax_s3 = _robust_vmax(final_map)
            pred3 = float(final_map.sum().item())
            abs3 = abs(pred3 - gt_count)

            pth = _dataset_get_path(ds, idx)
            name_show = os.path.basename(str(pth)) if pth else f"idx={idx}"

            # ---- COL 1: Input
            ax = axes[row, 0]
            ax.imshow(img_dn.permute(1, 2, 0).numpy())
            ax.set_title(f"Input | {name_show}\nGT={gt_count}")
            ax.axis("off")

            # ---- COL 2: Stage1 mask + grid
            ax = axes[row, 1]
            ax.imshow(mask_bin.numpy(), cmap="gray", vmin=0, vmax=1)
            _draw_patch_grid(ax, H, W, patch_px=patch_px, lw=0.45)
            ax.set_title(f"Stage1 Binary Mask\nthr={mask_threshold:.2f} | keep={keep:.2f}")
            ax.axis("off")

            # ---- COL 3: Stage2 standalone (RAW)
            ax = axes[row, 2]
            ax.imshow(img_dn.permute(1, 2, 0).numpy())
            ax.imshow(raw_map.numpy(), cmap="jet", vmin=0, vmax=vmax_s2, alpha=float(args.alpha_density))
            ax.set_title(f"Stage 2 (standalone)\nPred={pred2:.1f} | GT={gt_count} | |err|={abs2:.1f}")
            ax.axis("off")

            # ---- COL 4: Stage3 final
            ax = axes[row, 3]
            ax.imshow(img_dn.permute(1, 2, 0).numpy())
            ax.imshow(final_map.numpy(), cmap="jet", vmin=0, vmax=vmax_s3, alpha=float(args.alpha_density))
            ax.set_title(f"Stage 3 (final)\nPred={pred3:.1f} | GT={gt_count} | |err|={abs3:.1f}")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"✅ Saved visualization to: {out_png}")


if __name__ == "__main__":
    main()