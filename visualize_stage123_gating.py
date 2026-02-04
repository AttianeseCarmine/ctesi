#!/usr/bin/env python3
# visualize_stage123_gating_grid.py
#
# 4 colonne:
#  (1) Input + GT count
#  (2) Stage1 binary gating mask (white=crowd, black=bg) + patch-grid
#  (3) Stage2 HARD-GATED (raw * binary_mask) overlay + Pred/GT/|err|   [rinominata "Stage 2"]
#  (4) Stage3 final density overlay + Pred/GT/|err| (+ ratio)          [rinominata "Stage 2" come richiesto]
#
# NOTE:
# - Usiamo split="val" internamente (niente --split)
# - Usiamo upsample NEAREST della mask per preservare la struttura a patch

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
from utils import get_dataloader


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
# Dataloader args (stile tuo visualize_stage2)
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


def _find_indices_by_names(ds, names):
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


def _parse_indices(indices_str, num_rows):
    if not indices_str:
        return None
    parts = [p.strip() for p in indices_str.split(",") if p.strip()]
    if not parts:
        return None
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            raise ValueError(f"--indices contiene un valore non intero: '{p}'")
    return out[:num_rows]


def _choose_indices(ds, num_rows, indices, names, seed):
    if indices:
        chosen = _parse_indices(indices, num_rows)
        N = len(ds)
        bad = [i for i in chosen if i < 0 or i >= N]
        if bad:
            raise IndexError(f"--indices fuori range (0..{N-1}): {bad}")
        return chosen

    if names:
        name_list = [x.strip() for x in names.split(",") if x.strip()]
        idxs = _find_indices_by_names(ds, name_list)
        if not idxs:
            print("⚠️ Nessuna immagine trovata con quei nomi (o dataset non espone i path). Userò random dal validation set.")
        else:
            return idxs[:num_rows]

    rng = np.random.default_rng(seed)
    N = len(ds)
    if N <= num_rows:
        return list(range(N))
    return rng.choice(N, size=num_rows, replace=False).tolist()


# -------------------------
# Misc helpers
# -------------------------
def _denorm_image(img_chw, mean, std):
    if mean is None or std is None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    mean = torch.tensor(mean, device=img_chw.device).view(3, 1, 1)
    std = torch.tensor(std, device=img_chw.device).view(3, 1, 1)
    x = img_chw * std + mean
    return x.clamp(0, 1)


def _unwrap_density(stage2_out):
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


def load_state_dict_flexible(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)


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
        or 8.0
    )
    return float(base_steep), float(max_steep)


def main():
    p = argparse.ArgumentParser("Visualize Stage1 mask + Stage2(hard) + Stage3(final)")

    p.add_argument("--s1_config", type=str, required=True)
    p.add_argument("--s1_ckpt", type=str, required=True)

    p.add_argument("--s2_config", type=str, required=True)
    p.add_argument("--s2_ckpt", type=str, required=True)

    p.add_argument("--s3_config", type=str, required=True)
    p.add_argument("--s3_ckpt", type=str, required=True)

    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--model", type=str, required=True)

    # selection
    p.add_argument("--num_rows", type=int, default=3)
    p.add_argument("--indices", type=str, default=None)
    p.add_argument("--names", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)

    # visuals
    p.add_argument("--alpha_density", type=float, default=0.55)
    p.add_argument("--out_png", type=str, default="stage12_stage2_grid.png")
    p.add_argument("--device", type=str, default="cuda:0")

    args = p.parse_args()

    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cpu":
        print("ℹ️ CUDA non disponibile: uso CPU.")

    # configs
    cfg_s1 = load_config(args.s1_config)
    cfg_s2 = load_config(args.s2_config)
    cfg_s3 = load_config(args.s3_config)

    # patch grid
    patch_px = _safe_int(cfg_s1.get("reduction", cfg_s1.get("REDUCTION", None)), None)
    if patch_px is None:
        patch_px = 8
        print(f"⚠️ Stage1 config senza reduction: fallback patch_px={patch_px}")
    print(f"🧩 patch_px={patch_px}")

    # mask thr
    mask_threshold = cfg_s1.get("eval_threshold", None)
    if mask_threshold is None:
        mask_threshold = 0.5
        print("ℹ️ Stage1 eval_threshold è null/non presente -> uso mask_threshold=0.5")
    mask_threshold = float(mask_threshold)
    print(f"🎚️ mask_threshold={mask_threshold}")

    # stage3 steepness
    _, max_steep = _read_stage3_steepness(cfg_s3)
    print(f"📌 Stage3 eval steepness={max_steep}")

    # stage2 params
    input_size = _safe_int(cfg_s2.get("input_size", cfg_s2.get("INPUT_SIZE", 224)), 224)
    reduction = _safe_int(cfg_s2.get("reduction", cfg_s2.get("REDUCTION", 8)), 8)

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

    # dataloader val
    dl_args = _make_args_for_dataloader(cfg_s2, dataset=args.dataset, input_size=input_size)
    val_loader = get_dataloader(dl_args, split="val", ddp=False)
    ds = val_loader.dataset

    chosen = _choose_indices(ds, args.num_rows, args.indices, args.names, args.seed)
    print(f"🧾 Selected indices: {chosen}")

    # Stage1 build from stage3 cfg backbone
    zip_head_cfg = cfg_s3.get("ZIP_HEAD", {"HIDDEN_DIM": 256})
    if "BACKBONE" not in cfg_s3:
        raise KeyError("config_stage3.yaml non contiene BACKBONE (necessario per costruire ZIPModel come nel training).")

    zip_cfg = {
        "BACKBONE": cfg_s3["BACKBONE"],
        "ZIP_HEAD": zip_head_cfg,
        "REDUCTION": int(reduction),
    }
    stage1 = ZIPModel(zip_cfg).to(device)
    load_state_dict_flexible(stage1, args.s1_ckpt)
    stage1.eval()


    num_vpt_val = cfg_s2.get("num_vpt", cfg_s2.get("NUM_VPT", 32))
    num_vpt_val = 32 if num_vpt_val is None else int(num_vpt_val)

    vpt_drop_val = cfg_s2.get("vpt_drop", cfg_s2.get("VPT_DROP", 0.0))
    vpt_drop_val = 0.0 if vpt_drop_val is None else float(vpt_drop_val)

    # Stage2
    stage2 = get_model(
        backbone=args.model,
        input_size=input_size,
        reduction=reduction,
        bins=bins,
        anchor_points=anchor_points,
        prompt_type=cfg_s2.get("prompt_type", cfg_s2.get("PROMPT_TYPE", "word")),
        num_vpt=num_vpt_val,
        vpt_drop=vpt_drop_val,
        deep_vpt=not bool(cfg_s2.get("shallow_vpt", cfg_s2.get("SHALLOW_VPT", False))),
    ).to(device)

    load_state_dict_flexible(stage2, args.s2_ckpt)
    stage2.eval()

    # Stage3
    stage3 = ZIPCLIPJointModel(stage1, stage2, steepness=float(max_steep)).to(device)
    load_state_dict_flexible(stage3, args.s3_ckpt)
    stage3.eval()

    out_png = args.out_png
    if not out_png.lower().endswith(".png"):
        out_png += ".png"
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    # 4 colonne
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

            # Stage1 mask
            out1 = stage1(image_b)
            pi_logits = _get_stage1_logits(out1)
            pi_prob = torch.sigmoid(pi_logits)

            pi_up = F.interpolate(pi_prob.detach().cpu(), size=(H, W), mode="nearest")
            mask_bin = (pi_up[0, 0] >= mask_threshold).float()
            keep = float(mask_bin.mean().item())

            # Stage2 raw -> serve solo per fare hard-gated
            out2 = stage2(image_b)
            raw_density = _unwrap_density(out2)
            raw_map = _resize_density_preserve_sum(raw_density[0:1].detach().cpu(), H, W)[0, 0]

            # Stage2 HARD-GATED (questa sarà colonna 3)
            raw_map_hard = raw_map * mask_bin
            vmax_hard = _robust_vmax(raw_map_hard)
            pred2_hard = float(raw_map_hard.sum().item())
            abs2_hard = abs(pred2_hard - gt_count)

            # Stage3 final (colonna 4, ma la chiami "Stage 2" come richiesto)
            stage3.steepness = float(max_steep)
            out3 = stage3(image_b)
            final_density = out3["final_density"]
            final_map = _resize_density_preserve_sum(final_density[0:1].detach().cpu(), H, W)[0, 0]
            vmax_final = _robust_vmax(final_map)
            pred3 = float(final_density.sum().item())
            abs3 = abs(pred3 - gt_count)

            ratio = float(pred3 / (pred2_hard + 1e-6))

            pth = _dataset_get_path(ds, idx)
            name_show = os.path.basename(str(pth)) if pth else f"idx={idx}"

            # Col1: Input
            ax = axes[row, 0]
            ax.imshow(img_dn.permute(1, 2, 0).numpy())
            ax.set_title(f"Input | {name_show}\nGT={gt_count}")
            ax.axis("off")

            # Col2: Mask + grid
            ax = axes[row, 1]
            ax.imshow(mask_bin.numpy(), cmap="gray", vmin=0, vmax=1)
            _draw_patch_grid(ax, H, W, patch_px=patch_px, lw=0.45)
            ax.set_title(f"Stage1 Binary Mask\nthr={mask_threshold:.2f} | keep={keep:.2f}")
            ax.axis("off")

            # Col3: (ex Stage2 hard-gated) -> chiamala "Stage 2"
            ax = axes[row, 2]
            ax.imshow(img_dn.permute(1, 2, 0).numpy())
            ax.imshow(raw_map_hard.numpy(), cmap="jet", vmin=0, vmax=vmax_hard, alpha=float(args.alpha_density))
            ax.set_title(f"Stage 2\nPred={pred2_hard:.1f} | GT={gt_count} | |err|={abs2_hard:.1f}")
            ax.axis("off")

            # Col4: (ex Stage3 final) -> chiamala "Stage 2" come richiesto
            ax = axes[row, 3]
            ax.imshow(img_dn.permute(1, 2, 0).numpy())
            ax.imshow(final_map.numpy(), cmap="jet", vmin=0, vmax=vmax_final, alpha=float(args.alpha_density))
            ax.set_title(
                f"Stage 2\n"
                f"Pred={pred3:.1f} | GT={gt_count} | |err|={abs3:.1f}\n"
                f"ratio={ratio:.2f}"
            )
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"✅ Saved visualization to: {out_png}")


if __name__ == "__main__":
    main()



#python visualize_stage123_gating.py --s1_config checkpoints/qnrf/resnet50/stage1_v5/config_stage1.yaml --s1_ckpt checkpoints/qnrf/resnet50/stage1_v5/best_model.pth --s2_config checkpoints/qnrf/resnet50/stage2_official/config.yaml --s2_ckpt checkpoints/qnrf/resnet50/stage2_official/best_mae_0.pth  --s3_config checkpoints/qnrf/resnet50/stage3_v2/config_stage3.yaml  --s3_ckpt checkpoints/qnrf/resnet50/stage3_v2/best_model.pth  --dataset sha   --model clip_resnet50  --num_rows 3  --indices "17,22,55"  --out_png stage123_qnrf_resnet50.png
