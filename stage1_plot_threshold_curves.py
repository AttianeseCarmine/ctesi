#!/usr/bin/env python3
# plot_stage1_threshold_curves.py

import argparse
import os
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from types import SimpleNamespace

from models.zip_model import ZIPModel
from utils import get_dataloader


# -----------------------------
# Config helpers
# -----------------------------
def cfg_get(cfg, path, default=None):
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def infer_dataset(cfg):
    return cfg.get("dataset") or cfg.get("DATASET") or "unknown"


def infer_backbone(cfg):
    return (
        cfg.get("backbone")
        or cfg.get("model")
        or cfg.get("encoder")
        or cfg.get("zip_backbone")
        or cfg_get(cfg, ("BACKBONE", "TYPE"), None)
        or "unknown_backbone"
    )


def build_args_for_dataloader(cfg):
    """
    Crea un args compatibile con get_dataloader() prendendo i parametri dalla config.
    Supporta sia config flat (tipo config_stage1.yaml salvato) sia nested.
    """
    a = SimpleNamespace()

    a.dataset = cfg.get("dataset", None) or cfg_get(cfg, ("DATASET",), None) or "sha"
    a.data_dir = cfg.get("data_dir", None) or cfg_get(cfg, ("DATA_DIR",), None) or "./data"
    a.input_size = cfg.get("input_size", None) or cfg_get(cfg, ("INPUT_SIZE",), None) or 224
    a.reduction = cfg.get("reduction", None) or cfg_get(cfg, ("REDUCTION",), None) or 8
    a.batch_size = cfg.get("batch_size", None) or cfg_get(cfg, ("TRAIN_STAGE1", "BATCH_SIZE"), None) or 16
    a.num_workers = cfg.get("num_workers", None) or cfg_get(cfg, ("TRAIN_STAGE1", "NUM_WORKERS"), None) or 8

    # Flag usati spesso nel progetto
    a.sliding_window = bool(cfg.get("sliding_window", False))
    a.resize_to_multiple = bool(cfg.get("resize_to_multiple", False))
    a.zero_pad_to_multiple = bool(cfg.get("zero_pad_to_multiple", False))
    a.regression = bool(cfg.get("regression", False))
    a.prompt_type = cfg.get("prompt_type", None)

    a.stride = cfg.get("stride", None)
    a.window_size = cfg.get("window_size", None)

    return a


def parse_threshold_grid(cfg):
    th_start = cfg.get("th_start") or cfg.get("TH_START") or cfg_get(cfg, ("EVAL", "TH_START"), None) or 0.0
    th_end = cfg.get("th_end") or cfg.get("TH_END") or cfg_get(cfg, ("EVAL", "TH_END"), None) or 1.0
    th_step = cfg.get("th_step") or cfg.get("TH_STEP") or cfg_get(cfg, ("EVAL", "TH_STEP"), None) or 0.01

    th_step = float(th_step)
    if th_step <= 0:
        th_step = 0.01

    thresholds = np.arange(float(th_start), float(th_end) + 1e-12, th_step, dtype=np.float64)
    thresholds = np.clip(thresholds, 0.0, 1.0)
    return thresholds


def parse_eval_params(cfg):
    """
    Legge parametri "robusti" dalla config se ci sono, altrimenti default.
    """
    target_recall = (
        cfg.get("target_recall")
        or cfg.get("TARGET_RECALL")
        or cfg_get(cfg, ("TRAIN_STAGE1", "TARGET_RECALL"), None)
        or 0.90
    )
    max_pos_rate = (
        cfg.get("max_pos_rate")
        or cfg.get("MAX_POS_RATE")
        or cfg_get(cfg, ("EVAL", "MAX_POS_RATE"), None)
        or 0.60
    )
    gt_thr = (
        cfg.get("gt_thr")
        or cfg.get("GT_THR")
        or cfg_get(cfg, ("EVAL", "GT_THR"), None)
        or 1e-3
    )
    return float(target_recall), float(max_pos_rate), float(gt_thr)


# -----------------------------
# Eval patch-level
# -----------------------------
@torch.no_grad()
def evaluate_patch_level(model, dataloader, device, thresholds, gt_thr):
    model.eval()
    stats = [{"tp": 0, "tn": 0, "fp": 0, "fn": 0} for _ in thresholds]

    for batch in tqdm(dataloader, desc="Eval (patch-level)"):
        img = None
        gt_density = None

        if isinstance(batch, dict):
            img = batch.get("image", None)
            gt_density = batch.get("density", batch.get("labels", None))
        elif isinstance(batch, (list, tuple)):
            img = batch[0] if len(batch) > 0 else None
            if len(batch) >= 3 and torch.is_tensor(batch[2]):
                gt_density = batch[2]
            elif len(batch) >= 2 and torch.is_tensor(batch[1]):
                gt_density = batch[1]

        if img is None or gt_density is None:
            continue

        img = img.to(device)
        gt_density = gt_density.to(device)

        out = model(img)
        logits = out["pi_logits"] if isinstance(out, dict) else out
        probs = torch.sigmoid(logits)

        h_out, w_out = logits.shape[-2], logits.shape[-1]
        H, W = gt_density.shape[-2], gt_density.shape[-1]
        r_h = max(1, H // h_out)
        r_w = max(1, W // w_out)

        gt_block = F.max_pool2d(gt_density, kernel_size=(r_h, r_w), stride=(r_h, r_w))
        gt_bin = (gt_block > gt_thr).float()

        for i, th in enumerate(thresholds):
            pred = (probs > th).float()
            stats[i]["tp"] += ((pred == 1) & (gt_bin == 1)).sum().item()
            stats[i]["tn"] += ((pred == 0) & (gt_bin == 0)).sum().item()
            stats[i]["fp"] += ((pred == 1) & (gt_bin == 0)).sum().item()
            stats[i]["fn"] += ((pred == 0) & (gt_bin == 1)).sum().item()

    return stats


def compute_precision_recall(stats, thresholds):
    eps = 1e-7
    prec = []
    rec = []

    for s in stats:
        tp, tn, fp, fn = s["tp"], s["tn"], s["fp"], s["fn"]
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        prec.append(p)
        rec.append(r)

    return {
        "thresholds": np.asarray(thresholds, dtype=np.float64),
        "precision": np.asarray(prec, dtype=np.float64),
        "recall": np.asarray(rec, dtype=np.float64),
    }


def choose_threshold(stats, thresholds, target_recall, max_pos_rate):
    """
    Scelta robusta:
    1) recall >= target_recall
    2) pos_rate <= max_pos_rate  (pos_rate = (TP+FP)/tot)
    3) tra le valide: minimizza FP, tie-break: massimizza Precision
    Fallback:
    - se recall ok ma pos_rate no: min FP tra recall-ok
    - se recall non ok: max recall
    """
    eps = 1e-7
    candidates = []

    for i, th in enumerate(thresholds):
        tp, tn, fp, fn = stats[i]["tp"], stats[i]["tn"], stats[i]["fp"], stats[i]["fn"]
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        total = tp + tn + fp + fn
        pos_rate = (tp + fp) / (total + eps)

        candidates.append(
            dict(i=i, th=float(th), tp=tp, tn=tn, fp=fp, fn=fn, prec=prec, rec=rec, pos_rate=pos_rate)
        )

    valid_recall = [c for c in candidates if c["rec"] >= target_recall]
    valid = [c for c in valid_recall if c["pos_rate"] <= max_pos_rate]

    if valid:
        chosen = min(valid, key=lambda x: (x["fp"], -x["prec"]))
        status = "✅ constrained (recall ok, pos_rate ok)"
    elif valid_recall:
        chosen = min(valid_recall, key=lambda x: (x["fp"], -x["prec"]))
        status = "⚠️ fallback (recall ok, pos_rate not met)"
    else:
        chosen = max(candidates, key=lambda x: x["rec"])
        status = "❌ fallback (max recall)"

    return chosen, status


# -----------------------------
# Plot helper (solo andamento)
# -----------------------------
def plot_metric_vs_threshold(th, y, title, y_label, out_path):
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(th, y, marker="o", linewidth=1.5, markersize=3, label="Stage 1")

    # inverti asse x (threshold decrescente)
    plt.gca().invert_xaxis()

    plt.title(title)
    plt.xlabel("Detector confidence threshold")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.35)

    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser("Stage1 curves (precision/recall vs threshold)")
    parser.add_argument("--config", type=str, required=True, help="Stage1 config yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint .pth")
    parser.add_argument("--backbone", type=str, default=None, help="Override backbone (es: vit_b_16, resnet50)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # -----------------------------
    # BACKBONE FIX (compatibile con config flat e nested)
    # -----------------------------
    bb = args.backbone or cfg.get("model") or cfg.get("backbone") or cfg.get("encoder") or cfg.get("zip_backbone")

    cfg["BACKBONE"] = cfg.get("BACKBONE", {}) or {}
    if bb is not None:
        cfg["BACKBONE"]["TYPE"] = bb

    dl_args = build_args_for_dataloader(cfg)
    dataset_name = getattr(dl_args, "dataset", infer_dataset(cfg))
    backbone_name = infer_backbone(cfg)

    thresholds = parse_threshold_grid(cfg)
    target_recall, max_pos_rate, gt_thr = parse_eval_params(cfg)

    print("[*] Config-derived settings")
    print(f"    dataset       : {dataset_name}")
    print(f"    backbone      : {backbone_name}")
    print(f"    input_size    : {getattr(dl_args, 'input_size', 'n/a')}")
    print(f"    reduction     : {getattr(dl_args, 'reduction', 'n/a')}")
    print(f"    batch_size    : {getattr(dl_args, 'batch_size', 'n/a')}")
    print(f"    num_workers   : {getattr(dl_args, 'num_workers', 'n/a')}")
    print(f"    gt_thr        : {gt_thr:g}")
    print(f"    target_recall : {target_recall:.3f}")
    print(f"    max_pos_rate  : {max_pos_rate:.3f}")
    if len(thresholds) > 1:
        print(
            f"    thresholds    : {thresholds[0]:.2f}..{thresholds[-1]:.2f} "
            f"step={thresholds[1]-thresholds[0]:.3f} (N={len(thresholds)})"
        )

    # dataloader
    loader = get_dataloader(dl_args, split="val", ddp=False)

    # model
    model = ZIPModel(cfg).to(device)
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()}, strict=False)

    # eval
    stats = evaluate_patch_level(model, loader, device, thresholds, gt_thr=gt_thr)
    chosen, status = choose_threshold(stats, thresholds, target_recall=target_recall, max_pos_rate=max_pos_rate)

    print(status)
    print(
        f"[*] BEST th={chosen['th']:.2f} | P={chosen['prec']:.4f} R={chosen['rec']:.4f} "
        f"| FP={chosen['fp']} | pos_rate={chosen['pos_rate']:.3f}"
    )

    # compute arrays
    m = compute_precision_recall(stats, thresholds)
    th = m["thresholds"]

    title_prefix = f"Stage1 ({dataset_name}, {backbone_name})"

    out_prec = os.path.join(
        args.out_dir,
        f"precision_vs_threshold_{dataset_name}_{backbone_name}.png".replace("/", "_"),
    )
    out_rec = os.path.join(
        args.out_dir,
        f"recall_vs_threshold_{dataset_name}_{backbone_name}.png".replace("/", "_"),
    )

    plot_metric_vs_threshold(
        th, m["precision"],
        title=f"{title_prefix}: precision vs detector threshold",
        y_label="Precision",
        out_path=out_prec,
    )

    plot_metric_vs_threshold(
        th, m["recall"],
        title=f"{title_prefix}: recall vs detector threshold",
        y_label="Recall",
        out_path=out_rec,
    )

    print("[*] Saved plots:")
    print(f"    {out_prec}")
    print(f"    {out_rec}")


if __name__ == "__main__":
    main()
