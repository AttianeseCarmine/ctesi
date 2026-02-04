#!/usr/bin/env python3
# eval_threshold.py
"""
Valutazione Stage 1 (ZIP) a livello patch + generazione Confusion Matrix.

Obiettivo:
- Calcolare TP/TN/FP/FN su griglia patch (stessa risoluzione di pi_logits)
- Selezionare la soglia in modo "robusto":
    1) vincolo di recall minima (target_recall)
    2) vincolo "non far passare tutto" (max_pos_rate = frazione max di patch predette positive)
    3) tra le soglie valide: minimizza FP, tie-break: massimizza F1
- Salvare confusion matrix in stile "Blues" con TN/FP/FN/TP dentro le celle

COERENZA COL TRAINING STAGE1:
- Generazione label patch con max_pool2d sulla density GT
- Soglia label GT: 1e-3
"""

import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.zip_model import ZIPModel
from utils import get_dataloader


# -----------------------------
# Argomenti
# -----------------------------
def get_args():
    parser = argparse.ArgumentParser(
        description="Valutazione Stage 1 - Patch Level + Confusion Matrix (coerente col training)."
    )
    parser.add_argument("--config", type=str, default="config_stage1.yaml", help="Path al file di config")
    parser.add_argument("--gpu", default="0", type=str, help="ID della GPU (CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--ckpt", type=str, default=None, help="Path al checkpoint (best_model.pth)")

    # Confusion matrix output
    parser.add_argument("--cm_out", type=str, default=None, help="Path output PNG confusion matrix")
    parser.add_argument("--backbone", type=str, default=None, help="Nome backbone per intestazione (override)")

    # Scelta soglia (criterio robusto)
    parser.add_argument("--target_recall", type=float, default=0.90, help="Recall minima per selezione soglia")
    parser.add_argument(
        "--max_pos_rate",
        type=float,
        default=0.60,
        help="Massima frazione di patch predette positive (evita 'passa tutto')",
    )

    # Griglia soglie
    parser.add_argument("--th_start", type=float, default=0.0, help="Soglia start")
    parser.add_argument("--th_end", type=float, default=1.0, help="Soglia end (inclusa circa)")
    parser.add_argument("--th_step", type=float, default=0.05, help="Step soglia")

    return parser.parse_args()


# -----------------------------
# Eval patch-level
# -----------------------------
@torch.no_grad()
def evaluate_patch_level(model, dataloader, device, thresholds, gt_thr=1e-3):
    model.eval()

    stats = [
        {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "retained_density": 0.0, "total_density": 0.0}
        for _ in thresholds
    ]

    print("[*] Avvio valutazione PATCH-LEVEL (coerente col training Stage1)...")

    for batch in tqdm(dataloader, desc="Valutazione"):
        img = None
        gt_density = None

        # --- 1) Recupero dati ---
        if isinstance(batch, dict):
            img = batch.get("image", None)
            gt_density = batch.get("density", batch.get("labels", None))
        elif isinstance(batch, (list, tuple)):
            img = batch[0] if len(batch) > 0 else None
            # spesso: (img, points, density)
            if len(batch) >= 3 and torch.is_tensor(batch[2]):
                gt_density = batch[2]
            elif len(batch) >= 2 and torch.is_tensor(batch[1]):
                gt_density = batch[1]

        if img is None or gt_density is None:
            continue

        img = img.to(device)
        gt_density = gt_density.to(device)

        # --- 2) Forward ---
        output = model(img)
        logits = output["pi_logits"] if isinstance(output, dict) else output
        probs = torch.sigmoid(logits)

        # --- 3) GT patch-level (COERENTE col training: max_pool + 1e-3) ---
        h_out, w_out = logits.shape[-2], logits.shape[-1]
        H, W = gt_density.shape[-2], gt_density.shape[-1]

        # ratio tra densità GT e griglia output
        r_h = H // h_out
        r_w = W // w_out
        if r_h <= 0 or r_w <= 0:
            # fallback (non dovrebbe succedere)
            r_h, r_w = 1, 1

        gt_block = F.max_pool2d(gt_density, kernel_size=(r_h, r_w), stride=(r_h, r_w))
        gt_binary_patch = (gt_block > gt_thr).float()

        # --- 4) metriche per ogni soglia ---
        for i, th in enumerate(thresholds):
            pred_binary = (probs > th).float()

            tp = ((pred_binary == 1) & (gt_binary_patch == 1)).sum().item()
            tn = ((pred_binary == 0) & (gt_binary_patch == 0)).sum().item()
            fp = ((pred_binary == 1) & (gt_binary_patch == 0)).sum().item()
            fn = ((pred_binary == 0) & (gt_binary_patch == 1)).sum().item()

            stats[i]["tp"] += tp
            stats[i]["tn"] += tn
            stats[i]["fp"] += fp
            stats[i]["fn"] += fn

            # Quanto densità reale "sopravvive" al filtro (solo per informazione)
            mask_up = F.interpolate(pred_binary, size=gt_density.shape[-2:], mode="nearest")
            retained = (gt_density * mask_up).sum().item()
            total = gt_density.sum().item()

            stats[i]["retained_density"] += retained
            stats[i]["total_density"] += total

    return stats


# -----------------------------
# Confusion Matrix Plot
# -----------------------------
def plot_confusion_matrix_binary(tn, fp, fn, tp, dataset_name, backbone_name, out_path, th=None):
    """
    Matrice 2x2:
        righe = True (0/1), colonne = Pred (0/1)
        [[TN, FP],
         [FN, TP]]
    """
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int64)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=["0", "1"],
        yticklabels=["0", "1"],
        xlabel="Pred",
        ylabel="True",
    )

    title = f"Confusion Matrix - {dataset_name} - {backbone_name}"
    if th is not None:
        title += f" (th={th:.2f})"
    ax.set_title(title)

    cell_labels = np.array([["TN", "FP"], ["FN", "TP"]])
    thresh_val = cm.max() / 2.0 if cm.max() > 0 else 0

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            ax.text(
                j,
                i,
                f"{cell_labels[i, j]}\n{val}",
                ha="center",
                va="center",
                color="white" if val > thresh_val else "black",
                fontsize=11,
            )

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Selezione soglia "robusta"
# -----------------------------
def choose_threshold(stats, thresholds, target_recall=0.90, max_pos_rate=0.60):
    """
    Selezione:
    1) recall >= target_recall
    2) pos_rate <= max_pos_rate  (pos_rate = (TP+FP)/tot)
    3) tra le valide: minimizza FP, tie-break: massimizza F1
    Fallback:
    - se (1) ok ma (2) no: min FP tra quelle con recall ok
    - se (1) no: scegli max recall
    """
    eps = 1e-7
    candidates = []

    for i, th in enumerate(thresholds):
        s = stats[i]
        tp, tn, fp, fn = s["tp"], s["tn"], s["fp"], s["fn"]

        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * (prec * rec) / (prec + rec + eps)
        total = tp + tn + fp + fn

        pos_rate = (tp + fp) / (total + eps)
        fpr = fp / (fp + tn + eps)

        candidates.append(
            {
                "i": i,
                "th": float(th),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "prec": prec,
                "rec": rec,
                "f1": f1,
                "pos_rate": pos_rate,
                "fpr": fpr,
            }
        )

    valid_recall = [c for c in candidates if c["rec"] >= target_recall]
    valid = [c for c in valid_recall if c["pos_rate"] <= max_pos_rate]

    if valid:
        chosen = min(valid, key=lambda x: (x["fp"], -x["f1"]))
        status = "✅ constrained: min FP (recall ok, pos_rate ok)"
    elif valid_recall:
        chosen = min(valid_recall, key=lambda x: (x["fp"], -x["f1"]))
        status = "⚠️ fallback: min FP con recall ok (pos_rate vincolo NON soddisfatto)"
    else:
        chosen = max(candidates, key=lambda x: x["rec"])
        status = "❌ fallback: target recall NON raggiunto, scelto max recall"

    return chosen, status


# -----------------------------
# Main
# -----------------------------
def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Aggiorna args con campi del config (se presenti)
    for k, v in (config.items() if isinstance(config, dict) else []):
        setattr(args, k, v)

    # Defaults utili per loader (come avevi già fatto)
    if not hasattr(args, "dataset"):
        args.dataset = "sha"
    args.sliding_window = False
    args.resize_to_multiple = False
    args.zero_pad_to_multiple = False
    args.regression = False
    args.prompt_type = None

    print("[*] Loading Dataset & Model...")
    loader = get_dataloader(args, split="val", ddp=False)

    # Costruzione modello: nel tuo codice usavi ZIPModel(config)
    # Mantengo quello, ma se il tuo ZIPModel si aspetta cfg/dict, va bene.
    model = ZIPModel(config).to(device)

    # Checkpoint
    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_path = os.path.join(config.get("ckpt_dir", ""), "best_model.pth")

    if not os.path.exists(ckpt_path):
        print(f"[!] Errore: Checkpoint {ckpt_path} non trovato.")
        return

    print(f"[*] Loading weights: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()}, strict=False)

    # Soglie
    thresholds = np.arange(args.th_start, args.th_end + 1e-9, args.th_step)

    # Eval
    stats = evaluate_patch_level(model, loader, device, thresholds, gt_thr=1e-3)

    # Tabella riassuntiva
    print("\n" + "=" * 110)
    print(
        f"{'Thr':<6} | {'F1':<8} | {'Acc':<8} | {'Prec':<8} | {'Rec':<8} | {'FP':<8} | {'FN':<8} | {'pos_rate':<9} | {'DensKept%':<10}"
    )
    print("-" * 110)

    eps = 1e-7
    for i, th in enumerate(thresholds):
        s = stats[i]
        tp, tn, fp, fn = s["tp"], s["tn"], s["fp"], s["fn"]
        acc = (tp + tn) / (tp + tn + fp + fn + eps)
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * (prec * rec) / (prec + rec + eps)
        total = tp + tn + fp + fn
        pos_rate = (tp + fp) / (total + eps)
        dens_kept_pct = (s["retained_density"] / (s["total_density"] + eps)) * 100.0

        print(
            f"{th:<6.2f} | {f1:<8.4f} | {acc:<8.4f} | {prec:<8.4f} | {rec:<8.4f} | {fp:<8d} | {fn:<8d} | {pos_rate:<9.3f} | {dens_kept_pct:<10.2f}"
        )

    print("-" * 110)

    # Scelta soglia robusta (FP basso + vincoli)
    chosen, status = choose_threshold(
        stats,
        thresholds,
        target_recall=args.target_recall,
        max_pos_rate=args.max_pos_rate,
    )

    print(status)
    print(
        f"[*] Scelto th={chosen['th']:.2f} | FP={chosen['fp']} | FPR={chosen['fpr']:.4f} | "
        f"R={chosen['rec']:.4f} | P={chosen['prec']:.4f} | F1={chosen['f1']:.4f} | pos_rate={chosen['pos_rate']:.3f}"
    )
    print("=" * 110 + "\n")

    # Dataset + backbone intestazione
    dataset_name = getattr(args, "dataset", "unknown")

    backbone_name = args.backbone
    if backbone_name is None:
        backbone_name = (
            config.get("backbone")
            or config.get("model")
            or config.get("encoder")
            or config.get("zip_backbone")
            or getattr(args, "model", None)
            or "unknown_backbone"
        )

    # Output path confusion matrix
    if args.cm_out is None:
        args.cm_out = f"confusion_matrix_{dataset_name}_{backbone_name}.png".replace("/", "_")

    # Plot confusion matrix con soglia scelta
    print(f"[*] Saving confusion matrix -> {args.cm_out}")
    plot_confusion_matrix_binary(
        tn=chosen["tn"],
        fp=chosen["fp"],
        fn=chosen["fn"],
        tp=chosen["tp"],
        dataset_name=dataset_name,
        backbone_name=backbone_name,
        out_path=args.cm_out,
        th=chosen["th"],
    )

    print(f"    TN={chosen['tn']}  FP={chosen['fp']}  FN={chosen['fn']}  TP={chosen['tp']}")


if __name__ == "__main__":
    main()
