import argparse
import yaml
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import necessari dal tuo progetto
from models.zip_model import ZIPModel
from datasets.builder import build_dataset
from datasets.transforms import build_transforms


def crowd_collate(batch):
    """Gestisce batch con immagini di dimensioni diverse (se necessario)"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "density": torch.stack([item["density"] for item in batch]),
        "img_path": [item["img_path"] for item in batch],
    }


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.2):
    """
    Valuta il modello Stage 1 (Binary Segmentation).
    Logica allineata al 100% con train_stage1.py.
    """
    model.eval()

    tp, tn, fp, fn = 0, 0, 0, 0

    for batch in tqdm(loader, desc=f"Calculating Metrics (thr={threshold})"):
        if batch is None:
            continue

        images = batch["image"].to(device)
        gt_density = batch["density"].to(device)

        # 1. Forward
        outputs = model(images)
        pi_logits = outputs["pi_logits"]  # [B, 1, H_out, W_out]
        probs = torch.sigmoid(pi_logits)

        # 2. Prepara Ground Truth Binaria (Allineamento dimensioni)
        h_out, w_out = pi_logits.shape[2:]

        # Scaling factor per mantenere la somma della densità corretta dopo il pooling
        scale_factor = (images.shape[2] * images.shape[3]) / (h_out * w_out)

        # Downsample della densità GT alla risoluzione dell'output del modello
        gt_down = F.adaptive_avg_pool2d(gt_density, (h_out, w_out)) * scale_factor

        # Target binario:
        gt_binary = (gt_down > 0.001).float()

        # 3. Predizione binaria
        pred_binary = (probs > threshold).float()

        # 4. Aggiorna statistiche
        tp += ((pred_binary == 1) & (gt_binary == 1)).sum().item()
        tn += ((pred_binary == 0) & (gt_binary == 0)).sum().item()
        fp += ((pred_binary == 1) & (gt_binary == 0)).sum().item()
        fn += ((pred_binary == 0) & (gt_binary == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "threshold": threshold,
        "f1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def print_results_table(results):
    # header
    print("\n" + "=" * 92)
    print("📊 THRESHOLD SWEEP RESULTS (Stage 1)")
    print("=" * 92)
    print(f"{'thr':>5} | {'F1':>8} | {'Acc':>8} | {'Prec':>8} | {'Rec':>8} | {'TP':>9} | {'FP':>9} | {'FN':>9}")
    print("-" * 92)

    # rows
    for r in results:
        print(
            f"{r['threshold']:>5.2f} | "
            f"{100*r['f1']:>7.2f}% | "
            f"{100*r['accuracy']:>7.2f}% | "
            f"{100*r['precision']:>7.2f}% | "
            f"{100*r['recall']:>7.2f}% | "
            f"{r['TP']:>9} | "
            f"{r['FP']:>9} | "
            f"{r['FN']:>9}"
        )
    print("=" * 92)

    # best by precision
    best_prec = max(results, key=lambda x: x["precision"])
    best_f1 = max(results, key=lambda x: x["f1"])

    print("\n🏁 Best by Precision:")
    print(f"   thr={best_prec['threshold']:.2f} | Prec={best_prec['precision']:.2%} | Rec={best_prec['recall']:.2%} | F1={best_prec['f1']:.2%}")

    print("\n🏁 Best by F1:")
    print(f"   thr={best_f1['threshold']:.2f} | F1={best_f1['f1']:.2%} | Prec={best_f1['precision']:.2%} | Rec={best_f1['recall']:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 Model (ZIP) with threshold sweep")
    parser.add_argument("--config", type=str, required=True, help="Path to config file (yaml)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pth model")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Num dataloader workers")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        help="List of thresholds to evaluate",
    )

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")

    # 1. Config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"🏗️  Building Dataset & Model... ({config['BACKBONE']['TYPE']})")

    # 2. Dataset (val)
    val_transforms = build_transforms(config["DATA"], is_train=False)
    val_dataset = build_dataset(config, "val", val_transforms)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=crowd_collate,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Model + checkpoint
    model = ZIPModel(config).to(device)

    if not os.path.isfile(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return

    print(f"📥 Loading weights from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"📦 Checkpoint load -> Missing: {len(missing)} | Unexpected: {len(unexpected)}")
    if len(missing) > 0:
        print("   Missing sample:", missing[:20])
    if len(unexpected) > 0:
        print("   Unexpected sample:", unexpected[:20])

    # 4. Sweep thresholds
    print(f"🚀 Starting Evaluation on {len(val_dataset)} images...")
    results = []
    for thr in args.thresholds:
        metrics = evaluate(model, val_loader, device, threshold=thr)
        results.append(metrics)

    # 5. Print table
    print("\n" + "=" * 40)
    print(f"📊 EVALUATION RESULTS ({config['BACKBONE']['TYPE']})")
    print("=" * 40)
    print_results_table(results)


if __name__ == "__main__":
    main()
