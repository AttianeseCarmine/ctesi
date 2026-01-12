#!/usr/bin/env python3
"""
Evaluate Stage 3: Joint Model Evaluation + per-image logging
============================================================
Stampa per ogni immagine:
- conteggio predetto
- conteggio GT
- errore assoluto
- errore percentuale

Supporta sliding window per immagini grandi.
"""

import os
import yaml
import argparse
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.zip_model import ZIPModel
from models.clip_ebc_model import CLIPEBCModel
from models.joint_model import ZIPCLIPJointModel
from datasets.sha import SHA  # per SHA/SHB nel tuo repo
from datasets.transforms import build_transforms


@torch.no_grad()
def sliding_window_predict_count(model, img_cpu, window_size=448, stride=448, device="cuda", amp=False):
    """
    Sliding window che restituisce direttamente il conteggio.
    Importante: usa divisor_map se stride < window_size (overlap).
    """
    model.eval()
    img = img_cpu.to(device)
    B, C, H, W = img.shape
    assert B == 1, "sliding_window_predict_count supporta batch=1"

    count_map = torch.zeros((H, W), device=device)
    divisor_map = torch.zeros((H, W), device=device)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_end = min(y + window_size, H)
            x_end = min(x + window_size, W)
            y_start = max(y_end - window_size, 0)
            x_start = max(x_end - window_size, 0)

            crop = img[:, :, y_start:y_end, x_start:x_end]

            out = model(crop)
            # densità finale (gated)
            pred_density = out["final_density"]  # [1,1,h,w]

            # riallinea a crop size per accumulare correttamente
            pred_density = F.interpolate(
                pred_density,
                size=(crop.shape[2], crop.shape[3]),
                mode="bilinear",
                align_corners=False
            )

            count_map[y_start:y_end, x_start:x_end] += pred_density.squeeze(0).squeeze(0)
            divisor_map[y_start:y_end, x_start:x_end] += 1.0

    final_density = count_map / torch.clamp(divisor_map, min=1.0)
    return final_density.sum().item()


def evaluate(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(config.get("TRAIN_STAGE3", {}).get("AMP", False)) and (device.type == "cuda")

    print(f"🔧 Device: {device} | Dataset: {config.get('DATASET', 'N/A')} | AMP={amp}", flush=True)

    # 1) Build joint model
    stage1 = ZIPModel(config).to(device)
    stage2 = CLIPEBCModel(config).to(device)

    steep = float(config.get("TRAIN_STAGE3", {}).get("STEEPNESS", 20.0))
    model = ZIPCLIPJointModel(stage1, stage2, steepness=steep).to(device)

    # 2) Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"⚠️ Missing keys: {len(missing)} (spesso ok)", flush=True)
    if unexpected:
        print(f"⚠️ Unexpected keys: {len(unexpected)}", flush=True)

    model.eval()

    # 3) Dataset/loader
    dataset_name = str(config.get("DATASET", "sha")).lower()
    root_dir = config["DATA"]["ROOT"]
    split = args.split

    val_transforms = build_transforms(config["DATA"], is_train=False)

    if "sha" in dataset_name or "shb" in dataset_name:
        dataset = SHA(root_dir, split, val_transforms)
    else:
        raise ValueError(f"Dataset {dataset_name} non supportato da questo script (per ora).")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # sliding window params
    crop_size = int(config["DATA"].get("CROP_SIZE", 448))
    window = int(args.window or crop_size)
    stride = int(args.stride or window)

    mae_accum = 0.0
    mse_accum = 0.0

    print(f"🚀 Start eval | split={split} | steepness={steep} | window={window} stride={stride}", flush=True)

    pbar = tqdm(loader)
    for idx, batch in enumerate(pbar):
        img = batch["image"]          # CPU tensor
        gt_count = len(batch["points"][0])

        # pred count: sliding window se molto grande
        if img.shape[2] > args.max_side or img.shape[3] > args.max_side:
            pred_count = sliding_window_predict_count(
                model, img, window_size=window, stride=stride, device=device, amp=amp
            )
        else:
            img_dev = img.to(device)
            out = model(img_dev)
            pred_count = out["final_density"].sum().item()

        err = abs(pred_count - gt_count)
        mae_accum += err
        mse_accum += err ** 2

        pct = (err / max(1, gt_count)) * 100.0  # se gt=0 evita divisione per 0 (interpreta come % su 1)

        # nome immagine se disponibile
        img_name = None
        if isinstance(batch, dict):
            img_name = batch.get("img_path", None)
        if isinstance(img_name, list) and len(img_name) > 0:
            img_name = img_name[0]

        # stampa dettagli (evita spam)
        if args.print_every > 0 and (idx % args.print_every == 0):
            prefix = f"[{idx:05d}]"
            if img_name:
                prefix += f" {os.path.basename(str(img_name))}"
            print(
                f"{prefix}  pred={pred_count:.2f}  gt={gt_count}  "
                f"abs_err={err:.2f}  pct_err={pct:.1f}%",
                flush=True
            )

        pbar.set_postfix({"MAE": f"{mae_accum/(idx+1):.2f}", "RMSE": f"{math.sqrt(mse_accum/(idx+1)):.2f}"})

    final_mae = mae_accum / max(1, len(dataset))
    final_rmse = math.sqrt(mse_accum / max(1, len(dataset)))

    print("\n" + "=" * 40)
    print(f"🏆 FINAL RESULTS: {str(config.get('DATASET','')).upper()} | split={split}")
    print(f"   MAE:  {final_mae:.2f}")
    print(f"   RMSE: {final_rmse:.2f}")
    print("=" * 40 + "\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_shb.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to stage3 best_model/last_model.pth")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_side", type=int, default=1024, help="Se H o W supera questo, usa sliding window")
    parser.add_argument("--window", type=int, default=None, help="override window size (default = CROP_SIZE)")
    parser.add_argument("--stride", type=int, default=None, help="override stride (default = window)")
    parser.add_argument("--print_every", type=int, default=1, help="stampa una riga ogni N immagini (1=sempre, 0=mai)")
    args = parser.parse_args()

    evaluate(args)
