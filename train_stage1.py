import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import os
import yaml

# --- IMPORT DAI TUOI FILE ---
from models.zip_clip_ebc_model import build_model
from losses import build_stage1_loss
from train_utils import init_seeds, get_optimizer, get_scheduler, resume_if_exists, save_checkpoint, collate_fn
from datasets import get_dataset
from datasets.transforms import build_transforms

def load_config(config_path: str):
    with open(config_path, "r") as handle:
        return yaml.safe_load(handle)

def build_count_sampler(dataset, sampler_cfg):
    """
    Costruisce un sampler che pesca più spesso le immagini con folla.
    Preso dal progetto di riferimento (funziona molto bene per ZIP).
    """
    if not sampler_cfg or not sampler_cfg.get("ENABLE", False):
        return None
    
    # Verifica che il dataset supporti l'accesso ai punti
    if not hasattr(dataset, "image_list") or not hasattr(dataset, "load_points"):
        print("ℹ️ Sampler ZIP disattivato: il dataset non supporta image_list/load_points.")
        return None

    print("⏳ Costruzione Count Sampler (lettura annotazioni)...")
    counts = []
    for img_path in tqdm(dataset.image_list, desc="Scanning dataset"):
        try:
            pts = dataset.load_points(img_path)
            counts.append(len(pts))
        except Exception as exc:
            counts.append(0)

    if not counts:
        return None

    counts_arr = np.array(counts, dtype=np.float64)
    log_offset = max(1e-3, float(sampler_cfg.get("LOG_OFFSET", 1.0)))
    base_weight = max(1e-6, float(sampler_cfg.get("BASE_WEIGHT", 1.0)))
    power = float(sampler_cfg.get("POWER", 1.0))

    # Calcola pesi: log(count + 1) -> bilancia senza sovrappesare troppo le folle enormi
    scaled = np.log1p(counts_arr + log_offset)
    if power != 1.0:
        scaled = np.power(scaled, power)
    weights = base_weight * scaled
    weights = np.clip(weights, 1e-6, None)

    torch_weights = torch.as_tensor(weights, dtype=torch.double)
    sampler = WeightedRandomSampler(torch_weights, num_samples=len(dataset), replacement=True)
    
    print(
        "ℹ️ Count-aware sampler attivo: avg_w={:.3f}, max_count={:.1f}".format(
            float(weights.mean()), float(counts_arr.max())
        )
    )
    return sampler

def train_one_epoch(
    model,
    criterion,
    dataloader,
    optimizer,
    scheduler,
    device,
    clip_grad_norm: float = 1.0,
):
    model.train()
    total_loss = 0.0
    
    # Progress bar
    progress_bar = tqdm(dataloader, desc="Train Stage 1 (Pi-Head)")

    for batch in progress_bar:
        # Gestione flessibile del batch (dict o tupla)
        if isinstance(batch, dict):
            images = batch['image']
            gt_density = batch['density']
        else:
            images, gt_density = batch[0], batch[1]

        images = images.to(device)
        gt_density = gt_density.to(device)

        optimizer.zero_grad()
        
        # Forward pass (Stage 1 usa solo pi-head idealmente, ma il modello gestisce tutto)
        # Usiamo forward completo o forward_pi_only? 
        # Per semplicità usiamo forward completo, la loss ignorerà l'output EBC non necessario
        # MA per efficienza, se hai implementato forward_pi_only, usiamolo:
        predictions = model.forward_pi_only(images)
        
        # Calcolo Loss (La tua PiHeadLoss gestisce internamente BCE + Regolarizzazione)
        loss, loss_dict = criterion(predictions, gt_density)

        loss.backward()
        
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        
        optimizer.step()

        total_loss += loss.item()

        # Logging nella progress bar
        postfix = {
            'loss': f"{loss.item():.4f}",
            'bce': f"{loss_dict.get('pi_bce_loss', 0):.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        }
        # Aggiungi info reg se presente
        if 'pi_reg_loss' in loss_dict:
            postfix['reg'] = f"{loss_dict['pi_reg_loss']:.4f}"
            
        progress_bar.set_postfix(postfix)

    if scheduler:
        scheduler.step()
        
    return total_loss / len(dataloader)

def validate(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0.0
    
    # Metriche
    mae = 0.0
    mse = 0.0
    active_blocks_mean = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validate Stage 1"):
            if isinstance(batch, dict):
                images = batch['image']
                gt_density = batch['density']
            else:
                images, gt_density = batch[0], batch[1]
                
            images = images.to(device)
            gt_density = gt_density.to(device)

            # MODIFICA: Usiamo il modello completo per avere anche il conteggio (pred_count)
            # Anche se EBC è congelato, ci darà una stima basata su CLIP
            preds = model(images)
            
            # Loss (La PiHeadLoss userà solo le chiavi che le servono)
            loss, _ = criterion(preds, gt_density)
            total_loss += loss.item()

            # Calcolo MAE e RMSE
            pred_count = preds['pred_count'] # [B]
            gt_count = gt_density.sum(dim=[1, 2, 3]) # [B]
            
            mae += torch.abs(pred_count - gt_count).sum().item()
            mse += ((pred_count - gt_count) ** 2).sum().item()

            # Monitoraggio attività Pi
            active_blocks_mean += preds["pi_prob"].mean().item()

    # Medie finali
    N = len(dataloader.dataset)
    avg_loss = total_loss / len(dataloader)
    avg_active = active_blocks_mean / len(dataloader)
    
    avg_mae = mae / N
    avg_rmse = (mse / N) ** 0.5
    
    return avg_loss, avg_active, avg_mae, avg_rmse

def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 1 (ZIP-CLIP-EBC)")
    parser.add_argument("--config", default="config_sha.yaml", help="Path to config file.")
    return parser.parse_args()

def main(config_path: str):
    # 1. Carica Config
    config = load_config(config_path)
    device = torch.device(config["DEVICE"])
    init_seeds(config["SEED"])
    
    print(f"🚀 Avvio Stage 1: Training Pi-Head (Zero-Inflation)")
    print(f"   Config: {config_path}")

    # 2. Costruisci Modello (Il tuo ZIP-CLIP-EBC)
    model = build_model(config).to(device)
    
    # 3. Congelamento (Freezing Strategy per Stage 1)
    # Vogliamo allenare SOLO la Pi-Head e un po' il backbone
    # L'EBC Head deve stare ferma (non ha senso allenarla su blocchi vuoti/pieni)
    model.freeze_ebc_head()
    
    # Gestione backbone: leggi config
    train_stage1_cfg = config.get("TRAIN_STAGE1", {})
    lr_backbone = train_stage1_cfg.get("LR_BACKBONE", 0.0)
    
    if lr_backbone > 0:
        print("🔓 Backbone SCONGELATO (Fine-tuning)")
        model.unfreeze_backbone()
    else:
        print("🧊 Backbone CONGELATO")
        model.freeze_backbone()
        
    model.unfreeze_pi_head() # Assicuriamoci che questa sia attiva

    # 4. Loss Function
    # Usiamo la factory function dal tuo losses.py
    criterion = build_stage1_loss(config).to(device)

    # 5. Optimizer e Scheduler
    # Usiamo il metodo del tuo modello per raggruppare i parametri correttamente
    lr_pi = train_stage1_cfg.get("LR_PI_HEAD", 1e-4)
    
    param_groups = model.get_param_groups(
        lr_backbone=lr_backbone,
        lr_pi_head=lr_pi,
        lr_ebc_head=0.0 # EBC è congelato
    )
    
    # Rimuovi gruppi vuoti (es. se ebc_head ha lr 0 o params vuoti)
    param_groups = [g for g in param_groups if g['params']]
    
    optimizer = get_optimizer(param_groups, train_stage1_cfg)
    scheduler = get_scheduler(optimizer, train_stage1_cfg, max_epochs=train_stage1_cfg.get("EPOCHS", 100))

    # 6. Dataset e Dataloader
    data_cfg = config["DATA"]
    
    # Trasformazioni (Stage 1 usa crop più grandi solitamente)
    train_tf = build_transforms(
        data_cfg, 
        is_train=True, 
        override_crop_size=data_cfg.get("CROP_SIZE_STAGE1"),
        override_crop_scale=data_cfg.get("CROP_SCALE_STAGE1")
    )
    val_tf = build_transforms(data_cfg, is_train=False)
    
    DatasetClass = get_dataset(config["DATASET"])
    
    train_set = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["TRAIN_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=train_tf,
    )
    val_set = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=val_tf,
    )

    # Sampler per bilanciare immagini vuote/piene
    sampler_cfg = data_cfg.get("COUNT_SAMPLER", {})
    train_sampler = build_count_sampler(train_set, sampler_cfg)

    train_loader = DataLoader(
        train_set,
        batch_size=train_stage1_cfg["BATCH_SIZE"],
        shuffle=(train_sampler is None), # Shuffle solo se non usiamo il sampler
        sampler=train_sampler,
        num_workers=train_stage1_cfg["NUM_WORKERS"],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=1, # Val sempre batch 1 per coerenza
        shuffle=False,
        num_workers=train_stage1_cfg["NUM_WORKERS"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 7. Setup Esperimento
    out_dir = os.path.join(config["EXP"]["OUT_DIR"], config["RUN_NAME"], "stage1")
    os.makedirs(out_dir, exist_ok=True)
    
    # Nomi file personalizzati
    best_ckpt_name = "best_stage1_model.pth"
    last_ckpt_name = "last_stage1_model.pth"
    
    # Resume automatico
    start_epoch = 1
    best_loss = float("inf")
    if train_stage1_cfg.get("RESUME_LAST", True):
        # Passiamo il nome del file 'last' personalizzato
        s_ep, b_loss = resume_if_exists(model, optimizer, out_dir, device, filename=last_ckpt_name)
        if s_ep > 1:
            start_epoch = s_ep
            best_loss = b_loss

    # 8. Training Loop
    max_epochs = train_stage1_cfg.get("EPOCHS", 100)
    val_interval = train_stage1_cfg.get("VAL_INTERVAL", 5)
    
    for epoch in range(start_epoch, max_epochs + 1):
        print(f"\n--- Epoch {epoch}/{max_epochs} ---")
        
        train_loss = train_one_epoch(
            model, criterion, train_loader, optimizer, scheduler, device,
            clip_grad_norm=train_stage1_cfg.get("CLIP_GRAD_NORM", 1.0)
        )
        
        if epoch % val_interval == 0 or epoch == max_epochs:
            val_loss, avg_pi, val_mae, val_rmse = validate(model, criterion, val_loader, device)
            print(f"🔍 Val: Loss={val_loss:.4f} | Avg Pi={avg_pi:.3f} | MAE={val_mae:.2f} | RMSE={val_rmse:.2f}")
            
            # Save Best
            if val_mae < best_loss:
                best_loss = val_mae
                save_checkpoint(
                    model, optimizer, epoch, val_mae, best_loss, out_dir, 
                    is_best=True,
                    best_name=best_ckpt_name,
                    last_name=last_ckpt_name
                )
                print(f"⭐ New Best Model Saved (MAE={val_mae:.2f})!")
            
            # Save Last (sempre)
            save_checkpoint(
                model, optimizer, epoch, val_mae, best_loss, out_dir, 
                is_best=False,
                best_name=best_ckpt_name,
                last_name=last_ckpt_name
            )

    print("✅ Stage 1 Completato!")

if __name__ == "__main__":
    args = parse_args()
    main(args.config)