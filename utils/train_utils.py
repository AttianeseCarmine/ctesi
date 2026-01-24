import torch
from torch import nn, Tensor

from torch.optim import Adam
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR

from functools import partial
from argparse import ArgumentParser

import os, sys, math
from typing import Union, Tuple, Dict, List
from collections import OrderedDict

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import losses


def cosine_annealing_warm_restarts(
    epoch: int,
    base_lr: float,
    warmup_epochs: int,
    warmup_lr: float,
    T_0: int,
    T_mult: int,
    eta_min: float,
) -> float:
    """
    Learning rate scheduler.
    The learning rate will linearly increase from warmup_lr to lr in the first warmup_epochs epochs.
    Then, the learning rate will follow the cosine annealing with warm restarts strategy.
    """
    assert epoch >= 0, f"epoch must be non-negative, got {epoch}."
    assert isinstance(warmup_epochs, int) and warmup_epochs >= 0, f"warmup_epochs must be non-negative, got {warmup_epochs}."
    assert isinstance(warmup_lr, float) and warmup_lr > 0, f"warmup_lr must be positive, got {warmup_lr}."
    assert isinstance(T_0, int) and T_0 >= 1, f"T_0 must be greater than or equal to 1, got {T_0}."
    assert isinstance(T_mult, int) and T_mult >= 1, f"T_mult must be greater than or equal to 1, got {T_mult}."
    assert isinstance(eta_min, float) and eta_min > 0, f"eta_min must be positive, got {eta_min}."
    assert isinstance(base_lr, float) and base_lr > 0, f"base_lr must be positive, got {base_lr}."
    assert base_lr > eta_min, f"base_lr must be greater than eta_min, got base_lr={base_lr} and eta_min={eta_min}."
    assert warmup_lr >= eta_min, f"warmup_lr must be greater than or equal to eta_min, got warmup_lr={warmup_lr} and eta_min={eta_min}."

    if epoch < warmup_epochs:
        lr = warmup_lr + (base_lr - warmup_lr) * epoch / warmup_epochs
    else:
        epoch -= warmup_epochs
        if T_mult == 1:
            T_cur = epoch % T_0
            T_i = T_0
        else:
            n = int(math.log((epoch / T_0 * (T_mult - 1) + 1), T_mult))
            T_cur = epoch - T_0 * (T_mult ** n - 1) / (T_mult - 1)
            T_i = T_0 * T_mult ** (n)
        
        lr = eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2

    return lr / base_lr


def get_loss_fn(args: ArgumentParser) -> nn.Module:
    if args.bins is None:
        assert args.weight_ot is not None and args.weight_tv is not None, f"Expected weight_ot and weight_tv to be not None, got {args.weight_ot} and {args.weight_tv}"
        loss_fn = losses.DMLoss(
            input_size=args.input_size,
            reduction=args.reduction,
        )
    else:
        loss_fn = losses.DACELoss(
            bins=args.bins,
            reduction=args.reduction,
            weight_count_loss=args.weight_count_loss,
            count_loss=args.count_loss,
            input_size=args.input_size,
        )
    return loss_fn


def get_optimizer(args: ArgumentParser, model: nn.Module) -> Tuple[Adam, LambdaLR]:
    optimizer = Adam(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=partial(
            cosine_annealing_warm_restarts,
            warmup_epochs=args.warmup_epochs,
            warmup_lr=args.warmup_lr,
            T_0=args.T_0,
            T_mult=args.T_mult,
            eta_min=args.eta_min,
            base_lr=args.lr
        ),
    )

    return optimizer, scheduler


# In utils/train_utils.py
def load_checkpoint(
    args,
    model,
    optimizer,
    scheduler,
    grad_scaler,
):
    # --- Gestione Path Checkpoint ---
    if hasattr(args, "resume") and args.resume is not None:
        ckpt_path = args.resume
        print(f"🔄 Resuming from specific checkpoint: {ckpt_path}")
    else:
        ckpt_path = os.path.join(args.ckpt_dir, "last_model.pth")

    # Inizializza valori di default (nel caso non si trovi il file o sia solo pesi)
    start_epoch = 1
    loss_info = None
    hist_scores = {"mae": [], "rmse": []}
    # Calcola k per i best scores
    k = args.save_best_k if hasattr(args, "save_best_k") else 3
    best_scores = {key: [float('inf')] * k for key in hist_scores.keys()}

    if os.path.exists(ckpt_path):
        print(f"📂 Loading checkpoint from {ckpt_path}...")
        # Usa weights_only=False per compatibilità, ma attenzione alla sicurezza se scarichi file da internet
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        # --- CASO 1: Checkpoint Completo (Resume) ---
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            
            # Carichiamo anche lo stato dell'addestramento
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            
            if "loss_info" in ckpt: loss_info = ckpt["loss_info"]
            if "hist_scores" in ckpt: hist_scores = ckpt["hist_scores"]
            if "best_scores" in ckpt: best_scores = ckpt["best_scores"]

            if scheduler is not None and "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if grad_scaler is not None and "grad_scaler_state_dict" in ckpt:
                grad_scaler.load_state_dict(ckpt["grad_scaler_state_dict"])
            
            print(f"✅ Full Training State Restored. Resuming from Epoch {start_epoch}.")

        # --- CASO 2: Solo Pesi (Fine-tuning o Pre-trained) ---
        else:
            # Se il file è direttamente il dizionario dei pesi
            # Rimuoviamo il prefisso "module." se presente (caso salvataggio DDP)
            state_dict = ckpt
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Carica solo i pesi, ignora epoch e optimizer (si riparte da zero)
            msg = model.load_state_dict(state_dict, strict=False)
            print(f"⚠️ Loaded ONLY Model Weights (No optimizer/epoch info found).")
            print(f"   Missing keys: {msg.missing_keys}")
            print(f"   Unexpected keys: {msg.unexpected_keys}")
            print(f"🚀 Starting training from Epoch 1 (Fine-tuning mode).")

    else:
        # File non trovato
        if hasattr(args, "resume") and args.resume is not None:
             print(f"⚠️ WARNING: Checkpoint '{args.resume}' not found!")
        print(f"🚀 No checkpoint found at {ckpt_path}, starting training from scratch.")

    return model, optimizer, scheduler, grad_scaler, start_epoch, loss_info, hist_scores, best_scores


def save_checkpoint(
    epoch: int,
    model_state_dict: OrderedDict[str, Tensor],
    optimizer_state_dict: OrderedDict[str, Tensor],
    scheduler_state_dict: OrderedDict[str, Tensor],
    grad_scaler_state_dict: OrderedDict[str, Tensor],
    loss_info: Dict[str, List[float]],
    hist_scores: Dict[str, List[float]],
    best_scores: Dict[str, float],
    ckpt_dir: str,
) -> None:
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "grad_scaler_state_dict": grad_scaler_state_dict,
        "loss_info": loss_info,
        "hist_scores": hist_scores,
        "best_scores": best_scores,
    }
    torch.save(ckpt, os.path.join(ckpt_dir, "ckpt.pth"))

def seed_everything(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False