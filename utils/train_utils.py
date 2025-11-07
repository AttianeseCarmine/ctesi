# utils/train_utils.py
from argparse import ArgumentParser
import torch
from torch import nn, Tensor
from torch.optim import SGD, Adam, AdamW, RAdam
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, MultiStepLR

from functools import partial
import os, sys, math
from typing import Union, Tuple, Dict, List, Optional
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel as DDP

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)


# === FUNZIONI HELPER PER SCHEDULER (COPIATE) ===
def _check_lr(lr: float, eta_min: float) -> None:
    assert lr > eta_min > 0, f"lr and eta_min must satisfy 0 < eta_min < lr, got lr={lr} and eta_min={eta_min}."

def _check_warmup(warmup_epochs: int, warmup_lr: float) -> None:
    assert warmup_epochs >= 0, f"warmup_epochs must be non-negative, got {warmup_epochs}."
    assert warmup_lr > 0, f"warmup_lr must be positive, got {warmup_lr}."

def _warmup_lr(
    epoch: int,
    base_lr: float,
    warmup_epochs: int,
    warmup_lr: float,
) -> float:
    base_lr, warmup_lr = float(base_lr), float(warmup_lr)
    assert epoch >= 0, f"epoch must be non-negative, got {epoch}."
    _check_warmup(warmup_epochs, warmup_lr)
    if epoch < warmup_epochs:
        return (base_lr - warmup_lr) * epoch / warmup_epochs + warmup_lr
    return base_lr

def _cosine_lr(
    epoch: int,
    base_lr: float,
    eta_min: float,
    num_epochs: int,
) -> float:
    base_lr, eta_min = float(base_lr), float(eta_min)
    _check_lr(base_lr, eta_min)
    if epoch < 0:
        return base_lr
    return eta_min + 0.5 * (base_lr - eta_min) * (1.0 + math.cos(math.pi * epoch / num_epochs))

def _multistep_lr(
    epoch: int,
    base_lr: float,
    milestones: List[int],
    gamma: float,
) -> float:
    base_lr = float(base_lr)
    assert all(milestones[i] < milestones[i + 1] for i in range(len(milestones) - 1)), "milestones must be a list of increasing integers."
    assert gamma > 0, f"gamma must be positive, got {gamma}."
    if epoch < 0:
        return base_lr
    power = 0
    for m in milestones:
        if epoch >= m:
            power += 1
    return base_lr * (gamma ** power)


# === FUNZIONE PRINCIPALE (MODIFICATA) ===
def get_optimizer_and_scheduler(
    params: List[nn.Parameter],
    cfg: Dict, # <--- MODIFICA: Accetta un dizionario (lo stage_cfg)
    len_loader: int, # <--- Aggiunto len_loader (anche se non usato, per compatibilità)
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    
    # Legge i parametri dal dizionario cfg (es. cfg['train_stage1'])
    optimizer_name = cfg.get("optimizer", "adamw").lower()
    lr = float(cfg.get("lr", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    
    optimizer: torch.optim.Optimizer
    if optimizer_name == "sgd":
        momentum = cfg.get("momentum", 0.9)
        optimizer = SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "radam":
        optimizer = RAdam(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported.")

    # Scheduler
    scheduler_name = cfg.get("scheduler", "cosine").lower()
    num_epochs = int(cfg.get("num_epochs", 100))
    eta_min = float(cfg.get("eta_min", 0.0))
    warmup_epochs = int(cfg.get("warmup_epochs", 0))
    warmup_lr = float(cfg.get("warmup_lr", 1e-6))
    
    scheduler: torch.optim.lr_scheduler._LRScheduler
    if scheduler_name == "none":
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0) # Dummy scheduler
        
    elif scheduler_name == "cosine":
        if warmup_epochs > 0:
            lr_lambda = partial(
                _warmup_lr,
                base_lr=lr,
                warmup_epochs=warmup_epochs,
                warmup_lr=warmup_lr,
            )
            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)
            
    elif scheduler_name == "multistep":
        milestones = cfg.get("milestones", [int(num_epochs * 0.6), int(num_epochs * 0.8)])
        gamma = cfg.get("gamma", 0.1)
        if warmup_epochs > 0:
            lr_lambda = partial(
                _warmup_lr,
                base_lr=lr,
                warmup_epochs=warmup_epochs,
                warmup_lr=warmup_lr,
            )
            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
            
    else:
        raise ValueError(f"Scheduler {scheduler_name} not supported.")

    return optimizer, scheduler

# === ALTRE UTILS (MANTENUTE) ===

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# (Mantieni qui le tue vecchie funzioni 'load_checkpoint' e 'save_checkpoint'
#  se servono ad altri script come 'evaluate.py', anche se 'train.py' non le usa più)

def load_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    grad_scaler: Optional[GradScaler] = None,
    args: Optional[ArgumentParser] = None,
) -> Tuple[
    nn.Module,
    Optional[torch.optim.Optimizer],
    Optional[torch.optim.lr_scheduler._LRScheduler],
    Optional[GradScaler],
    int,
    Optional[Dict[str, List[float]]],
    Dict[str, List[float]],
    Dict[str, float],
]:
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        
        # Gestisce checkpoint salvati da Trainer e checkpoint standard
        if "model" in ckpt:
            model_state_dict = ckpt["model"]
        elif "model_state_dict" in ckpt:
            model_state_dict = ckpt["model_state_dict"]
        else:
            model_state_dict = ckpt

        # Carica il modello
        if isinstance(model, DDP):
            model.module.load_state_dict(model_state_dict)
        else:
            model.load_state_dict(model_state_dict)

        # Carica componenti opzionali
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        elif optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        elif scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            
        if grad_scaler is not None and "grad_scaler" in ckpt:
            grad_scaler.load_state_dict(ckpt["grad_scaler"])
        elif grad_scaler is not None and "grad_scaler_state_dict" in ckpt:
            grad_scaler.load_state_dict(ckpt["grad_scaler_state_dict"])

        start_epoch = ckpt.get("epoch", 1) + 1
        loss_info = ckpt.get("loss_info", None)
        hist_scores = ckpt.get("hist_scores", {"mae": [], "rmse": [], "nae": []})
        best_scores = ckpt.get("best_scores", {k: [torch.inf] * args.save_best_k for k in hist_scores.keys()})

        print(f"Loaded checkpoint from {ckpt_path}.")

    else:
        start_epoch = 1
        loss_info, hist_scores = None, {"mae": [], "rmse": [], "nae": []}
        if args is not None:
             best_scores = {k: [torch.inf] * args.save_best_k for k in hist_scores.keys()}
        else:
             best_scores = {k: [torch.inf] for k in hist_scores.keys()}
        print(f"Checkpoint not found at {ckpt_path}.")

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
    torch.save(ckpt, os.path.join(ckpt_dir, "last.pth"))