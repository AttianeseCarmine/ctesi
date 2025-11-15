# train.py (Aggiornato per il modello ibrido ConvZIP + CLIP-EBC)

import os
import torch
import torch.nn as nn
import argparse
import yaml
from torch.utils.data import DataLoader
from typing import Optional, Dict
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from utils import set_seed
from datasets import build_dataloader
from losses import build_loss
from trainer import Trainer 

# Importa il modello "contenitore" ibrido
# (Questo nome è corretto, anche se ora carica teste diverse)
from models.zip_clip_ebc_model import ZIP_CLIP_EBC_Model 


def main(args, cfg: Dict): 
    # Carica la configurazione base (condivisa)
    train_cfg_base = cfg['train_base']
    
    # Carica la configurazione specifica per lo stadio corrente
    train_cfg_stage = cfg[f'train_stage{args.stage}']
    
    # Unisci la config base con quella dello stadio
    train_cfg = {**train_cfg_base, **train_cfg_stage}
    
    # Ora usa 'train_cfg' per tutti i parametri
    set_seed(train_cfg['seed']) 
    device = torch.device(train_cfg['device'] if torch.cuda.is_available() else "cpu") 

    print(f"🚀 AVVIO DELLO STADIO {args.stage} 🚀")

    # --- 1. Definizione Percorsi Checkpoint ---
    output_dir = cfg['train_base']['output_dir']
    stage1_ckpt_path = os.path.join(output_dir, "stage1_best.pth")
    stage2_ckpt_path = os.path.join(output_dir, "stage2_best.pth")
    final_ckpt_path = os.path.join(output_dir, "best_mae.pth") 

    load_path = args.load_ckpt
    save_path = final_ckpt_path 

    # ==================================================================
    # === INIZIO MODIFICA (La tua unica modifica richiesta) ===
    # ==================================================================
    
    # --- 2. Creazione Modello ---
    print("Costruzione del modello IBRIDO (ConvZIP + CLIP-EBC)...")
    model = ZIP_CLIP_EBC_Model(
        # Parametri Backbone
        model_name=cfg['model']['name'],
        weight_name=cfg['model']['weight_name'],
        block_size=cfg['model'].get('block_size'),
        input_size=cfg['model'].get('input_size'), 
        
        # Parametri per EBC (ClipEBHead)
        ebc_bins=cfg['model']['ebc_bins'],
        ebc_bin_centers=cfg['model']['ebc_bin_centers'],
        text_prompts=cfg['model']['text_prompts'],
        
        # Parametri per ZIP (ConvZIPHead)
        zip_bins=cfg['model']['zip_bins'],
        zip_bin_centers=cfg['model']['zip_bin_centers'],
        zip_lambda_max=cfg['model'].get('zip_lambda_max', 8.0),
        
        # Parametri di Gating
        pi_thresh=cfg['model'].get('pi_thresh', 0.5),
        gate_mode=cfg['model'].get('gate_mode', 'multiply'),
        
        # Parametri LoRA/VPT
        num_vpt=cfg['model'].get('num_vpt'),
        vpt_drop=cfg['model'].get('vpt_drop'),
        adapter=cfg['model'].get('adapter', False),
        adapter_reduction=cfg['model'].get('adapter_reduction'),
        lora=cfg['model'].get('lora', False),
        lora_rank=cfg['model'].get('lora_rank'),
        lora_alpha=cfg['model'].get('lora_alpha'),
        lora_dropout=cfg['model'].get('lora_dropout'),
        norm=cfg['model'].get('norm', 'none'),
        act=cfg['model'].get('act', 'none')
    ).to(device)
    
    # ==================================================================
    # === FINE MODIFICA ===
    # ==================================================================


    # --- 3. Logica di Stadio (Congelamento e Caricamento) ---
    # (Questa sezione è GIA' CORRETTA per la nuova architettura)
    if args.stage == 1:
        print("--- Configurazione STADIO 1: Pre-training PI Head (ConvZIP) + Backbone ---")
        cfg['loss']['weight_cls'] = 0.0  # Disattiva loss EBC
        cfg['loss']['weight_reg'] = 1.0  # Attiva loss ZIP
        cfg['loss']['weight_aux'] = 0.0  
        print(f"Pesi loss sovrascritti: CLS=0.0, REG=1.0, AUX=0.0")

        # Si addestrano Backbone e la nuova zip_head (ConvZIPHead)
        print("Congelamento: ebc_head")
        for param in model.ebc_head.parameters():
            param.requires_grad = False
        
        save_path = stage1_ckpt_path

    elif args.stage == 2:
        print("--- Configurazione STADIO 2: Pre-training LAMBDA Head (ClipEBC) ---")
        
        if load_path is None: load_path = stage1_ckpt_path
        if os.path.exists(load_path):
            print(f"✅ Caricamento checkpoint Stage 1 da: {load_path}")
            state_dict = torch.load(load_path, map_location=device)
            if 'model' in state_dict: state_dict = state_dict['model']
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"⚠️ Checkpoint Stage 1 non trovato in {load_path}. Addestro da zero (sconsigliato).")

        cfg['loss']['weight_cls'] = 1.0  # Attiva loss EBC
        cfg['loss']['weight_reg'] = 0.0  # Disattiva loss ZIP
        cfg['loss']['weight_aux'] = 0.0 
        print(f"Pesi loss sovrascritti: CLS=1.0, REG=0.0, AUX=0.0")

        # Si addestra solo la ebc_head (ClipEBHead)
        print("Congelamento: backbone, zip_head")
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.zip_head.parameters():
            param.requires_grad = False
        
        for param in model.ebc_head.parameters():
            param.requires_grad = True

        save_path = stage2_ckpt_path

    elif args.stage == 3:
        print("--- Configurazione STADIO 3: Joint Fine-tuning ---")
        
        if load_path is None: load_path = stage2_ckpt_path
        if os.path.exists(load_path):
            print(f"✅ Caricamento checkpoint Stage 2 da: {load_path}")
            state_dict = torch.load(load_path, map_location=device)
            if 'model' in state_dict: state_dict = state_dict['model']
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"⚠️ Checkpoint Stage 2 non trovato in {load_path}. Addestro da zero (sconsigliato).")
            
        print(f"Pesi loss da config: CLS={cfg['loss']['weight_cls']}, REG={cfg['loss']['weight_reg']}, AUX={cfg['loss']['weight_aux']}")

        # Scongela tutto
        print("Scongelamento: Tutti i parametri sono addestrabili.")
        for param in model.parameters():
            param.requires_grad = True
        
        save_path = final_ckpt_path 

    # --- 4. Creazione Dataloaders ---
    train_loader = build_dataloader(cfg, split='train', stage_cfg=train_cfg)
    val_loader = build_dataloader(cfg, split='eval', stage_cfg=train_cfg)

    # --- 5. Creazione Optimizer e Scheduler ---
    # (Questa sezione è GIA' CORRETTA)
    params_to_train = [p for p in model.parameters() if p.requires_grad]
    print(f"Numero di parametri da addestrare in questo stadio: {sum(p.numel() for p in params_to_train)}")
    
    main_lr = train_cfg['lr']
    backbone_lr = train_cfg.get('lr_backbone', main_lr) 
    weight_decay = train_cfg.get('weight_decay', 0)
    optimizer_name = train_cfg.get('optimizer', 'adamw').lower()

    backbone_params = []
    head_params = [] # Conterrà zip_head (Conv) e ebc_head (CLIP)
    BACKBONE_NAME = 'backbone' 

    print(f"Separazione parametri: LR testine={main_lr}, LR backbone ('{BACKBONE_NAME}')={backbone_lr}")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if BACKBONE_NAME in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': main_lr}
    ]

    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(param_groups, lr=main_lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(param_groups, lr=main_lr, momentum=train_cfg.get('momentum', 0.9), weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_name} non supportato.")

    scheduler_name = train_cfg.get('scheduler', 'none').lower()
    if scheduler_name == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg['num_epochs'])
    else:
        scheduler = None 
        if scheduler is None:
            print("Nessuno scheduler ('cosine' o 'step') specificato, si procede senza scheduler.")

    # --- 6. Creazione Loss ---
    criterion = build_loss(cfg).to(device)

    # --- 7. Creazione Trainer ---
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        cfg=cfg, 
        stage_cfg=train_cfg, 
        best_ckpt_path=save_path 
    )

    print(f"Checkpoint per 'best_mae' sarà salvato in: {save_path}")
    
    # --- 8. Avvio Addestramento ---
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to output directory')
    
    parser.add_argument(
        '--stage', 
        type=int, 
        default=3, 
        choices=[1, 2, 3],
        help='Stadio di addestramento (1: Pre-train PI, 2: Pre-train LAMBDA, 3: Joint Fine-tuning)'
    )
    parser.add_argument(
        '--load_ckpt', 
        type=str, 
        default=None,
        help='Percorso checkpoint da caricare (sovrascrive la logica di default degli stadi)'
    )
    
    args = parser.parse_args()

    # Caricamento con PyYAML
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Sovrascrivi output_dir (nella config base) se fornito da linea di comando
    if args.output_dir:
        cfg['train_base']['output_dir'] = args.output_dir
    os.makedirs(cfg['train_base']['output_dir'], exist_ok=True)
    
    main(args, cfg)