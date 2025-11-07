# train.py (Senza Omegaconf, aggiornato per Stadi)
import os
import torch
import torch.nn as nn
import argparse
import yaml  # Importa la libreria standard YAML
from utils import get_optimizer_and_scheduler, set_seed # Importa la funzione corretta
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataloader
from losses import build_loss
from trainer import Trainer # Importa la *classe* Trainer
from typing import Optional, Dict

def main(args, cfg: Dict): 
    # Carica la configurazione base (condivisa)
    train_cfg_base = cfg['train_base']
    
    # Carica la configurazione specifica per lo stadio corrente
    train_cfg_stage = cfg[f'train_stage{args.stage}']
    
    # Unisci la config base con quella dello stadio
    # (train_cfg_stage sovrascrive train_cfg_base se ci sono duplicati)
    train_cfg = {**train_cfg_base, **train_cfg_stage}
    
    # Ora usa 'train_cfg' per tutti i parametri
    set_seed(train_cfg['seed']) 
    device = torch.device(train_cfg['device'] if torch.cuda.is_available() else "cpu") 

    print(f"🚀 AVVIO DELLO STADIO {args.stage} 🚀")

    # --- 1. Definizione Percorsi Checkpoint ---
    output_dir = cfg['train_base']['output_dir'] # Leggi dalla config base
    stage1_ckpt_path = os.path.join(output_dir, "stage1_best.pth")
    stage2_ckpt_path = os.path.join(output_dir, "stage2_best.pth")
    final_ckpt_path = os.path.join(output_dir, "best_mae.pth") 

    load_path = args.load_ckpt
    save_path = final_ckpt_path 

    # --- 2. Creazione Modello ---
    model = build_model(cfg).to(device)
    
    # --- 3. Logica di Stadio (Congelamento e Caricamento) ---
    if args.stage == 1:
        print("--- Configurazione STADIO 1: Pre-training PI Head (ZIP) ---")
        # Loss: Addestra solo con la loss ZIPNLL (reg_loss)
        cfg['loss']['weight_cls'] = 0.0  
        cfg['loss']['weight_reg'] = 1.0  
        cfg['loss']['weight_aux'] = 0.0  
        print(f"Pesi loss sovrascritti: CLS=0.0, REG=1.0, AUX=0.0")

        # Congelamento: Congela la lambda_head
        print("Congelamento: lambda_head, lambda_logit_scale")
        for param in model.lambda_head.parameters():
            param.requires_grad = False
        model.lambda_logit_scale.requires_grad = False
        
        save_path = stage1_ckpt_path

    elif args.stage == 2:
        print("--- Configurazione STADIO 2: Pre-training LAMBDA Head (EBC) ---")
        
        # Caricamento: Carica i pesi dello Stadio 1
        if load_path is None: load_path = stage1_ckpt_path
        if os.path.exists(load_path):
            print(f"✅ Caricamento checkpoint Stage 1 da: {load_path}")
            state_dict = torch.load(load_path, map_location=device)
            if 'model' in state_dict: state_dict = state_dict['model']
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"⚠️ Checkpoint Stage 1 non trovato in {load_path}. Addestro da zero (sconsigliato).")

        # Loss: Addestra solo con ZICE (cls_loss)
        cfg['loss']['weight_cls'] = 1.0 
        cfg['loss']['weight_reg'] = 0.0 
        cfg['loss']['weight_aux'] = 0.0 
        print(f"Pesi loss sovrascritti: CLS=1.0, REG=0.0, AUX=0.0")

        # Congelamento: Congela Backbone e pi_head
        print("Congelamento: backbone, pi_head, pi_logit_scale")
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.pi_head.parameters():
            param.requires_grad = False
        model.pi_logit_scale.requires_grad = False
        
        # Scongela la lambda_head per sicurezza
        for param in model.lambda_head.parameters():
            param.requires_grad = True
        model.lambda_logit_scale.requires_grad = True

        save_path = stage2_ckpt_path 

    elif args.stage == 3:
        print("--- Configurazione STADIO 3: Joint Fine-tuning ---")
        
        # Caricamento: Carica i pesi dello Stadio 2
        if load_path is None: load_path = stage2_ckpt_path
        if os.path.exists(load_path):
            print(f"✅ Caricamento checkpoint Stage 2 da: {load_path}")
            state_dict = torch.load(load_path, map_location=device)
            if 'model' in state_dict: state_dict = state_dict['model']
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"⚠️ Checkpoint Stage 2 non trovato in {load_path}. Addestro da zero (sconsigliato).")
            
        # Loss: Usa i pesi finali dal config
        print(f"Pesi loss da config: CLS={cfg['loss']['weight_cls']}, REG={cfg['loss']['weight_reg']}, AUX={cfg['loss']['weight_aux']}")

        # Congelamento: Scongela tutto
        print("Scongelamento: Tutti i parametri sono addestrabili.")
        for param in model.parameters():
            param.requires_grad = True
        
        save_path = final_ckpt_path 

    # --- 4. Creazione Dataloaders ---
    # Usa 'train_cfg' per i parametri di Dataloader
    train_loader = build_dataloader(cfg, split='train', stage_cfg=train_cfg)
    val_loader = build_dataloader(cfg, split='eval', stage_cfg=train_cfg)

    # --- 5. Creazione Optimizer e Scheduler ---
    params_to_train = [p for p in model.parameters() if p.requires_grad]
    print(f"Numero di parametri da addestrare in questo stadio: {sum(p.numel() for p in params_to_train)}")
    
    # Passa la config dello stadio corrente
    optimizer, scheduler = get_optimizer_and_scheduler(params_to_train, train_cfg, len(train_loader))

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
        cfg=cfg, # Passa la config *globale* (contiene 'train_base', 'eval', ecc.)
        stage_cfg=train_cfg, # Passa la config *dello stadio* (contiene 'num_epochs', 'eval_freq')
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