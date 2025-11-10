# datasets/__init__.py
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any

# Importa le tue classi Dataset
from .crowd import Crowd, InMemoryCrowd, available_datasets, standardize_dataset_name, NWPUTest, ShanghaiTech

# Importa la funzione 'build_transforms' che abbiamo aggiunto a transforms.py
from .transforms import build_transforms, RandomCrop, Resize, RandomResizedCrop, RandomHorizontalFlip, Resize2Multiple, ZeroPad2Multiple
from .transforms import ColorJitter, RandomGrayscale, GaussianBlur, RandomApply, PepperSaltNoise

from .utils import collate_fn


# === FUNZIONE NUOVA (Wrapper per il Dataloader) ===
def build_dataloader(
    cfg: Dict,             # Configurazione globale (per 'dataset' e 'eval')
    split: str,            # 'train' o 'eval'
    stage_cfg: Dict        # Configurazione dello stadio (per 'batch_size', 'num_workers')
) -> DataLoader:
    
    dataset_cfg = cfg['dataset']
    dataset_name = standardize_dataset_name(dataset_cfg['name'])
    
    # Impostazioni specifiche per split
    if split == 'train':
        split_name = dataset_cfg['train_split']
        batch_size = stage_cfg['batch_size']
        num_workers = stage_cfg['num_workers']
        shuffle = True
    elif split == 'eval':
        split_name = dataset_cfg['eval_split']
        # Per la validazione, usiamo i parametri globali da 'eval'
        batch_size = cfg['eval']['batch_size'] 
        num_workers = cfg['eval']['num_workers'] 
        shuffle = False
    else:
        raise ValueError(f"Split '{split}' non riconosciuto.")

    # --- INIZIO BLOCCO CORRETTO ---
    
    input_size = dataset_cfg['input_size']

    # Costruisci le trasformazioni
    transforms = build_transforms(
        input_size=input_size,
        aug_config=dataset_cfg.get('aug_config', 'aug_config_1'),
        is_train=(split == 'train')
    )
    
    DatasetClass = Crowd
    
    # --- CHIAMATA CORRETTA ---
    # Questa chiamata ora corrisponde a quella in datasets/crowd.py
    # Passa 'dataset_name' all'argomento 'dataset'
    dataset = DatasetClass(
        dataset=dataset_name,  # <--- 'dataset' (la stringa, es. "sha")
        split=split_name,      # <--- 'split' (la stringa, es. "train")
        transforms=transforms  # <--- 'transforms' (l'oggetto Compose)
    )
    # --- FINE BLOCCO CORRETTO ---
    
    print(f"Split '{split}' caricato: {dataset_name}/{split_name}, {len(dataset)} campioni.")
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return loader

# === EXPORTS ===
__all__ = [
    "Crowd", "InMemoryCrowd", "available_datasets", "standardize_dataset_name", "NWPUTest", "ShanghaiTech",
    "RandomCrop", "Resize", "RandomResizedCrop", "RandomHorizontalFlip", "Resize2Multiple", "ZeroPad2Multiple",
    "ColorJitter", "RandomGrayscale", "GaussianBlur", "RandomApply", "PepperSaltNoise",
    "collate_fn",
    "build_dataloader", 
    "build_transforms", 
]