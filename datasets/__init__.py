# datasets/__init__.py
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any

# Importa le tue classi Dataset (adatta i percorsi se .crowd è in un'altra cartella)
from .crowd import Crowd, InMemoryCrowd, available_datasets, standardize_dataset_name, NWPUTest, ShanghaiTech
from .transforms import RandomCrop, Resize, RandomResizedCrop, RandomHorizontalFlip, Resize2Multiple, ZeroPad2Multiple
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

    # (Assumiamo che il tuo dataset 'Crowd' sia quello corretto da usare)
    # Se usi 'ShanghaiTech' o altri, cambia la classe qui
    # NOTA: Assicurati che il tuo dataset restituisca un dizionario 
    # con 'image', 'density_map', e 'points'
    DatasetClass = Crowd # o ShanghaiTech, a seconda di cosa usi
    
    dataset = DatasetClass(
        dataset_name=dataset_name,
        split=split_name,
        input_size=dataset_cfg['input_size'],
        aug_config=dataset_cfg.get('aug_config', 'aug_config_1'), 
    )
    
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

# === MANTIENI GLI EXPORT ORIGINALI E AGGIUNGI IL NUOVO ===
__all__ = [
    "Crowd", "InMemoryCrowd", "available_datasets", "standardize_dataset_name", "NWPUTest", "ShanghaiTech",
    "RandomCrop", "Resize", "RandomResizedCrop", "RandomHorizontalFlip", "Resize2Multiple", "ZeroPad2Multiple",
    "ColorJitter", "RandomGrayscale", "GaussianBlur", "RandomApply", "PepperSaltNoise",
    "collate_fn",
    "build_dataloader", # Aggiungi la nuova funzione agli export
]