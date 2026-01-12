# datasets/builder.py
import os
from .sha import SHA  # Assumi che SHA gestisca anche SHB se la struttura è identica
from .ucf_qnrf import UCF_QNRF

def build_dataset(config, split, transforms):
    """
    Factory per istanziare il dataset corretto in base al config.
    """
    dataset_name = config.get('DATASET', 'sha').lower()
    root_dir = config['DATA']['ROOT']

    print(f"📂 Loading Dataset: {dataset_name.upper()} | Split: {split}")

    if 'sha' in dataset_name or 'shb' in dataset_name:
        # ShanghaiTech A e B usano solitamente la stessa classe loader
        return SHA(root_dir, split, transforms)
    
    elif 'qnrf' in dataset_name or 'ucf' in dataset_name:
        # UCF-QNRF ha un loader specifico (quello che hai scritto tu)
        return UCF_QNRF(root_dir, split, transforms)
    
    else:
        raise ValueError(f"❌ Dataset '{dataset_name}' non supportato o non trovato nel builder.")