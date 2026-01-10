import os
import shutil
import torch
import numpy as np
import random
import yaml

class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, filepath='checkpoint.pth'):
    """
    Saves the training checkpoint.
    If is_best is True, copies the checkpoint to 'model_best.pth'.
    """
    # Crea la directory se non esiste
    dirname = os.path.dirname(filepath)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
        
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(dirname, 'model_best.pth')
        shutil.copyfile(filepath, best_filepath)
        print(f"[*] Saved new best model to {best_filepath}")

def load_config(config_path):
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

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