import torch
import torch.nn as nn

class ZIPCLIPJointModel(nn.Module):
    """
    Modello congiunto per lo Stage 3.
    Esegue sia il modello ZIP (Stage 1) che CLIP-EBC (Stage 2)
    e unisce i loro output in un unico dizionario.
    """
    def __init__(self, stage1_model, stage2_model):
        super().__init__()
        self.stage1 = stage1_model
        self.stage2 = stage2_model
        
    def forward(self, x):
        # 1. Forward Stage 1 (ZIP - ResNet50)
        # Restituisce: {'pi_logits': ..., 'features': ...}
        out1 = self.stage1(x)
        
        # 2. Forward Stage 2 (CLIP - EBC)
        # Restituisce: {'ebc_density': ..., 'ebc_logits': ..., 'bin_probs': ...}
        out2 = self.stage2(x)
        
        # 3. Merge dei risultati
        outputs = {}
        outputs.update(out1)  # Inserisce pi_logits
        outputs.update(out2)  # Inserisce ebc_density
        
        return outputs