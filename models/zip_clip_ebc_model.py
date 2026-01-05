import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# Assicurati che questi import funzionino nel tuo progetto
from .backbone import build_vgg_backbone
from .pi_head import ZIPHead
from .clip_ebc_head import build_clip_ebc_head

class ZIPCLIPEBCModel(nn.Module):
    """
    Modello Stage 1: VGG Backbone + ZIP Head + CLIP-EBC Head.
    
    Questo modello impara a:
    1. Distinguere sfondo/folla (ZIP Head -> pi)
    2. Stimare la densità usando semantica CLIP (EBC Head)
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # 1. Backbone (VGG16 con Batch Norm)
        self.backbone = build_vgg_backbone(
            backbone_name=config.get('BACKBONE', {}).get('TYPE', 'vgg16_bn'),
            pretrained=config.get('BACKBONE', {}).get('PRETRAINED', True),
            freeze_bn=config.get('BACKBONE', {}).get('FREEZE_BN', False)
        )
        
        # Canali in uscita dal backbone (es. 512 per VGG16)
        # Nota: VGG riduce di 16x, quindi per input 448x448 -> 28x28
        in_channels = 512 
        
        # 2. ZIP Head (stima Pi e Lambda)
        zip_cfg = config.get('ZIP_HEAD', {})
        self.zip_head = ZIPHead(
            in_channels=in_channels,
            hidden_dim=zip_cfg.get('HIDDEN_DIM', 256)
        )
        
        # 3. CLIP-EBC Head (stima densità semantica)
        # Questa head proietta le feature VGG nello spazio CLIP
        self.ebc_head = build_clip_ebc_head(config, in_channels)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. Feature Extraction
        features = self.backbone(x) # [B, 512, H/16, W/16]
        
        # 2. ZIP Prediction
        zip_out = self.zip_head(features)
        pi_logits = zip_out['pi']       # Logits per probabilità "vuoto"
        lambda_logits = zip_out['lambda_'] # Logits per rate Poisson (non usato in Stage 3 ma utile)
        
        # 3. CLIP-EBC Prediction
        ebc_out = self.ebc_head(features)
        
        # --- FIX QUI SOTTO ---
        # Prima cercava 'expected_count', ora la chiave corretta è 'ebc_density'
        if 'ebc_density' in ebc_out:
            ebc_density = ebc_out['ebc_density']
        elif 'expected_count' in ebc_out: # Fallback per compatibilità
            ebc_density = ebc_out['expected_count']
        else:
            raise KeyError(f"Chiave densità non trovata in output EBC. Chiavi disponibili: {ebc_out.keys()}")
            
        ebc_logits = ebc_out['ebc_logits']
        
        # 4. Return Dictionary
        return {
            'pi_logits': pi_logits,
            'lambda_logits': lambda_logits,
            'ebc_density': ebc_density,
            'ebc_logits': ebc_logits,
            # Passiamo anche le feature se servissero
            'features': features
        }

if __name__ == '__main__':
    # Test rapido
    print("Testing ZIPCLIPEBCModel...")
    config = {
        'BACKBONE': {'TYPE': 'vgg16_bn', 'PRETRAINED': False},
        'ZIP_HEAD': {'HIDDEN_DIM': 256},
        'CLIP_EBC_HEAD': {'CLIP_MODEL': 'RN50', 'PRETRAINED': 'openai'},
        'BINS': [[0,0], [1,1]], 'BIN_CENTERS': [0.0, 1.0]
    }
    model = ZIPCLIPEBCModel(config)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print("Keys:", out.keys())
    print("✅ Test OK")