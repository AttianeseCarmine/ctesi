# models/zip_model.py
import torch
import torch.nn as nn
from typing import Dict

# Importa il nuovo builder e la tua ZIPHead esistente
from .backbone import build_backbone
from .pi_head import ZIPHead

class ZIPModel(nn.Module):
    """
    Modello Stage 1 PURO (No CLIP).
    Usa Backbone (VGG/ResNet) + ZIPHead per predire la maschera binaria (pi).
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # 1. Backbone Dinamico
        self.backbone = build_backbone(config)
        
        # 2. ZIP Head (Stima Pi per la maschera)
        # Usa out_channels del backbone (512 per VGG, 2048 per ResNet)
        zip_cfg = config.get('ZIP_HEAD', {})
        self.zip_head = ZIPHead(
            in_channels=self.backbone.out_channels,
            hidden_dim=zip_cfg.get('HIDDEN_DIM', 256)
        )
        
        print(f"✅ ZIPModel Inizializzato: Backbone={config.get('BACKBONE', {}).get('TYPE')} -> OutCh={self.backbone.out_channels}")
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. Feature Extraction
        features = self.backbone(x)
        
        # 2. ZIP Prediction
        zip_out = self.zip_head(features)
        
        # FIX: Prendiamo 'logit_pi' (pre-sigmoid) per la BCEWithLogitsLoss
        pi_logits = zip_out['logit_pi']
        
        return {
            'pi_logits': pi_logits, # Per la Loss e per lo Stage 3
            'features': features    # Opzionale
        }