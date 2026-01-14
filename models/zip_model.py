import torch
import torch.nn as nn
from typing import Dict

# Importa il nuovo builder e la tua ZIPHead esistente
from .backbone import build_backbone
from .pi_head import ZIPHead

class ZIPModel(nn.Module):
    """
    Modello Stage 1 PURO (No CLIP).
    Usa Backbone (VGG/ResNet/ViT) + ZIPHead per predire la maschera binaria (pi).
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # 1. Backbone Dinamico (ResNet, ViT, VGG...)
        self.backbone = build_backbone(config)
        
        # Recupera canali dinamicamente
        in_channels = self.backbone.out_channels 
        
        print(f"✅ ZIPModel Inizializzato: Backbone={self.backbone.name} -> OutCh={in_channels}")

        # 2. ZIP Head (Stima Pi per la maschera)
        zip_cfg = config.get('ZIP_HEAD', {})
        self.zip_head = ZIPHead(
            in_channels=in_channels,
            hidden_dim=zip_cfg.get('HIDDEN_DIM', 256)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. Feature Extraction
        features = self.backbone(x)
        
        # 2. ZIP Prediction
        zip_out = self.zip_head(features)
        
        # Restituisci tutto ciò che serve (inclusi i logits grezzi per la loss)
        return {
            'pi_logits': zip_out['logit_pi'],  # Per la BCE Loss
            'pi': zip_out['pi'],               # Per visualizzazione/inference
            'features': features               # Utile per debug
        }