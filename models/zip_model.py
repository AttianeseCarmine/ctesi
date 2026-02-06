# models/zip_model.py

import torch
import torch.nn as nn
from typing import Dict
import torch.nn.functional as F
from .backbone import build_backbone
from .pi_head import ZIPHead

class ZIPModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        
        self.backbone = build_backbone(config)
        
        # --- FIX GEOMETRICO ---
        # Determiniamo la riduzione nativa della backbone
        # ViT-B/16 -> 16. ResNet -> Solitamente 8 (se dilated) o 32.
        self.native_reduction = getattr(self.backbone, "native_reduction", 16) 
        
        # Il target per la localizzazione deve essere 8 per competere con ResNet
        self.target_reduction = 8 
        
        self.upsampler = nn.Identity()
        
        if self.native_reduction == 16 and self.target_reduction == 8:
            print("🔧 Using Stable Bilinear Upsampler + Refinement for ViT")
            self.upsampler = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(self.backbone.out_channels, self.backbone.out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.backbone.out_channels),
                nn.ReLU(inplace=True)
            )

        # La ZIP Head ora lavorerà sempre a stride 8
        zip_cfg = config.get('ZIP_HEAD', {})
        self.zip_head = ZIPHead(
            in_channels=self.backbone.out_channels,
            hidden_dim=zip_cfg.get('HIDDEN_DIM', 256)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. Backbone (esce a 16x16 per ViT)
        features = self.backbone(x)
        
        # 2. Upsampling (porta a 8x8 se necessario)
        # Questo permette alla ZIP Head di avere la risoluzione spaziale necessaria
        features = self.upsampler(features)
        
        # 3. ZIP Head
        zip_out = self.zip_head(features)
        pi_logits = zip_out['logit_pi']
        
        # Controllo dimensioni finali (sicurezza)
        H, W = x.shape[-2], x.shape[-1]
        target_h, target_w = H // self.target_reduction, W // self.target_reduction
        
        if pi_logits.shape[-2:] != (target_h, target_w):
            pi_logits = F.interpolate(pi_logits, size=(target_h, target_w), mode="bilinear", align_corners=False)

        return {
            'pi_logits': pi_logits,
            'features': features
        }