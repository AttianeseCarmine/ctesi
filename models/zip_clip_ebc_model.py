import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple

from .backbone import build_vgg_backbone
from .pi_head import build_zip_head
from .clip_ebc_head import CLIPEBCHead 

class ZIPCLIPEBCModel(nn.Module):
    """
    Modello ZIP-CLIP-EBC con Backbone VGG16.
    
    I parametri dei BIN vengono letti direttamente dal config file.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # ============================================================
        # 1. Backbone (VGG16)
        # ============================================================
        backbone_cfg = config.get('BACKBONE', {})
        
        # Gestione robusta per il dizionario BACKBONE
        if isinstance(backbone_cfg, str):
            bk_type = backbone_cfg
            pretrained = True
            freeze_bn = False
        else:
            bk_type = backbone_cfg.get('TYPE', 'vgg16_bn')
            pretrained = backbone_cfg.get('PRETRAINED', True)
            freeze_bn = backbone_cfg.get('FREEZE_BN', False)

        self.backbone = build_vgg_backbone(
            backbone_name=bk_type,
            pretrained=pretrained,
            freeze_bn=freeze_bn
        )
        
        self.in_channels = 512  # VGG16 output channels
        
        # ============================================================
        # 2. ZIP Head (π-Head)
        # ============================================================
        zip_cfg = config.get('ZIP_HEAD', {})
        self.pi_head = build_zip_head(
            in_channels=self.in_channels,
            hidden_dim=zip_cfg.get('HIDDEN_DIM', 256),
            version="v1" 
        )
        
        # ============================================================
        # 3. CLIP-EBC Head (Counting)
        # ============================================================
        ebc_cfg = config.get('CLIP_EBC_HEAD', {})
        
        # --- CARICAMENTO BINS DAL CONFIG ---
        # Priorità: config radice -> config EBC_HEAD -> default (errore se manca)
        raw_bins = config.get('BINS', ebc_cfg.get('BINS', None))
        bin_centers = config.get('BIN_CENTERS', ebc_cfg.get('BIN_CENTERS', None))
        
        if raw_bins is None or bin_centers is None:
            raise ValueError(
                "❌ ERRORE: 'BINS' e 'BIN_CENTERS' devono essere definiti nel config_sha.yaml! "
                "Non usare valori hardcoded nel codice."
            )
            
        # Conversione sicura: YAML carica liste di liste [[0,0], [1,1]], 
        # convertiamo in lista di tuple [(0,0), (1,1)] per coerenza
        bins = [tuple(b) for b in raw_bins]
        
        self.clip_ebc_head = CLIPEBCHead(
            in_channels=self.in_channels,
            bins=bins,
            bin_centers=bin_centers,
            clip_model_name=ebc_cfg.get('CLIP_MODEL', 'ViT-B-16'), 
            pretrained=ebc_cfg.get('PRETRAINED', 'openai'),
            prompt_type=ebc_cfg.get('PROMPT_TYPE', 'word')
        )

        # Modalità di combinazione
        self.density_mode = config.get('DENSITY_MODE', 'zip_gated_ebc')

    def forward(self, x, return_intermediates=False):
        # 1. Feature Extraction
        features = self.backbone(x)
        
        # 2. ZIP Head (Probabilità Vuoto)
        zip_out = self.pi_head(features)
        pi_logits = zip_out['pi'] 
        pi_prob = torch.sigmoid(pi_logits)  # Probabilità VUOTO
        prob_presence = 1.0 - pi_prob       # Probabilità PIENO
        
        # 3. EBC Head (Conteggio)
        ebc_out = self.clip_ebc_head(features)
        ebc_density = ebc_out['density']
        
        # 4. Gating
        if self.density_mode == 'zip_gated_ebc':
            final_density = ebc_density * prob_presence
        elif self.density_mode == 'ebc_only':
            final_density = ebc_density
        else:
            final_density = ebc_density
            
        final_count = final_density.sum(dim=[1, 2, 3])
        
        outputs = {
            'final_count': final_count,
            'final_density': final_density,
            'pi': pi_prob,
            'pi_logits': pi_logits,
            'ebc_density': ebc_density,
            'ebc_logits': ebc_out.get('logits', None),
            'features': features if return_intermediates else None
        }
        
        return outputs

    def predict_ebc_only(self, x):
        features = self.backbone(x)
        ebc_out = self.clip_ebc_head(features)
        return ebc_out['density'].sum(dim=[1, 2, 3])