# ============================================================
# ZIP-CLIP-EBC: Modello Completo
# ============================================================
# Combina:
# - VGG16-BN Backbone (feature extraction)
# - ZIP Head (π-head per zero-inflation)
# - CLIP-EBC Head (classificazione bins semantica)
#
# Training in 3 stage:
# 1. Stage 1: Train π-head (classificazione vuoto/pieno)
# 2. Stage 2: Train CLIP-EBC head (conteggio per bins)
# 3. Stage 3: Fine-tuning congiunto
# ============================================================

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
    
    Architettura:
    - VGG16-BN Backbone: estrae feature [B, 512, H/16, W/16]
    - ZIP Head (π-Head): predice probabilità blocco vuoto/pieno
    - CLIP-EBC Head: classifica blocchi in bins di conteggio
    
    I parametri dei BIN vengono letti direttamente dal config file.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # ============================================================
        # 1. Backbone (VGG16)
        # ============================================================
        backbone_cfg = config.get('BACKBONE', {})
        
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
        
        # Caricamento BINS dal config
        raw_bins = config.get('BINS', ebc_cfg.get('BINS', None))
        bin_centers = config.get('BIN_CENTERS', ebc_cfg.get('BIN_CENTERS', None))
        
        if raw_bins is None or bin_centers is None:
            raise ValueError(
                "❌ ERRORE: 'BINS' e 'BIN_CENTERS' devono essere definiti nel config! "
            )
            
        bins = [tuple(b) for b in raw_bins]
        
        self.clip_ebc_head = CLIPEBCHead(
            in_channels=self.in_channels,
            bins=bins,
            bin_centers=bin_centers,
            clip_model=ebc_cfg.get('CLIP_MODEL', 'ViT-B-16'),
            clip_pretrained=ebc_cfg.get('PRETRAINED', 'openai'),
            prompt_type=ebc_cfg.get('PROMPT_TYPE', 'word')
        )

        # Modalità di combinazione density
        self.density_mode = config.get('DENSITY_MODE', 'zip_gated_ebc')

    def forward(self, x: torch.Tensor, return_intermediates: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass del modello ZIP-CLIP-EBC.
        
        Args:
            x: Input images [B, 3, H, W]
            return_intermediates: Se True, include feature intermedie
            
        Returns:
            dict con tutti gli output
        """
        # 1. Feature Extraction
        features = self.backbone(x)
        
        # 2. ZIP Head (Probabilità Vuoto)
        zip_out = self.pi_head(features)
        pi_logits = zip_out['logit_pi']  # pre-sigmoid
        pi_prob = zip_out['pi']          # post-sigmoid, P(vuoto)
        prob_presence = 1.0 - pi_prob    # P(pieno)
        
        # 3. EBC Head (Conteggio)
        ebc_out = self.clip_ebc_head(features)
        ebc_density = ebc_out['expected_count']  # [B, 1, H, W]
        
        # 4. Gating: combina π con EBC
        if self.density_mode == 'zip_gated_ebc':
            final_density = ebc_density * prob_presence
        elif self.density_mode == 'ebc_only':
            final_density = ebc_density
        elif self.density_mode == 'zip_only':
            final_density = zip_out['expected_count']
        else:
            final_density = ebc_density
            
        final_count = final_density.sum(dim=[1, 2, 3])
        
        outputs = {
            'final_count': final_count,
            'final_density': final_density,
            'pi': pi_prob,
            'pi_logits': pi_logits,
            'lambda_': zip_out.get('lambda_', None),
            'zip_expected_count': zip_out.get('expected_count', None),
            'ebc_density': ebc_density,
            'ebc_logits': ebc_out.get('logits', None),
            'bin_probs': ebc_out.get('bin_probs', None),
            'features': features if return_intermediates else None
        }
        
        return outputs

    def forward_stage1(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass per Stage 1 (solo π-head training).
        """
        features = self.backbone(x)
        zip_out = self.pi_head(features)
        
        return {
            'pi': zip_out['pi'],
            'pi_logits': zip_out['logit_pi'],
            'lambda_': zip_out['lambda_'],
            'expected_count': zip_out['expected_count'],
            'features': features
        }
    
    def forward_stage2(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass per Stage 2 (solo EBC head training).
        """
        features = self.backbone(x)
        ebc_out = self.clip_ebc_head(features)
        
        return {
            'ebc_density': ebc_out['expected_count'],
            'ebc_logits': ebc_out['logits'],
            'bin_probs': ebc_out['bin_probs'],
            'visual_features': ebc_out['visual_features'],
            'features': features
        }

    def predict_count(self, x: torch.Tensor) -> torch.Tensor:
        """Predice il conteggio totale."""
        outputs = self.forward(x)
        return outputs['final_count']
    
    def predict_ebc_only(self, x: torch.Tensor) -> torch.Tensor:
        """Predice usando solo EBC (ignora π)."""
        features = self.backbone(x)
        ebc_out = self.clip_ebc_head(features)
        return ebc_out['expected_count'].sum(dim=[1, 2, 3])
    
    def get_trainable_params(self, stage: int = 0) -> List[nn.Parameter]:
        """
        Restituisce i parametri trainabili per ogni stage.
        
        Args:
            stage: 0=all, 1=backbone+pi_head, 2=backbone+ebc_head, 3=all
        """
        if stage == 0 or stage == 3:
            return list(self.parameters())
        elif stage == 1:
            return list(self.backbone.parameters()) + list(self.pi_head.parameters())
        elif stage == 2:
            return list(self.backbone.parameters()) + list(self.clip_ebc_head.parameters())
        else:
            return list(self.parameters())
    
    def freeze_for_stage(self, stage: int):
        """
        Congela i parametri non necessari per un dato stage.
        
        Args:
            stage: 1=freeze EBC, 2=freeze π-head, 3=unfreeze all
        """
        if stage == 1:
            for param in self.clip_ebc_head.parameters():
                param.requires_grad = False
            for param in self.backbone.parameters():
                param.requires_grad = True
            for param in self.pi_head.parameters():
                param.requires_grad = True
                
        elif stage == 2:
            for param in self.pi_head.parameters():
                param.requires_grad = False
            for param in self.backbone.parameters():
                param.requires_grad = True
            for param in self.clip_ebc_head.parameters():
                param.requires_grad = True
                
        elif stage == 3:
            for param in self.parameters():
                param.requires_grad = True
    
    def count_parameters(self) -> Dict[str, int]:
        """Conta i parametri del modello."""
        def count(module):
            return sum(p.numel() for p in module.parameters())
        
        def count_trainable(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        return {
            'backbone': count(self.backbone),
            'backbone_trainable': count_trainable(self.backbone),
            'pi_head': count(self.pi_head),
            'pi_head_trainable': count_trainable(self.pi_head),
            'clip_ebc_head': count(self.clip_ebc_head),
            'clip_ebc_head_trainable': count_trainable(self.clip_ebc_head),
            'total': count(self),
            'total_trainable': count_trainable(self),
        }


def build_zip_clip_ebc_model(config: Dict) -> ZIPCLIPEBCModel:
    """Factory function per costruire il modello."""
    return ZIPCLIPEBCModel(config)