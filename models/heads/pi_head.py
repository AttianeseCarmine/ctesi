# models/heads/pi_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PiHead(nn.Module):
    """
    Testa π (Pi) - Predice solo "blocco vuoto o pieno"
    NON predice λ (questo è compito di EBC)
    """
    def __init__(
        self,
        in_ch: int,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.epsilon = epsilon
        
        # Architettura semplice per classificazione binaria
        inter_ch = max(128, in_ch // 2)

        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, 3, padding=1),
            nn.BatchNorm2d(inter_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, inter_ch, 3, padding=1),
            nn.BatchNorm2d(inter_ch),
            nn.ReLU(inplace=True),
        )
        
        # Solo testa π (binaria: vuoto vs pieno)
        self.pi_head = nn.Conv2d(inter_ch, 2, 1)
        
        self._init_weights()

    def _init_weights(self):
        """Inizializzazione attenta per evitare NaN"""
        for m in self.shared.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Inizializzazione neutrale per π
        nn.init.xavier_uniform_(self.pi_head.weight, gain=0.01)
        if self.pi_head.bias is not None:
            nn.init.constant_(self.pi_head.bias, 0.0)

    def forward(self, feat: torch.Tensor):
        """
        Forward pass della π-head.
        
        Args:
            feat: Feature dal backbone [B, C, H, W]
        
        Returns:
            Dictionary con:
                - logit_pi_maps: Logits π [B, 2, H, W]
        """
        h = self.shared(feat)
        
        # Solo logits π (classe 0 = vuoto, classe 1 = pieno)
        logit_pi_maps = torch.clamp(self.pi_head(h), -10, 10)
        
        return {
            "logit_pi_maps": logit_pi_maps,  # [B, 2, H, W]
        }