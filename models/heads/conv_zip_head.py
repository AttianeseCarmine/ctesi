# models/heads/conv_zip_head.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class ConvZIPHead(nn.Module):
    """
    Testa ZIP Convoluzionale ULTRA-STABILE
    """
    def __init__(
        self,
        in_ch: int,
        bins: List[Tuple[float, float]],
        bin_centers: List[float],
        lambda_max: float = 8.0,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        if not all(len(b) == 2 for b in bins):
            raise ValueError("I bin devono essere tuple di lunghezza 2")
        
        # Registra i bin_centers
        self.register_buffer(
            "bin_centers",
            torch.tensor(bin_centers, dtype=torch.float32).view(1, -1, 1, 1)
        )
        self.lambda_max = lambda_max
        self.epsilon = epsilon
        
        # Canale intermedio
        inter_ch = max(64, in_ch // 4)

        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, 3, padding=1),
            nn.BatchNorm2d(inter_ch),
            nn.ReLU(inplace=True),
        )
        
        # Teste
        self.pi_head = nn.Conv2d(inter_ch, 2, 1)
        self.bin_head = nn.Conv2d(inter_ch, len(bins), 1)
        
        # ✅ INIZIALIZZAZIONE CRITICA - Evita NaN all'inizio
        self._init_weights()

    def _init_weights(self):
        """Inizializzazione attenta per evitare NaN"""
        # Inizializza shared layers
        for m in self.shared.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # ✅ Inizializza pi_head con valori NEUTRI
        nn.init.xavier_uniform_(self.pi_head.weight, gain=0.01)  # Gain piccolo!
        if self.pi_head.bias is not None:
            # Bias neutro: log(0.5/0.5) = 0 per entrambe le classi
            nn.init.constant_(self.pi_head.bias, 0.0)
        
        # ✅ Inizializza bin_head con valori PICCOLI
        nn.init.xavier_uniform_(self.bin_head.weight, gain=0.01)
        if self.bin_head.bias is not None:
            nn.init.constant_(self.bin_head.bias, 0.0)

    def forward(self, feat: torch.Tensor):
        # ✅ DEBUG: Verifica input
        if torch.isnan(feat).any() or torch.isinf(feat).any():
            print("💀 ConvZIPHead: INPUT ha NaN/Inf!")
            feat = torch.nan_to_num(feat, nan=0.0, posinf=10.0, neginf=-10.0)
        
        h = self.shared(feat)
        
        # ✅ DEBUG: Verifica dopo shared
        if torch.isnan(h).any() or torch.isinf(h).any():
            print("💀 ConvZIPHead: SHARED output ha NaN/Inf!")
            h = torch.nan_to_num(h, nan=0.0, posinf=10.0, neginf=-10.0)

        # 1. Logit di 'pi'
        logit_pi_maps = self.pi_head(h)
        
        # ✅ Clamp per sicurezza
        logit_pi_maps = torch.clamp(logit_pi_maps, min=-10, max=10)
        
        # 2. Logit dei 'bin'
        logit_bin_maps = self.bin_head(h)
        
        # ✅ Clamp anche qui
        logit_bin_maps = torch.clamp(logit_bin_maps, min=-10, max=10)
        
        # 3. Calcola 'lambda' - VERSIONE ULTRA-STABILE
        # Usa softmax invece di operazioni pericolose
        p_bins = F.softmax(logit_bin_maps, dim=1)
        
        # ✅ Assicura che bin_centers sia sul device corretto
        if self.bin_centers.device != p_bins.device:
            self.bin_centers = self.bin_centers.to(p_bins.device)
        
        # Calcola lambda come media pesata
        lambda_maps = (p_bins * self.bin_centers).sum(dim=1, keepdim=True)
        
        # ✅ Clamp lambda in range sicuro
        lambda_maps = torch.clamp(lambda_maps, min=self.epsilon, max=self.lambda_max)
        
        # ✅ DEBUG FINALE
        if torch.isnan(logit_pi_maps).any() or torch.isnan(lambda_maps).any():
            print("💀 ConvZIPHead OUTPUT ha NaN!")
            print(f"  logit_pi_maps: {torch.isnan(logit_pi_maps).sum()} NaN")
            print(f"  lambda_maps: {torch.isnan(lambda_maps).sum()} NaN")
            # Fallback estremo
            logit_pi_maps = torch.nan_to_num(logit_pi_maps, nan=0.0)
            lambda_maps = torch.nan_to_num(lambda_maps, nan=1.0)

        return {
            "logit_pi_maps": logit_pi_maps,
            "logit_bin_maps": logit_bin_maps,
            "lambda_maps": lambda_maps
        }