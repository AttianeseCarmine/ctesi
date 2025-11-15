# Salva come: models/heads/conv_zip_head.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class ConvZIPHead(nn.Module):
    """
    Testa ZIP Convoluzionale, adattata da christian1301/p2r_zip.
    Prende le features del backbone (es. 768 canali da ViT) e
    restituisce sia i logit per 'pi' (vuoto/non-vuoto) sia la mappa 'lambda' (stima del conteggio).
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
        
        # Registra i bin_centers per il calcolo di lambda
        self.register_buffer(
            "bin_centers",
            torch.tensor(bin_centers, dtype=torch.float32).view(1, -1, 1, 1)
        )
        self.lambda_max = lambda_max
        self.epsilon = epsilon
        
        # Canale intermedio (puoi aumentarlo se in_ch è grande)
        inter_ch = max(64, in_ch // 4)

        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, 3, padding=1),
            nn.BatchNorm2d(inter_ch),
            nn.ReLU(inplace=True),
        )
        
        # 1. Testa 'pi' (2 canali per [zero, non-zero])
        self.pi_head = nn.Conv2d(inter_ch, 2, 1)
        
        # 2. Testa 'bin' (N canali per i bin di regressione)
        self.bin_head = nn.Conv2d(inter_ch, len(bins), 1)
        
        # Inizializzazione bias per 'pi' (opzionale, ma aiuta)
        if self.pi_head.bias is not None:
            with torch.no_grad():
                self.pi_head.bias[0] = 1.5 # Tende a predire 'zero'
                self.pi_head.bias[1] = -1.5 # Tende a non predire 'non-zero'

    def forward(self, feat: torch.Tensor):
        h = self.shared(feat) # [B, C_inter, H, W]

        # 1. Calcola i logit di 'pi'
        logit_pi_maps = self.pi_head(h) # [B, 2, H, W]
        
        # 2. Calcola i logit dei 'bin'
        logit_bin_maps = self.bin_head(h) # [B, N_bins, H, W]
        
        # 3. Calcola 'lambda' (stima del conteggio) dalla media pesata dei bin
        #    (Questa è la stessa logica del file ZIP-EBC originale)
        p_bins = F.softmax(logit_bin_maps, dim=1)
        lambda_maps = (p_bins * self.bin_centers).sum(dim=1, keepdim=True)
        lambda_maps = torch.clamp(lambda_maps, min=self.epsilon, max=self.lambda_max)

        # Restituisce i due output necessari per la loss ZIP originale
        return {
            "logit_pi_maps": logit_pi_maps,   # Per pi_loss_bce
            "logit_bin_maps": logit_bin_maps, # Non usato direttamente dalla tua loss
            "lambda_maps": lambda_maps        # Per pi_loss_nll
        }