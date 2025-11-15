# models/heads/clip_ebc_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import conv1x1 # Assicurati che questo import sia corretto
import numpy as np

class ClipEBHead(nn.Module):
    """
    Testa EBC basata su CLIP.
    Prende le features del backbone (es. 768) e le features di testo (es. 512)
    e calcola i logits di classificazione dei bin.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        # Proiettore per far corrispondere le features ViT (es. 768) a quelle di CLIP (es. 512)
        self.projector = conv1x1(in_channels, out_channels, bias=bias)
        
        # Logit scale (parametro addestrabile)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)
        
        # Valore massimo per stabilità (previene NaN)
        self.logit_max = np.log(100.0) 

    def forward(self, image_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        """
        Input:
            image_feats: [B, C_in=768, H, W]
            text_feats:  [N_bins=13, C_out=512]
        Output:
            logit_map: [B, N_bins=13, H, W]
        """
        # Proietta le features dell'immagine
        image_feats_head = self.projector(image_feats) # [B, C_out=512, H, W]
        
        # Normalizza e calcola la similarità
        image_feats_norm = F.normalize(image_feats_head.permute(0, 2, 3, 1), p=2, dim=-1, eps=1e-8)
        
        # text_feats è già normalizzato (da _build_text_feats)
        
        # Applica il clamp al logit_scale (che è in spazio logaritmico)
        self.logit_scale.data.clamp_(max=self.logit_max)
        logit_scale = self.logit_scale.exp()
        
        logit_map = logit_scale * image_feats_norm @ text_feats.t() # [B, H, W, N_bins]
        
        return logit_map.permute(0, 3, 1, 2)  # [B, N_bins, H, W]