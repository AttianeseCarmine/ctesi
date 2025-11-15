import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import conv1x1
import numpy as np

class CLIPHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)
        self.projector = conv1x1(in_channels, out_channels, bias=bias)
        
        # Valore massimo per logit_scale (log(100) circa 4.605)
        self.logit_max = np.log(100.0) 

    def forward(self, image_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        image_feats_head = self.projector(image_feats)
        image_feats_norm = F.normalize(image_feats_head.permute(0, 2, 3, 1), p=2, dim=-1, eps=1e-8)
        
        # Applica il clamp al logit_scale (che è in spazio logaritmico)
        self.logit_scale.data.clamp_(max=self.logit_max)
        logit_scale = self.logit_scale.exp()
        
        logit_map = logit_scale * image_feats_norm @ text_feats.t()
        return logit_map.permute(0, 3, 1, 2)