# ============================================================
# ZIP-CLIP-EBC: VGG16-BN Backbone
# ============================================================
# Backbone VGG16 con Batch Normalization per crowd counting.
# Stride totale: 16 (output H/16 x W/16)
# Output channels: 512
#
# Riferimento: Paper ZIP (Yiming-M/ZIP)
# ============================================================

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple, Optional


class VGG16Backbone(nn.Module):
    """
    VGG16-BN Backbone per crowd counting.
    
    Estrae feature fino al layer conv4_3 (prima del quinto maxpool).
    Output: [B, 512, H/16, W/16]
    
    Args:
        pretrained: Se True, usa pesi pretrained su ImageNet
        freeze_bn: Se True, congela i layer BatchNorm (utile per fine-tuning)
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        freeze_bn: bool = False,
    ):
        super().__init__()
        
        # Carica VGG16 con BatchNorm
        if pretrained:
            weights = models.VGG16_BN_Weights.IMAGENET1K_V1
        else:
            weights = None
            
        vgg = models.vgg16_bn(weights=weights)
        
        # Prendi solo i layer fino a conv4_3 (escludi ultimo maxpool e conv5)
        # VGG16-BN structure:
        # features[0-6]:   conv1 (64) + maxpool   -> H/2
        # features[7-13]:  conv2 (128) + maxpool  -> H/4
        # features[14-23]: conv3 (256) + maxpool  -> H/8
        # features[24-33]: conv4 (512) + maxpool  -> H/16
        # features[34-43]: conv5 (512) + maxpool  -> H/32
        
        # Prendiamo fino al layer 33 (dopo conv4_3, prima del maxpool di conv4)
        # In realtà, per avere stride 16, dobbiamo includere il maxpool dopo conv4
        # Quindi prendiamo fino al layer 33 incluso
        
        # Layer indices per VGG16-BN:
        # 0-6:   Block 1 (conv-bn-relu x2 + maxpool)
        # 7-13:  Block 2 (conv-bn-relu x2 + maxpool)  
        # 14-23: Block 3 (conv-bn-relu x3 + maxpool)
        # 24-33: Block 4 (conv-bn-relu x3 + maxpool)
        # 34-43: Block 5 (conv-bn-relu x3 + maxpool)
        
        self.features = nn.Sequential(*list(vgg.features.children())[:34])
        
        # Output channels
        self.out_channels = 512
        self.stride = 16
        
        # Congela BN se richiesto
        if freeze_bn:
            self._freeze_bn()
        
        print(f"✅ VGG16Backbone inizializzato:")
        print(f"   Pretrained: {pretrained}")
        print(f"   Output channels: {self.out_channels}")
        print(f"   Stride: {self.stride}")
        print(f"   Freeze BN: {freeze_bn}")
    
    def _freeze_bn(self):
        """Congela tutti i layer BatchNorm."""
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
    
    def train(self, mode: bool = True):
        """Override train per mantenere BN in eval se freeze_bn."""
        super().train(mode)
        # Se BN è congelato, mantienilo in eval
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if not any(p.requires_grad for p in module.parameters()):
                    module.eval()
        return self
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            features: [B, 512, H/16, W/16]
        """
        return self.features(x)
    
    def get_output_shape(self, input_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calcola la shape dell'output data una shape di input.
        
        Args:
            input_shape: (H, W) dell'input
            
        Returns:
            (H_out, W_out) dell'output
        """
        H, W = input_shape
        return H // self.stride, W // self.stride



def build_vgg_backbone(
    backbone_name: str = "vgg16_bn",
    pretrained: bool = True,
    freeze_bn: bool = False,
) -> nn.Module:
    """
    Factory function per costruire il backbone VGG.
    
    Args:
        backbone_name: "vgg16_bn" o "vgg19_bn"
        pretrained: Se True, usa pesi pretrained
        freeze_bn: Se True, congela BatchNorm
        
    Returns:
        Backbone module
    """
    backbone_name = backbone_name.lower()
    
    if backbone_name in ["vgg16_bn", "vgg16"]:
        return VGG16Backbone(pretrained=pretrained, freeze_bn=freeze_bn)
    else:
        raise ValueError(f"Backbone non supportato: {backbone_name}. Usa 'vgg16_bn' o 'vgg19_bn'")


if __name__ == "__main__":
    # Test
    print("Testing VGG Backbones...")
    
    # Test VGG16
    backbone16 = VGG16Backbone(pretrained=True)
    x = torch.randn(2, 3, 256, 256)
    out = backbone16(x)
    print(f"\nVGG16 Input: {x.shape}")
    print(f"VGG16 Output: {out.shape}")  # Expected: [2, 512, 16, 16]
    
    # Test con input diversi
    x2 = torch.randn(1, 3, 448, 448)
    out2 = backbone16(x2)
    print(f"\nVGG16 Input 448x448: {x2.shape}")
    print(f"VGG16 Output: {out2.shape}")  # Expected: [1, 512, 28, 28]
    
    print("\n✅ Test completato!")
