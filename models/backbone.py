# models/backbone.py
import torch
import torch.nn as nn
from torchvision import models

class VGG16Backbone(nn.Module):
    def __init__(self, pretrained=True, freeze_bn=True):
        super().__init__()
        weights = models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = models.vgg16_bn(weights=weights)
        
        # VGG16 standard ha stride 32 alla fine.
        # Per crowd counting spesso si rimuove l'ultimo pooling o si usano solo i primi layer.
        # Qui prendiamo le features complete (stride 32).
        self.features = vgg.features
        
        self.out_channels = 512
        self.stride = 32 
        
        if freeze_bn:
            self._freeze_bn()

    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
    
    def forward(self, x):
        return self.features(x)

class ResNetBackbone(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True, freeze_bn=True):
        super().__init__()
        
        # Pesi
        if 'resnet50' in backbone_name:
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            # --- MODIFICA CRITICA SOTA: STRIDE 16 ---
            # Usiamo la dilatazione nell'ultimo blocco invece dello stride.
            # Questo mantiene la risoluzione alta (64x64 su img 1024) per matchare CLIP.
            self.resnet = models.resnet50(
                weights=weights,
                replace_stride_with_dilation=[False, False, True] 
            )
            self.out_channels = 2048
        elif 'resnet101' in backbone_name:
            weights = models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            self.resnet = models.resnet101(
                weights=weights,
                replace_stride_with_dilation=[False, False, True]
            )
            self.out_channels = 2048
        else:
            # Fallback per resnet18/34 (che non supportano dilation array facilmente)
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.resnet = models.resnet18(weights=weights)
            self.out_channels = 512

        # Costruiamo il backbone senza FC e AvgPool
        self.features = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        )
        
        self.stride = 16  # <--- ORA E' 16 GRAZIE ALLA DILATAZIONE!
        
        if freeze_bn:
            self._freeze_bn()

    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, x):
        return self.features(x)

def build_backbone(config):
    """Costruisce il backbone in base al config."""
    # Recuperiamo il tipo dal config, default a resnet50
    if 'BACKBONE' in config:
        bk_type = config['BACKBONE'].get('TYPE', 'resnet50').lower()
        pretrained = config['BACKBONE'].get('PRETRAINED', True)
        freeze_bn = config['BACKBONE'].get('FREEZE_BN', True)
    else:
        # Fallback se la struttura del config è diversa
        bk_type = 'resnet50'
        pretrained = True
        freeze_bn = True

    print(f"🏗️  Building Backbone: {bk_type} (Pretrained={pretrained})")

    if 'vgg' in bk_type:
        return VGG16Backbone(pretrained=pretrained, freeze_bn=freeze_bn)
    elif 'resnet' in bk_type:
        return ResNetBackbone(backbone_name=bk_type, pretrained=pretrained, freeze_bn=freeze_bn)
    else:
        raise ValueError(f"Backbone {bk_type} non supportato.")