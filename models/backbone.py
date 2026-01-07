# models/backbone.py
import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple

class VGG16Backbone(nn.Module):
    def __init__(self, pretrained=True, freeze_bn=False):
        super().__init__()
        weights = models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = models.vgg16_bn(weights=weights)
        # Prendi feature fino a prima dell'ultimo maxpool (stride 16)
        self.features = nn.Sequential(*list(vgg.features.children())[:34])
        self.out_channels = 512
        self.stride = 16
        if freeze_bn: self._freeze_bn()

    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters(): p.requires_grad = False

    def forward(self, x):
        return self.features(x)

class ResNetBackbone(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True, freeze_bn=False):
        super().__init__()
        if backbone_name == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet50(weights=weights)
        elif backbone_name == 'resnet101':
            weights = models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet101(weights=weights)
        else:
            raise ValueError(f"ResNet non supportato: {backbone_name}")

        # Rimuoviamo FC e AvgPool finali.
        # Layer 4 di ResNet ha stride 32. 
        # Se vogliamo stride 16 (come VGG) dobbiamo modificare la dilatazione o stride, 
        # ma per ora teniamo standard ResNet (stride 32).
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        self.out_channels = 2048
        self.stride = 32 # ResNet standard riduce di 32x
        
        if freeze_bn: self._freeze_bn()

    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters(): p.requires_grad = False

    def forward(self, x):
        return self.features(x)

def build_backbone(config):
    """Costruisce il backbone in base al config."""
    bk_conf = config.get('BACKBONE', {})
    bk_type = bk_conf.get('TYPE', 'vgg16_bn').lower()
    pretrained = bk_conf.get('PRETRAINED', True)
    freeze_bn = bk_conf.get('FREEZE_BN', False)

    if 'vgg' in bk_type:
        return VGG16Backbone(pretrained, freeze_bn)
    elif 'resnet' in bk_type:
        return ResNetBackbone(bk_type, pretrained, freeze_bn)
    else:
        raise ValueError(f"Backbone sconosciuto: {bk_type}")