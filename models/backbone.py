import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
from typing import Dict

class Backbone(nn.Module):
    def __init__(self, name: str, pretrained: bool = True, freeze_bn: bool = True):
        super().__init__()
        self.name = name.lower()
        
        # --- VISION TRANSFORMER (ViT) ---
        if 'vit' in self.name:
            if 'b_16' in self.name:
                weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
                self.model = models.vit_b_16(weights=weights)
                self.out_channels = 768
                self.patch_size = 16
                self.native_reduction = self.patch_size  # <-- ADD
            elif 'l_16' in self.name:
                weights = models.ViT_L_16_Weights.DEFAULT if pretrained else None
                self.model = models.vit_l_16(weights=weights)
                self.out_channels = 1024
                self.patch_size = 16
                self.native_reduction = self.patch_size  # <-- ADD
            else:
                raise ValueError(f"ViT variant {self.name} not supported yet.")

            self.model.heads = nn.Identity()

            
        # --- CNN (ResNet / VGG) ---
        elif 'resnet' in self.name:
            if '50' in self.name:
                weights = models.ResNet50_Weights.DEFAULT if pretrained else None
                base = models.resnet50(weights=weights)
                self.out_channels = 2048
            elif '101' in self.name:
                weights = models.ResNet101_Weights.DEFAULT if pretrained else None
                base = models.resnet101(weights=weights)
                self.out_channels = 2048
            
            # Rimuoviamo FC e AvgPool finali
            self.model = nn.Sequential(*list(base.children())[:-2])
            
        elif 'vgg' in self.name:
            weights = models.VGG16_BN_Weights.DEFAULT if pretrained else None
            base = models.vgg16_bn(weights=weights).features
            self.model = base
            self.out_channels = 512
            
        else:
            raise ValueError(f"Backbone {name} non supportato.")

        # Freeze BatchNorm
        if freeze_bn:
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                if hasattr(m, 'weight'): m.weight.requires_grad = False
                if hasattr(m, 'bias'): m.bias.requires_grad = False

    def forward_vit(self, x):
        """
        Gestione Custom per ViT:
        1. Patch Embedding (Senza controlli rigidi di dimensione)
        2. Interpolazione Positional Embedding (Fix per 448x448)
        3. Encoder
        4. Reshape a 2D
        """
        # x: [B, 3, H, W]
        b, c, h, w = x.shape
        
        # 1. Patch Embedding (Conv2d interna di torchvision)
        # USARE QUESTO al posto di _process_input evita l'AssertionError
        x = self.model.conv_proj(x)  # [B, 768, H/16, W/16]
        
        # Catturiamo le dimensioni della griglia
        h_grid = x.shape[2]
        w_grid = x.shape[3]
        
        # Flatten: [B, C, H', W'] -> [B, C, N] -> [B, N, C]
        x = x.flatten(2).transpose(1, 2)
        
        # 2. Add CLS Token
        batch_class_token = self.model.class_token.expand(b, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # 3. Positional Embedding Interpolation
        # I pesi originali sono per 224x224 (14x14 patches + 1 cls)
        pos_embed = self.model.encoder.pos_embedding # [1, 197, 768]
        
        # Se il numero di patch attuali è diverso da quello di training
        if pos_embed.shape[1] != x.shape[1]:
            # Separiamo CLS e Patch
            cls_pos = pos_embed[:, 0:1]
            patch_pos = pos_embed[:, 1:]
            
            # Calcoliamo la dimensione originale (es. 14x14)
            num_orig = patch_pos.shape[1]
            h_orig = int(math.sqrt(num_orig))
            
            # Reshape a griglia 2D [1, 768, 14, 14]
            patch_pos = patch_pos.permute(0, 2, 1).reshape(1, -1, h_orig, h_orig)
            
            # Interpolazione Bicubica alla nuova dimensione (es. 28x28)
            patch_pos = F.interpolate(patch_pos, size=(h_grid, w_grid), mode='bicubic', align_corners=False)
            
            # Flatten di nuovo [1, 768, 28*28] -> [1, 784, 768]
            patch_pos = patch_pos.flatten(2).transpose(1, 2)
            
            # Riuniamo
            pos_embed = torch.cat([cls_pos, patch_pos], dim=1)
            
        # Aggiungiamo i pos embedding interpolati
        x = x + pos_embed
        
        # 4. Encoder Layers
        # ATTENZIONE: Chiamiamo direttamente i layers per evitare che l'encoder
        # ri-aggiunga i pos_embedding originali sbagliati.
        x = self.model.encoder.dropout(x)
        x = self.model.encoder.layers(x)
        x = self.model.encoder.ln(x)
        
        # 5. Reshape back to Image
        # Rimuoviamo CLS
        x = x[:, 1:] 
        
        # Reshape: [B, H*W, C] -> [B, C, H, W]
        x = x.permute(0, 2, 1).reshape(b, self.out_channels, h_grid, w_grid)
        
        return x

    def forward(self, x):
        if 'vit' in self.name:
            return self.forward_vit(x)
        else:
            return self.model(x)

def build_backbone(config):
    bb_conf = config.get('BACKBONE', {})
    name = bb_conf.get('TYPE', 'resnet50')
    pretrained = bb_conf.get('PRETRAINED', True)
    freeze_bn = bb_conf.get('FREEZE_BN', True)
    
    return Backbone(name, pretrained, freeze_bn)