# models/clip_ebc_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, Union
import math
import numpy as np

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    print("⚠️ open_clip non installato.")

# ============================================================
# 1. OFFICIAL UTILS
# ============================================================
def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(identity)
        return self.relu(out)

# ============================================================
# 2. CLIP-EBC MODEL (MAIN CLASS)
# ============================================================
class CLIPEBCModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        ebc_cfg = config.get('CLIP_EBC_HEAD', {})
        model_name = ebc_cfg.get('CLIP_MODEL', 'ViT-B/16')
        pretrained = ebc_cfg.get('PRETRAINED', 'openai')
        
        # 1. Load CLIP
        print(f"🔄 Loading CLIP model: {model_name} ({pretrained})...")
        clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.visual = clip_model.visual
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # Freeze CLIP
        for p in self.visual.parameters(): p.requires_grad = False
        for p in clip_model.parameters(): p.requires_grad = False
        self.clip_model = clip_model

        # 2. Setup Dimensions & Hooks
        if 'RN' in model_name: # ResNet (RN50)
            self.channels = 2048 # ResNet50 output layer4 channels
            self.is_vit = False
            # HOOK: Cattura l'output di layer4 PRIMA che entri in attnpool
            self._resnet_features = None
            self.visual.layer4.register_forward_hook(self._hook_fn)
        else: # ViT
            self.channels = 768  # ViT-B/16 output
            self.is_vit = True
            
        self.embed_dim = clip_model.text_projection.shape[1] # usually 512

        # 3. Decoder with UPSAMPLE logic for ResNet
        if self.is_vit:
            # ViT esce già a 1/16 (28x28), non serve upsample
            self.decoder = nn.Sequential(
                BasicBlock(self.channels, 768),
                BasicBlock(768, 768)
            )
            self.channels = 768
        else:
            # ResNet esce a 1/32 (14x14). Dobbiamo portarlo a 1/16 (28x28)
            # Aggiungiamo un Upsample nel decoder
            self.decoder = nn.Sequential(
                nn.Conv2d(self.channels, 512, kernel_size=1, bias=False), # Riduci canali 2048->512
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # Upsample 14->28
                BasicBlock(512, 512) # Raffina le feature
            )
            self.channels = 512 # Nuova dimensione canali

        # 4. Projection
        self.projection = nn.Conv2d(self.channels, self.embed_dim, kernel_size=1)
        self._init_weights(self.projection)
        
        # 5. Prompts & Bins
        bin_centers = config.get('BIN_CENTERS', [])
        if not bin_centers:
            # Fallback
            bin_centers = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
            
        self.register_buffer('bin_centers', torch.tensor(bin_centers, dtype=torch.float32))
        self.prompts = [self._fmt_prompt(c) for c in bin_centers]
        print(f"✅ Bins initialized: {bin_centers}")
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self._txt_embed = None

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _fmt_prompt(self, num):
        if num < 1e-3: return "There is no person."
        if abs(num - 1.0) < 1e-3: return "There is one person."
        return f"There are {str(int(num))} people." if num < 9000 else "There is a crowd."

    def get_text_features(self, device):
        if self._txt_embed is None or self._txt_embed.device != device:
            tok = self.tokenizer(self.prompts).to(device)
            self._txt_embed = F.normalize(self.clip_model.encode_text(tok), dim=-1)
        return self._txt_embed

    # --- HOOK FUNCTION ---
    def _hook_fn(self, module, input, output):
        self._resnet_features = output

    def forward_resnet_features(self, x):
        """
        Esegue la ResNet con Hook per catturare layer4 (14x14).
        """
        try:
            _ = self.visual(x)
        except RuntimeError:
            pass
            
        if self._resnet_features is None:
            raise RuntimeError("Hook failed: features not captured from ResNet layer4")
            
        features = self._resnet_features
        self._resnet_features = None 
        return features

    def forward_vit_reshaped(self, x):
        """
        Logica ViT con interpolazione posizionale.
        """
        x = self.visual.conv1(x)  
        B, C, H, W = x.shape 
        x = x.reshape(B, C, -1).permute(0, 2, 1) 
        
        class_token = self.visual.class_embedding.to(x.dtype) + torch.zeros(B, 1, C, dtype=x.dtype, device=x.device)
        x = torch.cat([class_token, x], dim=1) 
        
        pos_embed = self.visual.positional_embedding.to(x.dtype)
        if x.shape[1] != pos_embed.shape[0]:
            cls_pos = pos_embed[0:1] 
            grid_pos = pos_embed[1:] 
            orig_size = int(math.sqrt(grid_pos.shape[0]))
            
            grid_pos = grid_pos.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
            grid_pos = F.interpolate(grid_pos, size=(H, W), mode='bicubic', align_corners=False)
            grid_pos = grid_pos.permute(0, 2, 3, 1).reshape(-1, C)
            
            pos_embed = torch.cat([cls_pos, grid_pos], dim=0)

        x = x + pos_embed
        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)
        
        x = x[:, 1:, :] 
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x

    def forward(self, x):
        # 1. Extract Features
        if self.is_vit:
            img_feat = self.forward_vit_reshaped(x)
        else:
            img_feat = self.forward_resnet_features(x) # Output 14x14
            
        # 2. Decode & Project (Upsample avviene qui dentro per ResNet)
        img_feat = self.decoder(img_feat) # Output diventa 28x28
        img_feat = self.projection(img_feat) 
        
        # 3. Similarity
        txt_feat = self.get_text_features(x.device) 
        img_feat = F.normalize(img_feat, dim=1)
        
        logits = torch.einsum('bchw,nc->bnhw', img_feat, txt_feat) * self.logit_scale.exp()
        prob = F.softmax(logits, dim=1)
        
        # 4. Density & Count
        density = (prob * self.bin_centers.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        final_count = density.sum(dim=(1, 2, 3))
        
        return {
            'ebc_density': density,
            'ebc_logits': logits,
            'bin_probs': prob,
            'final_count': final_count
        }