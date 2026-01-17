# models/clip_ebc_model.py
# ============================================================
# CLIP Visual Encoder & EBC Model - VI T 448 FIX
# ============================================================

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
def conv3x3(in_channels: int, out_channels: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 1, base_width: int = 64, dilation: int = 1, expansion: int = 4, norm_layer=None):
        super().__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        width = int(out_channels * (base_width / 64.0)) * groups
        self.expansion = expansion
        self.conv1 = conv1x1(in_channels, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, out_channels * self.expansion) 
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Identity()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels * self.expansion, stride),
                norm_layer(out_channels * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, expansion=1, norm_layer=None):
        super().__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(in_channels, out_channels, stride, dilation=dilation)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, dilation=dilation)
        self.bn2 = norm_layer(out_channels)
        self.downsample = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                norm_layer(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out

def _init_weights(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None: nn.init.constant_(m.bias, 0.)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1.)
            if m.bias is not None: nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None: nn.init.constant_(m.bias, 0.)

def make_resnet_layers(block, cfg, in_channels, dilation=1, expansion=1):
    layers = []
    for v in cfg:
        layers.append(block(in_channels=in_channels, out_channels=v, dilation=dilation, expansion=expansion))
        in_channels = v * expansion
    layers = nn.Sequential(*layers)
    layers.apply(_init_weights)
    return layers

# ============================================================
# 2. PROMPT UTILS
# ============================================================
NUM_TO_WORD = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", 
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten", 
    "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen", "15": "fifteen", 
    "16": "sixteen", "17": "seventeen", "18": "eighteen", "19": "nineteen", 
    "20": "twenty", "21": "twenty-one", "22": "twenty-two", "23": "twenty-three", 
    "24": "twenty-four", "25": "twenty-five", "26": "twenty-six", "27": "twenty-seven", 
    "28": "twenty-eight", "29": "twenty-nine", "30": "thirty", "40": "forty", 
    "50": "fifty", "60": "sixty", "70": "seventy", "80": "eighty", "90": "ninety",
    "100": "one hundred"
}

def num2word(num: Union[int, str]) -> str:
    num = str(int(num))
    return NUM_TO_WORD.get(num, num)

def format_count(count: Union[float, Tuple[float, float]], prompt_type: str = "word") -> str:
    if count == 0 or (isinstance(count, (list, tuple)) and count == [0, 0]):
        return "There is no person." if prompt_type == "word" else "There is 0 person."
    elif count == 1 or (isinstance(count, (list, tuple)) and count == [1, 1]):
        return "There is one person." if prompt_type == "word" else "There is 1 person."
    elif isinstance(count, (int, float)):
        val = int(count)
        word = num2word(val) if prompt_type == "word" else str(val)
        return f"There are {word} people."
    elif isinstance(count, (list, tuple)) and (count[1] == float("inf") or count[1] > 999):
        val = int(count[0])
        word = num2word(val) if prompt_type == "word" else str(val)
        return f"There are more than {word} people."
    else:  
        left, right = int(count[0]), int(count[1])
        if left == right:
            word = num2word(left) if prompt_type == "word" else str(left)
            return f"There are {word} people."
        w_left = num2word(left) if prompt_type == "word" else str(left)
        w_right = num2word(right) if prompt_type == "word" else str(right)
        return f"There are between {w_left} and {w_right} people."

# ============================================================
# 3. CLIP VISUAL ENCODER (Con FIX per Risoluzione 448+)
# ============================================================
class CLIPVisualEncoder(nn.Module):
    def __init__(self, model_name="ViT-B/16", pretrained="openai", frozen=False, output_layer="layer3", num_vpt=10, deep_vpt=True, vpt_drop=0.0):
        super().__init__()
        # Carica il modello CLIP originale
        clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, force_quick_gelu=True)
        self.visual = clip_model.visual
        self.output_layer = output_layer
        self.num_vpt = num_vpt
        self.deep_vpt = deep_vpt
        
        if 'RN' in model_name or 'ResNet' in model_name:
            # Ramo ResNet: No VPT
            if output_layer == 'layer2': self.out_channels, self.reduction = 512, 8
            elif output_layer == 'layer3': self.out_channels, self.reduction = 1024, 16
            else: self.out_channels, self.reduction = 2048, 32
        else:
            # Ramo ViT: Inizializzazione Visual Prompt Tuning (VPT)
            self.image_encoder_depth = len(self.visual.transformer.resblocks)
            width = self.visual.transformer.width
            
            # Inizializzazione stile Xavier/Yiming
            val = math.sqrt(6. / float(3 * 16 + width)) 
            num_layers = self.image_encoder_depth if self.deep_vpt else 1
            
            self.vpt_params = nn.ParameterList([
                nn.Parameter(torch.empty(self.num_vpt, width).uniform_(-val, val)) 
                for _ in range(num_layers)
            ])
            self.vpt_drop = nn.Dropout(vpt_drop)
            
            self.out_channels = width
            self.reduction = 16
            
        if frozen:
            for p in self.visual.parameters(): 
                p.requires_grad = False

    def _prepare_vpt(self, layer_idx, batch_size, device, dtype):
        # Fondamentale per AMP: assicura che i prompt abbiano lo stesso tipo dei dati immagine
        vpt = self.vpt_params[layer_idx].to(device).to(dtype)
        vpt = vpt.unsqueeze(0).expand(batch_size, -1, -1)
        vpt = self.vpt_drop(vpt)
        return vpt.permute(1, 0, 2) # [Num_VPT, Batch, Width]

    def forward(self, x):
        # --- LOGICA RESNET ---
        if hasattr(self.visual, 'layer1'): 
            x = self.visual.conv1(x)
            x = self.visual.bn1(x)
            x = F.relu(x)
            x = self.visual.conv2(x)
            x = self.visual.bn2(x)
            x = F.relu(x)
            x = self.visual.conv3(x)
            x = self.visual.bn3(x)
            x = F.relu(x)
            x = self.visual.avgpool(x)
            x = self.visual.layer1(x)
            x = self.visual.layer2(x)
            if self.output_layer in ['layer3', 'layer4']: x = self.visual.layer3(x)
            if self.output_layer == 'layer4': x = self.visual.layer4(x)
            return x
            
        # --- LOGICA ViT CON VPT E SUPPORTO RETTANGOLARE ---
        else: 
            # 1. Patch Embedding
            x = self.visual.conv1(x) 
            B, width, grid_h, grid_w = x.shape 
            
            # 2. Flatten & Class Token
            x = x.reshape(B, width, -1).permute(0, 2, 1) 
            class_embed = self.visual.class_embedding.to(x.dtype) + torch.zeros(B, 1, width, dtype=x.dtype, device=x.device)
            x = torch.cat([class_embed, x], dim=1) 
            
            # 3. Interpolazione Positional Embedding (per risoluzioni diverse da 224x224)
            pos_embed = self.visual.positional_embedding.to(x.dtype)
            if x.shape[1] != pos_embed.shape[0]:
                cls_pos = pos_embed[0:1]
                grid_pos = pos_embed[1:]
                orig_size = int(math.sqrt(grid_pos.shape[0]))
                grid_pos = grid_pos.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
                grid_pos = F.interpolate(grid_pos, size=(grid_h, grid_w), mode='bicubic', align_corners=False)
                grid_pos = grid_pos.permute(0, 2, 3, 1).reshape(grid_h * grid_w, -1)
                pos_embed = torch.cat([cls_pos, grid_pos], dim=0)
            
            x = x + pos_embed
            x = self.visual.ln_pre(x).permute(1, 0, 2) # [Seq, Batch, Width]
            
            # 4. Ciclo Transformer con iniezione VPT
            vpt = self._prepare_vpt(0, B, x.device, x.dtype)
            
            for idx in range(self.image_encoder_depth):
                # Assemble: [CLS, VPT, TOKENS]
                x = torch.cat([x[:1, :, :], vpt, x[1:, :, :]], dim=0)
                
                x = self.visual.transformer.resblocks[idx](x)

                # Disassemble: rimuove VPT per passare allo strato successivo
                if idx < self.image_encoder_depth - 1:
                    if self.deep_vpt:
                        vpt = self._prepare_vpt(idx + 1, B, x.device, x.dtype)
                    else:
                        vpt = x[1 : (self.num_vpt + 1), :, :]
                
                x = torch.cat([x[:1, :, :], x[(self.num_vpt + 1):, :, :]], dim=0)
            
            # 5. Ricostruzione Mappa Spaziale
            x = x.permute(1, 0, 2) 
            x = self.visual.ln_post(x)
            # Rimuove il CLS token e reshape a griglia [B, C, H, W]
            return x[:, 1:, :].permute(0, 2, 1).reshape(B, width, grid_h, grid_w)

            
class CLIPEBCModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        ebc_cfg = config.get('CLIP_EBC_HEAD', {})
        model_name = ebc_cfg.get('CLIP_MODEL', 'RN50')
        pretrained = ebc_cfg.get('PRETRAINED', 'openai')
        self.prompt_type = ebc_cfg.get('PROMPT_TYPE', 'word')
        
        # Recupera il target reduction dal config
        target_reduction = ebc_cfg.get('REDUCTION', 16)
        
        # Diciamo esplicitamente al modello qual è la sua riduzione finale
        # così sliding_window_predict non si confonde.
        self.reduction = target_reduction 
        
        block_size = config.get('DATA', {}).get('ZIP_BLOCK_SIZE', 16)
        output_layer = 'layer3' if block_size == 16 else 'layer4'
        self.visual_encoder = CLIPVisualEncoder(model_name, pretrained, frozen=False, output_layer=output_layer)
        
        self.clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, force_quick_gelu=True)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        for p in self.clip_model.parameters(): p.requires_grad = False
        
        self.channels = self.visual_encoder.out_channels
        self.clip_embed_dim = self.clip_model.text_projection.shape[1]
        
        # ====================================================
        # FIX: QUI MANCAVA IL BLOCCO DI DEFINIZIONE UPSAMPLE!
        # ====================================================
        if target_reduction == 8:
            self.upsample = nn.ConvTranspose2d(
                in_channels=self.channels, 
                out_channels=self.channels, 
                kernel_size=2, stride=2
            )
            nn.init.normal_(self.upsample.weight, std=0.01)
            if self.upsample.bias is not None:
                nn.init.constant_(self.upsample.bias, 0)
            print(f"✨ Enabled 2x Upsampling for Reduction 8 (Model: {model_name})")
        else:
            self.upsample = nn.Identity()

        if 'RN50' in model_name or 'ResNet50' in model_name:
            decoder_cfg = [2048]
            self.image_decoder = make_resnet_layers(Bottleneck, decoder_cfg, in_channels=self.channels, expansion=1)
            self.channels = decoder_cfg[-1]
        elif 'ViT-B/16' in model_name or 'vit-b/16' in model_name:
            # Implementazione stile Yiming per ViT
            decoder_cfg = [768] # Mantiene la dimensione canali del ViT
            self.image_decoder = make_resnet_layers(BasicBlock, decoder_cfg, in_channels=self.channels, expansion=1)
            self.channels = decoder_cfg[-1]
        else:
            self.image_decoder = nn.Identity()

        if self.channels != self.clip_embed_dim:
            self.projection = nn.Conv2d(in_channels=self.channels, out_channels=self.clip_embed_dim, kernel_size=1)
            self.projection.apply(_init_weights)
        else:
            self.projection = nn.Identity()

        raw_bins = config.get('BINS', [])
        self.register_buffer('bin_centers', torch.tensor(config.get('BIN_CENTERS', []), dtype=torch.float32))
        self.prompts = [format_count(b, self.prompt_type) for b in raw_bins]
        print(f"✅ Model Aligned. Prompts ({self.prompt_type}): {self.prompts[:3]}...")
        self._text_embeddings = None
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @torch.no_grad()
    def _get_text_features(self, device):
        if self._text_embeddings is None or self._text_embeddings.device != device:
            tokens = self.tokenizer(self.prompts).to(device)
            self._text_embeddings = F.normalize(self.clip_model.encode_text(tokens), dim=-1)
        return self._text_embeddings

    def forward(self, x):
        B = x.shape[0]
        # 1. Feature Extraction (Esce a 1/16 per ViT)
        img_feat = self.visual_encoder(x)
        
        # 2. Upsampling (Diventa 1/8 se attivato)
        img_feat = self.upsample(img_feat)

        # 3. Decoder & Projection
        img_feat = self.image_decoder(img_feat)
        img_feat = self.projection(img_feat)
        
        img_feat = img_feat.permute(0, 2, 3, 1)
        img_feat = F.normalize(img_feat, p=2, dim=-1)
        
        txt_feat = self._get_text_features(x.device)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * torch.matmul(img_feat, txt_feat.t())
        logits = logits.permute(0, 3, 1, 2)
        
        prob = F.softmax(logits, dim=1)
        density = (prob * self.bin_centers.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        final_count = density.sum(dim=[1, 2, 3])
        
        return {
            'ebc_density': density,
            'ebc_logits': logits,
            'bin_probs': prob,
            'final_count': final_count  
        }