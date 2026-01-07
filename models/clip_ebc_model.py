# models/clip_ebc_model.py
# ============================================================
# CLIP Visual Encoder per Crowd Counting - FIXED (Official Align)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import math
from torchvision.models.resnet import Bottleneck  # Import necessario per il Decoder

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    print("⚠️ open_clip non installato. Installa con: pip install open-clip-torch")


class CLIPVisualEncoder(nn.Module):
    """
    CLIP Visual Encoder per crowd counting.
    Estrae feature dense dal visual encoder di CLIP.
    """
    
    def __init__(
        self,
        model_name: str = "RN50",
        pretrained: str = "openai",
        frozen: bool = False,
        output_layer: str = "layer3",
    ):
        super().__init__()
        
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError("open_clip non disponibile")
        
        # Carica modello CLIP completo
        clip_model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, force_quick_gelu=True
        )
        
        self.model_name = model_name
        self.output_layer = output_layer
        
        # Estrai visual encoder
        self.visual = clip_model.visual
        
        # Determina tipo e proprietà
        if model_name.startswith('RN'):
            self._setup_resnet()
        elif model_name.startswith('ViT'):
            self._setup_vit()
        else:
            raise ValueError(f"Modello non supportato: {model_name}")
        
        if frozen:
            self._freeze()
        
        print(f"✅ CLIPVisualEncoder inizializzato:")
        print(f"   Model: {model_name} ({pretrained})")
        print(f"   Output channels: {self.out_channels}")
        print(f"   Output layer: {output_layer}")
    
    def _setup_resnet(self):
        """Setup per CLIP ModifiedResNet."""
        if self.output_layer == 'layer2':
            self.out_channels = 512
            self.reduction = 8
        elif self.output_layer == 'layer3':
            self.out_channels = 1024
            self.reduction = 16
        else:  # layer4
            self.out_channels = 2048
            self.reduction = 32

    
    def _setup_vit(self):
        """Setup per CLIP ViT."""
        if '16' in self.model_name:
            self.reduction = 16
        elif '32' in self.model_name:
            self.reduction = 32
        elif '14' in self.model_name:
            self.reduction = 14
        else:
            self.reduction = 16
        
        self.out_channels = self.visual.transformer.width
    
    def _freeze(self):
        """Congela tutti i parametri."""
        for param in self.visual.parameters():
            param.requires_grad = False
    
    def forward_resnet(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward per CLIP ModifiedResNet.
        Estrae feature dense senza attention pooling.
        """
        visual = self.visual
        
        # Stem
        x = F.relu(visual.bn1(visual.conv1(x)), inplace=True)
        x = F.relu(visual.bn2(visual.conv2(x)), inplace=True)
        x = F.relu(visual.bn3(visual.conv3(x)), inplace=True)
        x = visual.avgpool(x)
        
        x = visual.layer1(x)
        x = visual.layer2(x)

        if self.output_layer in ['layer3', 'layer4']:
            x = visual.layer3(x)

        if self.output_layer == 'layer4':
            x = visual.layer4(x)

        
        return x
    
    def forward_vit(self, x: torch.Tensor) -> torch.Tensor:
        """Forward per CLIP ViT - estrae feature dense."""
        B, C, H, W = x.shape
        visual = self.visual
        
        # Patch embedding
        x = visual.conv1(x)
        grid_h, grid_w = x.shape[2], x.shape[3]
        
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        # Class token
        class_token = visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([class_token, x], dim=1)
        
        # Positional embedding
        pos_embed = visual.positional_embedding.to(x.dtype)
        # Interpolazione pos embedding se size diversa
        if pos_embed.shape[0] != x.shape[1]:
            # Semplificazione: assume grid quadrata per resize
            pass 
        x = x + pos_embed
        
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        
        x = x[:, 1:, :]
        x = x.permute(0, 2, 1)
        x = x.reshape(B, -1, grid_h, grid_w)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_name.startswith('RN'):
            return self.forward_resnet(x)
        else:
            return self.forward_vit(x)


class CLIPEBCModel(nn.Module):
    """
    Modello CLIP-EBC completo per crowd counting.
    ALLINEATO AL REPO UFFICIALE: Include Decoder e Bias=True.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Parametri CLIP
        ebc_cfg = config.get('CLIP_EBC_HEAD', {})
        clip_model = ebc_cfg.get('CLIP_MODEL', 'RN50')
        clip_pretrained = ebc_cfg.get('PRETRAINED', 'openai')
        
        # Normalizza nome modello
        model_map = {
            'ViT-B-16': 'ViT-B-16', 'ViT-B-32': 'ViT-B-32', 'ViT-L-14': 'ViT-L-14',
            'RN50': 'RN50', 'RN101': 'RN101',
            'ResNet50': 'RN50', 'resnet50': 'RN50',
        }
        clip_model = model_map.get(clip_model, clip_model)
        
        # Determina output layer
        block_size = config.get('DATA', {}).get('ZIP_BLOCK_SIZE', 16)

        if block_size == 8:
            output_layer = 'layer2'   # stride 8
        elif block_size == 16:
            output_layer = 'layer3'   # stride 16
        else:
            output_layer = 'layer4'   # stride 32 (fallback)

        
        # 1. Visual Encoder (Frozen o no)
        self.visual_encoder = CLIPVisualEncoder(
            model_name=clip_model,
            pretrained=clip_pretrained,
            frozen=False,
            output_layer=output_layer,
        )
        
        self.reduction = self.visual_encoder.reduction
        self.visual_dim = self.visual_encoder.out_channels # es. 1024
        
        # 2. Text Encoder (Frozen)
        self.clip_model_full, _, _ = open_clip.create_model_and_transforms(
        clip_model, pretrained=clip_pretrained, force_quick_gelu=True
        )
        self.tokenizer = open_clip.get_tokenizer(clip_model)
        
        for param in self.clip_model_full.parameters():
            param.requires_grad = False
            
        # 3. Bins & Centers
        raw_bins = config.get('BINS', [])
        bin_centers = config.get('BIN_CENTERS', [])
        if not raw_bins or not bin_centers:
            raise ValueError("BINS e BIN_CENTERS devono essere definiti!")
        
        self.bins = [tuple(b) for b in raw_bins]
        self.num_bins = len(self.bins)
        self.register_buffer('bin_centers', torch.tensor(bin_centers, dtype=torch.float32))
        
        # 4. Prompts (Official Logic)
        self.prompt_type = ebc_cfg.get('PROMPT_TYPE', 'word')
        self.prompts = self._generate_prompts_official() # Usa il metodo nuovo
        self._text_embeddings = None
        
        # --- FIX UFFICIALE 1: DECODER ---
        # Aggiunge un blocco Bottleneck che porta i canali a 2048 (se RN50)
        # Questo è CRUCIALE per replicare le performance.
        if clip_model == 'RN50':
            self.decoder_dim = 2048
            self.image_decoder = nn.Sequential(
                Bottleneck(
                    inplanes=self.visual_dim, # 1024
                    planes=512,               # 512 * expansion(4) = 2048
                    stride=1,
                    downsample=nn.Sequential(
                        nn.Conv2d(self.visual_dim, self.decoder_dim, kernel_size=1, stride=1, bias=False),
                        nn.BatchNorm2d(self.decoder_dim),
                    )
                )
            )
            # Inizializzazione pesi decoder
            for m in self.image_decoder.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            # Per ViT o altri, controlla il repo ufficiale (solitamente Identity o diverso)
            self.image_decoder = nn.Identity()
            self.decoder_dim = self.visual_dim

        # --- FIX UFFICIALE 2: PROJECTION CON BIAS ---
        text_dim = self.clip_model_full.text_projection.shape[1]
        
        self.visual_projection = nn.Conv2d(
            self.decoder_dim,  # Input è ora l'output del decoder (2048)
            text_dim, 
            kernel_size=1, 
            bias=True  # IMPORTANTE: Ufficiale usa bias=True
        )
        # Inizializzazione corretta
        nn.init.kaiming_normal_(self.visual_projection.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.visual_projection.bias, 0)
        
        print(f"✅ CLIPEBCModel Allineato all'Ufficiale:")
        print(f"   Visual Dim: {self.visual_dim} -> Decoder Dim: {self.decoder_dim} -> Text Dim: {text_dim}")
        print(f"   Bias in Projection: True")
        
        # Temperature
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.07)))
    
    def _generate_prompts_official(self) -> List[str]:
        """Genera prompt usando il dizionario esteso e la logica ufficiale."""
        
        # Dizionario esteso (copiato dall'ufficiale)
        NUM_TO_WORD = {
            0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
            5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
            10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
            14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
            18: "eighteen", 19: "nineteen", 20: "twenty", 21: "twenty-one",
            22: "twenty-two", 23: "twenty-three", 24: "twenty-four", 25: "twenty-five",
            30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 
            70: "seventy", 80: "eighty", 90: "ninety", 100: "one hundred"
        }
        
        def num2word(n):
            if n in NUM_TO_WORD: return NUM_TO_WORD[n]
            return str(n) # Fallback semplice

        prompts = []
        for min_count, max_count in self.bins:
            if min_count == max_count:
                if min_count == 0:
                    prompts.append("There is no person.")
                elif min_count == 1:
                    prompts.append("There is one person.")
                else:
                    word = num2word(min_count)
                    prompts.append(f"There are {word} people.")
            elif max_count >= 9999:
                # FIX LOGICA: Ufficiale usa 'min_count', non 'min_count - 1'
                word = num2word(min_count) 
                prompts.append(f"There are more than {word} people.")
            else:
                w_min = num2word(min_count)
                w_max = num2word(max_count)
                prompts.append(f"There are between {w_min} and {w_max} people.")
        
        return prompts
    
    @torch.no_grad()
    def _encode_text(self, prompts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(prompts)
        device = next(self.clip_model_full.parameters()).device
        tokens = tokens.to(device)
        text_features = self.clip_model_full.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)
        return text_features
    
    def _ensure_text_embeddings(self, device: torch.device):
        if self._text_embeddings is None or self._text_embeddings.device != device:
            self._text_embeddings = self._encode_text(self.prompts).to(device)
    
    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.shape[0]
        device = x.device
        
        # 1. Visual Encoder
        x = self.visual_encoder(x)
        
        # 2. Decoder (Nuovo!)
        x = self.image_decoder(x)
        
        # 3. Projection
        visual_features = self.visual_projection(x)
        visual_features_norm = F.normalize(visual_features, dim=1)
        
        # 4. Text Similarity
        self._ensure_text_embeddings(device)
        text_embeddings = self._text_embeddings
        
        _, C, H, W = visual_features_norm.shape
        visual_flat = visual_features_norm.permute(0, 2, 3, 1).reshape(B * H * W, -1)
        
        logits = torch.matmul(visual_flat, text_embeddings.T) / self.temperature
        logits = logits.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        # 5. Density & Count
        bin_probs = F.softmax(logits, dim=1)
        centers = self.bin_centers.view(1, -1, 1, 1)
        ebc_density = (bin_probs * centers).sum(dim=1, keepdim=True)
        final_count = ebc_density.sum(dim=[1, 2, 3])
        
        return {
            'ebc_density': ebc_density,
            'ebc_logits': logits,
            'bin_probs': bin_probs,
            'final_count': final_count,
            'visual_features': visual_features,
        }