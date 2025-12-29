# ============================================================
# CLIP Visual Encoder per Crowd Counting - FIXED v2
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import math

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
            model_name, pretrained=pretrained
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
        print(f"   Reduction: {self.reduction}")
        print(f"   Frozen: {frozen}")
    
    def _setup_resnet(self):
        """Setup per CLIP ModifiedResNet."""
        if self.output_layer == 'layer3':
            self.out_channels = 1024
            self.reduction = 16
        else:
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
        
        # Stem: CLIP ModifiedResNet usa F.relu, non un modulo relu
        # La struttura è: conv1->bn1, conv2->bn2, conv3->bn3, avgpool
        x = F.relu(visual.bn1(visual.conv1(x)), inplace=True)
        x = F.relu(visual.bn2(visual.conv2(x)), inplace=True)
        x = F.relu(visual.bn3(visual.conv3(x)), inplace=True)
        x = visual.avgpool(x)
        
        # ResNet layers
        x = visual.layer1(x)
        x = visual.layer2(x)
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
        if pos_embed.shape[0] != x.shape[1]:
            pos_embed = self._resize_pos_embed(pos_embed, grid_h, grid_w)
        x = x + pos_embed
        
        x = visual.ln_pre(x)
        
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        
        x = x[:, 1:, :]
        x = x.permute(0, 2, 1)
        x = x.reshape(B, -1, grid_h, grid_w)
        
        return x
    
    def _resize_pos_embed(self, pos_embed, grid_h, grid_w):
        """Resize positional embedding."""
        cls_pos = pos_embed[:1, :]
        patch_pos = pos_embed[1:, :]
        
        old_grid_size = int(math.sqrt(patch_pos.shape[0]))
        patch_pos = patch_pos.reshape(1, old_grid_size, old_grid_size, -1)
        patch_pos = patch_pos.permute(0, 3, 1, 2)
        
        patch_pos = F.interpolate(
            patch_pos, size=(grid_h, grid_w),
            mode='bilinear', align_corners=False
        )
        
        patch_pos = patch_pos.permute(0, 2, 3, 1)
        patch_pos = patch_pos.reshape(1, grid_h * grid_w, -1)
        
        return torch.cat([cls_pos.unsqueeze(0), patch_pos], dim=1).squeeze(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_name.startswith('RN'):
            return self.forward_resnet(x)
        else:
            return self.forward_vit(x)


class CLIPEBCModel(nn.Module):
    """
    Modello CLIP-EBC completo per crowd counting.
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
        output_layer = 'layer3' if block_size == 16 else 'layer4'
        
        # Visual Encoder
        self.visual_encoder = CLIPVisualEncoder(
            model_name=clip_model,
            pretrained=clip_pretrained,
            frozen=False,
            output_layer=output_layer,
        )
        
        # Text Encoder
        self.clip_model_full, _, _ = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(clip_model)
        
        for param in self.clip_model_full.parameters():
            param.requires_grad = False
        
        # Bins
        raw_bins = config.get('BINS', [])
        bin_centers = config.get('BIN_CENTERS', [])
        
        if not raw_bins or not bin_centers:
            raise ValueError("BINS e BIN_CENTERS devono essere definiti!")
        
        self.bins = [tuple(b) for b in raw_bins]
        self.num_bins = len(self.bins)
        self.register_buffer('bin_centers', torch.tensor(bin_centers, dtype=torch.float32))
        
        # Prompts
        self.prompt_type = ebc_cfg.get('PROMPT_TYPE', 'word')
        self.prompts = self._generate_prompts()
        self._text_embeddings = None
        
        # Projection
        visual_dim = self.visual_encoder.out_channels
        text_dim = self.clip_model_full.text_projection.shape[1]
        
        if visual_dim != text_dim:
            self.visual_projection = nn.Conv2d(visual_dim, text_dim, kernel_size=1, bias=False)
            nn.init.kaiming_normal_(self.visual_projection.weight)
            print(f"   Visual projection: {visual_dim} → {text_dim}")
        else:
            self.visual_projection = nn.Identity()
        
        # Temperature
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.07)))
        self.reduction = self.visual_encoder.reduction
        
        print(f"✅ CLIPEBCModel inizializzato:")
        print(f"   Num bins: {self.num_bins}")
        print(f"   Reduction: {self.reduction}")
        print(f"   Prompts: {self.prompts[0]} ... {self.prompts[-1]}")
    
    def _generate_prompts(self) -> List[str]:
        NUM_TO_WORD = {
            0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
            5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
            10: "ten", 11: "eleven", 12: "twelve",
        }
        
        prompts = []
        for min_count, max_count in self.bins:
            if min_count == max_count:
                if min_count == 0:
                    prompts.append("There is no person.")
                elif min_count == 1:
                    prompts.append("There is one person.")
                else:
                    word = NUM_TO_WORD.get(min_count, str(min_count))
                    prompts.append(f"There are {word} people.")
            elif max_count >= 9999:
                word = NUM_TO_WORD.get(min_count - 1, str(min_count - 1))
                prompts.append(f"There are more than {word} people.")
            else:
                prompts.append(f"There are between {min_count} and {max_count} people.")
        
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
        
        # Visual features
        visual_features = self.visual_encoder(x)
        visual_features = self.visual_projection(visual_features)
        visual_features_norm = F.normalize(visual_features, dim=1)
        
        # Text embeddings
        self._ensure_text_embeddings(device)
        text_embeddings = self._text_embeddings
        
        # Cosine similarity
        _, C, H, W = visual_features_norm.shape
        visual_flat = visual_features_norm.permute(0, 2, 3, 1).reshape(B * H * W, -1)
        
        logits = torch.matmul(visual_flat, text_embeddings.T) / self.temperature
        logits = logits.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        # Bin probabilities e expected count
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
    
    def forward_stage2(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.forward(x)
        outputs['logits'] = outputs['ebc_logits']
        return outputs


def build_clip_ebc_model(config: Dict) -> CLIPEBCModel:
    return CLIPEBCModel(config)


if __name__ == "__main__":
    print("Testing CLIPEBCModel...")
    
    config = {
        'CLIP_EBC_HEAD': {'CLIP_MODEL': 'RN50', 'PRETRAINED': 'openai', 'PROMPT_TYPE': 'word'},
        'DATA': {'ZIP_BLOCK_SIZE': 16},
        'BINS': [[0, 0], [1, 1], [2, 2], [3, 3], [4, 9999]],
        'BIN_CENTERS': [0.0, 1.0, 2.0, 3.0, 17.0],
    }
    
    model = CLIPEBCModel(config)
    x = torch.randn(2, 3, 448, 448)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x = x.to(device)
    
    with torch.no_grad():
        outputs = model(x)
    
    print("\nOutputs:")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
    
    print(f"\n  Predicted counts: {outputs['final_count'].tolist()}")
    print("\n✅ Test completato!")
