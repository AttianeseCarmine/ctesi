# models/clip_ebc_head.py
# ============================================================
# ZIP-CLIP-EBC: CLIP-EBC Head (Official Prompts & Builder)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    print("⚠️ open_clip non installato. Installa con: pip install open-clip-torch")

# --- UTILS PROMPT UFFICIALI ---
NUM_TO_WORD = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
    14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
    18: "eighteen", 19: "nineteen", 20: "twenty", 
    30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 
    70: "seventy", 80: "eighty", 90: "ninety", 100: "one hundred"
}

def num_to_word(n: int) -> str:
    """Converte un numero in parola (semplificato)."""
    return NUM_TO_WORD.get(n, str(n))

def generate_bin_prompts(bins: List[Tuple[int, int]], prompt_type: str = "word") -> List[str]:
    """Genera i prompt ufficiali per ogni bin."""
    prompts = []
    
    for min_count, max_count in bins:
        if min_count == max_count:
            # Bin singolo
            if min_count == 0:
                prompts.append("There is no person.")
            elif min_count == 1:
                prompts.append("There is one person.")
            else:
                prompts.append(f"There are {num_to_word(min_count)} people.")
        
        elif max_count >= 9999:
            # Ultimo bin: "More than N" (Ufficiale usa min_count, non min_count-1)
            prompts.append(f"There are more than {num_to_word(min_count)} people.")
        
        else:
            # Range
            prompts.append(f"There are between {num_to_word(min_count)} and {num_to_word(max_count)} people.")
    
    return prompts


# ============================================================
# Visual Feature Projector
# ============================================================

class VisualProjector(nn.Module):
    """Proietta le feature VGG nello spazio CLIP."""
    def __init__(
        self,
        in_dim: int = 512,
        out_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        layers = []
        current_dim = in_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Conv2d(current_dim, hidden_dim, kernel_size=1),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Dropout2d(dropout),
            ])
            current_dim = hidden_dim
        
        # Layer finale
        layers.extend([
            nn.Conv2d(current_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
        ])
        
        self.projector = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


# ============================================================
# CLIP Text Encoder Wrapper
# ============================================================

class CLIPTextEncoder(nn.Module):
    """Wrapper per il text encoder di CLIP."""
    def __init__(self, model_name: str = "ViT-B-16", pretrained: str = "openai"):
        super().__init__()
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError("open_clip non disponibile. Installa con: pip install open-clip-torch")
        
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.embed_dim = self.clip_model.text_projection.shape[1]
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self._text_embeddings_cache = None
        self._cached_prompts = None
    
    @torch.no_grad()
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(prompts)
        device = next(self.clip_model.parameters()).device
        tokens = tokens.to(device)
        text_features = self.clip_model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)
        return text_features
    
    def get_text_embeddings(self, prompts: List[str], force_recompute: bool = False) -> torch.Tensor:
        if not force_recompute and self._text_embeddings_cache is not None:
            if self._cached_prompts == prompts:
                return self._text_embeddings_cache
        self._text_embeddings_cache = self.encode_text(prompts)
        self._cached_prompts = prompts
        return self._text_embeddings_cache


# ============================================================
# CLIP-EBC Head (Standalone)
# ============================================================

class CLIPEBCHead(nn.Module):
    """CLIP-based Expected Bin Counting Head."""
    def __init__(
        self,
        in_channels: int = 512,
        clip_model: str = "ViT-B-16",
        clip_pretrained: str = "openai",
        bins: Optional[List[Tuple[int, int]]] = None,
        bin_centers: Optional[List[float]] = None,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
        prompt_type: str = "word",
    ):
        super().__init__()
        
        if bins is None:
            bins = [(0, 0)] + [(i, i) for i in range(1, 11)] + [(11, 15), (16, 9999)]
        if bin_centers is None:
            bin_centers = [0.0] + [float(i) for i in range(1, 11)] + [13.0, 20.0]
        
        self.bins = bins
        self.num_bins = len(bins)
        self.prompt_type = prompt_type
        
        self.register_buffer("bin_centers", torch.tensor(bin_centers, dtype=torch.float32))
        
        self.text_encoder = CLIPTextEncoder(model_name=clip_model, pretrained=clip_pretrained)
        clip_dim = self.text_encoder.embed_dim
        
        self.visual_projector = VisualProjector(
            in_dim=in_channels,
            out_dim=clip_dim,
            hidden_dim=clip_dim,
            num_layers=2,
        )
        
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))
        else:
            self.register_buffer("log_temperature", torch.tensor(math.log(temperature)))
        
        self.prompts = generate_bin_prompts(bins, prompt_type)
        self._text_embeddings = None
        
        print(f"✅ CLIPEBCHead initialized: {in_channels} -> {clip_dim}, {self.num_bins} bins")
    
    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)
    
    def _ensure_text_embeddings(self, device: torch.device):
        if self._text_embeddings is None or self._text_embeddings.device != device:
            self._text_embeddings = self.text_encoder.get_text_embeddings(self.prompts).to(device)
            
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.shape[0]
        device = x.device
        
        visual_features = self.visual_projector(x)
        visual_features_norm = F.normalize(visual_features, dim=1)
        
        self._ensure_text_embeddings(device)
        text_embeddings = self._text_embeddings
        
        _, C, H, W = visual_features_norm.shape
        visual_flat = visual_features_norm.permute(0, 2, 3, 1).reshape(B * H * W, -1)
        
        logits = torch.matmul(visual_flat, text_embeddings.T) / self.temperature
        logits = logits.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        bin_probs = F.softmax(logits, dim=1)
        centers = self.bin_centers.view(1, -1, 1, 1)
        ebc_density = (bin_probs * centers).sum(dim=1, keepdim=True)
        final_count = ebc_density.sum(dim=[1, 2, 3])
        
        return {
            'ebc_density': ebc_density,
            'ebc_logits': logits,
            'bin_probs': bin_probs,
            'final_count': final_count
        }

# --- FIX: Builder Function ---
def build_clip_ebc_head(config: Dict, in_channels: int) -> CLIPEBCHead:
    """Factory function per costruire la head."""
    cfg = config.get('CLIP_EBC_HEAD', {})
    
    return CLIPEBCHead(
        in_channels=in_channels,
        clip_model=cfg.get('CLIP_MODEL', 'ViT-B-16'),
        clip_pretrained=cfg.get('PRETRAINED', 'openai'),
        bins=config.get('BINS'),
        bin_centers=config.get('BIN_CENTERS'),
        temperature=cfg.get('TEMPERATURE', 0.07),
        learnable_temperature=cfg.get('LEARNABLE_TEMP', True),
        prompt_type=cfg.get('PROMPT_TYPE', 'word'),
    )