# ============================================================
# ZIP-CLIP-EBC: CLIP-EBC Head
# ============================================================
# Expected Bin Counting Head basato su CLIP.
#
# Idea: Invece di fare regressione diretta del conteggio,
# classifichiamo ogni blocco in "bins" di conteggio usando
# la similarity tra feature visive e prompt testuali CLIP.
#
# Es. prompts: "There is no person", "There is one person",
#              "There are two people", ..., "There are more than 15 people"
#
# Riferimento: CLIP-EBC (Yiming-M/CLIP-EBC)
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


# ============================================================
# Utility per generare i prompt testuali
# ============================================================

NUM_TO_WORD = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
    14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
    18: "eighteen", 19: "nineteen", 20: "twenty",
}


def num_to_word(n: int) -> str:
    """Converte un numero in parola inglese."""
    if n in NUM_TO_WORD:
        return NUM_TO_WORD[n]
    return str(n)


def generate_count_prompt(count: int, prompt_type: str = "word") -> str:
    """
    Genera il prompt testuale per un dato conteggio.
    
    Args:
        count: Numero di persone
        prompt_type: "word" (usa parole) o "number" (usa cifre)
        
    Returns:
        Prompt testuale
    """
    if count == 0:
        return "There is no person."
    elif count == 1:
        return "There is one person."
    else:
        if prompt_type == "word" and count <= 20:
            return f"There are {num_to_word(count)} people."
        else:
            return f"There are {count} people."


def generate_bin_prompts(
    bins: List[Tuple[int, int]],
    prompt_type: str = "word"
) -> List[str]:
    """
    Genera i prompt per ogni bin di conteggio.
    
    Args:
        bins: Lista di tuple (min_count, max_count) per ogni bin
              Es. [(0,0), (1,1), (2,2), ..., (10,10), (11,15), (16,9999)]
        prompt_type: "word" o "number"
        
    Returns:
        Lista di prompt testuali
    """
    prompts = []
    
    for min_count, max_count in bins:
        if min_count == max_count:
            # Bin singolo: "There are five people."
            prompts.append(generate_count_prompt(min_count, prompt_type))
        elif max_count >= 9999:
            # Ultimo bin: "There are more than N people."
            if prompt_type == "word" and min_count <= 20:
                prompts.append(f"There are more than {num_to_word(min_count - 1)} people.")
            else:
                prompts.append(f"There are more than {min_count - 1} people.")
        else:
            # Range: "There are between N and M people."
            if prompt_type == "word" and max_count <= 20:
                prompts.append(f"There are between {num_to_word(min_count)} and {num_to_word(max_count)} people.")
            else:
                prompts.append(f"There are between {min_count} and {max_count} people.")
    
    return prompts


# ============================================================
# Visual Feature Projector
# ============================================================

class VisualProjector(nn.Module):
    """
    Proietta le feature VGG nello spazio CLIP.
    
    Il backbone VGG produce feature di dimensione 512.
    CLIP usa embedding di dimensione 512 (per ViT-B) o 768 (per ViT-L).
    
    Questo modulo proietta le feature VGG nello spazio comune
    per poter calcolare la similarity con i text embeddings.
    """
    
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
        
        # Layer finale (senza activation per preservare range)
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
        """
        Args:
            x: [B, C_in, H, W]
        Returns:
            [B, C_out, H, W] - feature proiettate
        """
        return self.projector(x)


# ============================================================
# CLIP Text Encoder Wrapper
# ============================================================

class CLIPTextEncoder(nn.Module):
    """
    Wrapper per il text encoder di CLIP.
    
    Carica il modello CLIP e usa solo il text encoder per
    generare gli embedding dei prompt di conteggio.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
    ):
        super().__init__()
        
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError("open_clip non disponibile. Installa con: pip install open-clip-torch")
        
        # Carica CLIP
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # Dimensione embedding
        self.embed_dim = self.clip_model.text_projection.shape[1]
        
        # Congela il text encoder (non viene trainato)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Cache per text embeddings
        self._text_embeddings_cache = None
        self._cached_prompts = None
        
        print(f"✅ CLIPTextEncoder inizializzato:")
        print(f"   Model: {model_name} ({pretrained})")
        print(f"   Embedding dim: {self.embed_dim}")
    
    @torch.no_grad()
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        """
        Codifica una lista di prompt testuali.
        
        Args:
            prompts: Lista di stringhe
            
        Returns:
            text_embeddings: [num_prompts, embed_dim] normalizzati
        """
        # Tokenizza
        tokens = self.tokenizer(prompts)
        device = next(self.clip_model.parameters()).device
        tokens = tokens.to(device)
        
        # Encode
        text_features = self.clip_model.encode_text(tokens)
        
        # Normalizza
        text_features = F.normalize(text_features, dim=-1)
        
        return text_features
    
    def get_text_embeddings(
        self,
        prompts: List[str],
        force_recompute: bool = False
    ) -> torch.Tensor:
        """
        Ottiene text embeddings con caching.
        """
        if not force_recompute and self._text_embeddings_cache is not None:
            if self._cached_prompts == prompts:
                return self._text_embeddings_cache
        
        self._text_embeddings_cache = self.encode_text(prompts)
        self._cached_prompts = prompts
        
        return self._text_embeddings_cache


# ============================================================
# CLIP-EBC Head
# ============================================================

class CLIPEBCHead(nn.Module):
    """
    CLIP-based Expected Bin Counting Head.
    
    Per ogni blocco spaziale:
    1. Proietta le feature VGG nello spazio CLIP
    2. Calcola similarity con tutti i text embeddings dei bins
    3. Applica softmax per ottenere distribuzione sui bins
    4. Calcola expected count come media pesata dei bin centers
    
    Args:
        in_channels: Canali delle feature in input (dal backbone)
        clip_model: Nome modello CLIP ("ViT-B-16", "ViT-L-14", etc.)
        clip_pretrained: Pesi pretrained ("openai", "laion2b", etc.)
        bins: Lista di tuple (min, max) per ogni bin
        bin_centers: Valori centrali per ogni bin (per calcolare expected count)
        temperature: Temperature per softmax (learnable)
        prompt_type: "word" o "number"
    """
    
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
        
        # Default bins (come in CLIP-EBC paper)
        if bins is None:
            bins = [(0, 0)] + [(i, i) for i in range(1, 11)] + [(11, 15), (16, 9999)]
        if bin_centers is None:
            bin_centers = [0.0] + [float(i) for i in range(1, 11)] + [13.0, 20.0]
        
        self.bins = bins
        self.num_bins = len(bins)
        self.prompt_type = prompt_type
        
        # Registra bin centers come buffer
        self.register_buffer(
            "bin_centers",
            torch.tensor(bin_centers, dtype=torch.float32)
        )
        
        # CLIP Text Encoder
        self.text_encoder = CLIPTextEncoder(
            model_name=clip_model,
            pretrained=clip_pretrained
        )
        clip_dim = self.text_encoder.embed_dim
        
        # Visual Projector (VGG → CLIP space)
        self.visual_projector = VisualProjector(
            in_dim=in_channels,
            out_dim=clip_dim,
            hidden_dim=clip_dim,
            num_layers=2,
        )
        
        # Temperature
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))
        else:
            self.register_buffer("log_temperature", torch.tensor(math.log(temperature)))
        
        # Genera prompt e codificali
        self.prompts = generate_bin_prompts(bins, prompt_type)
        
        # Pre-calcola text embeddings (verranno ricalcolati al primo forward)
        self._text_embeddings = None
        
        print(f"✅ CLIPEBCHead inizializzato:")
        print(f"   Input channels: {in_channels}")
        print(f"   CLIP dim: {clip_dim}")
        print(f"   Num bins: {self.num_bins}")
        print(f"   Temperature: {temperature} (learnable={learnable_temperature})")
        print(f"   Prompts[0]: {self.prompts[0]}")
        print(f"   Prompts[-1]: {self.prompts[-1]}")
    
    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)
    
    def _ensure_text_embeddings(self, device: torch.device):
        """Assicura che i text embeddings siano sul device corretto."""
        if self._text_embeddings is None or self._text_embeddings.device != device:
            self._text_embeddings = self.text_encoder.get_text_embeddings(self.prompts)
            self._text_embeddings = self._text_embeddings.to(device)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Feature map dal backbone [B, C, H, W]
            
        Returns:
            dict con:
                - visual_features: [B, clip_dim, H, W] - feature proiettate
                - logits: [B, num_bins, H, W] - similarity scores
                - bin_probs: [B, num_bins, H, W] - distribuzione sui bins
                - expected_count: [B, 1, H, W] - conteggio atteso per blocco
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Proietta feature nello spazio CLIP
        visual_features = self.visual_projector(x)  # [B, clip_dim, H, W]
        
        # Normalizza feature visive
        visual_features_norm = F.normalize(visual_features, dim=1)  # [B, clip_dim, H, W]
        
        # Assicura text embeddings
        self._ensure_text_embeddings(device)
        text_embeddings = self._text_embeddings  # [num_bins, clip_dim]
        
        # Reshape per calcolare similarity
        # visual: [B, clip_dim, H, W] -> [B, H*W, clip_dim]
        visual_flat = visual_features_norm.permute(0, 2, 3, 1).reshape(B * H * W, -1)
        
        # Cosine similarity: [B*H*W, num_bins]
        logits = torch.matmul(visual_flat, text_embeddings.T) / self.temperature
        
        # Reshape back: [B, num_bins, H, W]
        logits = logits.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        # Softmax per distribuzione sui bins
        bin_probs = F.softmax(logits, dim=1)  # [B, num_bins, H, W]
        
        # Expected count: somma pesata dei bin centers
        # bin_centers: [num_bins] -> [1, num_bins, 1, 1]
        centers = self.bin_centers.view(1, -1, 1, 1)
        expected_count = (bin_probs * centers).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        
        return {
            "visual_features": visual_features,
            "logits": logits,
            "bin_probs": bin_probs,
            "expected_count": expected_count,
        }
    
    def get_predicted_bins(self, x: torch.Tensor) -> torch.Tensor:
        """
        Restituisce il bin predetto (argmax) per ogni blocco.
        
        Args:
            x: Feature map [B, C, H, W]
            
        Returns:
            predicted_bins: [B, H, W] - indice del bin predetto
        """
        outputs = self.forward(x)
        return outputs["logits"].argmax(dim=1)


# ============================================================
# Factory Function
# ============================================================

def build_clip_ebc_head(
    in_channels: int = 512,
    clip_model: str = "ViT-B-16",
    clip_pretrained: str = "openai",
    bins_config: Optional[Dict] = None,
    temperature: float = 0.07,
    learnable_temperature: bool = True,
    prompt_type: str = "word",
) -> CLIPEBCHead:
    """
    Factory function per costruire CLIP-EBC Head.
    
    Args:
        in_channels: Canali input dal backbone
        clip_model: Nome modello CLIP
        clip_pretrained: Pesi pretrained
        bins_config: Dict con "bins" e "bin_centers"
        temperature: Temperature iniziale
        learnable_temperature: Se la temperature è learnable
        prompt_type: "word" o "number"
        
    Returns:
        CLIPEBCHead instance
    """
    bins = None
    bin_centers = None
    
    if bins_config is not None:
        bins = bins_config.get("bins")
        bin_centers = bins_config.get("bin_centers")
    
    return CLIPEBCHead(
        in_channels=in_channels,
        clip_model=clip_model,
        clip_pretrained=clip_pretrained,
        bins=bins,
        bin_centers=bin_centers,
        temperature=temperature,
        learnable_temperature=learnable_temperature,
        prompt_type=prompt_type,
    )


if __name__ == "__main__":
    # Test
    print("Testing CLIP-EBC Head...")
    
    if not OPEN_CLIP_AVAILABLE:
        print("❌ open_clip non disponibile, skip test")
    else:
        # Test prompts generation
        bins = [(0, 0)] + [(i, i) for i in range(1, 6)] + [(6, 10), (11, 9999)]
        prompts = generate_bin_prompts(bins, prompt_type="word")
        print("\nGenerated prompts:")
        for i, p in enumerate(prompts):
            print(f"  Bin {i}: {p}")
        
        # Test head
        B, C, H, W = 2, 512, 16, 16
        x = torch.randn(B, C, H, W)
        
        bin_centers = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 15.0]
        
        head = CLIPEBCHead(
            in_channels=C,
            bins=bins,
            bin_centers=bin_centers,
        )
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        head = head.to(device)
        x = x.to(device)
        
        outputs = head(x)
        
        print("\nOutputs:")
        for key, val in outputs.items():
            print(f"  {key}: {val.shape}")
        
        print(f"\n  Bin probs sum: {outputs['bin_probs'].sum(dim=1).mean():.4f} (should be ~1.0)")
        print(f"  Expected count range: [{outputs['expected_count'].min():.2f}, {outputs['expected_count'].max():.2f}]")
        print(f"  Temperature: {head.temperature.item():.4f}")
        
        print("\n✅ Test completato!")
