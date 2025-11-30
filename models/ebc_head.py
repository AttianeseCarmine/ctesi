# ============================================================
# ZIP-CLIP-EBC: EBC Head (CLIP-based Counting)
# ============================================================
# Classifica ogni blocco in bins di conteggio usando CLIP.
# Confronta le feature visive con prompt testuali per ogni bin.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


class FeatureRefiner(nn.Module):
    """
    Raffina le feature visive prima del matching con il testo.
    Può aiutare ad adattare le feature CLIP al task di counting.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        out_dim = out_dim or in_dim
        
        layers = []
        current_dim = in_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim
        
        # Layer finale
        layers.extend([
            nn.Linear(current_dim, out_dim),
            nn.LayerNorm(out_dim),
        ])
        
        self.refiner = nn.Sequential(*layers)
        
        # Residual connection se le dimensioni sono compatibili
        self.use_residual = (in_dim == out_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] o [B, D, H, W]
        Returns:
            refined: stessa shape di input
        """
        input_shape = x.shape
        is_4d = (x.dim() == 4)
        
        if is_4d:
            B, D, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B * H * W, D)  # [B*H*W, D]
        
        refined = self.refiner(x)
        
        if self.use_residual:
            refined = refined + x
        
        if is_4d:
            refined = refined.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        return refined


class EBCHead(nn.Module):
    """
    EBC (Expected Bin Counting) Head basato su CLIP.
    
    Per ogni blocco:
    1. Prende le feature visive (dal backbone CLIP)
    2. Le confronta con le text features dei prompts di conteggio
    3. Calcola la similarity per ottenere una distribuzione sui bins
    4. Calcola il conteggio atteso come valore atteso sui bin centers
    
    Args:
        visual_dim: Dimensione delle feature visive
        text_dim: Dimensione delle feature testuali
        bin_centers: Lista dei centri dei bins (es. [1, 2, 3, ..., 17])
        temperature: Temperature per il softmax
        learnable_temp: Se la temperature è apprendibile
        use_refiner: Se usare il refiner per le feature visive
        refiner_hidden: Dimensione hidden del refiner
    """
    
    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        bin_centers: List[float],
        temperature: float = 0.07,
        learnable_temp: bool = True,
        use_refiner: bool = True,
        refiner_hidden: int = 256,
    ):
        super().__init__()
        
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.num_bins = len(bin_centers)
        
        # Registra i bin centers come buffer (non parametro)
        self.register_buffer(
            "bin_centers",
            torch.tensor(bin_centers, dtype=torch.float32)
        )
        
        # Temperature
        if learnable_temp:
            # Inizializza con il log della temperature iniziale
            self.log_temp = nn.Parameter(torch.tensor(math.log(temperature)))
        else:
            self.register_buffer("log_temp", torch.tensor(math.log(temperature)))
        
        # Feature refiner (opzionale)
        self.use_refiner = use_refiner
        if use_refiner:
            self.refiner = FeatureRefiner(
                in_dim=visual_dim,
                hidden_dim=refiner_hidden,
                out_dim=text_dim,
                num_layers=2,
            )
        elif visual_dim != text_dim:
            # Proiezione lineare se le dimensioni non matchano
            self.proj = nn.Linear(visual_dim, text_dim)
            nn.init.eye_(self.proj.weight[:min(visual_dim, text_dim), :min(visual_dim, text_dim)])
        else:
            self.proj = nn.Identity()
        
        # Buffer per le text features (calcolate dal backbone)
        self.register_buffer("text_features", None)
        
        print(f"✅ EBCHead inizializzato:")
        print(f"   Visual dim: {visual_dim}")
        print(f"   Text dim: {text_dim}")
        print(f"   Num bins: {self.num_bins}")
        print(f"   Bin centers: {bin_centers[:5]}...{bin_centers[-3:]}")
        print(f"   Temperature: {temperature} (learnable={learnable_temp})")
        print(f"   Use refiner: {use_refiner}")
    
    @property
    def temperature(self) -> torch.Tensor:
        """Restituisce la temperature corrente."""
        return torch.exp(self.log_temp)
    
    def set_text_features(self, text_features: torch.Tensor):
        """
        Imposta le text features precalcolate.
        
        Args:
            text_features: [num_bins, text_dim] - normalizzate
        """
        if text_features.shape[0] != self.num_bins:
            raise ValueError(
                f"Numero di text features ({text_features.shape[0]}) "
                f"non corrisponde al numero di bins ({self.num_bins})"
            )
        
        # Assicurati che siano normalizzate
        text_features = F.normalize(text_features, dim=-1)
        
        # Registra come buffer (non parametro)
        self.register_buffer("text_features", text_features)
    
    def forward(
        self,
        visual_features: torch.Tensor,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            visual_features: [B, D, H, W] o [B, N, D] - feature visive
            return_logits: Se True, restituisce anche i logits raw
        
        Returns:
            lambda_map: [B, 1, H, W] - conteggio atteso per blocco
            bin_probs: [B, num_bins, H, W] - distribuzione sui bins
        """
        if self.text_features is None:
            raise RuntimeError(
                "Text features non impostate. Chiamare set_text_features() prima del forward."
            )
        
        is_4d = (visual_features.dim() == 4)
        
        if is_4d:
            B, D, H, W = visual_features.shape
            # Reshape per il processing
            visual_features = visual_features.permute(0, 2, 3, 1)  # [B, H, W, D]
            spatial_shape = (H, W)
        else:
            B, N, D = visual_features.shape
            spatial_shape = None
        
        # Raffina/proietta le feature visive
        if self.use_refiner:
            visual_features = self.refiner(visual_features.reshape(-1, D))
            visual_features = visual_features.reshape(B, -1, self.text_dim)
        elif hasattr(self, 'proj'):
            original_shape = visual_features.shape
            visual_features = visual_features.reshape(-1, D)
            visual_features = self.proj(visual_features)
            visual_features = visual_features.reshape(*original_shape[:-1], -1)
        
        # Normalizza le feature visive
        visual_features = F.normalize(visual_features, dim=-1)  # [B, H*W, text_dim] o [B, N, text_dim]
        
        # Calcola similarity con le text features
        # text_features: [num_bins, text_dim]
        # visual_features: [B, N, text_dim]
        logits = torch.matmul(visual_features, self.text_features.T)  # [B, N, num_bins]
        
        # Scala per temperature
        logits = logits / self.temperature
        
        # Softmax per ottenere distribuzione sui bins
        bin_probs = F.softmax(logits, dim=-1)  # [B, N, num_bins]
        
        # Calcola il conteggio atteso come valore atteso sui bin centers
        # bin_centers: [num_bins]
        lambda_map = torch.matmul(bin_probs, self.bin_centers)  # [B, N]
        
        # Reshape all'output desiderato
        if is_4d:
            H, W = spatial_shape
            lambda_map = lambda_map.reshape(B, H, W, 1).permute(0, 3, 1, 2)  # [B, 1, H, W]
            bin_probs = bin_probs.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # [B, num_bins, H, W]
            if return_logits:
                logits = logits.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        else:
            lambda_map = lambda_map.unsqueeze(-1)  # [B, N, 1]
            if return_logits:
                pass  # logits already [B, N, num_bins]
        
        if return_logits:
            return lambda_map, bin_probs, logits
        
        return lambda_map, bin_probs
    
    def get_count_with_confidence(
        self,
        visual_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcola il conteggio atteso insieme a una misura di confidenza.
        
        La confidenza è basata sull'entropia della distribuzione sui bins:
        - Alta entropia = bassa confidenza (distribuzione uniforme)
        - Bassa entropia = alta confidenza (distribuzione peaked)
        
        Args:
            visual_features: [B, D, H, W]
        
        Returns:
            lambda_map: [B, 1, H, W] - conteggio atteso
            confidence: [B, 1, H, W] - confidenza (0-1)
        """
        lambda_map, bin_probs = self.forward(visual_features)
        
        # Calcola entropia normalizzata
        eps = 1e-8
        entropy = -torch.sum(bin_probs * torch.log(bin_probs + eps), dim=1, keepdim=True)
        max_entropy = math.log(self.num_bins)  # Entropia massima (distribuzione uniforme)
        
        # Confidenza = 1 - entropia_normalizzata
        confidence = 1.0 - (entropy / max_entropy)
        
        return lambda_map, confidence


class EBCHeadWithBinLogits(nn.Module):
    """
    Versione dell'EBC Head che restituisce anche i logits per ogni bin.
    Utile per la loss Cross-Entropy diretta sui bins.
    
    Questa versione è più simile all'architettura CLIP-EBC originale.
    """
    
    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        bins: List[Tuple[int, int]],
        bin_centers: List[float],
        temperature: float = 0.07,
        learnable_temp: bool = True,
        use_refiner: bool = True,
        refiner_hidden: int = 256,
    ):
        super().__init__()
        
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.bins = bins
        self.num_bins = len(bin_centers)
        
        # Il primo bin [0,0] è gestito da π, quindi i bin effettivi per EBC sono num_bins - 1
        # Ma se bin_centers include lo 0, lo escludiamo
        if bin_centers[0] == 0.0:
            self.ebc_bin_centers = bin_centers[1:]
            self.num_ebc_bins = len(bin_centers) - 1
        else:
            self.ebc_bin_centers = bin_centers
            self.num_ebc_bins = len(bin_centers)
        
        self.register_buffer(
            "bin_centers_tensor",
            torch.tensor(self.ebc_bin_centers, dtype=torch.float32)
        )
        
        # Temperature
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.tensor(math.log(temperature)))
        else:
            self.register_buffer("log_temp", torch.tensor(math.log(temperature)))
        
        # Feature refiner
        self.use_refiner = use_refiner
        if use_refiner:
            self.refiner = FeatureRefiner(
                in_dim=visual_dim,
                hidden_dim=refiner_hidden,
                out_dim=text_dim,
                num_layers=2,
            )
        elif visual_dim != text_dim:
            self.proj = nn.Linear(visual_dim, text_dim)
        else:
            self.proj = nn.Identity()
        
        # Buffer per text features
        self.register_buffer("text_features", None)
        
        print(f"✅ EBCHeadWithBinLogits inizializzato:")
        print(f"   Num EBC bins (escluso zero): {self.num_ebc_bins}")
    
    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temp)
    
    def set_text_features(self, text_features: torch.Tensor):
        """Imposta le text features (una per ogni bin EBC, escluso zero)."""
        if text_features.shape[0] != self.num_ebc_bins:
            raise ValueError(
                f"Attese {self.num_ebc_bins} text features, ricevute {text_features.shape[0]}"
            )
        text_features = F.normalize(text_features, dim=-1)
        self.register_buffer("text_features", text_features)
    
    def forward(self, visual_features: torch.Tensor) -> dict:
        """
        Forward pass.
        
        Args:
            visual_features: [B, D, H, W]
        
        Returns:
            dict con:
                - logit_bin_maps: [B, num_ebc_bins, H, W] - logits per CE loss
                - lambda_maps: [B, 1, H, W] - conteggio atteso
                - bin_probs: [B, num_ebc_bins, H, W] - distribuzione sui bins
        """
        if self.text_features is None:
            raise RuntimeError("Text features non impostate!")
        
        B, D, H, W = visual_features.shape
        
        # Reshape
        visual_flat = visual_features.permute(0, 2, 3, 1).reshape(B * H * W, D)
        
        # Raffina/proietta
        if self.use_refiner:
            visual_refined = self.refiner(visual_flat)
        elif hasattr(self, 'proj'):
            visual_refined = self.proj(visual_flat)
        else:
            visual_refined = visual_flat
        
        # Normalizza
        visual_refined = F.normalize(visual_refined, dim=-1)
        
        # Similarity
        logits = torch.matmul(visual_refined, self.text_features.T) / self.temperature
        logits = logits.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # [B, num_ebc_bins, H, W]
        
        # Probabilità
        bin_probs = F.softmax(logits, dim=1)
        
        # Lambda (conteggio atteso)
        # bin_centers_tensor: [num_ebc_bins]
        # bin_probs: [B, num_ebc_bins, H, W]
        lambda_maps = torch.einsum('bchw,c->bhw', bin_probs, self.bin_centers_tensor)
        lambda_maps = lambda_maps.unsqueeze(1)  # [B, 1, H, W]
        
        return {
            "logit_bin_maps": logits,
            "lambda_maps": lambda_maps,
            "bin_probs": bin_probs,
        }


def build_ebc_head(
    config: dict,
    visual_dim: int,
    text_dim: int,
) -> EBCHeadWithBinLogits:
    """
    Costruisce l'EBC Head dalla configurazione.
    """
    dataset_name = config.get("DATASET", "sha")
    bins_cfg = config.get("BINS_CONFIG", {}).get(dataset_name, {})
    ebc_cfg = config.get("EBC_HEAD", {})
    
    bins = bins_cfg.get("bins", [[0, 0], [1, 1], [2, 2]])
    bin_centers = bins_cfg.get("bin_centers", [0.0, 1.0, 2.0])
    
    return EBCHeadWithBinLogits(
        visual_dim=visual_dim,
        text_dim=text_dim,
        bins=bins,
        bin_centers=bin_centers,
        temperature=ebc_cfg.get("TEMPERATURE", 0.07),
        learnable_temp=ebc_cfg.get("LEARNABLE_TEMP", True),
        use_refiner=ebc_cfg.get("USE_REFINER", True),
        refiner_hidden=ebc_cfg.get("REFINER_HIDDEN", 256),
    )


if __name__ == "__main__":
    # Test
    print("Testing EBCHead...")
    
    B, D, H, W = 2, 512, 16, 16
    visual_features = torch.randn(B, D, H, W)
    
    # Simula text features (13 bins per conteggio)
    num_bins = 13
    text_dim = 512
    text_features = F.normalize(torch.randn(num_bins, text_dim), dim=-1)
    
    # Test EBCHead base
    bin_centers = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.5, 13.5, 17.0]
    
    ebc_head = EBCHead(
        visual_dim=D,
        text_dim=text_dim,
        bin_centers=bin_centers,
        use_refiner=True,
    )
    ebc_head.set_text_features(text_features)
    
    lambda_map, bin_probs = ebc_head(visual_features)
    print(f"Lambda map shape: {lambda_map.shape}")  # [2, 1, 16, 16]
    print(f"Bin probs shape: {bin_probs.shape}")  # [2, 13, 16, 16]
    print(f"Lambda range: [{lambda_map.min():.2f}, {lambda_map.max():.2f}]")
    print(f"Total count: {lambda_map.sum(dim=[1,2,3])}")
    
    # Test con confidenza
    lambda_map, confidence = ebc_head.get_count_with_confidence(visual_features)
    print(f"Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
    
    # Test EBCHeadWithBinLogits
    print("\nTesting EBCHeadWithBinLogits...")
    bins = [[0, 0]] + [[i, i] for i in range(1, 11)] + [[11, 12], [13, 14], [15, 9999]]
    bin_centers_full = [0.0] + [float(i) for i in range(1, 11)] + [11.5, 13.5, 17.0]
    
    ebc_head_logits = EBCHeadWithBinLogits(
        visual_dim=D,
        text_dim=text_dim,
        bins=bins,
        bin_centers=bin_centers_full,
        use_refiner=True,
    )
    # Imposta solo le text features per i bin non-zero
    ebc_head_logits.set_text_features(text_features)
    
    outputs = ebc_head_logits(visual_features)
    print(f"Logit bin maps shape: {outputs['logit_bin_maps'].shape}")
    print(f"Lambda maps shape: {outputs['lambda_maps'].shape}")
    print(f"Bin probs shape: {outputs['bin_probs'].shape}")
    
    print("\n✅ Test completato!")
