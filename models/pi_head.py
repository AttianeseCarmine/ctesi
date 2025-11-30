# ============================================================
# ZIP-CLIP-EBC: π-Head Convoluzionale
# ============================================================
# Classifica ogni blocco come vuoto (0) o contenente persone (1).
# Architettura puramente convoluzionale, senza dipendenze da CLIP text.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PiHead(nn.Module):
    """
    π-Head convoluzionale per la classificazione vuoto/pieno dei blocchi.
    
    Input: Feature map dal backbone [B, C, H, W]
    Output: Logits per classificazione binaria [B, 2, H, W]
            - Canale 0: logit per "vuoto"
            - Canale 1: logit per "pieno" (contiene persone)
    
    L'architettura è semplice ma efficace:
    - Convoluzioni 3x3 con BN e ReLU
    - Nessun pooling (mantiene la risoluzione spaziale)
    - Output a 2 canali per classificazione binaria
    
    Args:
        in_channels: Numero di canali in input (dal backbone)
        hidden_dim: Dimensione dei canali intermedi
        num_layers: Numero di layer convoluzionali
        dropout: Dropout rate
        use_bn: Se usare BatchNorm
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_bn: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Costruisci i layer
        layers = []
        
        # Primo layer: riduce i canali
        layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        # Layer intermedi
        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
        
        self.features = nn.Sequential(*layers)
        
        # Layer di output: 2 canali (vuoto/pieno)
        self.classifier = nn.Conv2d(hidden_dim, 2, kernel_size=1)
        
        # Inizializzazione
        self._init_weights()
        
        print(f"✅ PiHead inizializzato:")
        print(f"   In channels: {in_channels}")
        print(f"   Hidden dim: {hidden_dim}")
        print(f"   Num layers: {num_layers}")
    
    def _init_weights(self):
        """Inizializza i pesi in modo appropriato."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Inizializza il classificatore con bias verso "vuoto"
        # Questo aiuta all'inizio del training dato che la maggior parte dei blocchi è vuota
        if self.classifier.bias is not None:
            # Bias per classe "vuoto" (canale 0) positivo
            # Bias per classe "pieno" (canale 1) negativo
            nn.init.constant_(self.classifier.bias[0], 1.0)   # Favorisce "vuoto"
            nn.init.constant_(self.classifier.bias[1], -1.0)  # Sfavorisce "pieno"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Feature map [B, C, H, W]
        
        Returns:
            logits: [B, 2, H, W] - logits per classificazione vuoto/pieno
        """
        # Estrai features
        h = self.features(x)
        
        # Classifica
        logits = self.classifier(h)
        
        return logits
    
    def get_pi_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcola la probabilità che ogni blocco sia PIENO (contenga persone).
        
        Args:
            x: Feature map [B, C, H, W]
        
        Returns:
            pi_prob: [B, 1, H, W] - probabilità di essere pieno
        """
        logits = self.forward(x)  # [B, 2, H, W]
        probs = F.softmax(logits, dim=1)  # [B, 2, H, W]
        pi_prob = probs[:, 1:2, :, :]  # [B, 1, H, W] - prob di essere pieno
        return pi_prob
    
    def get_mask(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
        soft: bool = False,
        soft_min: float = 0.0,
        soft_power: float = 1.0,
    ) -> torch.Tensor:
        """
        Calcola la maschera per il gating delle features.
        
        Args:
            x: Feature map [B, C, H, W]
            threshold: Soglia per la maschera hard
            soft: Se True, usa soft gating invece di hard threshold
            soft_min: Valore minimo per soft gating (mantiene gradienti)
            soft_power: Potenza per soft gating
        
        Returns:
            mask: [B, 1, H, W] - maschera per il gating
        """
        pi_prob = self.get_pi_prob(x)  # [B, 1, H, W]
        
        if soft:
            # Soft gating: usa direttamente le probabilità
            mask = pi_prob
            
            # Applica soft_min per mantenere gradienti
            if soft_min > 0:
                mask = torch.clamp(mask, min=soft_min)
            
            # Applica potenza (opzionale)
            if soft_power != 1.0:
                mask = torch.pow(mask, soft_power)
        else:
            # Hard gating: soglia binaria
            mask = (pi_prob >= threshold).float()
        
        return mask


class PiHeadWithLambda(nn.Module):
    """
    Versione estesa del π-Head che produce anche una stima λ (Poisson rate).
    
    Questo è più vicino all'architettura ZIP originale dove:
    - π: probabilità che il blocco sia vuoto
    - λ: rate Poisson per il conteggio (usato quando il blocco non è vuoto)
    
    Nel nostro caso, λ viene stimato come valore atteso del conteggio
    nel blocco, indipendentemente da CLIP (che invece classifica nei bins).
    
    Args:
        in_channels: Numero di canali in input
        hidden_dim: Dimensione dei canali intermedi
        num_layers: Numero di layer convoluzionali
        dropout: Dropout rate
        use_bn: Se usare BatchNorm
        lambda_max: Valore massimo per λ (clipping)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_bn: bool = True,
        lambda_max: float = 20.0,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.lambda_max = lambda_max
        
        # Feature extractor condiviso
        layers = []
        
        # Primo layer
        layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        # Layer intermedi
        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
        
        self.features = nn.Sequential(*layers)
        
        # Head per π (classificazione binaria)
        self.pi_head = nn.Conv2d(hidden_dim, 2, kernel_size=1)
        
        # Head per λ (regressione del rate Poisson)
        self.lambda_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )
        
        # Inizializzazione
        self._init_weights()
        
        print(f"✅ PiHeadWithLambda inizializzato:")
        print(f"   In channels: {in_channels}")
        print(f"   Hidden dim: {hidden_dim}")
        print(f"   Lambda max: {lambda_max}")
    
    def _init_weights(self):
        """Inizializza i pesi."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Bias π verso "vuoto"
        if self.pi_head.bias is not None:
            nn.init.constant_(self.pi_head.bias[0], 1.0)
            nn.init.constant_(self.pi_head.bias[1], -1.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Feature map [B, C, H, W]
        
        Returns:
            pi_logits: [B, 2, H, W] - logits per π
            lambda_map: [B, 1, H, W] - rate Poisson λ
        """
        h = self.features(x)
        
        # π logits
        pi_logits = self.pi_head(h)  # [B, 2, H, W]
        
        # λ (rate Poisson)
        lambda_raw = self.lambda_head(h)  # [B, 1, H, W]
        lambda_map = F.softplus(lambda_raw)  # Assicura positività
        lambda_map = torch.clamp(lambda_map, min=1e-6, max=self.lambda_max)
        
        return pi_logits, lambda_map
    
    def get_pi_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Probabilità che ogni blocco sia PIENO."""
        pi_logits, _ = self.forward(x)
        probs = F.softmax(pi_logits, dim=1)
        return probs[:, 1:2, :, :]  # [B, 1, H, W]
    
    def get_expected_count(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcola il conteggio atteso per ogni blocco.
        E[count] = P(pieno) * λ
        
        Args:
            x: Feature map [B, C, H, W]
        
        Returns:
            expected_count: [B, 1, H, W]
        """
        pi_logits, lambda_map = self.forward(x)
        pi_prob = F.softmax(pi_logits, dim=1)[:, 1:2, :, :]  # P(pieno)
        
        expected_count = pi_prob * lambda_map
        return expected_count


def build_pi_head(config: dict, in_channels: int) -> PiHead:
    """
    Costruisce il π-Head dalla configurazione.
    
    Args:
        config: Dizionario di configurazione
        in_channels: Numero di canali in input
    
    Returns:
        PiHead instance
    """
    pi_cfg = config.get("PI_HEAD", {})
    
    return PiHead(
        in_channels=in_channels,
        hidden_dim=pi_cfg.get("HIDDEN_DIM", 256),
        num_layers=pi_cfg.get("NUM_LAYERS", 2),
        dropout=pi_cfg.get("DROPOUT", 0.1),
        use_bn=pi_cfg.get("USE_BN", True),
    )


if __name__ == "__main__":
    # Test
    print("Testing PiHead...")
    
    # Input simulato (feature map dal backbone CLIP)
    B, C, H, W = 2, 768, 16, 16  # CLIP ViT-B/16: 768 dim, 16x16 patches per 256x256 input
    x = torch.randn(B, C, H, W)
    
    # Test PiHead base
    pi_head = PiHead(in_channels=C, hidden_dim=256, num_layers=2)
    
    logits = pi_head(x)
    print(f"Logits shape: {logits.shape}")  # [2, 2, 16, 16]
    
    pi_prob = pi_head.get_pi_prob(x)
    print(f"Pi prob shape: {pi_prob.shape}")  # [2, 1, 16, 16]
    print(f"Pi prob range: [{pi_prob.min():.3f}, {pi_prob.max():.3f}]")
    
    # Test maschera soft
    mask_soft = pi_head.get_mask(x, soft=True, soft_min=0.1)
    print(f"Soft mask shape: {mask_soft.shape}")
    print(f"Soft mask range: [{mask_soft.min():.3f}, {mask_soft.max():.3f}]")
    
    # Test maschera hard
    mask_hard = pi_head.get_mask(x, soft=False, threshold=0.5)
    print(f"Hard mask shape: {mask_hard.shape}")
    print(f"Hard mask unique values: {torch.unique(mask_hard)}")
    
    # Test PiHeadWithLambda
    print("\nTesting PiHeadWithLambda...")
    pi_head_lambda = PiHeadWithLambda(in_channels=C, hidden_dim=256)
    
    pi_logits, lambda_map = pi_head_lambda(x)
    print(f"Pi logits shape: {pi_logits.shape}")
    print(f"Lambda map shape: {lambda_map.shape}")
    print(f"Lambda range: [{lambda_map.min():.3f}, {lambda_map.max():.3f}]")
    
    expected = pi_head_lambda.get_expected_count(x)
    print(f"Expected count shape: {expected.shape}")
    print(f"Total expected count: {expected.sum(dim=[1,2,3])}")
    
    print("\n✅ Test completato!")
