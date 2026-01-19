# ============================================================
# ZIP-CLIP-EBC: ZIP Head (Zero-Inflated Poisson)
# ============================================================
# Implementazione fedele al paper ZIP (Yiming-M/ZIP).
#
# La ZIP head modella il conteggio per blocco come:
#   P(Y=0) = π + (1-π) * e^{-λ}
#   P(Y=k) = (1-π) * (λ^k * e^{-λ}) / k!   per k > 0
#
# Dove:
#   - π: probabilità che il blocco sia "strutturalmente vuoto"
#   - λ: rate Poisson per blocchi non-vuoti (expected count)
#
# Output:
#   - logit_pi: [B, 1, H, W] - logits per π (sigmoid → probabilità vuoto)
#   - log_lambda: [B, 1, H, W] - log(λ) (exp → rate Poisson)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class ZIPHead(nn.Module):
    """
    Zero-Inflated Poisson Head per crowd counting.
    
    Architettura:
    - Shared convolutional layers per feature extraction
    - Due branch separati:
      1. π-branch: classifica blocchi vuoti/non-vuoti
      2. λ-branch: stima il rate Poisson per blocchi non-vuoti
    
    Args:
        in_channels: Numero di canali in input (dal backbone)
        hidden_dim: Dimensione dei canali nascosti
        num_layers: Numero di layer convoluzionali condivisi
        dropout: Dropout rate
        lambda_activation: Attivazione per λ ("softplus" o "exp")
        lambda_bias_init: Valore iniziale del bias per λ (controlla scala iniziale)
    """
    
    def __init__(
        self,
        in_channels: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        lambda_activation: str = "softplus",
        lambda_bias_init: float = 0.0,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.lambda_activation = lambda_activation
        
        # ========================
        # Shared Feature Extractor
        # ========================
        shared_layers = []
        current_channels = in_channels
        
        for i in range(num_layers):
            out_ch = hidden_dim if i == 0 else hidden_dim
            shared_layers.extend([
                nn.Conv2d(current_channels, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            if dropout > 0 and i < num_layers - 1:
                shared_layers.append(nn.Dropout2d(dropout))
            current_channels = out_ch
        
        self.shared = nn.Sequential(*shared_layers)
        
        # ========================
        # π-Branch (Zero-Inflation)
        # ========================
        # Output: probabilità che il blocco sia strutturalmente vuoto
        self.pi_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )
        
        # ========================
        # λ-Branch (Poisson Rate)
        # ========================
        # Output: rate Poisson (expected count per blocco non-vuoto)
        self.lambda_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )
        
        # Inizializzazione
        self._init_weights(lambda_bias_init)
        
        print(f"✅ ZIPHead inizializzato:")
        print(f"   In channels: {in_channels}")
        print(f"   Hidden dim: {hidden_dim}")
        print(f"   Lambda activation: {lambda_activation}")
    
    def _init_weights(self, lambda_bias_init: float):
        """Inizializza i pesi."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Inizializza π-head con bias negativo (favorisce "non-vuoto" all'inizio)
        # Questo aiuta perché la maggior parte dei blocchi È vuota, 
        # ma vogliamo che il modello impari a riconoscerli, non assumere tutto vuoto
        if self.pi_head[-1].bias is not None:
            nn.init.constant_(self.pi_head[-1].bias, -1.0)  # sigmoid(-1) ≈ 0.27
        
        # Inizializza λ-head per output ragionevole
        if self.lambda_head[-1].bias is not None:
            nn.init.constant_(self.lambda_head[-1].bias, lambda_bias_init)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Feature map dal backbone [B, C, H, W]
            
        Returns:
            dict con:
                - logit_pi: [B, 1, H, W] - logits per π (pre-sigmoid)
                - pi: [B, 1, H, W] - probabilità vuoto (post-sigmoid)
                - log_lambda: [B, 1, H, W] - log(λ) raw
                - lambda_: [B, 1, H, W] - rate Poisson (post-activation)
                - expected_count: [B, 1, H, W] - (1-π) * λ
        """
        # Shared features
        h = self.shared(x)
        
        # π-branch
        logit_pi = self.pi_head(h)  # [B, 1, H, W]
        pi = torch.sigmoid(logit_pi)  # P(blocco vuoto)
        
        # λ-branch
        log_lambda = self.lambda_head(h)  # [B, 1, H, W]
        
        # Attivazione per λ (deve essere > 0)
        if self.lambda_activation == "softplus":
            lambda_ = F.softplus(log_lambda)
        elif self.lambda_activation == "exp":
            lambda_ = torch.exp(log_lambda.clamp(max=10))  # Clamp per stabilità
        else:
            lambda_ = F.relu(log_lambda) + 1e-6
        
        # Expected count: E[Y] = (1-π) * λ
        expected_count = (1 - pi) * lambda_
        
        return {
            "logit_pi": logit_pi,
            "pi": pi,
            "log_lambda": log_lambda,
            "lambda_": lambda_,
            "expected_count": expected_count,
        }
    
    def get_density_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcola la density map per l'immagine.
        
        Args:
            x: Feature map dal backbone [B, C, H, W]
            
        Returns:
            density: [B, 1, H, W] - expected count per blocco
        """
        outputs = self.forward(x)
        return outputs["expected_count"]


class ZIPHeadV2(nn.Module):
    """
    ZIP Head V2 con architettura migliorata.
    
    Differenze da V1:
    - Usa classificazione binaria (2 classi) per π invece di regressione
    - Aggiunge skip connection
    - Supporto per multi-scale features
    
    Args:
        in_channels: Canali input
        hidden_dim: Dimensione hidden
        use_skip: Se usare skip connections
    """
    
    def __init__(
        self,
        in_channels: int = 512,
        hidden_dim: int = 256,
        use_skip: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.use_skip = use_skip
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # π-head: classificazione binaria [vuoto, non-vuoto]
        self.pi_head = nn.Conv2d(hidden_dim, 2, kernel_size=1)
        
        # λ-head: regressione del rate Poisson
        self.lambda_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )
        
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
        
        # Bias π verso "non-vuoto" inizialmente
        if self.pi_head.bias is not None:
            nn.init.constant_(self.pi_head.bias[0], 0.5)   # logit vuoto
            nn.init.constant_(self.pi_head.bias[1], -0.5)  # logit non-vuoto
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: [B, C, H, W] feature map
            
        Returns:
            dict con logit_pi, pi, lambda_, expected_count
        """
        # Project input
        h = self.input_proj(x)
        
        # Encode
        encoded = self.encoder(h)
        
        # Skip connection
        if self.use_skip:
            encoded = encoded + h
        
        # π: classificazione binaria
        logit_pi = self.pi_head(encoded)  # [B, 2, H, W]
        pi_probs = F.softmax(logit_pi, dim=1)
        pi = pi_probs[:, 0:1, :, :]  # P(vuoto)
        pi_not_empty = pi_probs[:, 1:2, :, :]  # P(non-vuoto)
        
        # λ: rate Poisson
        log_lambda = self.lambda_head(encoded)
        lambda_ = F.softplus(log_lambda)
        
        # Expected count
        expected_count = pi_not_empty * lambda_
        
        return {
            "logit_pi": logit_pi,  # [B, 2, H, W]
            "pi": pi,  # P(vuoto)
            "pi_not_empty": pi_not_empty,  # P(non-vuoto)
            "log_lambda": log_lambda,
            "lambda_": lambda_,
            "expected_count": expected_count,
        }


def build_zip_head(
    in_channels: int = 512,
    hidden_dim: int = 256,
    version: str = "v1",
    **kwargs
) -> nn.Module:
    """
    Factory function per costruire la ZIP head.
    
    Args:
        in_channels: Canali input
        hidden_dim: Dimensione hidden
        version: "v1" o "v2"
        
    Returns:
        ZIPHead module
    """
    if version.lower() == "v1":
        return ZIPHead(in_channels=in_channels, hidden_dim=hidden_dim, **kwargs)
    elif version.lower() == "v2":
        return ZIPHeadV2(in_channels=in_channels, hidden_dim=hidden_dim, **kwargs)
    else:
        raise ValueError(f"Versione non supportata: {version}")


if __name__ == "__main__":
    # Test
    print("Testing ZIP Heads...")
    
    B, C, H, W = 2, 512, 16, 16
    x = torch.randn(B, C, H, W)
    
    # Test ZIPHead V1
    print("\n--- ZIPHead V1 ---")
    zip_head_v1 = ZIPHead(in_channels=C, hidden_dim=256)
    out_v1 = zip_head_v1(x)
    
    for key, val in out_v1.items():
        print(f"  {key}: {val.shape}")
    
    print(f"\n  π range: [{out_v1['pi'].min():.3f}, {out_v1['pi'].max():.3f}]")
    print(f"  λ range: [{out_v1['lambda_'].min():.3f}, {out_v1['lambda_'].max():.3f}]")
    print(f"  Expected count sum: {out_v1['expected_count'].sum(dim=[1,2,3])}")
    
    # Test ZIPHead V2
    print("\n--- ZIPHead V2 ---")
    zip_head_v2 = ZIPHeadV2(in_channels=C, hidden_dim=256)
    out_v2 = zip_head_v2(x)
    
    for key, val in out_v2.items():
        print(f"  {key}: {val.shape}")
    
    print(f"\n  π (vuoto) range: [{out_v2['pi'].min():.3f}, {out_v2['pi'].max():.3f}]")
    print(f"  λ range: [{out_v2['lambda_'].min():.3f}, {out_v2['lambda_'].max():.3f}]")
    
    print("\n✅ Test completato!")
