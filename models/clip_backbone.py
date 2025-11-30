# ============================================================
# ZIP-CLIP-EBC: CLIP Backbone
# ============================================================
# Wrapper per CLIP che estrae:
#   - Feature visive (patch features dal ViT)
#   - Feature testuali (per i prompts di conteggio)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    print("⚠️ open_clip non disponibile. Installare con: pip install open-clip-torch")


class CLIPBackbone(nn.Module):
    """
    Backbone basato su CLIP per crowd counting.
    
    Estrae:
    - Feature visive a livello di patch (non il CLS token globale)
    - Feature testuali per i prompts di conteggio
    
    Args:
        model_name: Nome del modello CLIP (es. "ViT-B-16")
        pretrained: Nome dei pesi pretrained (es. "openai", "laion2b_s34b_b88k")
        freeze_text_encoder: Se True, congela il text encoder
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        freeze_text_encoder: bool = True,
    ):
        super().__init__()
        
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError("open_clip è richiesto. Installare con: pip install open-clip-torch")
        
        self.model_name = model_name
        self.pretrained = pretrained
        
        # Carica il modello CLIP
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # Estrai informazioni sul modello
        self.visual = self.clip_model.visual
        
        # Dimensione delle feature
        if hasattr(self.visual, 'output_dim'):
            self.visual_dim = self.visual.output_dim
        elif hasattr(self.visual, 'proj'):
            self.visual_dim = self.visual.proj.shape[1]
        else:
            # Fallback per ViT-B-16
            self.visual_dim = 512
        
        # Dimensione delle patch (per calcolare la griglia)
        if hasattr(self.visual, 'patch_size'):
            ps = self.visual.patch_size
            self.patch_size = ps if isinstance(ps, int) else ps[0]
        else:
            # Default per ViT-B-16
            self.patch_size = 16
        
        # Dimensione embedding intermedio (prima della proiezione)
        if hasattr(self.visual, 'transformer'):
            self.embed_dim = self.visual.transformer.width
        elif hasattr(self.visual, 'width'):
            self.embed_dim = self.visual.width
        else:
            self.embed_dim = 768  # Default per ViT-B-16
        
        self.text_dim = self.clip_model.text_projection.shape[1] if hasattr(self.clip_model, 'text_projection') else self.visual_dim
        
        # Congela il text encoder se richiesto
        if freeze_text_encoder:
            self._freeze_text_encoder()
        
        # Cache per le text features (calcolate una volta sola)
        self._text_features_cache = None
        self._cached_prompts = None
        
        print(f"✅ CLIPBackbone inizializzato:")
        print(f"   Model: {model_name} ({pretrained})")
        print(f"   Patch size: {self.patch_size}")
        print(f"   Visual dim: {self.visual_dim}")
        print(f"   Embed dim: {self.embed_dim}")
        print(f"   Text dim: {self.text_dim}")
    
    def _freeze_text_encoder(self):
        """Congela tutti i parametri del text encoder."""
        if hasattr(self.clip_model, 'transformer'):
            for param in self.clip_model.transformer.parameters():
                param.requires_grad = False
        if hasattr(self.clip_model, 'token_embedding'):
            for param in self.clip_model.token_embedding.parameters():
                param.requires_grad = False
        if hasattr(self.clip_model, 'positional_embedding'):
            if isinstance(self.clip_model.positional_embedding, nn.Parameter):
                self.clip_model.positional_embedding.requires_grad = False
        if hasattr(self.clip_model, 'ln_final'):
            for param in self.clip_model.ln_final.parameters():
                param.requires_grad = False
        if hasattr(self.clip_model, 'text_projection'):
            if isinstance(self.clip_model.text_projection, nn.Parameter):
                self.clip_model.text_projection.requires_grad = False
    
    def encode_image_patches(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Estrae le feature per ogni patch dell'immagine.
        
        Args:
            x: Immagine [B, 3, H, W]
        
        Returns:
            patch_features: [B, num_patches, embed_dim]
            grid_size: (H_grid, W_grid) dimensioni della griglia di patch
        """
        B, C, H, W = x.shape
        
        # Calcola la dimensione della griglia
        H_grid = H // self.patch_size
        W_grid = W // self.patch_size
        
        # Passa attraverso il visual encoder
        # Questo dipende dall'implementazione specifica di open_clip
        visual = self.visual
        
        # Per ViT: patch embedding -> transformer -> output
        if hasattr(visual, 'conv1'):
            # Patch embedding (conv)
            x = visual.conv1(x)  # [B, embed_dim, H_grid, W_grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, embed_dim, num_patches]
            x = x.permute(0, 2, 1)  # [B, num_patches, embed_dim]
        else:
            raise NotImplementedError(f"Patch embedding non supportato per {self.model_name}")
        
        # Aggiungi class token (se presente)
        if hasattr(visual, 'class_embedding'):
            cls_token = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            x = torch.cat([cls_token, x], dim=1)  # [B, 1 + num_patches, embed_dim]
        
        # Aggiungi positional embedding
        if hasattr(visual, 'positional_embedding'):
            pos_embed = visual.positional_embedding.to(x.dtype)
            # Gestisci il caso in cui l'immagine ha dimensioni diverse dal training
            if pos_embed.shape[0] != x.shape[1]:
                pos_embed = self._interpolate_pos_encoding(pos_embed, H_grid, W_grid)
            x = x + pos_embed
        
        # Pre-LayerNorm (se presente)
        if hasattr(visual, 'ln_pre'):
            x = visual.ln_pre(x)
        
        # Passa attraverso il transformer
        if hasattr(visual, 'transformer'):
            x = visual.transformer(x)
        
        # Post-LayerNorm (se presente)
        if hasattr(visual, 'ln_post'):
            x = visual.ln_post(x)
        
        # Rimuovi il class token e restituisci solo le patch features
        if hasattr(visual, 'class_embedding'):
            patch_features = x[:, 1:, :]  # [B, num_patches, embed_dim]
        else:
            patch_features = x
        
        return patch_features, (H_grid, W_grid)
    
    def _interpolate_pos_encoding(self, pos_embed: torch.Tensor, H_grid: int, W_grid: int) -> torch.Tensor:
        """
        Interpola il positional embedding per gestire dimensioni diverse.
        """
        # Il primo token è il class token
        cls_pos = pos_embed[:1]
        patch_pos = pos_embed[1:]
        
        # Dimensione originale (assumendo quadrata)
        orig_size = int(patch_pos.shape[0] ** 0.5)
        
        # Reshape a griglia
        patch_pos = patch_pos.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
        
        # Interpola
        patch_pos = F.interpolate(patch_pos, size=(H_grid, W_grid), mode='bicubic', align_corners=False)
        
        # Reshape back
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(H_grid * W_grid, -1)
        
        return torch.cat([cls_pos, patch_pos], dim=0)
    
    def project_visual_features(self, patch_features: torch.Tensor) -> torch.Tensor:
        """
        Proietta le feature visive nello spazio condiviso CLIP.
        
        Args:
            patch_features: [B, num_patches, embed_dim]
        
        Returns:
            projected: [B, num_patches, visual_dim]
        """
        if hasattr(self.visual, 'proj') and self.visual.proj is not None:
            # Proiezione lineare
            return patch_features @ self.visual.proj
        return patch_features
    
    @torch.no_grad()
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        """
        Codifica una lista di prompts testuali.
        
        Args:
            prompts: Lista di stringhe
        
        Returns:
            text_features: [num_prompts, text_dim] normalizzate
        """
        # Tokenizza
        tokens = self.tokenizer(prompts)
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.to(next(self.clip_model.parameters()).device)
        else:
            tokens = torch.tensor(tokens).to(next(self.clip_model.parameters()).device)
        
        # Codifica
        text_features = self.clip_model.encode_text(tokens)
        
        # Normalizza
        text_features = F.normalize(text_features, dim=-1)
        
        return text_features
    
    def get_text_features(self, prompts: List[str], force_recompute: bool = False) -> torch.Tensor:
        """
        Ottiene le text features, usando la cache se disponibile.
        
        Args:
            prompts: Lista di prompts
            force_recompute: Se True, ricalcola anche se in cache
        
        Returns:
            text_features: [num_prompts, text_dim]
        """
        # Controlla se abbiamo già calcolato queste features
        if not force_recompute and self._text_features_cache is not None:
            if self._cached_prompts == prompts:
                return self._text_features_cache
        
        # Calcola e metti in cache
        self._text_features_cache = self.encode_text(prompts)
        self._cached_prompts = prompts
        
        return self._text_features_cache
    
    def forward(
        self,
        x: torch.Tensor,
        return_projected: bool = True
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Forward pass: estrae le feature visive.
        
        Args:
            x: Immagine [B, 3, H, W]
            return_projected: Se True, restituisce feature proiettate nello spazio CLIP
        
        Returns:
            features: [B, num_patches, dim] o [B, dim, H_grid, W_grid]
            grid_size: (H_grid, W_grid)
        """
        patch_features, grid_size = self.encode_image_patches(x)
        
        if return_projected:
            patch_features = self.project_visual_features(patch_features)
        
        return patch_features, grid_size
    
    def get_feature_map(self, x: torch.Tensor, return_projected: bool = True) -> torch.Tensor:
        """
        Estrae le feature come mappa 2D.
        
        Args:
            x: Immagine [B, 3, H, W]
            return_projected: Se True, usa feature proiettate
        
        Returns:
            feature_map: [B, dim, H_grid, W_grid]
        """
        patch_features, (H_grid, W_grid) = self.forward(x, return_projected)
        B = patch_features.shape[0]
        dim = patch_features.shape[-1]
        
        # Reshape a mappa 2D
        feature_map = patch_features.reshape(B, H_grid, W_grid, dim).permute(0, 3, 1, 2)
        
        return feature_map


def build_clip_backbone(config: dict) -> CLIPBackbone:
    """
    Costruisce il backbone CLIP dalla configurazione.
    
    Args:
        config: Dizionario di configurazione
    
    Returns:
        CLIPBackbone instance
    """
    model_cfg = config.get("MODEL", {})
    
    return CLIPBackbone(
        model_name=model_cfg.get("BACKBONE", "ViT-B-16"),
        pretrained=model_cfg.get("CLIP_PRETRAINED", "openai"),
        freeze_text_encoder=True,  # Il text encoder è sempre congelato
    )


if __name__ == "__main__":
    # Test
    print("Testing CLIPBackbone...")
    
    backbone = CLIPBackbone("ViT-B-16", "openai")
    
    # Test image encoding
    x = torch.randn(2, 3, 256, 256)
    features, grid_size = backbone(x)
    print(f"Patch features shape: {features.shape}")
    print(f"Grid size: {grid_size}")
    
    # Test feature map
    feat_map = backbone.get_feature_map(x)
    print(f"Feature map shape: {feat_map.shape}")
    
    # Test text encoding
    prompts = ["one person", "two people", "three people"]
    text_features = backbone.get_text_features(prompts)
    print(f"Text features shape: {text_features.shape}")
    
    print("✅ Test completato!")
