# ============================================================
# ZIP-CLIP-EBC: Modello Unificato
# ============================================================
# Combina:
#   - VGG16-BN Backbone
#   - ZIP Head (π per zero-inflation, λ per Poisson rate)
#   - CLIP-EBC Head (classificazione in bins via text-vision matching)
#
# Architettura:
#
#   Input [B, 3, H, W]
#          │
#          ▼
#   ┌──────────────────┐
#   │  VGG16 Backbone  │
#   │  [B, 512, H/16]  │
#   └────────┬─────────┘
#            │
#      ┌─────┴─────┐
#      │           │
#      ▼           ▼
#   ┌──────┐   ┌──────────┐
#   │ ZIP  │   │ CLIP-EBC │
#   │ Head │   │   Head   │
#   └──┬───┘   └────┬─────┘
#      │            │
#      │ π, λ_zip   │ λ_ebc
#      │            │
#      └─────┬──────┘
#            ▼
#    Density = π × λ_ebc
#            (o combinazione configurabile)
#
# Loss: L = (1-α) * L_ZIP + α * L_CLIP-EBC
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union

from .backbone import VGG16Backbone, build_vgg_backbone
from .pi_head import ZIPHead, ZIPHeadV2, build_zip_head
from .clip_ebc_head import CLIPEBCHead, build_clip_ebc_head


class ZIPCLIPEBCModel(nn.Module):
    """
    Modello unificato ZIP-CLIP-EBC per crowd counting.
    
    Combina la modellazione Zero-Inflated Poisson (ZIP) per gestire
    lo squilibrio spaziale con CLIP-EBC per la classificazione
    semantica del conteggio.
    
    Args:
        backbone: Nome del backbone ("vgg16_bn" o "vgg19_bn")
        pretrained_backbone: Se usare pesi pretrained per il backbone
        clip_model: Nome modello CLIP per EBC head
        clip_pretrained: Pesi pretrained CLIP
        bins: Configurazione bins per EBC
        bin_centers: Centri dei bins per expected count
        zip_hidden_dim: Dimensione hidden per ZIP head
        temperature: Temperature iniziale per CLIP similarity
        density_mode: Come calcolare la density finale:
            - "zip_only": usa solo π × λ_zip
            - "ebc_only": usa solo λ_ebc  
            - "zip_gated_ebc": usa π × λ_ebc (default, raccomandato)
            - "ensemble": media di zip e ebc
        freeze_backbone: Se congelare il backbone
        freeze_clip_text: Se congelare il text encoder CLIP (default True)
    """
    
    def __init__(
        self,
        backbone: str = "vgg16_bn",
        pretrained_backbone: bool = True,
        clip_model: str = "ViT-B-16",
        clip_pretrained: str = "openai",
        bins: Optional[List[Tuple[int, int]]] = None,
        bin_centers: Optional[List[float]] = None,
        zip_hidden_dim: int = 256,
        zip_version: str = "v2",
        temperature: float = 0.07,
        learnable_temperature: bool = True,
        prompt_type: str = "word",
        density_mode: str = "zip_gated_ebc",
        freeze_backbone: bool = False,
        freeze_clip_text: bool = True,
    ):
        super().__init__()
        
        self.density_mode = density_mode
        
        # ========================
        # 1. Backbone
        # ========================
        self.backbone = build_vgg_backbone(
            backbone_name=backbone,
            pretrained=pretrained_backbone,
            freeze_bn=freeze_backbone,
        )
        backbone_channels = self.backbone.out_channels
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # ========================
        # 2. ZIP Head
        # ========================
        self.zip_head = build_zip_head(
            in_channels=backbone_channels,
            hidden_dim=zip_hidden_dim,
            version=zip_version,
        )
        
        # ========================
        # 3. CLIP-EBC Head
        # ========================
        self.ebc_head = build_clip_ebc_head(
            in_channels=backbone_channels,
            clip_model=clip_model,
            clip_pretrained=clip_pretrained,
            bins_config={"bins": bins, "bin_centers": bin_centers} if bins else None,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
            prompt_type=prompt_type,
        )
        
        # Info
        print(f"\n✅ ZIPCLIPEBCModel inizializzato:")
        print(f"   Backbone: {backbone}")
        print(f"   ZIP version: {zip_version}")
        print(f"   CLIP model: {clip_model}")
        print(f"   Density mode: {density_mode}")
        print(f"   Backbone frozen: {freeze_backbone}")
    
    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input image [B, 3, H, W]
            return_intermediates: Se restituire output intermedi
            
        Returns:
            dict con:
                - density: [B, 1, H_out, W_out] - density map finale
                - pred_count: [B] - conteggio predetto totale
                
            Se return_intermediates=True, anche:
                - features: [B, 512, H_out, W_out] - backbone features
                - zip_outputs: dict con output ZIP head
                - ebc_outputs: dict con output EBC head
        """
        B, C, H, W = x.shape
        
        # ========================
        # 1. Backbone Features
        # ========================
        features = self.backbone(x)  # [B, 512, H/16, W/16]
        
        # ========================
        # 2. ZIP Head
        # ========================
        zip_outputs = self.zip_head(features)
        
        # Estrai π (probabilità non-vuoto)
        if "pi_not_empty" in zip_outputs:
            pi_not_empty = zip_outputs["pi_not_empty"]
        else:
            pi_not_empty = 1 - zip_outputs["pi"]
        
        lambda_zip = zip_outputs["lambda_"]
        
        # ========================
        # 3. CLIP-EBC Head
        # ========================
        ebc_outputs = self.ebc_head(features)
        lambda_ebc = ebc_outputs["expected_count"]
        
        # ========================
        # 4. Calcola Density Finale
        # ========================
        if self.density_mode == "zip_only":
            density = zip_outputs["expected_count"]
        elif self.density_mode == "ebc_only":
            density = lambda_ebc
        elif self.density_mode == "zip_gated_ebc":
            # Usa π come gate per λ_ebc
            density = pi_not_empty * lambda_ebc
        elif self.density_mode == "ensemble":
            # Media dei due metodi
            density_zip = zip_outputs["expected_count"]
            density_ebc = lambda_ebc
            density = (density_zip + density_ebc) / 2
        else:
            raise ValueError(f"density_mode non valido: {self.density_mode}")
        
        # ========================
        # 5. Conteggio Totale
        # ========================
        pred_count = density.sum(dim=[1, 2, 3])
        
        # ========================
        # Output
        # ========================
        outputs = {
            "density": density,
            "pred_count": pred_count,
            "pi": zip_outputs.get("pi", 1 - pi_not_empty),
            "pi_not_empty": pi_not_empty,
            "lambda_zip": lambda_zip,
            "lambda_ebc": lambda_ebc,
            "bin_probs": ebc_outputs["bin_probs"],
        }
        
        if return_intermediates:
            outputs["features"] = features
            outputs["zip_outputs"] = zip_outputs
            outputs["ebc_outputs"] = ebc_outputs
        
        return outputs
    
    def get_density_map(
        self,
        x: torch.Tensor,
        upsample_to_input: bool = False,
    ) -> torch.Tensor:
        """
        Restituisce solo la density map.
        
        Args:
            x: Input image [B, 3, H, W]
            upsample_to_input: Se fare upsample alla dimensione input
            
        Returns:
            density: [B, 1, H_out, W_out] o [B, 1, H, W] se upsample
        """
        outputs = self.forward(x)
        density = outputs["density"]
        
        if upsample_to_input:
            density = F.interpolate(
                density,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
        
        return density
    
    def count(self, x: torch.Tensor) -> torch.Tensor:
        """
        Restituisce solo il conteggio totale.
        
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            count: [B] - conteggio per ogni immagine nel batch
        """
        return self.forward(x)["pred_count"]
    
    # ========================
    # Metodi per Training a Stage
    # ========================
    
    def freeze_backbone(self):
        """Congela il backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("🧊 Backbone congelato")
    
    def unfreeze_backbone(self):
        """Scongela il backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("🔓 Backbone scongelato")
    
    def freeze_zip_head(self):
        """Congela ZIP head."""
        for param in self.zip_head.parameters():
            param.requires_grad = False
        print("🧊 ZIP head congelata")
    
    def unfreeze_zip_head(self):
        """Scongela ZIP head."""
        for param in self.zip_head.parameters():
            param.requires_grad = True
        print("🔓 ZIP head scongelata")
    
    def freeze_ebc_head(self):
        """Congela EBC head (escluso text encoder che è sempre congelato)."""
        for name, param in self.ebc_head.named_parameters():
            if "text_encoder" not in name:
                param.requires_grad = False
        print("🧊 EBC head congelata")
    
    def unfreeze_ebc_head(self):
        """Scongela EBC head (escluso text encoder)."""
        for name, param in self.ebc_head.named_parameters():
            if "text_encoder" not in name:
                param.requires_grad = True
        print("🔓 EBC head scongelata")
    
    def get_trainable_params(self) -> int:
        """Conta i parametri trainabili."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_param_groups(
        self,
        lr_backbone: float = 1e-5,
        lr_zip_head: float = 1e-4,
        lr_ebc_head: float = 1e-4,
    ) -> List[Dict]:
        """
        Restituisce param groups per optimizer con LR differenziati.
        
        Args:
            lr_backbone: Learning rate per backbone
            lr_zip_head: Learning rate per ZIP head
            lr_ebc_head: Learning rate per EBC head (escluso text encoder)
            
        Returns:
            Lista di dict per optimizer
        """
        param_groups = []
        
        # Backbone
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": lr_backbone,
                "name": "backbone",
            })
        
        # ZIP head
        zip_params = [p for p in self.zip_head.parameters() if p.requires_grad]
        if zip_params:
            param_groups.append({
                "params": zip_params,
                "lr": lr_zip_head,
                "name": "zip_head",
            })
        
        # EBC head (escludi text encoder)
        ebc_params = []
        for name, param in self.ebc_head.named_parameters():
            if param.requires_grad and "text_encoder" not in name:
                ebc_params.append(param)
        if ebc_params:
            param_groups.append({
                "params": ebc_params,
                "lr": lr_ebc_head,
                "name": "ebc_head",
            })
        
        return param_groups


# ============================================================
# Factory Function
# ============================================================

def build_model(config: Dict) -> ZIPCLIPEBCModel:
    """
    Costruisce il modello dalla configurazione.
    
    Args:
        config: Dizionario di configurazione
        
    Returns:
        ZIPCLIPEBCModel instance
    """
    model_cfg = config.get("MODEL", {})
    dataset_name = config.get("DATASET", "sha")
    bins_cfg = config.get("BINS_CONFIG", {}).get(dataset_name, {})
    
    return ZIPCLIPEBCModel(
        backbone=model_cfg.get("BACKBONE", "vgg16_bn"),
        pretrained_backbone=model_cfg.get("PRETRAINED_BACKBONE", True),
        clip_model=model_cfg.get("CLIP_MODEL", "ViT-B-16"),
        clip_pretrained=model_cfg.get("CLIP_PRETRAINED", "openai"),
        bins=bins_cfg.get("bins"),
        bin_centers=bins_cfg.get("bin_centers"),
        zip_hidden_dim=model_cfg.get("ZIP_HIDDEN_DIM", 256),
        zip_version=model_cfg.get("ZIP_VERSION", "v2"),
        temperature=model_cfg.get("TEMPERATURE", 0.07),
        learnable_temperature=model_cfg.get("LEARNABLE_TEMPERATURE", True),
        prompt_type=model_cfg.get("PROMPT_TYPE", "word"),
        density_mode=model_cfg.get("DENSITY_MODE", "zip_gated_ebc"),
        freeze_backbone=model_cfg.get("FREEZE_BACKBONE", False),
    )


if __name__ == "__main__":
    # Test
    print("Testing ZIPCLIPEBCModel...")
    
    # Configura bins
    bins = [(0, 0)] + [(i, i) for i in range(1, 11)] + [(11, 15), (16, 9999)]
    bin_centers = [0.0] + [float(i) for i in range(1, 11)] + [13.0, 20.0]
    
    # Crea modello
    model = ZIPCLIPEBCModel(
        backbone="vgg16_bn",
        pretrained_backbone=True,
        bins=bins,
        bin_centers=bin_centers,
        density_mode="zip_gated_ebc",
    )
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Test forward
    x = torch.randn(2, 3, 256, 256).to(device)
    
    with torch.no_grad():
        outputs = model(x, return_intermediates=True)
    
    print("\nOutput shapes:")
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")
        elif isinstance(val, dict):
            print(f"  {key}: <dict>")
    
    print(f"\nPredicted counts: {outputs['pred_count']}")
    print(f"Density sum: {outputs['density'].sum(dim=[1,2,3])}")
    
    # Test param groups
    param_groups = model.get_param_groups(lr_backbone=1e-5, lr_zip_head=1e-4, lr_ebc_head=1e-4)
    print(f"\nParam groups:")
    for g in param_groups:
        n_params = sum(p.numel() for p in g["params"])
        print(f"  {g['name']}: {n_params:,} params, lr={g['lr']}")
    
    print(f"\nTotal trainable params: {model.get_trainable_params():,}")
    
    print("\n✅ Test completato!")
