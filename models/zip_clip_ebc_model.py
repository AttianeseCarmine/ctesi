# ============================================================
# ZIP-CLIP-EBC: Modello Completo
# ============================================================
# Combina:
#   - Backbone CLIP (feature extraction)
#   - π-Head (convoluzionale, classifica vuoto/pieno)
#   - EBC-Head (CLIP-based, conta persone nei blocchi non-vuoti)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .clip_backbone import CLIPBackbone, build_clip_backbone
from .pi_head import PiHead, build_pi_head
from .ebc_head import EBCHeadWithBinLogits, build_ebc_head


class ZIPCLIPEBCModel(nn.Module):
    """
    Modello completo per crowd counting con Zero-Inflated architecture.
    
    Architettura:
    ```
    Input Image
         │
         ▼
    ┌─────────────┐
    │ CLIP Backbone│ ──► Feature Map [B, D, H, W]
    └─────────────┘
         │
         ├──────────────────┐
         ▼                  ▼
    ┌─────────┐       ┌──────────┐
    │ π-Head  │       │ EBC-Head │
    │ (Conv)  │       │ (CLIP)   │
    └─────────┘       └──────────┘
         │                  │
         ▼                  ▼
    P(vuoto/pieno)    λ (count per bin)
         │                  │
         └───────┬──────────┘
                 ▼
        Density Map = (1 - P(vuoto)) × λ
    ```
    
    Args:
        config: Dizionario di configurazione
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        model_cfg = config.get("MODEL", {})
        
        # ========================
        # 1. Backbone CLIP
        # ========================
        self.backbone = build_clip_backbone(config)
        
        # Dimensioni
        self.visual_dim = self.backbone.visual_dim
        self.embed_dim = self.backbone.embed_dim
        self.text_dim = self.backbone.text_dim
        self.patch_size = self.backbone.patch_size
        
        # ========================
        # 2. π-Head (Convoluzionale)
        # ========================
        # Input: feature map dal backbone [B, embed_dim, H, W]
        self.pi_head = build_pi_head(config, in_channels=self.embed_dim)
        
        # ========================
        # 3. EBC-Head (CLIP-based)
        # ========================
        # Input: feature map proiettate [B, visual_dim, H, W]
        self.ebc_head = build_ebc_head(
            config,
            visual_dim=self.visual_dim,
            text_dim=self.text_dim,
        )
        
        # ========================
        # 4. Configurazione Gating
        # ========================
        self.gate_mode = model_cfg.get("GATE_MODE", "multiply")
        self.pi_thresh = model_cfg.get("PI_THRESH", 0.5)
        self.pi_soft_gate = model_cfg.get("PI_SOFT_GATE", True)
        self.pi_soft_min = model_cfg.get("PI_SOFT_MIN", 0.1)
        self.pi_soft_power = model_cfg.get("PI_SOFT_POWER", 1.0)
        self.upsample_to_input = model_cfg.get("UPSAMPLE_TO_INPUT", False)
        
        # ========================
        # 5. Inizializza Text Features
        # ========================
        self._init_text_features(config)
        
        print(f"✅ ZIPCLIPEBCModel inizializzato:")
        print(f"   Backbone: {model_cfg.get('BACKBONE', 'ViT-B-16')}")
        print(f"   Patch size: {self.patch_size}")
        print(f"   Gate mode: {self.gate_mode}")
        print(f"   Soft gate: {self.pi_soft_gate}")
    
    def _init_text_features(self, config: Dict):
        """Inizializza le text features per l'EBC head."""
        ebc_cfg = config.get("EBC_HEAD", {})
        prompts = ebc_cfg.get("TEXT_PROMPTS", [])
        
        if not prompts:
            # Usa prompts di default
            prompts = [
                "a photo showing exactly one person",
                "a photo showing exactly two people",
                "a photo showing exactly three people",
                "a photo showing exactly four people",
                "a photo showing exactly five people",
                "a photo showing exactly six people",
                "a photo showing exactly seven people",
                "a photo showing exactly eight people",
                "a photo showing exactly nine people",
                "a photo showing exactly ten people",
                "a photo showing about eleven or twelve people",
                "a photo showing about thirteen or fourteen people",
                "a dense crowd of fifteen or more people",
            ]
        
        # Verifica che il numero di prompts corrisponda ai bins
        expected_num = self.ebc_head.num_ebc_bins
        if len(prompts) != expected_num:
            raise ValueError(
                f"Numero di prompts ({len(prompts)}) non corrisponde "
                f"al numero di EBC bins ({expected_num})"
            )
        
        # Codifica i prompts
        with torch.no_grad():
            text_features = self.backbone.get_text_features(prompts)
        
        # Imposta nell'EBC head
        self.ebc_head.set_text_features(text_features)
        
        print(f"   Text prompts: {len(prompts)} inizializzati")
    
    def get_feature_maps(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """
        Estrae le feature maps dal backbone.
        
        Args:
            x: Immagine [B, 3, H, W]
        
        Returns:
            embed_features: [B, embed_dim, H_grid, W_grid] - per π-head
            proj_features: [B, visual_dim, H_grid, W_grid] - per EBC-head
            grid_size: (H_grid, W_grid)
        """
        B, C, H, W = x.shape
        
        # Estrai patch features
        patch_features, grid_size = self.backbone(x, return_projected=False)
        # patch_features: [B, num_patches, embed_dim]
        
        H_grid, W_grid = grid_size
        
        # Reshape a mappa 2D per π-head
        embed_features = patch_features.reshape(B, H_grid, W_grid, -1).permute(0, 3, 1, 2)
        # embed_features: [B, embed_dim, H_grid, W_grid]
        
        # Proietta per EBC-head
        proj_features = self.backbone.project_visual_features(patch_features)
        proj_features = proj_features.reshape(B, H_grid, W_grid, -1).permute(0, 3, 1, 2)
        # proj_features: [B, visual_dim, H_grid, W_grid]
        
        return embed_features, proj_features, grid_size
    
    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass completo.
        
        Args:
            x: Immagine [B, 3, H, W]
            return_intermediates: Se True, restituisce anche output intermedi
        
        Returns:
            dict con:
                - logit_pi_maps: [B, 2, H_grid, W_grid] - logits π
                - pi_prob: [B, 1, H_grid, W_grid] - P(pieno)
                - logit_bin_maps: [B, num_bins, H_grid, W_grid] - logits EBC
                - lambda_maps: [B, 1, H_grid, W_grid] - conteggio atteso per blocco
                - density_map: [B, 1, H_grid, W_grid] - mappa di densità finale
                - pred_count: [B] - conteggio totale predetto
        """
        B, C, H_in, W_in = x.shape
        
        # ========================
        # 1. Feature Extraction
        # ========================
        embed_features, proj_features, (H_grid, W_grid) = self.get_feature_maps(x)
        
        # ========================
        # 2. π-Head: Classificazione vuoto/pieno
        # ========================
        logit_pi_maps = self.pi_head(embed_features)  # [B, 2, H_grid, W_grid]
        
        # Probabilità
        pi_probs = F.softmax(logit_pi_maps, dim=1)
        pi_prob = pi_probs[:, 1:2, :, :]  # [B, 1, H_grid, W_grid] - P(pieno)
        
        # ========================
        # 3. Gating Mask
        # ========================
        if self.training and self.pi_soft_gate:
            # Soft gating durante training (mantiene gradienti)
            mask = pi_prob.clone()
            if self.pi_soft_min > 0:
                mask = torch.clamp(mask, min=self.pi_soft_min)
            if self.pi_soft_power != 1.0:
                mask = torch.pow(mask, self.pi_soft_power)
        else:
            # Hard gating durante inference
            mask = (pi_prob >= self.pi_thresh).float()
        
        # ========================
        # 4. EBC-Head: Conteggio CLIP-based
        # ========================
        # Opzione: gating delle feature prima dell'EBC
        if self.gate_mode == "multiply":
            gated_features = proj_features * mask
        else:
            gated_features = proj_features
        
        ebc_outputs = self.ebc_head(gated_features)
        logit_bin_maps = ebc_outputs["logit_bin_maps"]  # [B, num_bins, H_grid, W_grid]
        lambda_maps = ebc_outputs["lambda_maps"]  # [B, 1, H_grid, W_grid]
        bin_probs = ebc_outputs["bin_probs"]
        
        # ========================
        # 5. Density Map Finale
        # ========================
        # density = P(pieno) × λ
        density_map = pi_prob * lambda_maps
        
        # ========================
        # 6. Conteggio Totale
        # ========================
        pred_count = density_map.sum(dim=[1, 2, 3])  # [B]
        
        # ========================
        # 7. Upsample se richiesto
        # ========================
        if self.upsample_to_input:
            density_map = F.interpolate(
                density_map, size=(H_in, W_in),
                mode='bilinear', align_corners=False
            )
            # Normalizza per preservare la somma
            scale_factor = (H_grid * W_grid) / (H_in * W_in)
            density_map = density_map * scale_factor
        
        # ========================
        # Output
        # ========================
        outputs = {
            "logit_pi_maps": logit_pi_maps,
            "pi_prob": pi_prob,
            "logit_bin_maps": logit_bin_maps,
            "lambda_maps": lambda_maps,
            "density_map": density_map,
            "pred_count": pred_count,
        }
        
        if return_intermediates:
            outputs["embed_features"] = embed_features
            outputs["proj_features"] = proj_features
            outputs["gating_mask"] = mask
            outputs["bin_probs"] = bin_probs
        
        return outputs
    
    def forward_pi_only(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass solo per π-head (Stage 1).
        
        Args:
            x: Immagine [B, 3, H, W]
        
        Returns:
            dict con logit_pi_maps, pi_prob
        """
        embed_features, _, _ = self.get_feature_maps(x)
        logit_pi_maps = self.pi_head(embed_features)
        
        pi_probs = F.softmax(logit_pi_maps, dim=1)
        pi_prob = pi_probs[:, 1:2, :, :]
        
        return {
            "logit_pi_maps": logit_pi_maps,
            "pi_prob": pi_prob,
        }
    
    def forward_ebc_only(
        self,
        x: torch.Tensor,
        use_gt_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass solo per EBC-head (Stage 2).
        
        Args:
            x: Immagine [B, 3, H, W]
            use_gt_mask: Se fornito, usa questa maschera invece di π
        
        Returns:
            dict con logit_bin_maps, lambda_maps, ecc.
        """
        embed_features, proj_features, _ = self.get_feature_maps(x)
        
        # Calcola π (ma non lo addestriamo)
        with torch.no_grad():
            logit_pi_maps = self.pi_head(embed_features)
            pi_probs = F.softmax(logit_pi_maps, dim=1)
            pi_prob = pi_probs[:, 1:2, :, :]
        
        # Usa GT mask se fornito, altrimenti usa π
        if use_gt_mask is not None:
            mask = use_gt_mask
        else:
            if self.training and self.pi_soft_gate:
                mask = torch.clamp(pi_prob, min=self.pi_soft_min)
            else:
                mask = (pi_prob >= self.pi_thresh).float()
        
        # Gate features
        if self.gate_mode == "multiply":
            gated_features = proj_features * mask
        else:
            gated_features = proj_features
        
        # EBC forward
        ebc_outputs = self.ebc_head(gated_features)
        
        # Density map
        density_map = pi_prob * ebc_outputs["lambda_maps"]
        pred_count = density_map.sum(dim=[1, 2, 3])
        
        return {
            "logit_pi_maps": logit_pi_maps,
            "pi_prob": pi_prob,
            "logit_bin_maps": ebc_outputs["logit_bin_maps"],
            "lambda_maps": ebc_outputs["lambda_maps"],
            "bin_probs": ebc_outputs["bin_probs"],
            "density_map": density_map,
            "pred_count": pred_count,
            "gating_mask": mask,
        }
    
    def freeze_backbone(self):
        """Congela tutti i parametri del backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("🧊 Backbone congelato")
    
    def unfreeze_backbone(self, lr_scale: float = 0.1):
        """Scongela il backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print(f"🔓 Backbone scongelato")
    
    def freeze_pi_head(self):
        """Congela π-head."""
        for param in self.pi_head.parameters():
            param.requires_grad = False
        print("🧊 π-head congelato")
    
    def unfreeze_pi_head(self):
        """Scongela π-head."""
        for param in self.pi_head.parameters():
            param.requires_grad = True
        print("🔓 π-head scongelato")
    
    def freeze_ebc_head(self):
        """Congela EBC-head."""
        for param in self.ebc_head.parameters():
            param.requires_grad = False
        print("🧊 EBC-head congelato")
    
    def unfreeze_ebc_head(self):
        """Scongela EBC-head."""
        for param in self.ebc_head.parameters():
            param.requires_grad = True
        print("🔓 EBC-head scongelato")
    
    def get_param_groups(
        self,
        lr_backbone: float,
        lr_pi_head: float,
        lr_ebc_head: float,
    ) -> List[Dict]:
        """
        Restituisce gruppi di parametri con LR differenziati.
        
        Args:
            lr_backbone: LR per il backbone
            lr_pi_head: LR per π-head
            lr_ebc_head: LR per EBC-head
        
        Returns:
            Lista di dicts per l'optimizer
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
        
        # π-head
        pi_params = [p for p in self.pi_head.parameters() if p.requires_grad]
        if pi_params:
            param_groups.append({
                "params": pi_params,
                "lr": lr_pi_head,
                "name": "pi_head",
            })
        
        # EBC-head
        ebc_params = [p for p in self.ebc_head.parameters() if p.requires_grad]
        if ebc_params:
            param_groups.append({
                "params": ebc_params,
                "lr": lr_ebc_head,
                "name": "ebc_head",
            })
        
        return param_groups


def build_model(config: Dict) -> ZIPCLIPEBCModel:
    """Costruisce il modello dalla configurazione."""
    return ZIPCLIPEBCModel(config)


if __name__ == "__main__":
    # Test
    import yaml
    
    print("Testing ZIPCLIPEBCModel...")
    
    # Config minimale per test
    config = {
        "DATASET": "sha",
        "MODEL": {
            "BACKBONE": "ViT-B-16",
            "CLIP_PRETRAINED": "openai",
            "PI_THRESH": 0.5,
            "PI_SOFT_GATE": True,
            "PI_SOFT_MIN": 0.1,
            "GATE_MODE": "multiply",
            "UPSAMPLE_TO_INPUT": False,
        },
        "PI_HEAD": {
            "HIDDEN_DIM": 256,
            "NUM_LAYERS": 2,
        },
        "EBC_HEAD": {
            "TEMPERATURE": 0.07,
            "LEARNABLE_TEMP": True,
            "USE_REFINER": True,
            "TEXT_PROMPTS": [
                "one person", "two people", "three people",
                "four people", "five people", "six people",
                "seven people", "eight people", "nine people",
                "ten people", "about eleven or twelve people",
                "about thirteen or fourteen people",
                "fifteen or more people",
            ],
        },
        "BINS_CONFIG": {
            "sha": {
                "bins": [[0, 0]] + [[i, i] for i in range(1, 11)] + [[11, 12], [13, 14], [15, 9999]],
                "bin_centers": [0.0] + [float(i) for i in range(1, 11)] + [11.5, 13.5, 17.0],
            }
        }
    }
    
    # Crea modello
    model = build_model(config)
    
    # Test forward
    x = torch.randn(2, 3, 256, 256)
    
    with torch.no_grad():
        outputs = model(x, return_intermediates=True)
    
    print("\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    print(f"\nPred counts: {outputs['pred_count']}")
    
    # Test forward separati per stage
    print("\nTesting forward_pi_only...")
    with torch.no_grad():
        pi_out = model.forward_pi_only(x)
    print(f"  logit_pi_maps: {pi_out['logit_pi_maps'].shape}")
    print(f"  pi_prob range: [{pi_out['pi_prob'].min():.3f}, {pi_out['pi_prob'].max():.3f}]")
    
    print("\nTesting forward_ebc_only...")
    with torch.no_grad():
        ebc_out = model.forward_ebc_only(x)
    print(f"  logit_bin_maps: {ebc_out['logit_bin_maps'].shape}")
    print(f"  lambda_maps range: [{ebc_out['lambda_maps'].min():.2f}, {ebc_out['lambda_maps'].max():.2f}]")
    
    # Test freeze/unfreeze
    print("\nTesting freeze/unfreeze...")
    model.freeze_backbone()
    model.freeze_ebc_head()
    model.unfreeze_pi_head()
    
    # Conta parametri trainabili
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parametri trainabili: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    print("\n✅ Test completato!")
