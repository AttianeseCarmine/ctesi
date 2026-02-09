import torch
import torch.nn as nn
from typing import Dict
import torch.nn.functional as F

from .backbone import build_backbone
from .pi_head import ZIPHead

class ZIPModel(nn.Module):
    """
    Stage 1 (ZIP): Backbone + ZIPHead -> pi_logits

    - ResNet/CNN: comportamento "vecchio" (config-driven, no forcing stride)
    - ViT-B/16: comportamento "nuovo" (porta da stride16 a stride8 con upsample+refine)
    """

    def __init__(self, config: Dict):
        super().__init__()

        self.config = config
        self.backbone = build_backbone(config)

        # Detect backbone type in modo robusto (senza dipendere da una singola chiave)
        bb_type = (
            config.get("BACKBONE", {}).get("TYPE")
            or config.get("model")
            or config.get("backbone")
            or "unknown"
        )
        bb_type = str(bb_type).lower()

        # ViT se il backbone espone native_reduction (es. 16) oppure il nome contiene 'vit'
        self.is_vit = hasattr(self.backbone, "native_reduction") or ("vit" in bb_type)

        # Target reduction:
        # - ResNet: come prima (può essere None se non vuoi forzare niente)
        # - ViT: di default 8 (ma se nel config metti REDUCTION=16 te lo rispetta)
        if self.is_vit:
            self.native_reduction = int(getattr(self.backbone, "native_reduction", 16))
            self.target_reduction = int(config.get("REDUCTION", 8))
        else:
            self.native_reduction = None
            self.target_reduction = config.get("REDUCTION", None)  # come prima (None ok)

        # Upsampler SOLO per ViT quando vuoi 16 -> 8
        self.upsampler = nn.Identity()
        if self.is_vit and self.native_reduction == 16 and self.target_reduction == 8:
            print("🔧 ZIPModel: ViT-B/16 -> using stable upsample+refine (16->8)")
            self.upsampler = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(self.backbone.out_channels, self.backbone.out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.backbone.out_channels),
                nn.ReLU(inplace=True),
            )

        # ZIP Head
        zip_cfg = config.get("ZIP_HEAD", {})
        self.zip_head = ZIPHead(
            in_channels=self.backbone.out_channels,
            hidden_dim=zip_cfg.get("HIDDEN_DIM", 256),
        )

        print(
            f"✅ ZIPModel init | bb={bb_type} | is_vit={self.is_vit} | "
            f"native_red={self.native_reduction} | target_red={self.target_reduction} | "
            f"out_ch={self.backbone.out_channels}"
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1) Backbone features
        features = self.backbone(x)

        # 2) (ViT only) upsample features if needed
        features_up = self.upsampler(features)

        # 3) ZIP head -> logits
        zip_out = self.zip_head(features_up)
        pi_logits = zip_out["logit_pi"]  # pre-sigmoid

        # 4) Geometry align: SOLO se target_reduction è definito
        #    (ResNet "vecchio": se target_reduction è None non tocchiamo shape)
        if self.target_reduction is not None:
            H, W = x.shape[-2], x.shape[-1]
            out_h, out_w = H // int(self.target_reduction), W // int(self.target_reduction)
            if out_h > 0 and out_w > 0 and pi_logits.shape[-2:] != (out_h, out_w):
                pi_logits = F.interpolate(
                    pi_logits, size=(out_h, out_w),
                    mode="bilinear", align_corners=False
                )

        return {
            "pi_logits": pi_logits,
            "features": features_up,
        }
