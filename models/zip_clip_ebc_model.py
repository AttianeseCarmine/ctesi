# models/zip_clip_ebc_model.py

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
from copy import deepcopy

# Importa i backbone
from .clip_ebc.vit import vit_names_and_weights, _vit
from .clip_ebc.convnext import convnext_names_and_weights, _convnext
from .clip_ebc.resnet import resnet_names_and_weights, _resnet
from .clip_ebc.mobileclip import mobileclip_names_and_weights, _mobileclip

# Importa la logica dei prompt
from .clip_ebc.utils import encode_text 

# Importa le teste
from .heads.pi_head import PiHead          # ✅ π-Head (filtro binario)
from .heads.clip_ebc_head import ClipEBHead  # EBC Head (CLIP)

# Mappa dei backbone
supported_models_and_weights = deepcopy(vit_names_and_weights)
supported_models_and_weights.update(convnext_names_and_weights)
supported_models_and_weights.update(resnet_names_and_weights)
supported_models_and_weights.update(mobileclip_names_and_weights)


class ZIP_CLIP_EBC_Model(nn.Module):
    """
    Modello ibrido π + CLIP-EBC:
    1. Backbone: CLIP ViT condiviso
    2. pi_head: PiHead (filtro binario: blocco vuoto/pieno)
    3. ebc_head: ClipEBHead (predice λ da blocchi non vuoti)
    
    Architettura seriale:
    - π Head predice probabilità blocco vuoto
    - Gating: usa π per mascherare le feature
    - EBC Head predice λ (1-13+ persone) dalle feature mascherate
    """
    def __init__(
        self,
        # === ARGOMENTI OBBLIGATORI ===
        model_name: str,
        weight_name: str,
        ebc_bins: List[Tuple[float, float]],
        ebc_bin_centers: List[float],
        
        # === ARGOMENTI OPZIONALI ===
        text_prompts: Optional[Dict[str, List[str]]] = None,
        
        # Parametri di Gating
        pi_thresh: float = 0.5,
        pi_soft_min: float = 0.0,
        gate_mode: str = "multiply",

        # Parametri Backbone
        block_size: Optional[int] = None,
        num_vpt: Optional[int] = None,
        vpt_drop: Optional[float] = None,
        input_size: Optional[int] = None,
        adapter: Optional[bool] = False,
        adapter_reduction: Optional[int] = None,
        lora: Optional[bool] = False,
        lora_rank: Optional[int] = None,
        lora_alpha: Optional[float] = None,
        lora_dropout: Optional[float] = None,
        norm: Optional[str] = "none",
        act: Optional[str] = "none",
        
        debug: bool = False,
    ) -> None:
        super().__init__()
        
        self.debug = debug
        
        # Validazione input
        if "mobileclip" in model_name.lower() or "vit" in model_name.lower():
            model_name = model_name.replace("_", "-")
        
        if model_name not in supported_models_and_weights:
            available = list(supported_models_and_weights.keys())
            raise AssertionError(
                f"Model '{model_name}' not supported.\n"
                f"Available models: {available}"
            )

        assert weight_name in supported_models_and_weights[model_name]
        assert len(ebc_bins) == len(ebc_bin_centers)
        assert text_prompts is not None and "lambda" in text_prompts
        assert len(ebc_bins) == len(text_prompts['lambda'])
        
        # Salva parametri
        self.model_name = model_name
        self.weight_name = weight_name
        self.text_prompts = text_prompts
        self.pi_thresh = pi_thresh
        self.pi_soft_min = pi_soft_min
        self.gate_mode = gate_mode

        # Registra bin centers EBC
        self.register_buffer(
            "ebc_bin_centers", 
            torch.tensor(ebc_bin_centers, dtype=torch.float32, requires_grad=False).view(1, -1, 1, 1)
        )
        
        # --- 1. COSTRUZIONE BACKBONE ---
        if model_name in vit_names_and_weights:
            if num_vpt is None: 
                num_vpt = 0
            if vpt_drop is None: 
                vpt_drop = 0.0
            
            self.backbone = _vit(
                model_name=model_name, 
                weight_name=weight_name, 
                num_vpt=num_vpt, 
                vpt_drop=vpt_drop,
                block_size=block_size, 
                adapter=adapter, 
                adapter_reduction=adapter_reduction,
                lora=lora, 
                lora_rank=lora_rank, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout,
                input_size=(input_size, input_size) if input_size else None, 
                norm=norm, 
                act=act
            )
        elif model_name in convnext_names_and_weights:
            self.backbone = _convnext(
                model_name=model_name, 
                weight_name=weight_name, 
                block_size=block_size, 
                adapter=adapter, 
                adapter_reduction=adapter_reduction,
                lora=lora, 
                lora_rank=lora_rank, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout,
                norm=norm, 
                act=act
            )
        else:
            raise NotImplementedError(f"Backbone '{model_name}' not supported")
            
        # --- 2. COSTRUZIONE HEADS ---
        in_channels = self.backbone.in_features      # Es. 768 per ViT-B
        out_channels_clip = self.backbone.out_features  # Es. 512 per ViT-B CLIP
        
        # π Head (filtro binario vuoto/pieno)
        self.pi_head = PiHead(in_ch=in_channels)
        
        # EBC Head (CLIP-based)
        self.ebc_head = ClipEBHead(
            in_channels=in_channels, 
            out_channels=out_channels_clip, 
            bias=False
        )

        # --- 3. TEXT FEATURES (solo per EBC) ---
        self._build_text_feats()

    def _build_text_feats(self):
        """Carica e processa i prompt di testo per EBC."""
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert "lambda" in self.text_prompts
        
        lambda_text_prompts_groups = self.text_prompts["lambda"]
        lambda_text_feats_list = []
        
        for group in lambda_text_prompts_groups:
            group_feats = encode_text(self.model_name, self.weight_name, group).to(device)
            avg_feat = group_feats.mean(dim=0)
            avg_feat /= avg_feat.norm(dim=-1, keepdim=True)
            lambda_text_feats_list.append(avg_feat)
        
        self.register_buffer("lambda_text_feats", torch.stack(lambda_text_feats_list))

    def forward(self, image: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass del modello ibrido.
        
        Args:
            image: Input tensor [B, 3, H, W]
        
        Returns:
            Dictionary con:
                - pred_logit_map: Logits EBC [B, num_bins_ebc, H_blocks, W_blocks]
                - pred_den_map: Density map finale [B, 1, H_blocks, W_blocks]
                - pred_logit_pi_map: Logits π [B, 2, H_blocks, W_blocks]
        """
        
        # 1. Backbone (feature extraction)
        image_feats = self.backbone(image)  # [B, C_feat, H_blocks, W_blocks]
        
        # 2. π Head (predice blocco vuoto/pieno)
        pi_outputs = self.pi_head(image_feats.float())
        pi_logit_map = pi_outputs["logit_pi_maps"]  # [B, 2, H_blocks, W_blocks]

        # 3. Gating (crea maschera dai logits π)
        pi_softmax = pi_logit_map.softmax(dim=1)
        pi_not_zero_prob = pi_softmax[:, 1:2]  # P(blocco NON vuoto)

        if self.training:
            # Soft gating in training (mantiene gradiente)
            mask = (1.0 - self.pi_soft_min) * pi_not_zero_prob + self.pi_soft_min
        else:
            # Hard gating in eval (binary mask)
            mask = (pi_not_zero_prob > self.pi_thresh).float()

        # 4. Applica Gating alle feature
        if self.gate_mode == "multiply":
            gated_feats = image_feats * mask
        else:
            gated_feats = image_feats

        # 5. EBC Head (predice λ dalle feature mascherate)
        if self.lambda_text_feats.device != image.device:
            self.lambda_text_feats = self.lambda_text_feats.to(image.device)
            
        lambda_logit_map_ebc = self.ebc_head(gated_feats, self.lambda_text_feats)

        # 6. Calcola density map finale
        if self.ebc_bin_centers.device != lambda_logit_map_ebc.device:
            self.ebc_bin_centers = self.ebc_bin_centers.to(lambda_logit_map_ebc.device)
            
        lambda_map_ebc = (lambda_logit_map_ebc.float().softmax(dim=1) * self.ebc_bin_centers).sum(dim=1, keepdim=True)
        
        # Density finale = λ_EBC * maschera
        den_map = lambda_map_ebc * mask
        
        return {
            "pred_logit_map": lambda_logit_map_ebc,  # Logits EBC (per loss Stage 2/3)
            "pred_den_map": den_map,                  # Density finale
            "pred_logit_pi_map": pi_logit_map,        # Logits π (per loss Stage 1/3)
        }