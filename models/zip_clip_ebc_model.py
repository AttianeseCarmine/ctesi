# Salva come: models/zip_clip_ebc_model.py
# CORREZIONE: Riordinati gli argomenti in __init__ per risolvere l'errore Pylance.

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
from copy import deepcopy
import json

# Importa i backbone
from .clip_ebc.vit import vit_names_and_weights, _vit
from .clip_ebc.convnext import convnext_names_and_weights, _convnext
from .clip_ebc.resnet import resnet_names_and_weights, _resnet
from .clip_ebc.mobileclip import mobileclip_names_and_weights, _mobileclip

# Importa la logica dei prompt
from .clip_ebc.utils import encode_text 

# Importa le DUE diverse teste
from .heads.conv_zip_head import ConvZIPHead  # La nuova testa CNN per ZIP
from .heads.clip_ebc_head import ClipEBHead   # La testa CLIP per EBC

# Mappa dei backbone
supported_models_and_weights = deepcopy(vit_names_and_weights)
supported_models_and_weights.update(convnext_names_and_weights)
supported_models_and_weights.update(resnet_names_and_weights)
supported_models_and_weights.update(mobileclip_names_and_weights)


class ZIP_CLIP_EBC_Model(nn.Module):
    """
    Modello ibrido:
    1. Backbone: CLIP ViT
    2. zip_head:  ConvZIPHead (CNN addestrata da zero)
    3. ebc_head:  ClipEBHead (basata su prompt di testo)
    """
    def __init__(
        self,
        # === INIZIO CORREZIONE: Argomenti NON-DEFAULT prima ===
        model_name: str,
        weight_name: str,
        
        # Parametri per EBC (CLIP-Head)
        ebc_bins: List[Tuple[float, float]],
        ebc_bin_centers: List[float],
        
        # Parametri per ZIP (Conv-Head)
        zip_bins: List[Tuple[float, float]],
        zip_bin_centers: List[float],
        
        # === Argomenti DEFAULT dopo ===
        text_prompts: Optional[Dict[str, List[str]]] = None,
        zip_lambda_max: float = 8.0,
        # === FINE CORREZIONE ===

        # Parametri di Gating
        pi_thresh: float = 0.5,
        gate_mode: str = "multiply",

        # Parametri Backbone (ViT, LoRA, etc)
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
        act: Optional[str] = "none"
    ) -> None:
        super().__init__()
        
        # Validazione input
        if "mobileclip" in model_name.lower() or "vit" in model_name.lower():
            model_name = model_name.replace("_", "-")
        
        if model_name not in supported_models_and_weights:
            print(f"ERRORE: model_name '{model_name}' (dal file YAML) non trovato.")
            print(json.dumps(list(supported_models_and_weights.keys()), indent=2))
            raise AssertionError(f"model_name '{model_name}' non è tra i modelli supportati.")

        assert weight_name in supported_models_and_weights[model_name]
        assert len(ebc_bins) == len(ebc_bin_centers)
        assert len(zip_bins) == len(zip_bin_centers)
        assert text_prompts is not None and "lambda" in text_prompts, "text_prompts con chiave 'lambda' è richiesto"
        assert len(ebc_bins) == len(text_prompts['lambda'])
        assert len(ebc_bins) == len(zip_bins) - 1 # EBC ha un bin in meno (non ha lo zero)
        
        # Salva i parametri
        self.model_name = model_name
        self.weight_name = weight_name
        self.text_prompts = text_prompts
        self.pi_thresh = pi_thresh
        self.gate_mode = gate_mode

        # Registra i bin EBC (per il calcolo della densità finale)
        self.register_buffer(
            "ebc_bin_centers", 
            torch.tensor(ebc_bin_centers, dtype=torch.float32, requires_grad=False).view(1, -1, 1, 1)
        )
        
        # --- 1. Costruzione Backbone ---
        if model_name in vit_names_and_weights:
            if num_vpt is None: num_vpt = 0
            assert num_vpt >= 0
            if vpt_drop is None: vpt_drop = 0.0
            
            self.backbone = _vit(
                model_name=model_name, weight_name=weight_name, 
                num_vpt=num_vpt, vpt_drop=vpt_drop,
                block_size=block_size, adapter=adapter, adapter_reduction=adapter_reduction,
                lora=lora, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                input_size=(input_size, input_size) if input_size else None, 
                norm=norm, act=act
            )
        # (Aggiungi qui elif per ConvNeXt, ResNet, etc. se ti servono)
        else:
            raise NotImplementedError(f"Backbone {model_name} non supportato in questa configurazione ibrida.")
            
        # --- 2. Costruzione Teste (Heads) ---
        in_channels = self.backbone.in_features  # Es. 768 per ViT-B
        out_channels_clip = self.backbone.out_features # Es. 512 per ViT-B CLIP
        
        # Testa ZIP Convoluzionale (Input: 768)
        self.zip_head = ConvZIPHead(
            in_ch=in_channels, 
            bins=zip_bins,
            bin_centers=zip_bin_centers,
            lambda_max=zip_lambda_max
        )
        
        # Testa EBC basata su CLIP (Input: 768, Output: 512)
        self.ebc_head = ClipEBHead(
            in_channels=in_channels, 
            out_channels=out_channels_clip, 
            bias=False
        )

        # --- 3. Costruzione Text Features (SOLO PER EBC) ---
        self._build_text_feats()

    def _build_text_feats(self):
        """ Carica solo i prompt 'lambda' (EBC) """
        model_name = self.model_name
        weight_name = self.weight_name
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert "lambda" in self.text_prompts, "La chiave 'lambda' è richiesta in text_prompts."
        lambda_text_prompts_groups = self.text_prompts["lambda"]
        lambda_text_feats_list = []
        
        for group in lambda_text_prompts_groups:
            group_feats = encode_text(model_name, weight_name, group).to(device)
            avg_feat = group_feats.mean(dim=0)
            avg_feat /= avg_feat.norm(dim=-1, keepdim=True)
            lambda_text_feats_list.append(avg_feat)
        
        self.register_buffer("lambda_text_feats", torch.stack(lambda_text_feats_list))
        self.lambda_text_feats = self.lambda_text_feats.to(device)


    def forward(self, image: Tensor):
        # 1. Backbone condiviso
        image_feats = self.backbone(image)  # [B, C_feat=768, H, W]
        
        # --- MODULO A: Conv-ZIP (zip_head) ---
        zip_outputs = self.zip_head(image_feats.float()) # Forza float32 per stabilità BNorm
        pi_logit_map = zip_outputs["pred_logit_pi_map"]   # [B, 2, H, W]
        lambda_map_zip = zip_outputs["pred_lambda_map"] # [B, 1, H, W]

        # 3. Crea la maschera di gating
        with torch.no_grad():
            pi_softmax = pi_logit_map.softmax(dim=1)
            pi_not_zero_prob = pi_softmax[:, 1:2]
            mask = (pi_not_zero_prob > self.pi_thresh).float()

        # 4. Applica il Gating alle feature
        if self.gate_mode == "multiply":
            gated_feats = image_feats * mask
        else:
            gated_feats = image_feats * mask

        # --- MODULO B: CLIP-EBC (ebc_head) ---
        if image.device != self.lambda_text_feats.device:
             self.lambda_text_feats = self.lambda_text_feats.to(image.device)
             
        lambda_logit_map_ebc = self.ebc_head(gated_feats, self.lambda_text_feats) # [B, 13, H, W]

        # 6. Calcola la mappa di densità (basata solo su EBC)
        if self.ebc_bin_centers.device != lambda_logit_map_ebc.device:
            self.ebc_bin_centers = self.ebc_bin_centers.to(lambda_logit_map_ebc.device)
            
        lambda_map_ebc = (lambda_logit_map_ebc.float().softmax(dim=1) * self.ebc_bin_centers).sum(dim=1, keepdim=True)
        
        den_map = lambda_map_ebc
        
        if self.training:
            # Restituiamo tutto ciò che serve alla QuadLoss
            return {
                "pred_logit_map": lambda_logit_map_ebc, # Per EBC loss (cls)
                "pred_den_map": den_map,               # Per Conteggio (aux)
                "pred_logit_pi_map": pi_logit_map,     # Per ZIP loss (reg)
                "pred_lambda_map": lambda_map_zip      # Per ZIP loss (reg)
            }
        else:
            return den_map * mask