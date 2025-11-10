import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Tuple
from copy import deepcopy

from .vit import vit_names_and_weights, _vit
from .convnext import convnext_names_and_weights, _convnext
from .resnet import resnet_names_and_weights, _resnet
from .mobileclip import mobileclip_names_and_weights, _mobileclip

from .utils import encode_text, optimize_text_prompts
from ..utils import conv1x1

supported_models_and_weights = deepcopy(vit_names_and_weights)
supported_models_and_weights.update(convnext_names_and_weights)
supported_models_and_weights.update(resnet_names_and_weights)
supported_models_and_weights.update(mobileclip_names_and_weights)


class CLIP_EBC(nn.Module):
    def __init__(
        self,
        model_name: str,
        weight_name: str,
        block_size: Optional[int] = None,
        bins: Optional[List[Tuple[float, float]]] = None,
        bin_centers: Optional[List[float]] = None,
        zero_inflated: Optional[bool] = True,
        num_vpt: Optional[int] = None,
        vpt_drop: Optional[float] = None,
        input_size: Optional[int] = None,
        adapter: Optional[bool] = False,
        adapter_reduction: Optional[int] = None,
        lora: Optional[bool] = False,
        lora_rank: Optional[int] = None,
        lora_alpha: Optional[float] = None,
        lora_dropout: Optional[float] = None,
        text_prompts: Optional[Dict[str, List[str]]] = None,
        norm: Optional[str] = "none",
        act: Optional[str] = "none",
        
        # === INIZIO MODIFICA: Aggiunti parametri di Gating (come P2R-ZIP) ===
        pi_thresh: float = 0.5,
        gate_mode: str = "multiply",
        # === FINE MODIFICA ===

    ) -> None:
        super().__init__()
        if "mobileclip" in model_name.lower() or "vit" in model_name.lower():
            model_name = model_name.replace("_", "-")
        assert model_name in supported_models_and_weights, f"Model name should be one of {list(supported_models_and_weights.keys())}, but got {model_name}."
        assert weight_name in supported_models_and_weights[model_name], f"Pretrained should be one of {supported_models_and_weights[model_name]}, but got {weight_name}."
        assert len(bins) == len(bin_centers), f"Expected bins and bin_centers to have the same length, got {len(bins)} and {len(bin_centers)}"
        assert len(bins) >= 2, f"Expected at least 2 bins, got {len(bins)}"
        assert all(len(b) == 2 for b in bins), f"Expected bins to be a list of tuples of length 2, got {bins}"
        bins = [(float(b[0]), float(b[1])) for b in bins]
        assert all(bin[0] <= p <= bin[1] for bin, p in zip(bins, bin_centers)), f"Expected bin_centers to be within the range of the corresponding bin, got {bins} and {bin_centers}"
        
        # === INIZIO MODIFICA: Asserzione per la logica seriale ===
        assert zero_inflated, "La logica seriale 'ZIP-in-serie-a-EBC' richiede zero_inflated=True per separare pi_head e lambda_head."
        # === FINE MODIFICA ===

        self.model_name = model_name
        self.weight_name = weight_name
        self.block_size = block_size
        self.bins = bins
        self.register_buffer("bin_centers", torch.tensor(bin_centers, dtype=torch.float32, requires_grad=False).view(1, -1, 1, 1))
        self.zero_inflated = zero_inflated
        self.text_prompts = text_prompts
        
        # === INIZIO MODIFICA: Salvataggio parametri di Gating ===
        self.pi_thresh = pi_thresh
        self.gate_mode = gate_mode
        # === FINE MODIFICA ===

        # Image encoder
        if model_name in vit_names_and_weights:
            assert num_vpt is not None and num_vpt >= 0, f"Number of VPT tokens should be greater than 0, but got {num_vpt}."
            vpt_drop = 0. if vpt_drop is None else vpt_drop
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
                input_size=(input_size, input_size),
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
        elif model_name in resnet_names_and_weights:
            self.backbone = _resnet(
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
        elif model_name in mobileclip_names_and_weights:
            self.backbone = _mobileclip(
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

        self._build_text_feats()
        self._build_head()

    def _build_text_feats(self):
        # --- FUNZIONE MODIFICATA PER GESTIRE I GRUPPI DI PROMPT ---
        import torch
        from .utils import encode_text # Assicurati che importi la funzione corretta
        
        model_name = self.model_name
        weight_name = self.weight_name
        device = next(self.backbone.parameters()).device
        
        # 1. Processa i prompt della PI-Head
        pi_text_prompts_groups = self.text_prompts["pi"]
        pi_text_feats_list = []
        
        for group in pi_text_prompts_groups:
            # group è una List[str], es: ["prompt1", "prompt2", ...]
            # encode_text si aspetta List[str] e ritorna [Num_Prompts, Feature_Dim]
            group_feats = encode_text(model_name, weight_name, group).to(device)
            
            # Calcola la media dei feature vector per questo gruppo
            avg_feat = group_feats.mean(dim=0)
            
            # Ri-normalizza il feature vector medio
            avg_feat /= avg_feat.norm(dim=-1, keepdim=True)
            pi_text_feats_list.append(avg_feat)
        
        # Registra il buffer finale (es. [2, Feature_Dim])
        self.register_buffer("pi_text_feats", torch.stack(pi_text_feats_list))

        # 2. Processa i prompt della LAMBDA-Head (EBC)
        lambda_text_prompts_groups = self.text_prompts["lambda"]
        lambda_text_feats_list = []
        
        for group in lambda_text_prompts_groups:
            # group è una List[str], es: ["one person"]
            group_feats = encode_text(model_name, weight_name, group).to(device)
            
            # Calcola la media (anche se c'è un solo prompt, è più robusto)
            avg_feat = group_feats.mean(dim=0)
            
            # Ri-normalizza
            avg_feat /= avg_feat.norm(dim=-1, keepdim=True)
            lambda_text_feats_list.append(avg_feat)
        
        # Registra il buffer finale (es. [13, Feature_Dim])
        self.register_buffer("lambda_text_feats", torch.stack(lambda_text_feats_list))

    def _build_head(self) -> None:
        # (Questo metodo rimane identico)
        in_channels = self.backbone.in_features
        out_channels = self.backbone.out_features
        if self.zero_inflated:
            self.pi_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)
            self.lambda_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)

            self.pi_head = conv1x1(in_channels, out_channels, bias=False)
            self.lambda_head = conv1x1(in_channels, out_channels, bias=False)

        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)
            self.head = conv1x1(in_channels, out_channels, bias=False)

    # ======================================================================
    # === INIZIO MODIFICA: Metodo forward() seriale (Stile P2R-ZIP) ===
    # ======================================================================
    def forward(self, image: Tensor):
        # 1. Backbone condiviso
        image_feats = self.backbone(image)  # [B, C_feat, H, W]
        
        # --- MODULO A: ZIP (pi_head) ---
        # 2. Calcola i logits di 'pi' (il modulo ZIP)
        pi_image_feats_head = self.pi_head(image_feats) # [B, C_out, H, W]
        pi_image_feats_norm = F.normalize(pi_image_feats_head.permute(0, 2, 3, 1), p=2, dim=-1) # [B, H, W, C_out]
        pi_logit_scale = self.pi_logit_scale.exp()
        pi_logit_map = pi_logit_scale * pi_image_feats_norm @ self.pi_text_feats.t() # [B, H, W, 2]
        pi_logit_map = pi_logit_map.permute(0, 3, 1, 2)  # [B, 2, H, W]

        # 3. Crea la maschera di gating (logica P2R-ZIP)
        with torch.no_grad():
            pi_softmax = pi_logit_map.softmax(dim=1)
            pi_not_zero_prob = pi_softmax[:, 1:2]  # [B, 1, H, W] (Probabilità di essere NON-vuoto)
            # Maschera binaria (hard-gating)
            mask = (pi_not_zero_prob > self.pi_thresh).float()

        # 4. Applica il Gating alle feature
        if self.gate_mode == "multiply":
            gated_feats = image_feats * mask
        else: # Fallback
            gated_feats = image_feats * mask # Default a multiply

        # --- MODULO B: CLIP-EBC (lambda_head) ---
        # 5. Calcola i logits di 'lambda' (il modulo EBC) dalle *feature mascherate*
        lambda_image_feats_head = self.lambda_head(gated_feats) # [B, C_out, H, W]
        lambda_image_feats_norm = F.normalize(lambda_image_feats_head.permute(0, 2, 3, 1), p=2, dim=-1) # [B, H, W, C_out]
        lambda_logit_scale = self.lambda_logit_scale.exp()
        lambda_logit_map = lambda_logit_scale * lambda_image_feats_norm @ self.lambda_text_feats.t() # [B, H, W, N-1]
        lambda_logit_map = lambda_logit_map.permute(0, 3, 1, 2) # [B, N-1, H, W]

        # 6. Calcola la mappa di densità (basata solo su lambda)
        lambda_map = (lambda_logit_map.softmax(dim=1) * self.bin_centers[:, 1:]).sum(dim=1, keepdim=True)
        
        # L'output di lambda_map è già "mascherato" perché ha ricevuto feature azzerate
        den_map = lambda_map
        
        if self.training:
            # Restituiamo entrambi i set di logits per le loss separate (ZIPNLL + ZICE)
            return pi_logit_map, lambda_logit_map, lambda_map, den_map
        else:
            # In modalità inferenza, applichiamo la maschera binaria
            # per assicurare che i blocchi vuoti siano *esattamente* zero.
            return den_map * mask
    # ======================================================================
    # === FINE MODIFICA ===
    # ======================================================================
        

def _clip_ebc(
    model_name: str,
    weight_name: str,
    block_size: Optional[int] = None,
    bins: Optional[List[Tuple[float, float]]] = None,
    bin_centers: Optional[List[float]] = None,
    zero_inflated: Optional[bool] = True,
    num_vpt: Optional[int] = None,
    vpt_drop: Optional[float] = None,
    input_size: Optional[int] = None,
    adapter: Optional[bool] = False,
    adapter_reduction: Optional[int] = None,
    lora: Optional[bool] = False,
    lora_rank: Optional[int] = None,
    lora_alpha: Optional[float] = None,
    lora_dropout: Optional[float] = None,
    text_prompts: Optional[List[str]] = None,
    norm: Optional[str] = "none",
    act: Optional[str] = "none",
    
    # === INIZIO MODIFICA: Aggiunti argomenti al costruttore helper ===
    pi_thresh: float = 0.5,
    gate_mode: str = "multiply",
    # === FINE MODIFICA ===
) -> CLIP_EBC:
    return CLIP_EBC(
        model_name=model_name,
        weight_name=weight_name,
        block_size=block_size,
        bins=bins,
        bin_centers=bin_centers,
        zero_inflated=zero_inflated,
        num_vpt=num_vpt,
        vpt_drop=vpt_drop,
        input_size=input_size,
        adapter=adapter,
        adapter_reduction=adapter_reduction,
        lora=lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        text_prompts=text_prompts,
        norm=norm,
        act=act,
        
        # === INIZIO MODIFICA: Passaggio nuovi argomenti ===
        pi_thresh=pi_thresh,
        gate_mode=gate_mode,
        # === FINE MODIFICA ===
    )