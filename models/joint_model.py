import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import types

# ==============================================================================
# 🛠️ PATCH DINAMICA PER VISION TRANSFORMER
# Questa funzione sostituisce il forward originale del ViT a runtime.
# Permette di gestire immagini di qualsiasi dimensione interpolando 
# correttamente i Positional Embeddings.
# ==============================================================================
def forward_vit_dynamic(self, x: torch.Tensor):
    # x: (Batch, 3, H, W)
    x = self.conv1(x)  # shape = [*, width, grid, grid]
    
    # Catturiamo la geometria attuale della griglia
    B, C, H_grid, W_grid = x.shape
    
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    
    # Aggiunta CLS Token
    cls_token = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls_token, x], dim=1)  # shape = [*, grid ** 2 + 1, width]

    # --- INTERPOLAZIONE POSITIONAL EMBEDDING ---
    pos_embed = self.positional_embedding.to(x.dtype)
    cls_pos_embed = pos_embed[0, :]
    patch_pos_embed = pos_embed[1:, :] # (NumPatchesTrain, Dim)
    
    orig_num_patches = patch_pos_embed.shape[0]
    # Assumiamo training quadrato (es. 14x14 = 196)
    orig_grid_size = int(math.sqrt(orig_num_patches)) 
    
    # Se la griglia attuale differisce da quella originale, interpoliamo
    if H_grid * W_grid != orig_num_patches:
        # Reshape a 2D: (1, Dim, OrigH, OrigW)
        patch_pos_embed = patch_pos_embed.reshape(1, orig_grid_size, orig_grid_size, -1).permute(0, 3, 1, 2)
        
        # Interpolazione BICUBIC (fondamentale per mantenere la struttura spaziale)
        patch_pos_embed = F.interpolate(
            patch_pos_embed, 
            size=(H_grid, W_grid), 
            mode='bicubic', 
            align_corners=False
        )
        
        # Torna flat: (NewN, Dim)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2).squeeze(0)
    
    # Somma finale embeddings
    pos_embed = torch.cat((cls_pos_embed.unsqueeze(0), patch_pos_embed), dim=0)
    x = x + pos_embed

    # Passaggi standard Transformer
    x = self.ln_pre(x)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    return x

# ==============================================================================
# CLASSE JOINT MODEL
# ==============================================================================
class ZIPCLIPJointModel(nn.Module):
    def __init__(self, stage1_model, stage2_model, steepness=0.5): 
        super().__init__()
        self.stage1 = stage1_model
        self.stage2 = stage2_model
        
        # Applica AUTOMATICAMENTE il fix se rileva un ViT
        self._apply_vit_fix()

        # Aligner per correggere micro-disallineamenti (ora funzionerà davvero!)
        self.mask_aligner = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        nn.init.dirac_(self.mask_aligner.weight)
        nn.init.zeros_(self.mask_aligner.bias)
        
        self.steepness = steepness

    def _apply_vit_fix(self):
        """
        Cerca il modulo VisionTransformer dentro stage2 e sostituisce il metodo forward.
        """
        vit_module = None
        # Lista di percorsi probabili dove si nasconde il ViT
        possible_paths = [
            ['backbone', 'visual'],        
            ['image_encoder'],             
            ['module', 'image_encoder'],
            ['backbone']
        ]
        
        found = False
        for path in possible_paths:
            m = self.stage2
            try:
                for attr in path:
                    m = getattr(m, attr)
                # Verifica "impronta digitale" del ViT
                if hasattr(m, 'positional_embedding') and hasattr(m, 'conv1') and hasattr(m, 'class_embedding'):
                    vit_module = m
                    found = True
                    break
            except AttributeError:
                continue
        
        if found and vit_module is not None:
            print(f"[JointModel] 🛠️  ViT Backbone rilevata ({type(vit_module).__name__}).")
            print("[JointModel] 💉 Iniezione Patch: Attivazione interpolazione dinamica dei Positional Embeddings.")
            
            # MONKEY PATCH: Sostituiamo il metodo sull'istanza specifica
            # Usiamo MethodType per legare la funzione all'istanza (self funzionerà correttamente)
            vit_module.forward = types.MethodType(forward_vit_dynamic, vit_module)
        else:
            # Se è una ResNet o altro, non facciamo nulla
            pass

    def forward(self, x):
        # 1. ZIP Stage (Filtro)
        out1 = self.stage1(x)
        if isinstance(out1, dict):
            pi_logits_raw = out1.get('pi_logits', out1.get('logit_pi', None))
        else:
            pi_logits_raw = out1

        # 2. CLIP Stage (Contatore)
        # Ora chiamerà la versione patchata se è un ViT
        out2 = self.stage2(x)

        if isinstance(out2, (tuple, list)):
            ebc_logits, raw_density = out2[0], out2[1]
        else:
            raw_density = out2
            ebc_logits = None
            
        # --- ALLINEAMENTO ---
        # Adattiamo la maschera ZIP alla dimensione di CLIP
        target_h, target_w = raw_density.shape[2:]
        if pi_logits_raw.shape[2:] != (target_h, target_w):
            pi_logits_raw = F.interpolate(
                pi_logits_raw, 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            )

        # 3. Gating Strategy
        pi_logits_aligned = self.mask_aligner(pi_logits_raw)
        
        # Sigmoide con steepness variabile
        pi_prob = torch.sigmoid(pi_logits_aligned * self.steepness)
        
        # 4. Applicazione Maschera
        alpha = 0.5
        final_density = raw_density * (alpha + (1.0 - alpha) * pi_prob)

        return {
            'pi_logits': pi_logits_aligned,
            'ebc_logits': ebc_logits,
            'raw_density': raw_density,
            'final_density': final_density,
            'pi_prob': pi_prob,
            'zip_out': out1 
        }