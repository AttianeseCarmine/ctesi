import torch
import torch.nn as nn
import torch.nn.functional as F

class ZIPCLIPJointModel(nn.Module):
    def __init__(self, stage1_model, stage2_model):
        super().__init__()
        self.stage1 = stage1_model  # Il tuo filtro (VGG/ResNet + pi_head)
        self.stage2 = stage2_model  # Il tuo contatore (CLIP-EBC)
        
        # Congeliamo parzialmente i backbone? 
        # All'inizio sì, per sicurezza. Poi si sbloccano.
        # Per ora lasciamo tutto trainabile ma con LR basso.

    def forward(self, x):
        # 1. Forward Stage 1 (Struttura)
        out1 = self.stage1(x) 
        pi_logits = out1['pi_logits']
        pi_prob = torch.sigmoid(pi_logits)
        
        # 2. Forward Stage 2 (Conteggio Semantico)
        out2 = self.stage2(x)
        raw_density = out2['ebc_density'] # Nome corretto
        ebc_logits = out2['ebc_logits']
        
        # --- FIX DIMENSIONI (Nuova Parte) ---
        # Se le dimensioni non coincidono (succede in validation), adattiamo pi_prob
        if raw_density.shape[2:] != pi_prob.shape[2:]:
            pi_prob = F.interpolate(
                pi_prob, 
                size=raw_density.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        # ------------------------------------

        # 3. Soft Gating (Fusione)
        final_density = raw_density * pi_prob
        
        return {
            'pi_logits': pi_logits,
            'ebc_logits': ebc_logits,
            'raw_density': raw_density,
            'final_density': final_density,
            'pi_prob': pi_prob
        }