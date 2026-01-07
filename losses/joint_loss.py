import torch
import torch.nn as nn
import torch.nn.functional as F

class ZIPCLIPJointModel(nn.Module):
    def __init__(self, stage1_model, stage2_model):
        super().__init__()
        self.stage1 = stage1_model  # ZIPModel (Filtro)
        self.stage2 = stage2_model  # CLIPEBCModel (Contatore)

    def forward(self, x):
        # 1. Forward Stage 1 (Struttura/Filtro)
        # Il modello ZIP restituisce 'pi_logits'
        out1 = self.stage1(x) 
        pi_logits = out1['pi_logits']
        pi_prob = torch.sigmoid(pi_logits)
        
        # 2. Forward Stage 2 (Conteggio Semantico)
        # Il modello CLIP-EBC restituisce 'ebc_density', 'ebc_logits', etc.
        out2 = self.stage2(x)
        raw_density = out2['ebc_density']
        ebc_logits = out2['ebc_logits']
        
        # Gestione dimensioni diverse (utile in validation/test con immagini di dimensioni varie)
        if raw_density.shape[2:] != pi_prob.shape[2:]:
            pi_prob = F.interpolate(
                pi_prob, 
                size=raw_density.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )

        # 3. Soft Gating (FUSIONE)
        # Applica la probabilità strutturale alla densità semantica
        final_density = raw_density * pi_prob
        
        # 4. Return Dictionary [IMPORTANTE]
        # La joint_loss si aspetta 'final_density', 'pi_logits' e 'ebc_logits'
        return {
            'pi_logits': pi_logits,       # Per la Loss Strutturale (ZIP)
            'ebc_logits': ebc_logits,     # Per la Loss Semantica (CLIP)
            'raw_density': raw_density,   # Densità pura (non filtrata)
            'final_density': final_density, # <--- QUESTA CHIAVE MANCAVA
            'pi_prob': pi_prob,           # Utile per debug/visualizzazione
            'bin_probs': out2.get('bin_probs', None) # Passa le probabilità dei bin se presenti
        }