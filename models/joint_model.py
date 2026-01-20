# models/joint_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ZIPCLIPJointModel(nn.Module):
    def __init__(self, stage1_model, stage2_model, steepness=1.0): # Inizia con steepness bassa
        super().__init__()
        self.stage1 = stage1_model
        self.stage2 = stage2_model
        
        # Aligner per correggere micro-disallineamenti
        self.mask_aligner = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        nn.init.dirac_(self.mask_aligner.weight)
        nn.init.zeros_(self.mask_aligner.bias)
        
        self.steepness = steepness

    def forward(self, x):
        # 1. ZIP Stage (Filtro)
        out1 = self.stage1(x)
        # Supporto per dizionari output diversi
        if isinstance(out1, dict):
            pi_logits_raw = out1.get('pi_logits', out1.get('logit_pi', None))
        else:
            pi_logits_raw = out1

        # 2. CLIP Stage (Contatore)
        out2 = self.stage2(x)

        if isinstance(out2, (tuple, list)):
            # Stage2 ritorna (pred_class, pred_density)
            ebc_logits, raw_density = out2[0], out2[1]
        else:
            # regression case: ritorna direttamente la density
            raw_density = out2
            ebc_logits = None
            
        # --- ALLINEAMENTO ---
        if raw_density.shape[2:] != pi_logits_raw.shape[2:]:
            pi_logits_raw = F.interpolate(
                pi_logits_raw, 
                size=raw_density.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )

        # --- GATING STRATEGY (P2R-ZIP Style) ---
        # 1. Aligner
        pi_logits_aligned = self.mask_aligner(pi_logits_raw)
        
        # 2. Sigmoide Dinamica
        # Training: steepness bassa (~1.0) -> Soft Mask -> Gradienti passano
        # Eval: steepness alta (~10.0) -> Hard Mask -> Pulizia rumore
        pi_prob = torch.sigmoid(pi_logits_aligned * self.steepness)
        
        # 3. Applicazione
        final_density = raw_density * pi_prob
        
        return {
            'pi_logits': pi_logits_aligned,
            'ebc_logits': ebc_logits,
            'raw_density': raw_density,
            'final_density': final_density,
            'pi_prob': pi_prob,
            
            # Passiamo anche i dati grezzi per le loss ausiliarie
            'zip_out': out1 
        }