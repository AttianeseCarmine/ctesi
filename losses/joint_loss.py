import torch
import torch.nn as nn
import torch.nn.functional as F

class JointLoss(nn.Module):
    def __init__(self, clip_loss_fn, lambda_zip=1.0, lambda_clip=1.0, lambda_count=10.0):
        super().__init__()
        # Loss per Stage 1 (Filtro) 
        self.zip_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0]).cuda())
        
        # Loss per Stage 2 (CLIP-EBC) 
        self.clip_loss = clip_loss_fn
        
        # Loss Finale (Consistenza Conteggio)
        self.count_loss = nn.L1Loss()
        
        self.w1 = lambda_zip
        self.w2 = lambda_clip
        self.w3 = lambda_count

    def forward(self, outputs, targets):
        """
        outputs: dizionario con 'pi_logits', 'ebc_logits', 'final_density'
        targets: dizionario con 'mask', 'counts', 'density', 'points'
        """
        
        # 1. Loss ZIP (Filtro Sfondo/Persone)
        l_zip = self.zip_loss(outputs['pi_logits'], targets['mask'])
        
        # 2. Loss CLIP (Conteggio Semantico + OT/TV)
        l_clip, clip_logs = self.clip_loss(
            outputs['ebc_logits'], 
            targets['counts'],          # Target Bin/Conteggi per blocco
            targets['density'],         # Target Densità (per Loss TV/OT)
            targets['points']           # Target Punti (per Loss OT)
        )
        
        # 3. Loss Congiunta (Risultato Finale) - FIX DIMENSIONI
        pred_density = outputs['final_density']
        gt_density = targets['density']
        
        # Se le dimensioni non coincidono (es. 28x28 vs 448x448), ridimensioniamo la GT
        if pred_density.shape[-2:] != gt_density.shape[-2:]:
            h_out, w_out = pred_density.shape[-2:]
            h_in, w_in = gt_density.shape[-2:]
            
            scale_factor = (h_in * w_in) / (h_out * w_out)
            
            gt_density_resized = F.adaptive_avg_pool2d(gt_density, (h_out, w_out)) * scale_factor
        else:
            gt_density_resized = gt_density

        l_final = self.count_loss(pred_density, gt_density_resized)
        
        # Somma pesata
        total_loss = (self.w1 * l_zip) + (self.w2 * l_clip) + (self.w3 * l_final)
        
        # Logging
        loss_dict = {
            'l_zip': l_zip.item(), 
            'l_clip': l_clip.item(), 
            'l_final': l_final.item(),
            'l_count': l_final.item() # Alias per compatibilità log
        }
        
        if isinstance(clip_logs, dict):
            loss_dict.update(clip_logs)
            
        return total_loss, loss_dict