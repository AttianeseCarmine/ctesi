import torch
import torch.nn as nn
import torch.nn.functional as F

class JointLoss(nn.Module):
    def __init__(self, clip_loss_fn, lambda_zip=1.0, lambda_clip=1.0, lambda_count=10.0):
        super().__init__()
        # Loss per Stage 1 (Filtro)
        # Nota: togliamo .cuda() qui per evitare errori di init. 
        # PyTorch sposterà il tensore quando farai joint_criterion.to(device)
        self.register_buffer('pos_weight', torch.tensor([5.0]))
        self.zip_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        
        # Loss per Stage 2 (Semantica dei Bin / DACELossWrapper)
        self.clip_loss = clip_loss_fn
        
        # Loss Finale (Consistenza Conteggio - L1 sulla mappa finale)
        self.count_loss = nn.L1Loss()
        
        self.w1 = lambda_zip
        self.w2 = lambda_clip
        self.w3 = lambda_count

    def forward(self, outputs, targets):
        """
        outputs: dizionario con keys ['pi_logits', 'ebc_logits', 'final_density']
        targets: dizionario con keys ['mask', 'counts', 'density']
        """
        
        # 1. Loss Strutturale (ZIP): Impara la maschera binaria
        l_zip = self.zip_loss(outputs['pi_logits'], targets['mask'])
        
        # 2. Loss Semantica (CLIP): Impara la densità grezza dai bin
        # Nota: clip_loss_fn (DACELossWrapper) deve ritornare una tupla (loss, dict)
        l_clip, _ = self.clip_loss(outputs['ebc_logits'], targets['counts'])
        
        # 3. Loss Congiunta (Refinement): Impara il conteggio finale pulito
        # Qui confrontiamo la densità GIA' FILTRATA con la Ground Truth
        l_count = self.count_loss(outputs['final_density'], targets['density'])
        
        # Somma pesata
        total_loss = (self.w1 * l_zip) + (self.w2 * l_clip) + (self.w3 * l_count)
        
        # Ritorniamo loss scalare + dizionario per logging
        return total_loss, {
            'l_zip': l_zip.item(), 
            'l_clip': l_clip.item(), 
            'l_count': l_count.item()
        }