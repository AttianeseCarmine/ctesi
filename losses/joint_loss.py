import torch
import torch.nn as nn

class JointLoss(nn.Module):
    def __init__(self, clip_loss_fn, lambda_zip=1.0, lambda_clip=1.0, lambda_count=10.0):
        super().__init__()
        # Loss per Stage 1 (Filtro)
        # Usa il pos_weight che hai trovato utile (es. 5.0 o 15.0)
        self.zip_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).cuda())
        
        # Loss per Stage 2 (Semantica dei Bin)
        self.clip_loss = clip_loss_fn
        
        # Loss Finale (Consistenza Conteggio)
        # L1 Loss sulla densità finale filtrata vs Ground Truth
        self.count_loss = nn.L1Loss()
        
        self.w1 = lambda_zip
        self.w2 = lambda_clip
        self.w3 = lambda_count

    def forward(self, outputs, targets):
        # targets: {'mask', 'counts', 'density'}
        
        # 1. Insegna allo Stage 1 a distinguere Muro/Folla
        l_zip = self.zip_loss(outputs['pi_logits'], targets['mask'])
        
        # 2. Insegna allo Stage 2 a capire "quante persone" (anche se vede il muro)
        l_clip, _ = self.clip_loss(outputs['ebc_logits'], targets['counts'])
        
        # 3. Loss Congiunta: Il conteggio FINALE (Filtrato) deve essere giusto.
        # Questa è la parte potente: se Stage 2 sbaglia ma Stage 1 corregge, la loss è bassa.
        # Se entrambi sbagliano, la loss è alta.
        l_count = self.count_loss(outputs['final_density'], targets['density'])
        
        total_loss = (self.w1 * l_zip) + (self.w2 * l_clip) + (self.w3 * l_count)
        
        return total_loss, {
            'l_zip': l_zip.item(), 
            'l_clip': l_clip.item(), 
            'l_count': l_count.item()
        }