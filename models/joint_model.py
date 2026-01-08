import torch
import torch.nn as nn
import torch.nn.functional as F

class ZIPCLIPJointModel(nn.Module):
    """
    Modello Unificato ZIP-CLIP.
    - Training: Soft Gating (i gradienti fluiscono ovunque).
    - Inference: Hard Filtering (risparmia calcoli e rimuove falsi positivi).
    """
    def __init__(self, zip_model, clip_ebc_model, patch_size=224, stride=224, threshold=0.5):
        super().__init__()
        self.zip_model = zip_model
        self.clip_ebc_model = clip_ebc_model
        
        # Parametri per inferenza Hard
        self.patch_size = patch_size
        self.stride = stride
        self.threshold = threshold

    def forward(self, x):
        # Decide automaticamente la modalità in base a model.train() o model.eval()
        if self.training:
            return self._forward_soft_train(x)
        else:
            return self._forward_hard_inference(x)

    def _forward_soft_train(self, x):
        """
        Flow differenziabile per il training.
        Densità Finale = Densità CLIP * Probabilità ZIP
        """
        # 1. ZIP (Filtro)
        zip_out = self.zip_model(x)
        pi_logits = zip_out['pi_logits']
        prob_presence = torch.sigmoid(pi_logits) # [0, 1]

        # 2. CLIP (Contatore)
        clip_out = self.clip_ebc_model(x)
        
        # --- FIX: RECUPERO ROBUSTO DELLE CHIAVI (Risolve il crash) ---
        # Cerca la densità con varie chiavi possibili
        raw_density = None
        for key in ['ebc_density', 'pred_density', 'final_density', 'density']:
            if key in clip_out:
                raw_density = clip_out[key]
                break
        if raw_density is None:
            raise KeyError(f"Nessuna chiave di densità valida trovata in CLIP output. Chiavi presenti: {clip_out.keys()}")

        # Cerca i logits con varie chiavi possibili
        ebc_logits = None
        for key in ['ebc_logits', 'logits']:
            if key in clip_out:
                ebc_logits = clip_out[key]
                break
        
        # 3. Allineamento Dimensioni (se necessario)
        if prob_presence.shape[-2:] != raw_density.shape[-2:]:
            prob_presence = F.interpolate(prob_presence, size=raw_density.shape[-2:], mode='bilinear', align_corners=False)

        # 4. Soft Gating
        refined_density = raw_density * prob_presence

        return {
            'pi_logits': pi_logits,           # Per ZIP Loss
            'ebc_logits': ebc_logits,         # Per CLIP Loss
            'final_density': refined_density, # Per Count Loss
            'raw_density': raw_density        # Debug
        }

    def _forward_hard_inference(self, x):
        """
        Flow "Divide et Impera" per l'inferenza.
        Taglia l'immagine -> Filtra i blocchi vuoti -> Conta solo su quelli pieni.
        """
        B, C, H, W = x.shape
        
        # 1. Pad per rendere l'immagine divisibile per il patch_size
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        x_padded = F.pad(x, (0, pad_w, 0, pad_h))
        
        # 2. Unfold (Tiling dell'immagine)
        patches = F.unfold(x_padded, kernel_size=self.patch_size, stride=self.stride)
        # Reshape: [N_patches, C, H_patch, W_patch]
        patches = patches.permute(0, 2, 1).contiguous().view(-1, C, self.patch_size, self.patch_size)
        
        # 3. ZIP Filter (Decisione Hard)
        with torch.no_grad():
            zip_out = self.zip_model(patches)
            pi_logits = zip_out['pi_logits']
            
            # Score del patch: c'è almeno un punto con alta probabilità?
            # Usiamo Max Pooling sulla mappa di probabilità del patch
            patch_scores = torch.sigmoid(pi_logits).amax(dim=(1, 2, 3))
            
            # Decisione: Tengo o butto?
            mask_keep = patch_scores > self.threshold
            
        valid_patches = patches[mask_keep]
        total_count = 0.0
        
        # 4. CLIP Counter (Solo sui sopravvissuti)
        if valid_patches.size(0) > 0:
            clip_out = self.clip_ebc_model(valid_patches)
            
            # Anche qui usiamo la ricerca robusta della chiave
            densities = None
            for key in ['ebc_density', 'pred_density', 'final_density', 'density']:
                if key in clip_out:
                    densities = clip_out[key]
                    break
            
            if densities is not None:
                total_count = densities.sum().item()
            
        return {
            'pred_count': torch.tensor([total_count], device=x.device),
            'n_patches_total': patches.size(0),
            'n_patches_kept': valid_patches.size(0)
        }