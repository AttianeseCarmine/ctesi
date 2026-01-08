import torch
import torch.nn as nn
import torch.nn.functional as F

class ZIPCLIPJointModel(nn.Module):
    """
    Modello congiunto per lo Stage 3.
    Esegue sia il modello ZIP (Stage 1) che CLIP-EBC (Stage 2)
    e unisce i loro output in un unico dizionario.
    """
    def __init__(self, stage1_model, stage2_model):
        super().__init__()
        self.stage1 = stage1_model
        self.stage2 = stage2_model
        
    def forward(self, x):
        # 1. Forward Stage 1 (ZIP - ResNet50)
        # Restituisce: {'pi_logits': ..., 'features': ...}
        out1 = self.stage1(x)
        
        # 2. Forward Stage 2 (CLIP - EBC)
        # Restituisce: {'ebc_density': ..., 'ebc_logits': ..., 'bin_probs': ...}
        out2 = self.stage2(x)
        
        # 3. Merge dei risultati
        outputs = {}
        outputs.update(out1)  # Inserisce pi_logits
        outputs.update(out2)  # Inserisce ebc_density
        
        return outputs

class DivideAndConquerStage3(nn.Module):
    def __init__(self, zip_model, clip_ebc_model, tile_size=224, threshold=0.5):
        """
        Args:
            zip_model: Modello Stage 1 (Filtro).
            clip_ebc_model: Modello Stage 2 (Contatore).
            tile_size: Dimensione del blocco (default 224 per CLIP).
            threshold: Soglia di probabilità ZIP (0-1). Se prob < threshold, il blocco è scartato.
        """
        super().__init__()
        self.zip_model = zip_model
        self.clip_ebc_model = clip_ebc_model
        self.tile_size = tile_size
        self.threshold = threshold
        
        # Impostiamo i sottomodelli in eval (Stage 3 Gate è solo inferenza)
        self.zip_model.eval()
        self.clip_ebc_model.eval()

    def forward(self, x):
        """
        x: Immagine intera (Batch, C, H, W). Il batch size deve essere 1 per ora.
        Return: (Conteggio totale, Numero blocchi tenuti, Numero blocchi totali)
        """
        B, C, H, W = x.shape
        
        # 1. Padding: Rende l'immagine perfettamente divisibile per tile_size
        pad_h = (self.tile_size - H % self.tile_size) % self.tile_size
        pad_w = (self.tile_size - W % self.tile_size) % self.tile_size
        x_padded = F.pad(x, (0, pad_w, 0, pad_h))
        
        # 2. DIVIDE: Estrazione Patches (Unfold)
        # Stride = Tile Size -> Nessuna sovrapposizione tra i blocchi
        patches = x_padded.unfold(2, self.tile_size, self.tile_size).unfold(3, self.tile_size, self.tile_size)
        # Shape: (B, C, Rows, Cols, Tile, Tile)
        
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(-1, C, self.tile_size, self.tile_size)
        # Ora patches è un batch di "mini-immagini": (N_Blocchi, 3, 224, 224)

        # 3. FILTER (ZIP): Passiamo tutto al filtro
        with torch.no_grad():
            zip_output = self.zip_model(patches) 
            
            # Calcoliamo uno score per ogni blocco (max probability all'interno del blocco)
            # zip_output è (N, 1, H_out, W_out). Adaptive Max Pool lo riduce a (N, 1, 1, 1)
            block_scores = F.adaptive_max_pool2d(zip_output, (1, 1)).view(-1)
            
            # Maschera binaria: Chi passa il test?
            keep_mask = block_scores > self.threshold
            
            num_kept = keep_mask.sum().item()
            num_total = patches.size(0)
            
            # Caso limite: Se nessun blocco passa, restituiamo 0
            if num_kept == 0:
                return torch.tensor(0.0, device=x.device), 0, num_total

            # Selezioniamo solo i blocchi "promossi"
            valid_patches = patches[keep_mask]

        # 4. COUNT (CLIP-EBC): Passiamo al contatore solo i sopravvissuti
        with torch.no_grad():
            clip_output = self.clip_ebc_model(valid_patches)
            # clip_output è la density map o il count dei blocchi. Sommiamo tutto.
            final_count = clip_output.sum()
            
        return final_count, num_kept, num_total