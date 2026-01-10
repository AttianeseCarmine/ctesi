import torch
import torch.nn as nn
import torch.nn.functional as F

class ZIPCLIPJointModel(nn.Module):
    def __init__(self, stage1_model, stage2_model, steepness=10.0):
        """
        Modello congiunto che unisce ZIP (Filtro) e CLIP (Contatore).
        
        Args:
            stage1_model: Modello ZIP pre-addestrato (backbone + head)
            stage2_model: Modello CLIP-EBC pre-addestrato
            steepness: Fattore di pendenza della sigmoide per il gating.
                       Valori alti (es. 10) rendono il filtro quasi binario (0 o 1).
        """
        super().__init__()
        self.stage1 = stage1_model
        self.stage2 = stage2_model
        
        # --- MASK ALIGNER ---
        # Piccolo layer convoluzionale che impara a "micro-spostare" la maschera 
        # di ZIP per allinearla perfettamente alla griglia di CLIP.
        # Inizializzato come identità (non fa nulla all'inizio).
        self.mask_aligner = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        nn.init.dirac_(self.mask_aligner.weight)
        nn.init.zeros_(self.mask_aligner.bias)
        
        self.steepness = steepness

    def forward(self, x):
        # 1. Forward Stage 1 (Il Filtro ZIP)
        # Otteniamo i logits grezzi (prima della sigmoide)
        out1 = self.stage1(x) 
        pi_logits_raw = out1['pi_logits'] 
        
        # 2. Forward Stage 2 (Il Contatore CLIP)
        out2 = self.stage2(x)
        raw_density = out2['ebc_density']
        ebc_logits = out2['ebc_logits']
        
        # --- ALLINEAMENTO DIMENSIONALE ---
        # Se le risoluzioni non coincidono (es. padding diverso o VGG vs ResNet),
        # interpoliamo la maschera di ZIP per matchare la densità di CLIP.
        if raw_density.shape[2:] != pi_logits_raw.shape[2:]:
            pi_logits_raw = F.interpolate(
                pi_logits_raw, 
                size=raw_density.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )

        # --- MASCHERA INTELLIGENTE ---
        
        # A. Allineamento Spaziale (Aligner)
        # Corregge piccoli errori di posizionamento tra i due modelli
        pi_logits_aligned = self.mask_aligner(pi_logits_raw)
        
        # B. Hard Gating (Sigmoide Ripida)
        # Trasforma i dubbi (0.4) in certezze (0.0) e le quasi-certezze (0.6) in (1.0).
        # Questo spegne completamente il rumore di fondo.
        gate = torch.sigmoid(pi_logits_aligned * self.steepness)
        
        # 3. Applicazione del Filtro
        # Se gate è 0 (sfondo), la densità diventa 0. Se è 1, passa il conteggio di CLIP.
        final_density = raw_density * gate
        
        return {
            'pi_logits': pi_logits_aligned, # Usiamo quelli allineati per la loss
            'ebc_logits': ebc_logits,
            'raw_density': raw_density,     # Cosa vedeva CLIP prima del filtro
            'final_density': final_density, # Il risultato finale pulito
            'pi_prob': gate                 # La maschera binaria usata
        }