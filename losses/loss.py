import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union

# Importa le uniche dipendenze necessarie e presenti nel tuo progetto
from .zero_inflated_poisson_nll import ZIPoissonNLL, ZICrossEntropy

EPS = 1e-8

# --- FUNZIONI HELPER (da .utils mancante) ---
# Queste funzioni erano in 'losses/utils.py' e causavano un errore.
# Le ho spostate qui per rimuovere la dipendenza.

def _reshape_density(gt_den_map: Tensor, block_size: int) -> Tensor:
    """Raggruppa la density map in blocchi."""
    B, C, H, W = gt_den_map.shape
    assert H % block_size == 0 and W % block_size == 0, \
        f"Le dimensioni H e W ({H}x{W}) non sono divisibili per block_size ({block_size})"
    
    gt_den_map = gt_den_map.view(
        B, C, H // block_size, block_size, W // block_size, block_size
    )
    gt_den_map = gt_den_map.permute(0, 1, 2, 4, 3, 5).contiguous()
    gt_den_map = gt_den_map.view(
        B, C, (H // block_size) * (W // block_size), block_size, block_size
    )
    gt_den_map = gt_den_map.sum(dim=(-1, -2))
    gt_den_map = gt_den_map.view(
        B, C, H // block_size, W // block_size
    )
    return gt_den_map
def _bin_count(gt_den_map: Tensor, bins: List[Tuple[float, float]]) -> Tensor:
    """Converte la density map in una mappa di classi (bin)."""
    
    # --- MODIFICA: Converti i bin in float PRIMA di usarli ---
    float_bins = [(float(low), float(high)) for low, high in bins]
    
    B, C, H, W = gt_den_map.shape
    num_bins = len(bins)
    
    # Inizializza la mappa delle classi (B, H, W)
    gt_class_map = torch.full(
        (B, H, W), fill_value=-1, dtype=torch.long, device=gt_den_map.device # Inizializza con -1 (invalido)
    )
    
    gt_den_map_flat = gt_den_map.squeeze(1) # Rimuovi canale C (B, H, W)
    
    # Assegna ogni blocco al bin corretto
    # NOTA: I bin si sovrappongono (es. [0,0] e [1,1]), usiamo la logica dei bin di EBC
    # che si basa sui *centri* dei bin.
    
    # Crea i bordi dei bin
    bin_edges = [float_bins[0][0]] # Inizia con 0
    for i in range(len(float_bins) - 1):
        # Il bordo è a metà tra i centri
        center_curr = (float_bins[i][0] + float_bins[i][1]) / 2
        center_next = (float_bins[i+1][0] + float_bins[i+1][1]) / 2
        border = (center_curr + center_next) / 2
        bin_edges.append(border)
    bin_edges.append(float('inf')) # Bordo finale

    # Esempio: bin_edges = [0.0, 0.5, 1.5, 2.5, ..., 14.5, inf]

    for i in range(num_bins):
        low = bin_edges[i]
        high = bin_edges[i+1]
        
        if i == 0:
             # Il primo bin (classe 0) include lo 0
            mask = (gt_den_map_flat >= low) & (gt_den_map_flat < high)
        else:
            mask = (gt_den_map_flat >= low) & (gt_den_map_flat < high)
            
        gt_class_map[mask] = i

    # Controlla se qualche valore non è stato assegnato (non dovrebbe succedere)
    if (gt_class_map == -1).any():
        print("ATTENZIONE: Alcuni valori di densità non sono stati assegnati a nessun bin.")
        # Assegna i non assegnati al bin più vicino (o all'ultimo bin)
        gt_class_map[gt_class_map == -1] = num_bins - 1 
        
    return gt_class_map
# --- CLASSE LOSS PRINCIPALE (Modificata) ---

class QuadLoss(nn.Module):
    def __init__(
        self,
        input_size: int,
        block_size: int,
        bins: List[Tuple[float, float]],
        weight_cls: float = 1.0,
        weight_reg: float = 1.0,
        weight_aux: float = 0.0,
        pi_loss_weight_bce: float = 1.0, # Peso per la BCE della PI-Head
        pi_mask_threshold: float = 0.5, # Soglia per la maschera
        # Rimuovi parametri non usati (numItermax, scales, alpha, etc.)
        **kwargs # Accetta altri argomenti (come reg_loss, aux_loss) ma non li usa
    ) -> None:
        super().__init__()
        assert input_size % block_size == 0, f"Expected input_size to be divisible by block_size, got {input_size} and {block_size}"
        
        self.input_size = input_size
        self.block_size = block_size
        self.bins = bins
        self.num_bins = len(bins)
        self.num_blocks_h = input_size // block_size
        self.num_blocks_w = input_size // block_size

        # Salva i pesi per i diversi stadi
        self.weight_cls = weight_cls
        self.weight_reg = weight_reg
        self.weight_aux = weight_aux
        self.pi_loss_weight_bce = pi_loss_weight_bce
        self.pi_mask_threshold = pi_mask_threshold

        # --- Inizializza le 3 componenti di loss ---
        
        # 1. Loss per PI-Head (ZIP) - Componente Regressione (NLL)
        # Usata in Stage 1 e 3 (controllata da weight_reg)
        self.pi_loss_nll = ZIPoissonNLL(reduction="mean")
        
        # 2. Loss per PI-Head (ZIP) - Componente Classificazione (BCE per 'vuoto')
        # Usata in Stage 1 e 3 (controllata da weight_reg)
        self.pi_loss_bce = ZICrossEntropy(reduction="mean")
        
        # 3. Loss per LAMBDA-Head (EBC) - Componente Classificazione Bin
        # Usata in Stage 2 e 3 (controllata da weight_cls)
        # reduction='none' è FONDAMENTALE per permettere il masking
        self.lambda_loss_ebc = nn.CrossEntropyLoss(reduction="none")
        
        # 4. Loss sul conteggio totale (AUX)
        # Usata in Stage 3 (controllata da weight_aux)
        self.cnt_loss_fn = nn.L1Loss(reduction="mean")

    def forward(
        self,
        pred_logit_map: Tensor, # Output EBC/Lambda-Head (B, Num_Bins, H, W)
        pred_den_map: Tensor,   # Output Densità Finale (B, 1, H, W)
        gt_den_map: Tensor,     # GT Densità (B, 1, H_full, W_full)
        gt_points: List[Tensor],
        pred_logit_pi_map: Optional[Tensor] = None, # Output PI-Head (B, 2, H, W)
        pred_lambda_map: Optional[Tensor] = None, # Output PI-Head (B, 1, H, W)
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        
        B = pred_den_map.shape[0]

        # Reshapa la GT density map per matchare i blocchi
        if gt_den_map.shape[-2:] != (self.num_blocks_h, self.num_blocks_w):
            assert gt_den_map.shape[-2:] == (self.input_size, self.input_size), f"Expected gt_den_map to have shape {B, 1, self.input_size, self.input_size}, got {gt_den_map.shape}"
            gt_den_map_blocks = _reshape_density(gt_den_map, block_size=self.block_size)
        else:
            gt_den_map_blocks = gt_den_map

        loss_info = {}
        total_loss = 0
        
        pi_loss = torch.tensor(0.0, device=pred_den_map.device)
        lambda_loss = torch.tensor(0.0, device=pred_den_map.device)
        cnt_loss = torch.tensor(0.0, device=pred_den_map.device)

        # --- 1. Calcolo Loss PI-Head (ZIP) ---
        # Questa è la 'reg_loss' in train.py (Stage 1 e 3)
        if self.weight_reg > 0:
            assert pred_logit_pi_map is not None and pred_lambda_map is not None, \
                "PI-Head outputs (pred_logit_pi_map, pred_lambda_map) are required when weight_reg > 0"
            
            # Loss NLL (solo per blocchi non-zero)
            pi_loss_n, n_info = self.pi_loss_nll(pred_logit_pi_map, pred_lambda_map, gt_den_map_blocks)
            # Loss BCE (zero vs non-zero)
            pi_loss_b, b_info = self.pi_loss_bce(pred_logit_pi_map, gt_den_map_blocks)
            
            # Combina le due componenti della loss ZIP
            pi_loss = pi_loss_n + (pi_loss_b * self.pi_loss_weight_bce)
            
            loss_info.update({
                "pi_nll_loss": n_info['nll'].detach(), 
                "pi_bce_loss": b_info['bce'].detach()
            })

        # --- 2. Calcolo Loss LAMBDA-Head (EBC) ---
        # Questa è la 'cls_loss' in train.py (Stage 2 e 3)
        if self.weight_cls > 0:
            assert pred_logit_pi_map is not None, \
                "PI-Head output (pred_logit_pi_map) is required for masking when weight_cls > 0"

            # Crea la GT per la classificazione EBC
            gt_class_map = _bin_count(gt_den_map_blocks, bins=self.bins)
            
            # Stabilizza l'input per prevenire NaN da logit estremi
            pred_logit_map = torch.clamp(pred_logit_map, min=-30, max=30)

            # Calcola la loss EBC non ridotta (B, H, W)
            lambda_loss_unreduced = self.lambda_loss_ebc(pred_logit_map, gt_class_map)
            
            # --- APPLICA IL MASKING (come da tua idea) ---
            # Usa l'output della PI-Head per mascherare la loss EBC
            with torch.no_grad():
                # Probabilità che un blocco sia NON-VUOTO (canale 1)
                pi_prob_nonzero = pred_logit_pi_map.softmax(dim=1)[:, 1, :, :]
                # Crea la maschera binaria (1.0 se non-vuoto, 0.0 se vuoto)
                # .detach() è fondamentale per non propagare il gradiente EBC alla PI-Head
                pi_mask = (pi_prob_nonzero > self.pi_mask_threshold).float()

            # Applica la maschera: la loss è 0 per i blocchi vuoti
            lambda_loss_masked = lambda_loss_unreduced * pi_mask
            
            # Riduci la loss (media sul batch)
            # Nota: dividiamo per pi_mask.sum() per fare la media solo sui blocchi non-vuoti
            # Questo evita che la loss scenda artificialmente in immagini molto vuote
            if pi_mask.sum() > 0:
                 lambda_loss = lambda_loss_masked.sum() / pi_mask.sum()
            else:
                 lambda_loss = lambda_loss_masked.sum() # Sarà 0
            
            loss_info["lambda_ebc_loss"] = lambda_loss.detach()
            loss_info["mask_active_blocks"] = pi_mask.sum().detach() / (B * self.num_blocks_h * self.num_blocks_w)

        # --- 3. Calcolo Loss Conteggio Totale (AUX) ---
        # Questa è la 'aux_loss' in train.py (Stage 3)
        if self.weight_aux > 0:
            gt_cnt_per_image = torch.tensor([len(p) for p in gt_points], dtype=torch.float32, device=pred_den_map.device)
            pred_cnt_per_image = pred_den_map.sum(dim=(1, 2, 3))
            
            cnt_loss = self.cnt_loss_fn(pred_cnt_per_image, gt_cnt_per_image)
            loss_info["cnt_loss"] = cnt_loss.detach()

        # --- Loss Totale Ponderata ---
        total_loss = (self.weight_reg * pi_loss) + \
                     (self.weight_cls * lambda_loss) + \
                     (self.weight_aux * cnt_loss)
        
        loss_info["total_loss"] = total_loss.detach()
        
        return total_loss, loss_info