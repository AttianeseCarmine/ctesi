import torch
from torch import nn, Tensor
import torch.nn.functional as F

# Epsilon aumentato per stabilità logaritmica
EPS = 1e-5

def _safe_mean(x: Tensor) -> Tensor:
    if x.numel() == 0:
        return torch.tensor(0.0, device=x.device)
    return x.mean()

class ZIPoissonNLL(nn.Module):
    """
    ZIP Loss ottimizzata per ConvZIPHead.
    Assume che 'pred_lambda_map' sia GIÀ positivo (output della testa).
    """
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred_logit_pi_map: Tensor,      # [B, 2, H, W] (Logits per Vuoto/Pieno)
        pred_lambda_map: Tensor,        # [B, 1, H, W] (Lambda GIÀ positivo da ConvZIPHead)
        gt_den_map_blocks: Tensor       # [B, 1, H, W] (Ground Truth)
    ):
        # 1. Sanitizzazione Input (Gradient Clipping implicito)
        # Clampiamo solo i logits di PI, Lambda è già clampato nella testa
        pred_logit_pi_map = torch.clamp(pred_logit_pi_map, min=-10, max=10)
        
        # 2. Calcolo Probabilità π (Log-Space)
        log_pi = F.log_softmax(pred_logit_pi_map, dim=1)
        log_pi0 = log_pi[:, 0, :, :]  # Log-prob Vuoto
        log_pi1 = log_pi[:, 1, :, :]  # Log-prob Pieno

        # 3. Recupero λ (Lambda)
        # CORREZIONE: Non usiamo softplus qui perché ConvZIPHead restituisce già lambda > 0.
        # Aggiungiamo solo EPS per sicurezza assoluta nel logaritmo.
        lam = pred_lambda_map.squeeze(1) + EPS 

        y = gt_den_map_blocks.squeeze(1)
        
        # Maschere per i casi Zero e Non-Zero
        zero_mask = (y == 0)
        nonzero_mask = ~zero_mask

        # 4. Caso y = 0 (Zero-Inflated)
        # Loss = -log( P(y=0) ) = -log( π0 + π1 * e^-λ )
        # Usiamo logaddexp: log(a + b) -> logaddexp(log_a, log_b)
        # log(π1 * e^-λ) = log_π1 - λ
        nll_zero = -torch.logaddexp(log_pi0, log_pi1 - lam)
        
        # Applichiamo la maschera: calcoliamo solo dove y=0
        loss_zero = nll_zero[zero_mask]

        # 5. Caso y > 0 (Poisson standard)
        # Loss = -log( P(y=k) ) = - (log_π1 + y*log(λ) - λ - log(y!))
        # Nota: P(y|non-vuoto) * π1 -> log_π1 + log_Poisson
        
        if nonzero_mask.any():
            y_pos = y[nonzero_mask]
            lam_pos = lam[nonzero_mask]
            log_pi1_pos = log_pi1[nonzero_mask]

            # Approssimazione Stirling per log(y!) su float: lgamma(y + 1)
            log_factorial = torch.lgamma(y_pos + 1.0)

            log_prob_pos = (
                log_pi1_pos 
                + (y_pos * torch.log(lam_pos))  # Qui lam_pos è sicuro (> EPS)
                - lam_pos 
                - log_factorial
            )
            loss_pos = -log_prob_pos
        else:
            loss_pos = torch.tensor([], device=y.device)

        # 6. Combinazione delle loss
        # Concateniamo i vettori appiattiti invece di sommare mappe (più sicuro per i NaN)
        all_losses = torch.cat([loss_zero.flatten(), loss_pos.flatten()])

        # Debug per NaN: Se crasha qui, sappiamo esattamente perché
        if torch.isnan(all_losses).any():
            print("!!! NaN rilevato dentro ZIPoissonNLL !!!")
            # Fallback di emergenza per non fermare il training, ma loggare l'errore
            all_losses = torch.nan_to_num(all_losses, nan=0.0, posinf=100.0)

        if self.reduction == "mean":
            loss = _safe_mean(all_losses)
        elif self.reduction == "sum":
            loss = all_losses.sum()
        else:
            loss = all_losses

        info = {"nll": loss.detach() if loss.numel() > 0 else 0.0}
        return loss, info


class ZICrossEntropy(nn.Module):
    """
    Versione stabile di Cross Entropy per la testa PI.
    """
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred_logit_pi_map: Tensor,      
        gt_den_map_blocks: Tensor       
    ):
        pred_logit_pi_map = torch.clamp(pred_logit_pi_map, min=-10, max=10)
        
        # Target: 0 se densità è 0, 1 se densità > 0
        target_long = (gt_den_map_blocks > 0).long().squeeze(1)
        
        loss = F.cross_entropy(pred_logit_pi_map, target_long, reduction=self.reduction)
        
        info = {"bce": loss.detach() if loss.numel() > 0 else 0.0}
        return loss, info