import torch
from torch import nn, Tensor
import torch.nn.functional as F

# Epsilon per stabilità numerica
EPS = 1e-6

def _safe_mean(x: Tensor) -> Tensor:
    if x.numel() == 0:
        return torch.tensor(0.0, device=x.device)
    return x.mean()

class ZIPoissonNLL(nn.Module):
    """
    ZIP Loss ottimizzata e STABILE.
    """
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred_logit_pi_map: Tensor,      # [B, 2, H, W]
        pred_lambda_map: Tensor,        # [B, 1, H, W]
        gt_den_map_blocks: Tensor       # [B, 1, H, W]
    ):
        # ✅ 1. SANITIZZAZIONE INPUT AGGRESSIVA
        # Clampa i logits PI per evitare overflow/underflow
        pred_logit_pi_map = torch.clamp(pred_logit_pi_map, min=-10, max=10)
        
        # ✅ 2. SANITIZZAZIONE LAMBDA
        # Lambda deve essere STRETTAMENTE positivo per evitare log(0)
        lam = pred_lambda_map.squeeze(1)
        lam = torch.clamp(lam, min=EPS, max=100.0)  # Limita anche superiormente
        
        # ✅ 3. SANITIZZAZIONE GT
        y = gt_den_map_blocks.squeeze(1)
        y = torch.clamp(y, min=0.0)  # Assicura non-negatività
        
        # ✅ 4. Calcolo Probabilità π (Log-Space STABILE)
        log_pi = F.log_softmax(pred_logit_pi_map, dim=1)
        log_pi0 = log_pi[:, 0, :, :]  # Log P(vuoto)
        log_pi1 = log_pi[:, 1, :, :]  # Log P(pieno)

        # Maschere
        zero_mask = (y < EPS)  # Considera "zero" anche valori molto piccoli
        nonzero_mask = ~zero_mask

        # ✅ 5. CASO y = 0 (Zero-Inflated) - VERSIONE STABILE
        # Loss = -log(π0 + π1 * e^(-λ))
        # Usiamo logaddexp per stabilità: log(a + b) = logaddexp(log_a, log_b)
        term1 = log_pi0[zero_mask]
        term2 = log_pi1[zero_mask] - lam[zero_mask]
        
        if term1.numel() > 0:
            loss_zero = -torch.logaddexp(term1, term2)
            # ✅ Rimuovi NaN/Inf anche qui
            loss_zero = torch.where(
                torch.isfinite(loss_zero),
                loss_zero,
                torch.zeros_like(loss_zero)
            )
        else:
            loss_zero = torch.tensor([], device=y.device)

        # ✅ 6. CASO y > 0 (Poisson) - VERSIONE ULTRA-STABILE
        if nonzero_mask.any():
            y_pos = y[nonzero_mask]
            lam_pos = lam[nonzero_mask]
            log_pi1_pos = log_pi1[nonzero_mask]

            # ✅ TRUCCO CRITICO: Usa lgamma solo su valori ARROTONDATI e SICURI
            # Arrotonda y_pos per evitare problemi con decimali strani
            y_pos_safe = torch.floor(y_pos) + 1.0  # +1 perché lgamma(n+1) = log(n!)
            y_pos_safe = torch.clamp(y_pos_safe, min=1.0, max=170.0)  # lgamma(171) overflow
            
            log_factorial = torch.lgamma(y_pos_safe)
            
            # ✅ Verifica NaN in log_factorial
            if torch.isnan(log_factorial).any():
                print("⚠️ NaN rilevato in lgamma! Sostituisco con 0.")
                log_factorial = torch.nan_to_num(log_factorial, nan=0.0)

            # Calcola log-probabilità Poisson
            log_prob_pos = (
                log_pi1_pos 
                + (y_pos * torch.log(lam_pos + EPS))  # +EPS per sicurezza
                - lam_pos 
                - log_factorial
            )
            
            loss_pos = -log_prob_pos
            
            # ✅ Rimuovi NaN/Inf
            loss_pos = torch.where(
                torch.isfinite(loss_pos),
                loss_pos,
                torch.zeros_like(loss_pos)
            )
        else:
            loss_pos = torch.tensor([], device=y.device)

        # ✅ 7. COMBINAZIONE E RIDUZIONE FINALE
        all_losses = torch.cat([loss_zero.flatten(), loss_pos.flatten()])

        # ✅ Ultimo check: Se ci sono ancora NaN, sostituisci con 0
        if torch.isnan(all_losses).any() or torch.isinf(all_losses).any():
            print(f"⚠️ NaN/Inf in all_losses: {torch.isnan(all_losses).sum()} NaN, {torch.isinf(all_losses).sum()} Inf")
            all_losses = torch.nan_to_num(all_losses, nan=0.0, posinf=10.0, neginf=0.0)

        if self.reduction == "mean":
            loss = _safe_mean(all_losses)
        elif self.reduction == "sum":
            loss = all_losses.sum()
        else:
            loss = all_losses

        info = {"nll": loss.detach() if loss.numel() > 0 else torch.tensor(0.0)}
        return loss, info


class ZICrossEntropy(nn.Module):
    """
    Cross Entropy STABILE per la testa PI.
    """
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred_logit_pi_map: Tensor,      
        gt_den_map_blocks: Tensor       
    ):
        # ✅ Sanitizzazione
        pred_logit_pi_map = torch.clamp(pred_logit_pi_map, min=-10, max=10)
        
        # Target: 0 se densità < EPS, 1 altrimenti
        target_long = (gt_den_map_blocks > EPS).long().squeeze(1)
        
        loss = F.cross_entropy(pred_logit_pi_map, target_long, reduction=self.reduction)
        
        # ✅ Controllo NaN finale
        if torch.isnan(loss):
            print("⚠️ NaN in ZICrossEntropy! Ritorno 0.")
            loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
        
        info = {"bce": loss.detach()}
        return loss, info