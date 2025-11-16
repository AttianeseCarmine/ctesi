# trainer.py
# (Corretto per Pylance e con Early Stopping)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from typing import Optional, Dict, Tuple

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        cfg: Dict, # Configurazione globale
        stage_cfg: Dict, # Configurazione dello stadio corrente (es. 'train_stage1')
        best_ckpt_path: Optional[str] = None, 
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.cfg = cfg # Config globale
        self.stage_cfg = stage_cfg # Config dello stadio
        
        self.best_mae = float('inf')
        self.start_epoch = 1
        
        # === INIZIO LOGICA EARLY STOPPING ===
        # Leggi la pazienza dalla config dello stadio, default 0 (disabilitato)
        self.patience = self.stage_cfg.get('early_stopping_patience', 0)
        self.no_improve_rounds = 0
        if self.patience > 0:
            print(f"✅ Early Stopping ATTIVO con pazienza={self.patience} epoche.")
        # === FINE LOGICA EARLY STOPPING ===

        
        # Usa l'output_dir dalla config globale 'train_base'
        output_dir = self.cfg['train_base']['output_dir']
        
        if best_ckpt_path is not None:
            self.best_ckpt_path = best_ckpt_path
        else:
            self.best_ckpt_path = os.path.join(output_dir, 'best_mae.pth') 
        
        self.last_ckpt_path = os.path.join(output_dir, 'last.pth') 
        
        self._resume_checkpoint()

    def _resume_checkpoint(self):
        """Carica l'ultimo checkpoint se esiste (per resume)"""
        if os.path.exists(self.last_ckpt_path) and self.stage_cfg.get('resume', True): 
            try:
                print(f"Caricamento checkpoint 'last.pth' da: {self.last_ckpt_path}")
                state = torch.load(self.last_ckpt_path, map_location=self.device)
                
                self.model.load_state_dict(state['model'], strict=False)
                
                if len(self.optimizer.param_groups) == len(state['optimizer']['param_groups']):
                    self.optimizer.load_state_dict(state['optimizer'])
                
                self.scheduler.load_state_dict(state['scheduler'])
                self.start_epoch = state['epoch'] + 1
                self.best_mae = state['best_mae']
                
                # === INIZIO LOGICA EARLY STOPPING ===
                # Riprendi il conteggio della pazienza
                self.no_improve_rounds = state.get('no_improve_rounds', 0)
                print(f"Resume da epoch {self.start_epoch}, Best MAE: {self.best_mae:.2f}, Round senza miglioramenti: {self.no_improve_rounds}")
                # === FINE LOGICA EARLY STOPPING ===
                
            except Exception as e:
                print(f"Errore nel caricamento di 'last.pth', si riparte da zero. Errore: {e}")

    def _save_checkpoint(self, epoch: int, is_best: bool):
        """Salva lo stato corrente in last.pth e il best in best_ckpt_path"""
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_mae': self.best_mae,
            'no_improve_rounds': self.no_improve_rounds # Salva il contatore
        }
        torch.save(state, self.last_ckpt_path)
        
        if is_best:
            print(f"🏅 Nuovo MAE migliore: {self.best_mae:.2f}. Salvo in {self.best_ckpt_path}")
            torch.save(state['model'], self.best_ckpt_path)

    def _train_one_epoch(self) -> float:
        """Esegue un'epoca di addestramento."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}/{self.stage_cfg['num_epochs']} [Train]")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            gt_den_map = batch['density_map'].to(self.device)
            gt_points = batch['points'] 
            
            self.optimizer.zero_grad()
            
            pi_logit_map, lambda_logit_map, lambda_map, den_map = self.model(images)
            
            loss, loss_info = self.criterion(
                pred_logit_map=lambda_logit_map, 
                pred_den_map=den_map,            
                gt_den_map=gt_den_map,
                gt_points=gt_points,
                pred_logit_pi_map=pi_logit_map, 
                pred_lambda_map=lambda_map      
            )
            
            loss.backward()
            
            if self.stage_cfg.get('clip_grad_norm', 0) > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.stage_cfg['clip_grad_norm'])
                
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.1e}")

        if self.scheduler:
            self.scheduler.step()
            
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Esegue la validazione."""
        self.model.eval()
        total_mae = 0.0
        total_mse = 0.0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [Val]")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            gt_points = batch['points'] 
            gt_count = torch.tensor([len(p) for p in gt_points], dtype=torch.float32, device=self.device)
            
            pred_den_map = self.model(images) 
            pred_count = pred_den_map.sum(dim=(1, 2, 3))
            
            total_mae += torch.abs(pred_count - gt_count).sum().item()
            total_mse += ((pred_count - gt_count) ** 2).sum().item()

        avg_mae = total_mae / len(self.val_loader.dataset)
        avg_rmse = (total_mse / len(self.val_loader.dataset)) ** 0.5
        
        return avg_mae, avg_rmse

    def train(self):
        """Loop di addestramento principale."""
        num_epochs = self.stage_cfg['num_epochs']
        eval_freq = self.stage_cfg['eval_freq']
        
        print(f"Inizio addestramento da epoch {self.start_epoch} a {num_epochs}")
        
        for epoch in range(self.start_epoch, num_epochs + 1):
            self.current_epoch = epoch
            
            train_loss = self._train_one_epoch()
            
            if epoch % eval_freq == 0 or epoch == num_epochs:
                val_mae, val_rmse = self.validate()
                
                is_best = val_mae < self.best_mae
                if is_best:
                    self.best_mae = val_mae
                    self.no_improve_rounds = 0 # Resetta il contatore
                else:
                    self.no_improve_rounds += 1 # Incrementa il contatore
                
                self._save_checkpoint(epoch, is_best)
                
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.2f} | Val RMSE: {val_rmse:.2f} | Best MAE: {self.best_mae:.2f}")
                
                # === INIZIO LOGICA EARLY STOPPING ===
                if self.patience > 0 and self.no_improve_rounds >= self.patience:
                    print(f"⛔ EARLY STOPPING: Nessun miglioramento del MAE per {self.no_improve_rounds} epoche di validazione consecutive.")
                    print(f"Addestramento interrotto all'epoca {epoch}.")
                    break # Interrompe il ciclo for delle epoche
                # === FINE LOGICA EARLY STOPPING ===

            else:
                self._save_checkpoint(epoch, is_best=False)
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f} | Valutazione saltata")