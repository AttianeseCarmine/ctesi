# trainer.py (Corretto per "Tuple is not defined")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# === INIZIO MODIFICA ===
from typing import Optional, Dict, List, Tuple # Aggiunto 'Tuple'
# === FINE MODIFICA ===
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os
import time
from tqdm import tqdm

from utils.log_utils import get_logger
from utils.eval_utils import AverageMeter, evaluate_mae_rmse

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[lr_scheduler._LRScheduler],
        device: torch.device,
        cfg: Dict,
        stage_cfg: Dict,
        best_ckpt_path: str
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.cfg = cfg
        self.stage_cfg = stage_cfg
        self.best_ckpt_path = best_ckpt_path

        # Impostazioni di logging
        self.output_dir = self.cfg['train_base']['output_dir']
        self.logger = get_logger(os.path.join(self.output_dir, 'train.log'))

        # Impostazioni di training
        self.start_epoch = 1
        self.num_epochs = self.stage_cfg.get('num_epochs', 100)
        self.eval_freq = self.cfg['train_base'].get('eval_freq', 1)
        self.clip_grad_norm = self.stage_cfg.get('clip_grad_norm', None)

        # Metriche
        self.best_mae = float('inf')
        self.best_rmse = float('inf')
        self.best_epoch = 0
        self.no_improve_epochs = 0
        
        # Early Stopping
        self.early_stopping_patience = self.stage_cfg.get('early_stopping_patience', 0)
        if self.early_stopping_patience > 0:
            self.logger.info(f"✅ Early Stopping ATTIVO con pazienza={self.early_stopping_patience} epoche.")

        # Carica checkpoint se 'last.pth' esiste (per resume)
        self._load_last_checkpoint()

    def _load_last_checkpoint(self):
        last_ckpt_path = os.path.join(self.output_dir, 'last.pth')
        if os.path.exists(last_ckpt_path):
            self.logger.info(f"Caricamento checkpoint 'last.pth' da: {last_ckpt_path}")
            try:
                checkpoint = torch.load(last_ckpt_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.scheduler and checkpoint.get('scheduler_state_dict'):
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.start_epoch = checkpoint.get('epoch', 1) + 1
                self.best_mae = checkpoint.get('best_mae', float('inf'))
                self.best_rmse = checkpoint.get('best_rmse', float('inf'))
                self.best_epoch = checkpoint.get('best_epoch', 0)
                self.no_improve_epochs = checkpoint.get('no_improve_epochs', 0)
                
                self.logger.info(
                    f"Resume da epoch {self.start_epoch}, Best MAE: {self.best_mae:.2f}, "
                    f"Round senza miglioramenti: {self.no_improve_epochs}"
                )
            except Exception as e:
                self.logger.warning(f"Errore nel caricamento di 'last.pth', si riparte da zero. Errore: {e}")
                self.start_epoch = 1

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_mae': self.best_mae,
            'best_rmse': self.best_rmse,
            'best_epoch': self.best_epoch,
            'no_improve_epochs': self.no_improve_epochs,
        }
        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Salva sempre 'last.pth'
        last_ckpt_path = os.path.join(self.output_dir, 'last.pth')
        torch.save(state, last_ckpt_path)

        if is_best:
            torch.save(state, self.best_ckpt_path)
            self.logger.info(f"💾 Nuovo best checkpoint salvato: {self.best_ckpt_path} (Epoch {epoch}, MAE {self.best_mae:.2f})")

    def _train_one_epoch(self) -> float:
        self.model.train()
        loss_meter = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}/{self.num_epochs} [Train]")
        
        for images, gt_density_map, gt_points in pbar:
            images = images.to(self.device)
            gt_density_map = gt_density_map.to(self.device)
            # gt_points rimane una lista di tensori sulla CPU, gestita dalla loss

            self.optimizer.zero_grad()

            # Il modello ora restituisce un DIZIONARIO.
            outputs_dict = self.model(images)
            
            # La tua QuadLoss si aspetta i seguenti argomenti:
            # (pred_logit_map, pred_den_map, gt_den_map, gt_points, pred_logit_pi_map, pred_lambda_map)
            
            loss, loss_info = self.criterion(
                pred_logit_map=outputs_dict["pred_logit_map"],
                pred_den_map=outputs_dict["pred_den_map"],
                gt_den_map=gt_density_map, 
                gt_points=gt_points,
                pred_logit_pi_map=outputs_dict["pred_logit_pi_map"],
                pred_lambda_map=outputs_dict["pred_lambda_map"]
            )

            if torch.isnan(loss):
                self.logger.warning(f"Rilevata loss NaN. Iterazione saltata.")
                continue

            loss.backward()

            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            
            self.optimizer.step()
            
            loss_meter.update(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.1e}")

        return loss_meter.avg

    def _validate_one_epoch(self) -> Tuple[float, float]: # <--- La riga 164 ora è corretta
        self.model.eval()
        mae_meter = AverageMeter()
        rmse_meter = AverageMeter()
        
        eval_cfg = self.cfg.get('eval', {})
        sliding_window = eval_cfg.get('sliding_window', False)
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [Val]")
        
        with torch.no_grad():
            for images, gt_density_map, gt_points in pbar:
                images = images.to(self.device)
                
                # Il modello in modalità .eval() restituisce solo la mappa di densità
                pred_den_map = self.model(images)
                
                # Calcola MAE/RMSE (presumendo che la tua funzione lo gestisca)
                try:
                    mae, rmse = evaluate_mae_rmse(pred_den_map, gt_points, sliding_window, **eval_cfg)
                    
                    if mae is not None:
                        mae_meter.update(mae)
                    if rmse is not None:
                        rmse_meter.update(rmse)
                        
                except Exception as e:
                    self.logger.warning(f"Errore during validazione: {e}")
                    pass 

        avg_mae = mae_meter.avg if mae_meter.count > 0 else float('nan')
        avg_rmse = rmse_meter.avg if rmse_meter.count > 0 else float('nan')
        
        return avg_mae, avg_rmse

    def train(self):
        self.logger.info(f"Inizio addestramento da epoch {self.start_epoch} a {self.num_epochs}")
        
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            self.current_epoch = epoch
            start_time = time.time()
            
            train_loss = self._train_one_epoch()
            
            if self.scheduler:
                self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            # Log e Validazione
            if epoch % self.eval_freq == 0 or epoch == self.num_epochs:
                val_start_time = time.time()
                val_mae, val_rmse = self._validate_one_epoch()
                val_time = time.time() - val_start_time
                
                self.logger.info(
                    f"Epoch {epoch}/{self.num_epochs}: Train Loss: {train_loss:.4f} | "
                    f"Val MAE: {val_mae:.2f} | Val RMSE: {val_rmse:.2f} | "
                    f"Epoch Time: {epoch_time:.1f}s | Val Time: {val_time:.1f}s"
                )
                
                # Controllo Best Model
                is_best = val_mae < self.best_mae
                if is_best:
                    self.best_mae = val_mae
                    self.best_rmse = val_rmse
                    self.best_epoch = epoch
                    self.no_improve_epochs = 0
                    self._save_checkpoint(epoch, is_best=True)
                else:
                    self.no_improve_epochs += self.eval_freq
                    self._save_checkpoint(epoch, is_best=False) # Salva 'last.pth'
                
                # Controllo Early Stopping
                if self.early_stopping_patience > 0 and self.no_improve_epochs >= self.early_stopping_patience:
                    self.logger.info(
                        f"⛔ Early Stopping! Nessun miglioramento del MAE per {self.no_improve_epochs} epoche. "
                        f"Best MAE: {self.best_mae:.2f} (Epoch {self.best_epoch})"
                    )
                    break
            else:
                # Log solo training
                self.logger.info(
                    f"Epoch {epoch}/{self.num_epochs}: Train Loss: {train_loss:.4f} | "
                    f"Valutazione saltata | Epoch Time: {epoch_time:.1f}s"
                )
                self._save_checkpoint(epoch, is_best=False)
        
        self.logger.info(f"Addestramento completato. Best MAE: {self.best_mae:.2f} (Epoch {self.best_epoch})")