import torch
import torch.nn as nn
import torch.nn.functional as F  # <--- AGGIUNTO per l'interpolazione
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple
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
            # Salva nel percorso specifico definito nel main (es. stage1_best.pth)
            torch.save(state, self.best_ckpt_path)
            self.logger.info(f"💾 Nuovo best checkpoint salvato: {self.best_ckpt_path} (Epoch {epoch}, MAE {self.best_mae:.2f})")

    def _train_one_epoch(self) -> float:
        self.model.train()
        loss_meter = AverageMeter()
        
        # Pbar con enumerate per debug
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {self.current_epoch}/{self.num_epochs} [Train]")
        
        for batch_idx, batch_data in pbar:
            # 1. Caricamento Dati
            if isinstance(batch_data, dict):
                images = batch_data['image'].to(self.device)
                gt_density_map = batch_data['density_map'].to(self.device)
                gt_points = batch_data['points']
            else:
                images, gt_points, gt_density_map = batch_data
                images = images.to(self.device)
                gt_density_map = gt_density_map.to(self.device)

            self.optimizer.zero_grad()

            # 2. Forward Pass
            outputs_dict = self.model(images)

            # === GESTIONE DIMENSIONI (Input vs Ground Truth) ===
            input_h, input_w = images.shape[2], images.shape[3]
            target_h, target_w = input_h // 16, input_w // 16

            # A. Pulizia Punti (Fondamentale per evitare crash se punti escono dal crop)
            cleaned_points = []
            for pts in gt_points:
                if pts.numel() > 0:
                    if pts.device != self.device: pts = pts.to(self.device)
                    # Filtra punti fuori dall'immagine corrente
                    mask_inside = (pts[:, 0] >= 0) & (pts[:, 0] < input_w) & \
                                  (pts[:, 1] >= 0) & (pts[:, 1] < input_h)
                    cleaned_points.append(pts[mask_inside])
                else:
                    cleaned_points.append(pts)
            gt_points = cleaned_points

            # B. Resize Predizioni del Modello (Feature Maps)
            # Helper function
            def safe_resize(tensor_val, t_h, t_w):
                if tensor_val is None: return None
                if tensor_val.shape[2] != t_h or tensor_val.shape[3] != t_w:
                    return F.interpolate(tensor_val, size=(t_h, t_w), mode='bilinear', align_corners=False)
                return tensor_val

            pred_pi = safe_resize(outputs_dict.get("pred_logit_pi_map"), target_h, target_w)
            pred_lam = safe_resize(outputs_dict.get("pred_lambda_map"), target_h, target_w)
            pred_den = safe_resize(outputs_dict.get("pred_den_map"), target_h, target_w)

            # C. Resize GROUND TRUTH MAP (IL PEZZO MANCANTE)
            # Forza la mappa di verità ad avere le stesse dimensioni spaziali dell'immagine di input.
            # Questo garantisce che la Loss generi una maschera della dimensione corretta.
            if gt_density_map.shape[2] != input_h or gt_density_map.shape[3] != input_w:
                if batch_idx == 0:
                    print(f"[FIX] Resizing GT Map {gt_density_map.shape} -> ({input_h}, {input_w})")
                gt_density_map = F.interpolate(
                    gt_density_map, 
                    size=(input_h, input_w), 
                    mode='bilinear', 
                    align_corners=False
                ) 
                # (Opzionale: moltiplicare per il fattore di scala per mantenere il count esatto, 
                # ma per risolvere il crash basta l'interpolate)

            # 3. Calcolo Loss
            try:
                loss, loss_info = self.criterion(
                    pred_logit_map=outputs_dict.get("pred_logit_map"),
                    pred_den_map=pred_den,
                    gt_den_map=gt_density_map,   # <--- Ora è sincronizzata 224x224
                    gt_points=gt_points,
                    pred_logit_pi_map=pred_pi,
                    pred_lambda_map=pred_lam
                )
            except IndexError as e:
                print(f"\n❌ ERRORE CRITICO DIMENSIONI:")
                print(f"Image Input: {input_h}x{input_w}")
                print(f"GT Map Final: {gt_density_map.shape}")
                print(f"Pred Shape: {pred_pi.shape if pred_pi is not None else 'None'}")
                raise e

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

    def _validate_one_epoch(self) -> Tuple[float, float]:
        self.model.eval()
        mae_meter = AverageMeter()
        rmse_meter = AverageMeter()
        
        eval_cfg = self.cfg.get('eval', {})
        sliding_window = eval_cfg.get('sliding_window', False)
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [Val]")
        
        with torch.no_grad():
            for batch_data in pbar:
                if isinstance(batch_data, dict):
                    images = batch_data['image'].to(self.device)
                    gt_points = batch_data['points']
                else:
                    images, gt_density_map, gt_points = batch_data
                    images = images.to(self.device)
                
                pred_den_map = self.model(images)
                
                try:
                    mae, rmse = evaluate_mae_rmse(pred_den_map, gt_points, **eval_cfg)
                    
                    if mae is not None:
                        mae_meter.update(mae)
                    if rmse is not None:
                        rmse_meter.update(rmse)
                        
                except Exception as e:
                    self.logger.warning(f"Errore durante validazione: {e}")
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
            
            if epoch % self.eval_freq == 0 or epoch == self.num_epochs:
                val_start_time = time.time()
                val_mae, val_rmse = self._validate_one_epoch()
                val_time = time.time() - val_start_time
                
                self.logger.info(
                    f"Epoch {epoch}/{self.num_epochs}: Train Loss: {train_loss:.4f} | "
                    f"Val MAE: {val_mae:.2f} | Val RMSE: {val_rmse:.2f} | "
                    f"Epoch Time: {epoch_time:.1f}s | Val Time: {val_time:.1f}s"
                )
                
                is_best = val_mae < self.best_mae
                if is_best:
                    self.best_mae = val_mae
                    self.best_rmse = val_rmse
                    self.best_epoch = epoch
                    self.no_improve_epochs = 0
                    self._save_checkpoint(epoch, is_best=True)
                else:
                    self.no_improve_epochs += self.eval_freq
                    self._save_checkpoint(epoch, is_best=False)
                
                if self.early_stopping_patience > 0 and self.no_improve_epochs >= self.early_stopping_patience:
                    self.logger.info(
                        f"⛔ Early Stopping! Nessun miglioramento del MAE per {self.no_improve_epochs} epoche. "
                        f"Best MAE: {self.best_mae:.2f} (Epoch {self.best_epoch})"
                    )
                    break
            else:
                self.logger.info(
                    f"Epoch {epoch}/{self.num_epochs}: Train Loss: {train_loss:.4f} | "
                    f"Valutazione saltata | Epoch Time: {epoch_time:.1f}s"
                )
                self._save_checkpoint(epoch, is_best=False)
        
        self.logger.info(f"Addestramento completato. Best MAE: {self.best_mae:.2f} (Epoch {self.best_epoch})")