import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os
import time
from tqdm import tqdm

from utils.log_utils import get_logger
from utils.eval_utils import AverageMeter
# MODIFICA QUI: Importa lo script evaluate che abbiamo appena corretto
from evaluate import evaluate 

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

        # Recupera lo stage corrente (default 3 se non specificato)
        self.current_stage = self.cfg.get('stage', 3)

        self.output_dir = self.cfg['train_base']['output_dir']
        self.log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Ora salva il log dentro la cartella logs
        self.logger = get_logger(os.path.join(self.log_dir, 'train.log'))

        self.start_epoch = 1
        self.num_epochs = self.stage_cfg.get('num_epochs', 100)
        self.eval_freq = self.cfg['train_base'].get('eval_freq', 1)
        self.clip_grad_norm = self.stage_cfg.get('clip_grad_norm', 1.0)

        self.best_mae = float('inf')
        self.best_rmse = float('inf')
        self.best_epoch = 0
        self.no_improve_epochs = 0
        
        self.early_stopping_patience = self.stage_cfg.get('early_stopping_patience', 0)
        if self.early_stopping_patience > 0:
            self.logger.info(f"✅ Early Stopping ATTIVO (pazienza={self.early_stopping_patience}, stage={self.current_stage})")

        self._load_last_checkpoint()

    def _load_last_checkpoint(self):
        last_ckpt_path = os.path.join(self.output_dir, 'last.pth')
        if os.path.exists(last_ckpt_path):
            self.logger.info(f"Caricamento checkpoint 'last.pth'...")
            try:
                checkpoint = torch.load(last_ckpt_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.scheduler and checkpoint.get('scheduler_state_dict'):
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.start_epoch = checkpoint.get('epoch', 1) + 1
                self.best_mae = checkpoint.get('best_mae', float('inf'))
                self.best_rmse = checkpoint.get('best_rmse', float('inf'))
                self.best_epoch = checkpoint.get('best_epoch', 0)
                self.no_improve_epochs = checkpoint.get('no_improve_epochs', 0)
            except Exception as e:
                self.logger.warning(f"Errore caricamento last.pth: {e}")
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
        
        torch.save(state, os.path.join(self.output_dir, 'last.pth'))
        if is_best:
            torch.save(state, self.best_ckpt_path)
            self.logger.info(f"💾 Best checkpoint salvato: {self.best_ckpt_path} (MAE {self.best_mae:.2f})")

# trainer.py - Metodo _train_one_epoch

    def _train_one_epoch(self) -> float:
        self.model.train()
        loss_meter = AverageMeter()
        
        # ✅ Contatori per debug
        batch_count = 0
        debug_freq = 50  # Stampa ogni 50 batch
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}/{self.num_epochs} [Train]")
        
        for batch in pbar:
            batch_count += 1
            
            if isinstance(batch, dict):
                images = batch['image'].to(self.device)
                gt_map = batch['density_map'].to(self.device)
                gt_points = batch['points']
            else:
                images, gt_points, gt_map = batch
                images = images.to(self.device)
                gt_map = gt_map.to(self.device)
            
            # Crea blocchi GT
            with torch.no_grad():
                kernel_size = 16 
                gt_den_map_blocks = F.avg_pool2d(
                    gt_map, 
                    kernel_size=kernel_size, 
                    stride=kernel_size, 
                    divisor_override=1
                )
            
            # Verifica batch
            if torch.isnan(images).any() or torch.isinf(images).any():
                self.logger.warning("💀 BATCH con NaN/Inf! Sanitizzo...")
                images = torch.nan_to_num(images, nan=0.0, posinf=10.0, neginf=-10.0)
            if torch.isnan(gt_map).any() or torch.isinf(gt_map).any():
                self.logger.warning("⚠️ Sanitizzo GT con NaN/Inf")
                gt_map = torch.nan_to_num(gt_map, nan=0.0, posinf=10.0, neginf=-10.0)
                
            self.optimizer.zero_grad()
            
            # Forward
            outputs = self.model(images)
            
            # ✅ DEBUG MODELLO - Ogni N batch
            if batch_count % debug_freq == 0:
                print(f"\n🔍 DEBUG BATCH {batch_count}:")
                print(f"  Image shape: {images.shape}")
                print(f"  GT count: {[len(p) for p in gt_points]}")
                print(f"  GT blocks sum: {gt_den_map_blocks.sum(dim=(1,2,3)).tolist()}")
                
                if isinstance(outputs, dict):
                    pred_den_map = outputs["pred_den_map"]
                    pred_cnt = pred_den_map.sum(dim=(1,2,3))
                    print(f"  Pred count: {pred_cnt.tolist()}")
                    
                    if "pred_logit_pi_map" in outputs and outputs["pred_logit_pi_map"] is not None:
                        pi_probs = outputs["pred_logit_pi_map"].softmax(dim=1)[:, 1:2]
                        mask = (pi_probs > 0.5).float()
                        print(f"  Active blocks: {mask.mean().item():.2%}")
            
            # Gestisci output
            if isinstance(outputs, dict):
                pred_logit_map = outputs["pred_logit_map"]
                pred_den_map = outputs["pred_den_map"]
                pred_logit_pi_map = outputs.get("pred_logit_pi_map")
                pred_lambda_map = outputs.get("pred_lambda_map")
            else:
                pred_den_map = outputs
                pred_logit_map = None
                pred_logit_pi_map = None
                pred_lambda_map = None
            
            # Loss
            loss, loss_info = self.criterion(
                pred_logit_map=pred_logit_map,
                pred_den_map=pred_den_map,
                gt_den_map=gt_den_map_blocks,
                gt_points=gt_points,
                pred_logit_pi_map=pred_logit_pi_map,
                pred_lambda_map=pred_lambda_map
            )
            
            # ✅ DEBUG LOSS - Ogni N batch
            if batch_count % debug_freq == 0:
                print(f"  Loss components:")
                for k, v in loss_info.items():
                    if isinstance(v, torch.Tensor):
                        print(f"    {k}: {v.item():.4f}")
                    else:
                        print(f"    {k}: {v:.4f}")
            
            if torch.isnan(loss):
                self.logger.warning(f"💀 Loss NaN at batch {batch_count}! Skipping.")
                continue
            
            loss.backward()
            
            # Gradient norm
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            
            # ✅ DEBUG GRADIENTI - Ogni N batch
            if batch_count % debug_freq == 0:
                print(f"  Gradient norm: {total_norm:.4f}")
            
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            
            self.optimizer.step()
            
            loss_meter.update(loss.item())
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", 
                avg_loss=f"{loss_meter.avg:.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.1e}"
            )
        
        return loss_meter.avg

    def _validate(self) -> Tuple[float, float]:
        # MODIFICA: Usa la funzione evaluate importata passando lo STAGE
        return evaluate(
            model=self.model,
            data_loader=self.val_loader,
            device=self.device,
            stage=self.current_stage, # Passa stage 1/2/3
            desc=f"Epoch {self.epoch} [Val]"
        )

    def train(self):
        self.logger.info(f"Start training Stage {self.current_stage} from {self.start_epoch} to {self.num_epochs}")
        
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            self.epoch = epoch
            start_t = time.time()
            
            train_loss = self._train_one_epoch()
            
            if self.scheduler:
                self.scheduler.step()
            
            epoch_t = time.time() - start_t
            
            if epoch % self.eval_freq == 0 or epoch == self.num_epochs:
                val_t_start = time.time()
                val_mae, val_rmse = self._validate()
                val_t = time.time() - val_t_start
                
                self.logger.info(
                    f"Epoch {epoch}: Train Loss {train_loss:.4f} | "
                    f"Val MAE {val_mae:.2f} RMSE {val_rmse:.2f} | "
                    f"Time: Train {epoch_t:.1f}s Val {val_t:.1f}s"
                )
                
                if val_mae < self.best_mae:
                    self.best_mae = val_mae
                    self.best_rmse = val_rmse
                    self.best_epoch = epoch
                    self.no_improve_epochs = 0
                    self._save_checkpoint(epoch, is_best=True)
                else:
                    self.no_improve_epochs += self.eval_freq
                    self._save_checkpoint(epoch, is_best=False)
                
                if self.early_stopping_patience > 0 and self.no_improve_epochs >= self.early_stopping_patience:
                    self.logger.info(f"⛔ Early Stopping dopo {self.no_improve_epochs} epoche.")
                    break
            else:
                self._save_checkpoint(epoch, is_best=False)