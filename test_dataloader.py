# test_dataloader.py

import torch
from datasets import build_dataloader
import yaml

with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

train_loader = build_dataloader(cfg, split='train', stage_cfg=cfg['train_stage1'])

print("Testing dataloader...")
nan_count = 0
total_batches = 0

for i, batch in enumerate(train_loader):
    if isinstance(batch, dict):
        images = batch['image']
    else:
        images, _, _ = batch
    
    total_batches += 1
    
    if torch.isnan(images).any():
        nan_count += 1
        print(f"❌ Batch {i}: NaN rilevato!")
    
    if i >= 50:  # Testa solo primi 50 batch
        break

print(f"\n✅ Test completato:")
print(f"  Batch totali: {total_batches}")
print(f"  Batch con NaN: {nan_count} ({nan_count/total_batches*100:.1f}%)")

if nan_count == 0:
    print("🎉 Nessun NaN! Dataset OK.")
else:
    print("💀 PROBLEMA: Il dataset ha NaN!")