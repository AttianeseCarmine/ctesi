# test_real_batch.py

import torch
import yaml
from models.zip_clip_ebc_model import ZIP_CLIP_EBC_Model
from losses import build_loss
from datasets import build_dataloader

def test_real_batch():
    print("="*60)
    print("TEST CON BATCH REALE DAL DATASET")
    print("="*60)
    
    # Carica config
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # 1. Crea modello
    print("\n1️⃣ Creazione modello...")
    model_cfg = cfg['model']
    
    model = ZIP_CLIP_EBC_Model(
        model_name=model_cfg['name'],
        weight_name=model_cfg['weight_name'],
        ebc_bins=model_cfg['ebc_bins'],
        ebc_bin_centers=model_cfg['ebc_bin_centers'],
        zip_bins=model_cfg['zip_bins'],
        zip_bin_centers=model_cfg['zip_bin_centers'],
        text_prompts=model_cfg.get('text_prompts'),
        zip_lambda_max=model_cfg.get('zip_lambda_max', 13.0),
        pi_thresh=model_cfg.get('pi_thresh', 0.5),
        pi_soft_min=model_cfg.get('pi_soft_min', 0.0),
        gate_mode=model_cfg.get('gate_mode', 'multiply'),
        block_size=model_cfg.get('block_size'),
        num_vpt=model_cfg.get('num_vpt'),
        vpt_drop=model_cfg.get('vpt_drop'),
        input_size=model_cfg.get('input_size'),
        adapter=model_cfg.get('adapter', False),
        adapter_reduction=model_cfg.get('adapter_reduction'),
        lora=model_cfg.get('lora', False),
        lora_rank=model_cfg.get('lora_rank'),
        lora_alpha=model_cfg.get('lora_alpha'),
        lora_dropout=model_cfg.get('lora_dropout'),
        norm=model_cfg.get('norm', 'none'),
        act=model_cfg.get('act', 'none'),
    ).to(device)
    model.train()
    print("✅ Modello creato e in training mode")
    
    # 2. Crea dataloader
    print("\n2️⃣ Creazione dataloader...")
    train_loader = build_dataloader(cfg, split='train', stage_cfg=cfg['train_stage1'])
    print(f"✅ Dataloader creato ({len(train_loader)} batches)")
    
    # 3. Prendi un batch reale
    print("\n3️⃣ Caricamento batch reale...")
    batch = next(iter(train_loader))
    
    if isinstance(batch, dict):
        images = batch['image'].to(device)
        gt_map = batch['density_map'].to(device)
        gt_points = batch['points']
    else:
        images, gt_points, gt_map = batch
        images = images.to(device)
        gt_map = gt_map.to(device)
    
    print(f"✅ Batch caricato:")
    print(f"  Images: {images.shape}")
    print(f"  GT density: {gt_map.shape}")
    print(f"  GT counts: {[len(p) for p in gt_points]}")
    
    # 4. Crea GT blocks
    import torch.nn.functional as F
    kernel_size = 16
    with torch.no_grad():
        gt_den_map_blocks = F.avg_pool2d(
            gt_map, 
            kernel_size=kernel_size, 
            stride=kernel_size, 
            divisor_override=1
        )
    print(f"  GT blocks: {gt_den_map_blocks.shape}")
    
    # 5. Forward
    print("\n4️⃣ Forward pass...")
    outputs = model(images)
    
    print("✅ Output shapes:")
    for k, v in outputs.items():
        print(f"  {k}: {v.shape}")
    
    # 6. Loss
    print("\n5️⃣ Calcolo loss...")
    criterion = build_loss(cfg).to(device)
    
    loss, loss_info = criterion(
        pred_logit_map=outputs["pred_logit_map"],
        pred_den_map=outputs["pred_den_map"],
        gt_den_map=gt_den_map_blocks,
        gt_points=gt_points,
        pred_logit_pi_map=outputs["pred_logit_pi_map"],
        pred_lambda_map=outputs["pred_lambda_map"]
    )
    
    print(f"✅ Loss: {loss.item():.4f}")
    print("✅ Loss components:")
    for k, v in loss_info.items():
        val = v.item() if isinstance(v, torch.Tensor) else v
        print(f"  {k}: {val:.4f}")
    
    # 7. Backward
    print("\n6️⃣ Backward pass...")
    loss.backward()
    
    # Analizza gradienti
    has_grad = 0
    no_grad = 0
    zero_grad = 0
    
    grad_info = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 1e-8:
                    has_grad += 1
                    # Salva i primi 10 con gradiente
                    if has_grad <= 10:
                        grad_info[name] = grad_norm
                else:
                    zero_grad += 1
                    if zero_grad <= 5:
                        print(f"  ⚠️ Zero gradient: {name}")
            else:
                no_grad += 1
                if no_grad <= 5:
                    print(f"  ⚠️ No gradient: {name}")
    
    print(f"\n✅ Analisi gradienti:")
    print(f"  Parametri totali addestrabili: {has_grad + zero_grad + no_grad}")
    print(f"  Con gradiente valido: {has_grad}")
    print(f"  Con gradiente zero: {zero_grad}")
    print(f"  Senza gradiente: {no_grad}")
    
    total_params = has_grad + zero_grad + no_grad
    grad_percentage = (has_grad / total_params * 100) if total_params > 0 else 0
    print(f"  Percentuale con gradiente: {grad_percentage:.1f}%")
    
    if grad_info:
        print("\n✅ Top 10 gradienti più grandi:")
        for name, grad_norm in sorted(grad_info.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {name}: {grad_norm:.6f}")
    
    # Verifica finale
    print("\n" + "="*60)
    if grad_percentage >= 50 and loss.item() < 50:
        print("✅ MODELLO PRONTO PER IL TRAINING!")
        print(f"   - Loss ragionevole: {loss.item():.2f}")
        print(f"   - Gradienti fluiscono: {grad_percentage:.1f}%")
        return True
    else:
        print("⚠️ POSSIBILI PROBLEMI:")
        if grad_percentage < 50:
            print(f"   - Troppi parametri senza gradiente: {100-grad_percentage:.1f}%")
        if loss.item() >= 50:
            print(f"   - Loss troppo alta: {loss.item():.2f}")
        return False

if __name__ == "__main__":
    try:
        success = test_real_batch()
        if success:
            print("\n🎉 PUOI INIZIARE IL TRAINING!")
        else:
            print("\n⚠️ RISOLVI I PROBLEMI PRIMA DI INIZIARE IL TRAINING")
    except Exception as e:
        print(f"\n💀 ERRORE:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()