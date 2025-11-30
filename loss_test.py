import torch
import yaml
from models.zip_clip_ebc_model import ZIP_CLIP_EBC_Model
from losses import build_loss

def test_model():
    print("="*60)
    print("TEST MODELLO E LOSS")
    print("="*60)
    
    # Carica config
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Crea modello
    print("\n1️⃣ Creazione modello...")
    model = ZIP_CLIP_EBC_Model(
        model_name=cfg['model']['name'],
        weight_name=cfg['model']['weight_name'],
        block_size=cfg['model']['block_size'],
        input_size=cfg['model']['input_size'],
        ebc_bins=cfg['model']['ebc_bins'],
        ebc_bin_centers=cfg['model']['ebc_bin_centers'],
        text_prompts=cfg['model']['text_prompts'],
        zip_bins=cfg['model']['zip_bins'],
        zip_bin_centers=cfg['model']['zip_bin_centers'],
        zip_lambda_max=cfg['model'].get('zip_lambda_max', 13.0),
        pi_thresh=cfg['model'].get('pi_thresh', 0.5),
        pi_soft_min=cfg['model'].get('pi_soft_min', 0.0),
        gate_mode=cfg['model'].get('gate_mode', 'multiply'),
        num_vpt=cfg['model'].get('num_vpt'),
        vpt_drop=cfg['model'].get('vpt_drop'),
    )
    model.eval()
    print("✅ Modello creato")
    
    # Test forward
    print("\n2️⃣ Test forward pass...")
    dummy_input = torch.randn(2, 3, 256, 256)
    outputs = model(dummy_input)
    
    print("✅ Output shapes:")
    for k, v in outputs.items():
        print(f"  {k}: {v.shape}")
    
    # Crea loss
    print("\n3️⃣ Test loss function...")
    criterion = build_loss(cfg)
    
    gt_den_map_blocks = torch.randn(2, 1, 16, 16).abs() * 5  # Simula densità 0-5
    gt_points = [torch.randn(10, 2), torch.randn(15, 2)]
    
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
    
    # Test backward
    print("\n4️⃣ Test backward pass...")
    model.train()
    loss.backward()
    
    has_grad = 0
    no_grad = 0
    zero_grad = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                if param.grad.norm().item() > 1e-8:
                    has_grad += 1
                else:
                    zero_grad += 1
            else:
                no_grad += 1
    
    print(f"✅ Gradienti:")
    print(f"  Con gradiente: {has_grad}")
    print(f"  Gradiente zero: {zero_grad}")
    print(f"  Senza gradiente: {no_grad}")
    
    if zero_grad > 0 or no_grad > 0:
        print("\n⚠️ WARNING: Alcuni parametri non hanno gradienti!")
    else:
        print("\n✅ TUTTO OK! Puoi iniziare il training.")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    test_model()