# test_model.py

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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Creazione modello
    print("1️⃣ Creazione modello...")
    model_cfg = cfg['model']
    
    try:
        model = ZIP_CLIP_EBC_Model(
            model_name=model_cfg['name'],
            weight_name=model_cfg['weight_name'],
            
            # ✅ SOLO parametri EBC (non più ZIP)
            ebc_bins=model_cfg['ebc_bins'],
            ebc_bin_centers=model_cfg['ebc_bin_centers'],
            text_prompts=model_cfg.get('text_prompts'),
            
            # Parametri di gating
            pi_thresh=model_cfg.get('pi_thresh', 0.5),
            pi_soft_min=model_cfg.get('pi_soft_min', 0.0),
            gate_mode=model_cfg.get('gate_mode', 'multiply'),
            
            # Parametri backbone
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
        print("✅ Modello creato")
    except Exception as e:
        print(f"💀 ERRORE DURANTE LA CREAZIONE DEL MODELLO:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 2. Test forward pass
    print("2️⃣ Test forward pass...")
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 256, 256).to(device)
    
    try:
        outputs = model(dummy_input)
        
        print("✅ Output shapes:")
        for k, v in outputs.items():
            print(f"  {k}: {v.shape}")
        
        # Verifica shapes
        expected_shapes = {
            "pred_logit_map": (batch_size, 13, 16, 16),  # EBC logits
            "pred_den_map": (batch_size, 1, 16, 16),     # Density finale
            "pred_logit_pi_map": (batch_size, 2, 16, 16) # π logits
        }
        
        all_correct = True
        for k, expected in expected_shapes.items():
            if k in outputs:
                actual = tuple(outputs[k].shape)
                if actual != expected:
                    print(f"❌ Shape mismatch per {k}: atteso {expected}, ottenuto {actual}")
                    all_correct = False
            else:
                print(f"❌ Chiave mancante: {k}")
                all_correct = False
        
        if all_correct:
            print("✅ Tutte le shapes sono corrette!")
        
    except Exception as e:
        print(f"💀 ERRORE DURANTE IL FORWARD:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. Test loss function
    print("3️⃣ Test loss function...")
    
    try:
        criterion = build_loss(cfg).to(device)
        
        # Crea GT dummy
        gt_den_map = torch.randn(batch_size, 1, 16, 16).abs().to(device) * 5
        gt_points = [
            torch.randn(10, 2) * 128 + 128,  # 10 punti per prima immagine
            torch.randn(15, 2) * 128 + 128   # 15 punti per seconda immagine
        ]
        
        # Calcola loss
        loss, loss_info = criterion(
            pred_logit_map=outputs["pred_logit_map"],
            pred_den_map=outputs["pred_den_map"],
            gt_den_map=gt_den_map,
            gt_points=gt_points,
            pred_logit_pi_map=outputs["pred_logit_pi_map"],
            # ❌ RIMOSSO: pred_lambda_map (non esiste più)
        )
        
        print(f"✅ Loss: {loss.item():.4f}")
        print("✅ Loss components:")
        for k, v in loss_info.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            print(f"  {k}: {val:.4f}")
        
    except Exception as e:
        print(f"💀 ERRORE DURANTE IL CALCOLO DELLA LOSS:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Test backward pass
    print("4️⃣ Test backward pass...")
    
    try:
        loss.backward()
        
        # Analizza gradienti
        has_grad = 0
        no_grad = 0
        zero_grad = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 1e-8:
                        has_grad += 1
                    else:
                        zero_grad += 1
                        if zero_grad <= 5:
                            print(f"  ⚠️ Zero gradient: {name}")
                else:
                    no_grad += 1
        
        total = has_grad + zero_grad + no_grad
        grad_percentage = (has_grad / total * 100) if total > 0 else 0
        
        print(f"✅ Gradienti:")
        print(f"  Con gradiente: {has_grad}")
        print(f"  Gradiente zero: {zero_grad}")
        print(f"  Senza gradiente: {no_grad}")
        print(f"  Percentuale con gradiente: {grad_percentage:.1f}%")
        
        if grad_percentage < 50:
            print("⚠️ WARNING: Alcuni parametri non hanno gradienti!")
            print("   Verifica che il freezing sia corretto per lo stage corrente.")
        
    except Exception as e:
        print(f"💀 ERRORE DURANTE IL BACKWARD:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("="*60)
    if loss.item() < 50 and grad_percentage > 40:
        print("🎉 TEST COMPLETATO CON SUCCESSO!")
        return True
    else:
        print("⚠️ TEST COMPLETATO MA CON POSSIBILI PROBLEMI")
        if loss.item() >= 50:
            print(f"   - Loss troppo alta: {loss.item():.2f}")
        if grad_percentage <= 40:
            print(f"   - Troppi parametri senza gradiente: {100-grad_percentage:.1f}%")
        return False

if __name__ == "__main__":
    try:
        success = test_model()
        if not success:
            print("\n⚠️ Risolvi i problemi prima di iniziare il training")
            exit(1)
    except Exception as e:
        print(f"\n💀 ERRORE DURANTE IL TEST:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)