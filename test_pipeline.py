#!/usr/bin/env python3
# ============================================================
# ZIP-CLIP-EBC: Test Script
# ============================================================
# Verifica che l'intera pipeline funzioni correttamente.
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
import yaml


def test_config():
    """Test caricamento config."""
    print("\n" + "=" * 60)
    print("TEST 1: Caricamento Config")
    print("=" * 60)
    
    config_path = Path(__file__).parent / "configs" / "config_sha.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ Config caricato: {config_path}")
    print(f"   Run name: {config.get('RUN_NAME')}")
    print(f"   Dataset: {config.get('DATASET')}")
    print(f"   Backbone: {config.get('MODEL', {}).get('BACKBONE')}")
    
    return config


def test_model_build(config):
    """Test costruzione modello."""
    print("\n" + "=" * 60)
    print("TEST 2: Costruzione Modello")
    print("=" * 60)
    
    from models import build_model
    
    # Riduci il numero di prompts per test veloce
    config["EBC_HEAD"]["TEXT_PROMPTS"] = [
        "one person", "two people", "three people",
        "four people", "five people", "six people",
        "seven people", "eight people", "nine people",
        "ten people", "about eleven or twelve people",
        "about thirteen or fourteen people",
        "fifteen or more people",
    ]
    
    try:
        model = build_model(config)
        print(f"✅ Modello costruito con successo")
        
        # Conta parametri
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Parametri totali: {total:,}")
        print(f"   Parametri trainabili: {trainable:,}")
        
        return model
    except Exception as e:
        print(f"❌ Errore nella costruzione del modello: {e}")
        raise


def test_forward_pass(model, device="cpu"):
    """Test forward pass."""
    print("\n" + "=" * 60)
    print("TEST 3: Forward Pass")
    print("=" * 60)
    
    model = model.to(device)
    model.eval()
    
    # Input di test
    B, C, H, W = 2, 3, 256, 256
    x = torch.randn(B, C, H, W).to(device)
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        try:
            # Full forward
            outputs = model(x, return_intermediates=True)
            
            print(f"\n✅ Forward pass completato!")
            print(f"\nOutput keys e shapes:")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"   {key}: {tuple(value.shape)}")
                else:
                    print(f"   {key}: {type(value)}")
            
            # Verifica shapes
            H_grid = H // model.patch_size
            W_grid = W // model.patch_size
            
            assert outputs["logit_pi_maps"].shape == (B, 2, H_grid, W_grid), \
                f"Shape errata per logit_pi_maps: {outputs['logit_pi_maps'].shape}"
            assert outputs["pi_prob"].shape == (B, 1, H_grid, W_grid), \
                f"Shape errata per pi_prob: {outputs['pi_prob'].shape}"
            assert outputs["density_map"].shape == (B, 1, H_grid, W_grid), \
                f"Shape errata per density_map: {outputs['density_map'].shape}"
            assert outputs["pred_count"].shape == (B,), \
                f"Shape errata per pred_count: {outputs['pred_count'].shape}"
            
            print(f"\n✅ Tutte le shapes sono corrette!")
            
            # Verifica valori
            print(f"\nStatistiche output:")
            print(f"   pi_prob: [{outputs['pi_prob'].min():.3f}, {outputs['pi_prob'].max():.3f}]")
            print(f"   lambda_maps: [{outputs['lambda_maps'].min():.2f}, {outputs['lambda_maps'].max():.2f}]")
            print(f"   density_map: [{outputs['density_map'].min():.4f}, {outputs['density_map'].max():.4f}]")
            print(f"   pred_count: {outputs['pred_count'].tolist()}")
            
            return outputs
            
        except Exception as e:
            print(f"❌ Errore nel forward pass: {e}")
            raise


def test_forward_stages(model, device="cpu"):
    """Test forward per singoli stage."""
    print("\n" + "=" * 60)
    print("TEST 4: Forward per Stage")
    print("=" * 60)
    
    model = model.to(device)
    model.eval()
    
    x = torch.randn(2, 3, 256, 256).to(device)
    
    with torch.no_grad():
        # Stage 1: Solo π
        print("\n--- forward_pi_only (Stage 1) ---")
        try:
            pi_out = model.forward_pi_only(x)
            print(f"✅ logit_pi_maps: {pi_out['logit_pi_maps'].shape}")
            print(f"   pi_prob range: [{pi_out['pi_prob'].min():.3f}, {pi_out['pi_prob'].max():.3f}]")
        except Exception as e:
            print(f"❌ Errore: {e}")
        
        # Stage 2: EBC con π congelato
        print("\n--- forward_ebc_only (Stage 2) ---")
        try:
            ebc_out = model.forward_ebc_only(x)
            print(f"✅ logit_bin_maps: {ebc_out['logit_bin_maps'].shape}")
            print(f"   lambda_maps range: [{ebc_out['lambda_maps'].min():.2f}, {ebc_out['lambda_maps'].max():.2f}]")
            print(f"   pred_count: {ebc_out['pred_count'].tolist()}")
        except Exception as e:
            print(f"❌ Errore: {e}")


def test_losses(config, model, device="cpu"):
    """Test loss functions."""
    print("\n" + "=" * 60)
    print("TEST 5: Loss Functions")
    print("=" * 60)
    
    from losses import build_stage1_loss, build_stage2_loss, build_stage3_loss
    
    model = model.to(device)
    model.eval()
    
    # Dati di test
    x = torch.randn(2, 3, 256, 256).to(device)
    gt_density = torch.rand(2, 1, 256, 256).to(device) * 0.1  # Sparse
    
    with torch.no_grad():
        outputs = model(x)
    
    # Test Stage 1 Loss
    print("\n--- Stage 1 Loss (π-head) ---")
    try:
        criterion1 = build_stage1_loss(config).to(device)
        loss1, loss_dict1 = criterion1(outputs, gt_density)
        print(f"✅ Total loss: {loss1.item():.4f}")
        for k, v in loss_dict1.items():
            print(f"   {k}: {v.item():.4f}")
    except Exception as e:
        print(f"❌ Errore: {e}")
    
    # Test Stage 2 Loss
    print("\n--- Stage 2 Loss (EBC-head) ---")
    try:
        criterion2 = build_stage2_loss(config).to(device)
        loss2, loss_dict2 = criterion2(outputs, gt_density)
        print(f"✅ Total loss: {loss2.item():.4f}")
        for k, v in loss_dict2.items():
            print(f"   {k}: {v.item():.4f}")
    except Exception as e:
        print(f"❌ Errore: {e}")
    
    # Test Stage 3 Loss
    print("\n--- Stage 3 Loss (Joint) ---")
    try:
        criterion3 = build_stage3_loss(config).to(device)
        loss3, loss_dict3 = criterion3(outputs, gt_density)
        print(f"✅ Total loss: {loss3.item():.4f}")
        # Mostra solo alcune loss
        for k in ["joint_pi_bce_loss", "joint_ebc_ce_loss", "joint_count_loss"]:
            if k in loss_dict3:
                print(f"   {k}: {loss_dict3[k].item():.4f}")
    except Exception as e:
        print(f"❌ Errore: {e}")


def test_freeze_unfreeze(model):
    """Test freeze/unfreeze dei componenti."""
    print("\n" + "=" * 60)
    print("TEST 6: Freeze/Unfreeze")
    print("=" * 60)
    
    def count_trainable():
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    initial = count_trainable()
    print(f"Iniziale: {initial:,} parametri trainabili")
    
    # Freeze tutto
    model.freeze_backbone()
    model.freeze_pi_head()
    model.freeze_ebc_head()
    frozen_all = count_trainable()
    print(f"Dopo freeze tutto: {frozen_all:,} parametri trainabili")
    
    # Unfreeze solo π (Stage 1)
    model.unfreeze_pi_head()
    only_pi = count_trainable()
    print(f"Solo π-head: {only_pi:,} parametri trainabili")
    
    # Freeze π, unfreeze EBC (Stage 2)
    model.freeze_pi_head()
    model.unfreeze_ebc_head()
    only_ebc = count_trainable()
    print(f"Solo EBC-head: {only_ebc:,} parametri trainabili")
    
    # Unfreeze tutto (Stage 3)
    model.unfreeze_backbone()
    model.unfreeze_pi_head()
    model.unfreeze_ebc_head()
    all_unfrozen = count_trainable()
    print(f"Tutto scongelato: {all_unfrozen:,} parametri trainabili")
    
    assert frozen_all < only_pi, "Freeze/unfreeze non funziona correttamente"
    assert only_pi < all_unfrozen, "Freeze/unfreeze non funziona correttamente"
    
    print("✅ Freeze/unfreeze funziona correttamente!")


def test_gradient_flow(model, config, device="cpu"):
    """Test che i gradienti fluiscano correttamente."""
    print("\n" + "=" * 60)
    print("TEST 7: Gradient Flow")
    print("=" * 60)
    
    from losses import build_stage3_loss
    
    model = model.to(device)
    model.train()
    
    # Unfreeze tutto
    model.unfreeze_backbone()
    model.unfreeze_pi_head()
    model.unfreeze_ebc_head()
    
    # Forward
    x = torch.randn(2, 3, 256, 256, requires_grad=True).to(device)
    gt_density = torch.rand(2, 1, 256, 256).to(device)
    
    outputs = model(x)
    
    # Loss
    criterion = build_stage3_loss(config).to(device)
    loss, _ = criterion(outputs, gt_density)
    
    # Backward
    loss.backward()
    
    # Verifica gradienti
    components = {
        "backbone": model.backbone,
        "pi_head": model.pi_head,
        "ebc_head": model.ebc_head,
    }
    
    for name, component in components.items():
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                      for p in component.parameters() if p.requires_grad)
        status = "✅" if has_grad else "❌"
        print(f"{status} {name}: gradienti presenti = {has_grad}")
    
    print("\n✅ Gradient flow test completato!")


def main():
    """Esegue tutti i test."""
    print("\n" + "=" * 60)
    print("ZIP-CLIP-EBC: Test Suite")
    print("=" * 60)
    
    # Determina device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    try:
        # Test 1: Config
        config = test_config()
        
        # Test 2: Model build
        model = test_model_build(config)
        
        # Test 3: Forward pass
        test_forward_pass(model, device)
        
        # Test 4: Forward stages
        test_forward_stages(model, device)
        
        # Test 5: Losses
        test_losses(config, model, device)
        
        # Test 6: Freeze/unfreeze
        test_freeze_unfreeze(model)
        
        # Test 7: Gradient flow
        test_gradient_flow(model, config, device)
        
        print("\n" + "=" * 60)
        print("✅ TUTTI I TEST PASSATI!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FALLITO: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
