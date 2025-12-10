#!/usr/bin/env python3
"""
Script di diagnostica per verificare la configurazione dei bins.
Esegui PRIMA di iniziare il training per assicurarti che tutto sia coerente.
"""

import yaml
import sys

def check_config(config_path):
    print(f"=" * 60)
    print(f"🔍 DIAGNOSTICA CONFIGURAZIONE: {config_path}")
    print(f"=" * 60)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset = config.get('DATASET', 'sha')
    bins_cfg = config.get('BINS_CONFIG', {}).get(dataset, {})
    
    bins = bins_cfg.get('bins', [])
    bin_centers = bins_cfg.get('bin_centers', [])
    
    print(f"\n📊 BINS CONFIGURATION:")
    print(f"   Dataset: {dataset}")
    print(f"   Numero bins definiti: {len(bins)}")
    print(f"   Numero bin_centers definiti: {len(bin_centers)}")
    
    # Il bin[0] = [0,0] è gestito da π-head, quindi EBC usa bins[1:]
    ebc_bins = bins[1:] if bins and bins[0] in ([0, 0], (0, 0)) else bins
    num_ebc_bins = len(ebc_bins)
    
    print(f"\n🎯 EBC HEAD (escludendo bin vuoto):")
    print(f"   Numero bin EBC effettivi: {num_ebc_bins}")
    print(f"   Numero bin_centers: {len(bin_centers)}")
    
    errors = []
    warnings = []
    
    # Check 1: bin_centers deve avere la stessa lunghezza di ebc_bins
    if len(bin_centers) != num_ebc_bins:
        errors.append(
            f"❌ MISMATCH: bin_centers ha {len(bin_centers)} elementi, "
            f"ma EBC bins sono {num_ebc_bins}!\n"
            f"   Devi avere esattamente {num_ebc_bins} centri."
        )
    
    # Check 2: Verifica che i centri siano coerenti con i bins
    print(f"\n📋 DETTAGLIO BINS:")
    for i, (lo, hi) in enumerate(bins):
        is_zero_bin = (lo == 0 and hi == 0)
        
        if is_zero_bin:
            print(f"   Bin {i}: [{lo}, {hi}] → Gestito da π-head (vuoto)")
        else:
            ebc_idx = i - 1 if bins[0] in ([0, 0], (0, 0)) else i
            if ebc_idx < len(bin_centers):
                center = bin_centers[ebc_idx]
                expected_center = (lo + hi) / 2 if hi < 1000 else lo + 5
                
                # Per bin finiti, il centro dovrebbe essere vicino alla media
                if hi < 1000:
                    if abs(center - expected_center) > 2:
                        warnings.append(
                            f"⚠️ Bin {i} [{lo}, {hi}]: centro={center}, "
                            f"ma media geometrica={expected_center:.1f}"
                        )
                
                print(f"   Bin {i}: [{lo}, {hi}] → EBC bin {ebc_idx}, centro={center}")
            else:
                print(f"   Bin {i}: [{lo}, {hi}] → MISSING CENTER!")
    
    # Check 3: NUM_EBC_BINS nel config MODEL
    model_num_bins = config.get('MODEL', {}).get('NUM_EBC_BINS')
    if model_num_bins is not None:
        total_bins = len(bins)
        if model_num_bins != total_bins:
            warnings.append(
                f"⚠️ MODEL.NUM_EBC_BINS={model_num_bins} ma bins totali={total_bins}"
            )
    
    # Check 4: Bins realistici per blocco 16x16
    block_size = config.get('DATA', {}).get('ZIP_BLOCK_SIZE', 16)
    max_realistic = block_size * block_size // 4  # ~64 per 16x16, ma realisticamente < 25
    
    for i, (lo, hi) in enumerate(bins):
        if lo > 30 and hi < 9000:  # Non è il bin "infinito"
            warnings.append(
                f"⚠️ Bin {i} [{lo}, {hi}]: range troppo alto! "
                f"In un blocco {block_size}x{block_size} raramente ci sono >20-25 teste."
            )
    
    # Check 5: Verifica contiguità bins
    for i in range(1, len(bins)):
        prev_hi = bins[i-1][1]
        curr_lo = bins[i][0]
        
        if prev_hi >= 9000:  # Il bin precedente era "infinito"
            continue
            
        if curr_lo != prev_hi + 1 and curr_lo != prev_hi:
            warnings.append(
                f"⚠️ Gap tra bin {i-1} (hi={prev_hi}) e bin {i} (lo={curr_lo})"
            )
    
    # Stampa risultati
    print(f"\n" + "=" * 60)
    
    if errors:
        print("❌ ERRORI CRITICI:")
        for e in errors:
            print(f"   {e}")
    
    if warnings:
        print("\n⚠️ WARNINGS:")
        for w in warnings:
            print(f"   {w}")
    
    if not errors and not warnings:
        print("✅ CONFIGURAZIONE OK!")
    
    print("=" * 60)
    
    return len(errors) == 0

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config_sha.yaml"
    success = check_config(config_path)
    sys.exit(0 if success else 1)