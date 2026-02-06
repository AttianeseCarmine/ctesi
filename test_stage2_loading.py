import torch
import yaml
import os
import sys

# --- MONKEY PATCH (Per evitare errori di CLIP se presenti) ---
try:
    import models.clip.utils as clip_utils
    def fixed_format_count(val, prompt_type):
        return str(val), str(val) # Dummy implementation
    clip_utils.format_count = fixed_format_count
except ImportError: pass

from models import get_model

# --- FUNZIONE HELPER ---
def load_weights_only(path, map_location="cpu"):
    try:
        # Tentativo per PyTorch moderni
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Fallback per PyTorch vecchi
        ckpt = torch.load(path, map_location=map_location)
    
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt

# --- CONFIGURAZIONE ---
config_path = "checkpoints/sha/vit_b_16/stage2_size224_56/config.yaml"
checkpoint_path = "checkpoints/sha/vit_b_16/stage2_size224_56/best_mae_0.pth"

# Usa la GPU se disponibile
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Test running on: {device}")

# 1. Carica Config (FIX YAML ERROR)
if not os.path.exists(config_path):
    print(f"❌ Config not found at: {config_path}")
    sys.exit(1)

with open(config_path, 'r') as f:
    # MODIFICA QUI: Usiamo UnsafeLoader per leggere !!python/tuple
    args_dict = yaml.load(f, Loader=yaml.UnsafeLoader)

# Simuliamo un oggetto args
class Args: pass
opt = Args()
for k, v in args_dict.items(): setattr(opt, k, v)

# Forziamo parametri critici se mancano nel yaml
if not hasattr(opt, 'reduction'): opt.reduction = 8
if not hasattr(opt, 'input_size'): opt.input_size = 224
if not hasattr(opt, 'bins'): opt.bins = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, float('inf'))]
if not hasattr(opt, 'anchor_points'): opt.anchor_points = [0.0, 1.0, 2.0, 3.0, 4.0]

# 2. Costruisci Modello
print("🏗️ Building Stage 2 Model...")
try:
    model = get_model(
        backbone=opt.model,
        input_size=opt.input_size,
        reduction=opt.reduction,
        bins=opt.bins, 
        anchor_points=opt.anchor_points,
        prompt_type=opt.prompt_type,
        num_vpt=opt.num_vpt,
        vpt_drop=opt.vpt_drop,
        deep_vpt=not opt.shallow_vpt
    ).to(device)
except Exception as e:
    print(f"❌ Error building model: {e}")
    # Stampiamo errore completo per debug
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. Carica Pesi (Test)
print(f"📂 Loading weights from {checkpoint_path}")
if not os.path.exists(checkpoint_path):
    print(f"❌ Checkpoint not found at: {checkpoint_path}")
    sys.exit(1)

state_dict = load_weights_only(checkpoint_path, map_location=device)

# Fix prefissi (simulazione Smart Loader)
new_state = {}
for k, v in state_dict.items():
    k_new = k
    if k_new.startswith("model."): k_new = k_new.replace("model.", "", 1)
    elif k_new.startswith("backbone."): k_new = k_new.replace("backbone.", "", 1)
    elif k_new.startswith("module."): k_new = k_new.replace("module.", "", 1)
    new_state[k_new] = v

missing, unexpected = model.load_state_dict(new_state, strict=False)

print("\n" + "="*40)
print(f"📊 REPORT CARICAMENTO PESI:")
print(f"   - Totale chiavi nel checkpoint: {len(state_dict)}")
print(f"   - Chiavi Mancanti (Missing): {len(missing)}")
print(f"   - Chiavi Inattese (Unexpected): {len(unexpected)}")

if len(missing) > 0:
    print(f"⚠️ Esempio Missing: {missing[:5]}")
    if len(missing) > 50:
        print("❌ CONCLUSIONE: Il caricamento è FALLITO. I nomi non combaciano.")
    else:
        print("⚠️ CONCLUSIONE: Caricamento parziale. Controllare se mancano layer critici (es. regression head).")
else:
    print("✅ CONCLUSIONE: Pesi caricati perfettamente!")
print("="*40 + "\n")

# 4. Test Predizione (Dummy Input)
model.eval()
dummy_img = torch.randn(1, 3, 224, 224).to(device) # Batch 1, RGB, 224x224

print("🧪 Testing Inference on Dummy Input...")
with torch.no_grad():
    try:
        output = model(dummy_img)
        # CLIP-EBC di solito ritorna (logits, density_map) o solo density_map
        if isinstance(output, (tuple, list)):
            density = output[1]
        else:
            density = output
        
        count = density.sum().item()
        print(f"   Input Shape: {dummy_img.shape}")
        print(f"   Output Density Shape: {density.shape}")
        print(f"   Predicted Count (on noise): {count:.4f}")
        
        if count == 0.0:
            print("❌ PROBLEMA GRAVE: Il modello predice esattamente ZERO. I pesi non stanno lavorando.")
        else:
            print("✅ Il modello è VIVO (predice numeri).")
            
    except Exception as e:
        print(f"❌ Inference Failed: {e}")
        import traceback
        traceback.print_exc()