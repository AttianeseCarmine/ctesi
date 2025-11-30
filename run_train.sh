#!/bin/bash
set -e  # Blocca lo script se un comando fallisce

# === CONFIGURAZIONE ===
CONFIG_FILE="config.yaml"
OUTPUT_DIR=$(grep 'output_dir:' $CONFIG_FILE | awk '{print $2}' | tr -d \"'\')
LOG_DIR="logs/train"
mkdir -p "$LOG_DIR"

echo "========================================================"
echo "🚀 AVVIO TRAINING COMPLETO (3 STADI) - CONFIG: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "========================================================"

# --- STADIO 1: Addestra PI Head ---
echo ""
echo "--- 1️⃣  STADIO 1: Training PI Head... ---"
# Esegue train.py. Questo creerà 'best.pth' nella cartella output
nohup python train.py --config "$CONFIG_FILE" --stage 1 > "$LOG_DIR/stage1.log" 2>&1

echo "✅ STADIO 1 completato. Rinomino i checkpoint..."
# Rinomina 'best.pth' in 'stage1_best.pth' così lo Stage 2 può trovarlo
mv "$OUTPUT_DIR/best.pth" "$OUTPUT_DIR/stage1_best.pth"

# --- STADIO 2: Addestra LAMBDA Head ---
echo ""
echo "--- 2️⃣  STADIO 2: Training LAMBDA Head... ---"
# Carica esplicitamente stage1_best.pth
nohup python train.py --config "$CONFIG_FILE" --stage 2 --load_ckpt "$OUTPUT_DIR/stage1_best.pth" > "$LOG_DIR/stage2.log" 2>&1

echo "✅ STADIO 2 completato. Rinomino i checkpoint..."
# Rinomina il nuovo 'best.pth' (prodotto dallo Stage 2) in 'stage2_best.pth'
mv "$OUTPUT_DIR/best.pth" "$OUTPUT_DIR/stage2_best.pth"

# --- STADIO 3: Joint Fine-tuning ---
echo ""
echo "--- 3️⃣  STADIO 3: Joint Fine-tuning... ---"
# Carica stage2_best.pth
nohup python train.py --config "$CONFIG_FILE" --stage 3 --load_ckpt "$OUTPUT_DIR/stage2_best.pth" > "$LOG_DIR/stage3.log" 2>&1

echo "✅ STADIO 3 completato. Rinomino i checkpoint finali..."
mv "$OUTPUT_DIR/best.pth" "$OUTPUT_DIR/final_best.pth"
mv "$OUTPUT_DIR/last.pth" "$OUTPUT_DIR/final_last.pth" 2>/dev/null || true

echo ""
echo "🏁🏁🏁 ADDESTRAMENTO COMPLETATO CON SUCCESSO! 🏁🏁🏁"