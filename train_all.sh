#!/bin/bash
set -e # Interrompe lo script se un comando fallisce

# --- CONFIGURAZIONE ---
CONFIG_FILE="configs/sha.yaml" 
OUTPUT_DIR="output/sha_staged_training_v1" # Cambia questo per ogni esperimento
# ----------------------

echo "========================================="
echo "🚀 AVVIO STADIO 1: Pre-training PI Head (ZIP)"
echo "Output in: $OUTPUT_DIR"
echo "========================================="
python train.py \
    --config $CONFIG_FILE \
    --output_dir $OUTPUT_DIR \
    --stage 1

echo "=============================================="
echo "🚀 AVVIO STADIO 2: Pre-training LAMBDA Head (EBC)"
echo "Caricamento da: $OUTPUT_DIR/stage1_best.pth"
echo "=============================================="
python train.py \
    --config $CONFIG_FILE \
    --output_dir $OUTPUT_DIR \
    --stage 2

echo "========================================="
echo "🚀 AVVIO STADIO 3: Joint Fine-tuning"
echo "Caricamento da: $OUTPUT_DIR/stage2_best.pth"
echo "========================================="
python train.py \
    --config $CONFIG_FILE \
    --output_dir $OUTPUT_DIR \
    --stage 3

echo "========================================="
echo "✅ Pipeline a 3 stadi completata!"
echo "Modello finale salvato in: $OUTPUT_DIR/best_mae.pth"
echo "========================================="