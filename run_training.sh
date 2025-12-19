#!/bin/bash
# ============================================================
# ZIP-CLIP-EBC: Full Training Pipeline
# ============================================================
# Esegue i 3 stage di training in sequenza:
#   Stage 1: π-Head (classificazione vuoto/pieno)
#   Stage 2: EBC-Head (classificazione bins)
#   Stage 3: Joint Fine-tuning
#
# Usage:
#   ./run_training.sh                    # Usa config di default
#   ./run_training.sh config_sha.yaml    # Specifica config
# ============================================================

set -e  # Exit on error

# Config file
CONFIG_FILE="${1:-configs/config_sha.yaml}"
LOG_DIR="logs_pipeline"

# Crea directory logs
mkdir -p $LOG_DIR

echo "========================================================"
echo "🚀 ZIP-CLIP-EBC TRAINING PIPELINE"
echo "========================================================"
echo "📅 Data: $(date)"
echo "📄 Config: $CONFIG_FILE"
echo "========================================================"

# Verifica che il config esista
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

# --- STAGE 1: π-Head Training ---
echo ""
echo "========================================================"
echo "▶️  STAGE 1: π-Head Training (Zero-Inflation)"
echo "========================================================"
start_time=$(date +%s)

python train_stage1.py --config $CONFIG_FILE 2>&1 | tee "$LOG_DIR/stage1.log"

end_time=$(date +%s)
echo "✅ Stage 1 completato in $((end_time - start_time)) secondi"

# --- STAGE 2: EBC-Head Training ---
echo ""
echo "========================================================"
echo "▶️  STAGE 2: CLIP-EBC Head Training"
echo "========================================================"
start_time=$(date +%s)

python train_stage2.py --config $CONFIG_FILE 2>&1 | tee "$LOG_DIR/stage2.log"

end_time=$(date +%s)
echo "✅ Stage 2 completato in $((end_time - start_time)) secondi"

# --- STAGE 3: Joint Fine-tuning ---
echo ""
echo "========================================================"
echo "▶️  STAGE 3: Joint Fine-tuning"
echo "========================================================"
start_time=$(date +%s)

python train_stage3.py --config $CONFIG_FILE 2>&1 | tee "$LOG_DIR/stage3.log"

end_time=$(date +%s)
echo "✅ Stage 3 completato in $((end_time - start_time)) secondi"

# --- EVALUATION ---
echo ""
echo "========================================================"
echo "📊 FINAL EVALUATION"
echo "========================================================"

python evaluate.py --config $CONFIG_FILE --stage 3 2>&1 | tee "$LOG_DIR/evaluation.log"

echo ""
echo "========================================================"
echo "🏆 TRAINING PIPELINE COMPLETATA!"
echo "========================================================"
echo "📁 Logs salvati in: $LOG_DIR/"
echo "========================================================"
