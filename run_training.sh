#!/bin/bash
# ============================================================
# ZIP-CLIP-EBC: Full Training & Evaluation Pipeline
# ============================================================

set -e  # Esce immediatamente se un comando fallisce

# Config file
CONFIG_FILE="${1:-configs/config_sha.yaml}"
LOG_DIR="logs_pipeline"
VIS_DIR="visualizations_results"

# Crea directory per log e immagini
mkdir -p $LOG_DIR
mkdir -p $VIS_DIR

echo "========================================================"
echo "🚀 ZIP-CLIP-EBC TRAINING & EVALUATION PIPELINE"
echo "========================================================"
echo "📅 Data: $(date)"
echo "📄 Config: $CONFIG_FILE"
echo "========================================================"

# --- STAGE 1: π-Head Training ---
echo "▶️  STAGE 1: Training π-Head..."
python train_stage1.py --config $CONFIG_FILE 2>&1 | tee "$LOG_DIR/stage1_train.log"

# --- STAGE 2: EBC-Head Training ---
echo "▶️  STAGE 2: Training CLIP-EBC Head..."
python train_stage2.py --config $CONFIG_FILE 2>&1 | tee "$LOG_DIR/stage2_train.log"

# --- STAGE 3: Joint Fine-tuning ---
echo "▶️  STAGE 3: Joint Fine-tuning End-to-End..."
python train_stage3.py --config $CONFIG_FILE 2>&1 | tee "$LOG_DIR/stage3_train.log"

