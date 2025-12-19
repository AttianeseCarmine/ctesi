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

echo "========================================================"
echo "📊 INIZIO VALUTAZIONE DETTAGLIATA"
echo "========================================================"

# Verifica l'esistenza dei checkpoint (assumendo i nomi standard salvati dai train)
CKPT_S1="outputs/sha_v3_aggressive/stage1/best_stage1_model.pth"
CKPT_S2="outputs/sha_v3_aggressive/stage2/best_stage2_model.pth"
CKPT_S3="outputs/sha_v3_aggressive/stage3/best_stage3_model.pth"

# 1. Valutazione Metriche Stage 1
echo "🔍 Valutazione Stage 1 (Filtro Strutturale)..."
python evaluation/evaluation_stage1.py --config $CONFIG_FILE --checkpoint $CKPT_S1 2>&1 | tee "$LOG_DIR/eval_stage1.log"

# 2. Valutazione Metriche Stage 2
echo "🔍 Valutazione Stage 2 (CLIP-EBC Count)..."
python evaluation/evaluation_stage2.py --config $CONFIG_FILE --checkpoint $CKPT_S2 2>&1 | tee "$LOG_DIR/eval_stage2.log"

# 3. Valutazione Metriche Stage 3 (End-to-End)
echo "🔍 Valutazione Stage 3 (Sistema Completo)..."
python evaluation/evaluation_stage3.py --config $CONFIG_FILE --checkpoint $CKPT_S3 2>&1 | tee "$LOG_DIR/eval_stage3.log"

# --- GENERAZIONE VISUALE ---
echo "🖼️ Generazione report visuale..."
# Scegliamo un'immagine di test significativa (es. IMG_160 citata nel tuo config)
SAMPLE_IMG="./data/sha/val/images/IMG_160.jpg"

if [ -f "$SAMPLE_IMG" ]; then
    python generate_visual_evaluation.py \
        --img $SAMPLE_IMG \
        --config $CONFIG_FILE \
        --checkpoint $CKPT_S3 \
        --output "$VIS_DIR/final_report_IMG160.png"
    echo "✅ Immagine di valutazione salvata in $VIS_DIR"
else
    echo "⚠️ Immagine $SAMPLE_IMG non trovata, salto generazione visuale."
fi

echo "========================================================"
echo "🏆 PIPELINE COMPLETATA CON SUCCESSO!"
echo "========================================================"