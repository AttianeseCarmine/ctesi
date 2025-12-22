#!/bin/bash
# ============================================================
# ZIP-CLIP-EBC: Evaluation Pipeline
# ============================================================
# Esegue valutazione e visualizzazione per Stage 1 e Stage 2
#
# Uso:
#   ./run_evaluation.sh                          # Usa config default
#   ./run_evaluation.sh --config configs/config_shb.yaml
#   ./run_evaluation.sh --num_samples 20         # Più visualizzazioni
# ============================================================
# ./run_evaluation.sh --config configs/config_sha.yaml --num_samples 15 --gpu 0 --threshold 0.5
set -e  # Esci se un comando fallisce

# === CONFIGURAZIONE DEFAULT ===
CONFIG="configs/config_sha.yaml"
NUM_SAMPLES=10
GPU=0
THRESHOLD_S1=0.5

# === DIRECTORY OUTPUT ===
OUTPUT_DIR="output"
OUTPUT_S1="${OUTPUT_DIR}/stage1"
OUTPUT_S2="${OUTPUT_DIR}/stage2"

# Crea directory
mkdir -p "$OUTPUT_S1"
mkdir -p "$OUTPUT_S2"

# === HEADER ===
echo ""
echo "========================================================"
echo "🚀 ZIP-CLIP-EBC EVALUATION PIPELINE"
echo "========================================================"
echo "📅 Data: $(date)"
echo "📄 Config: $CONFIG"
echo "🖥️  GPU: $GPU"
echo "📊 Num samples per visualizzazione: $NUM_SAMPLES"
echo "📁 Output Stage 1: $OUTPUT_S1"
echo "📁 Output Stage 2: $OUTPUT_S2"
echo "========================================================"
echo ""

# ============================================================
# STAGE 1: π-Head Evaluation
# ============================================================
echo ""
echo "========================================================"
echo "📊 STAGE 1: Valutazione π-Head (Classificazione Blocchi)"
echo "========================================================"
echo ""

# Valutazione metriche
echo "🔍 Esecuzione evaluation_stage1.py..."
python evaluations/evaluation_stage1.py \
    --config "$CONFIG" \
    --threshold "$THRESHOLD_S1" \
    --gpu "$GPU" \
    | tee "${OUTPUT_S1}/evaluation_results.txt"

echo ""
echo "🎨 Generazione visualizzazioni Stage 1..."
python visualizations_results/visualize_stage1.py \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_S1" \
    --num_samples "$NUM_SAMPLES" \
    --threshold "$THRESHOLD_S1" \
    --gpu "$GPU"

echo ""
echo "✅ Stage 1 completato! Risultati in: $OUTPUT_S1"

# ============================================================
# STAGE 2: CLIP-EBC Evaluation
# ============================================================
echo ""
echo "========================================================"
echo "📊 STAGE 2: Valutazione CLIP-EBC Head (Conteggio)"
echo "========================================================"
echo ""

# Valutazione metriche
echo "🔍 Esecuzione evaluation_stage2.py..."
python evaluations/evaluation_stage2.py \
    --config "$CONFIG" \
    --gpu "$GPU" \
    | tee "${OUTPUT_S2}/evaluation_results.txt"

echo ""
echo "🎨 Generazione visualizzazioni Stage 2..."
python visualizations_results/visualize_stage2.py \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_S2" \
    --num_samples "$NUM_SAMPLES" \
    --gpu "$GPU"

echo ""
echo "✅ Stage 2 completato! Risultati in: $OUTPUT_S2"

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "========================================================"
echo "🏆 EVALUATION PIPELINE COMPLETATA!"
echo "========================================================"

echo " risultati in: $OUTPUT_DIR/"

echo "========================================================"
echo "📅 Completato: $(date)"
echo "========================================================"