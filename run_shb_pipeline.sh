#!/bin/bash

# ==============================================================================
# SCRIPT DI AUTOMAZIONE TRAINING SHANGHAITECH PART B (SHB)
# ==============================================================================

# 1. Configurazione
# -----------------
CONFIG="config_shb.yaml"
EXP_DIR="experiments/shb_final_scale_aware"
LOG_DIR="logs_pipeline"

# Assicuriamoci che la GPU sia visibile
export CUDA_VISIBLE_DEVICES=0

# Creazione cartella per i log testuali
mkdir -p $LOG_DIR

echo "=========================================================="
echo "🚀 AVVIO PIPELINE NOTTURNA: ZIP-CLIP-EBC (SHB)"
echo "📅 Data: $(date)"
echo "⚙️  Config: $CONFIG"
echo "📂 Log dir: $LOG_DIR"
echo "=========================================================="

# 2. Pulizia Preventiva
# ---------------------
# Fondamentale per evitare mismatch di pesi se avevi run precedenti interrotte
if [ -d "$EXP_DIR" ]; then
    echo "🧹 [CLEANUP] Rimozione cartella esperimenti precedente: $EXP_DIR"
    rm -rf "$EXP_DIR"
fi

# 3. Stage 1: Pi-Head
# -------------------
echo ""
echo "▶️  [1/3] Avvio STAGE 1 (Pi-Head)..."
start_time=$(date +%s)

# Lancia python e redireziona stdout e stderr nel file di log E a video (tee)
python train_stage1.py --config $CONFIG 2>&1 | tee $LOG_DIR/stage1.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ ERRORE CRITICO in Stage 1. Interruzione pipeline."
    exit 1
fi
echo "✅ Stage 1 Completato."

# 4. Stage 2: EBC-Head
# --------------------
echo ""
echo "▶️  [2/3] Avvio STAGE 2 (EBC-Head)..."

python train_stage2.py --config $CONFIG 2>&1 | tee $LOG_DIR/stage2.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ ERRORE CRITICO in Stage 2. Interruzione pipeline."
    exit 1
fi
echo "✅ Stage 2 Completato."

# 5. Stage 3: Joint Fine-Tuning
# -----------------------------
echo ""
echo "▶️  [3/3] Avvio STAGE 3 (Joint Tuning)..."

python train_stage3.py --config $CONFIG 2>&1 | tee $LOG_DIR/stage3.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ ERRORE CRITICO in Stage 3. Interruzione pipeline."
    exit 1
fi
echo "✅ Stage 3 Completato."

# 6. Valutazione Finale
# ---------------------
echo ""
echo "📊 [EVAL] Avvio Valutazione Finale..."
python evaluate_stage3.py --config $CONFIG 2>&1 | tee $LOG_DIR/final_evaluation.log

end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "=========================================================="
echo "🎉 TUTTO COMPLETATO CON SUCCESSO!"
echo "⏱️  Tempo totale: $(($duration / 60)) minuti"
echo "📄 Controlla i risultati in: $LOG_DIR/final_evaluation.log"
echo "=========================================================="