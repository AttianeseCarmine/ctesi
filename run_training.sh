#!/bin/bash

# Attiva l'uscita immediata in caso di errore
set -e

# Configurazione
CONFIG_FILE="config_sha.yaml"
LOG_DIR="logs_pipeline"

# Crea cartella log se non esiste
mkdir -p $LOG_DIR

echo "========================================================"
echo "🚀 AVVIO TRAINING ZIP-CLIP-EBC COMPLETO"
echo "📅 Data: $(date)"
echo "📄 Config: $CONFIG_FILE"
echo "========================================================"

# --- PULIZIA (Opzionale: scommenta se vuoi cancellare automaticamente i vecchi esperimenti) ---
# echo "🧹 Cancellazione vecchi esperimenti..."
# 
#rm -rf experiments/sha_zip_clip_ebc/*
# echo "✅ Pulizia completata."

# --- STAGE 1 ---
echo ""
echo "--------------------------------------------------------"
echo "▶️  AVVIO STAGE 1: π-Head Training (Classificazione)"
echo "--------------------------------------------------------"
start_time=$(date +%s)

# Esegui train_stage1 e salva l'output sia a video che su file
#python train_stage1.py --config $CONFIG_FILE 2>&1 | tee "$LOG_DIR/stage1.log"

#end_time=$(date +%s)
echo "✅ Stage 1 Completato in $((end_time - start_time)) secondi."


# --- STAGE 2 ---
echo ""
echo "--------------------------------------------------------"
echo "▶️  AVVIO STAGE 2: EBC-Head Training (Conteggio)"
echo "--------------------------------------------------------"
start_time=$(date +%s)

python train_stage2.py --config $CONFIG_FILE 2>&1 | tee "$LOG_DIR/stage2.log"

end_time=$(date +%s)
echo "✅ Stage 2 Completato in $((end_time - start_time)) secondi."


# --- STAGE 3 ---
echo ""
echo "--------------------------------------------------------"
echo "▶️  AVVIO STAGE 3: Joint Fine-tuning"
echo "--------------------------------------------------------"
start_time=$(date +%s)

python train_stage3.py --config $CONFIG_FILE 2>&1 | tee "$LOG_DIR/stage3.log"

end_time=$(date +%s)
echo "✅ Stage 3 Completato in $((end_time - start_time)) secondi."

echo ""
echo "========================================================"
echo "🏆 TRAINING COMPLETATO CON SUCCESSO!"
echo "========================================================"

# nohup ./run_training.sh > main_log.out 2>&1 &
echo " REALIZZO I GRAFICI DEI RISULTATI... "
python visualize_stage1.py --checkpoint experiments/sha_final_scale_aware/stage1/best_stage1_model.pth
python visualize_stage2.py --checkpoint experiments/sha_final_scale_aware/stage2/best_stage2_model.pth
python visualize_stage3.py --checkpoint experiments/sha_final_scale_aware/stage3/best_stage3_model.pth

python logs/filter_logs.py logs_pipeline/stage1.log logs_pipeline/stage2.log logs_pipeline/stage3.log
python logs/plot_training_logs.py logs/stage1_clean.txt --out grafico_stage1.png
python logs/plot_training_logs.py logs/stage2_clean.txt --out grafico_stage2.png
python logs/plot_training_logs.py logs/stage3_clean.txt --out grafico_stage3.png

python evaluate_stage1.py --config config_sha.yaml
python evaluate_stage2.py --config config_sha.yaml --checkpoint experiments/sha_final_scale_aware/stage2/best_stage2_model.pth
python evaluate_stage3.py --config config_sha.yaml --checkpoint experiments/sha_final_scale_aware/stage3/best_stage3_model.pth

