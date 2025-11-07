#!/bin/bash
set -e # Interrompe lo script se un comando fallisce

# --- CONFIGURAZIONE ---
CONFIG_FILE="configs/sha.yaml"
OUTPUT_DIR="output/sha_staged_training_v1" # Cambia questo per ogni esperimento
LOG_DIR="logs/test"
LOG_FILE="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
# ----------------------

# Crea le directory necessarie
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# --- CORREZIONE ---
# Invece di 'nohup { ... }', eseguiamo l'intero blocco
# in una subshell ( ... ) e mandiamo *quella* in background.
# Redirige sia stdout (>) che stderr (2>&1) al file di log
# e va in background (&). Non serve 'nohup'.
(
    echo "========================================="
    echo "PIPELINE AVVIATA: $(date)"
    echo "CONFIG: $CONFIG_FILE"
    echo "OUTPUT: $OUTPUT_DIR"
    echo "LOG: $LOG_FILE"
    echo "========================================="
    echo " "
    
    echo "========================================="
    echo "🚀 AVVIO STADIO 1: Pre-training PI Head (ZIP)"
    echo "========================================="
    # Usiamo 'python3' per essere sicuri
    python3 train.py \
        --config $CONFIG_FILE \
        --output_dir $OUTPUT_DIR \
        --stage 1

    echo " "
    echo "=============================================="
    echo "🚀 AVVIO STADIO 2: Pre-training LAMBDA Head (EBC)"
    echo "Caricamento da: $OUTPUT_DIR/stage1_best.pth"
    echo "=============================================="
    python3 train.py \
        --config $CONFIG_FILE \
        --output_dir $OUTPUT_DIR \
        --stage 2

    echo " "
    echo "========================================="
    echo "🚀 AVVIO STADIO 3: Joint Fine-tuning"
    echo "Caricamento da: $OUTPUT_DIR/stage2_best.pth"
    echo "========================================="
    python3 train.py \
        --config $CONFIG_FILE \
        --output_dir $OUTPUT_DIR \
        --stage 3

    echo " "
    echo "========================================="
    echo "✅ Pipeline a 3 stadi completata!"
    echo "Modello finale salvato in: $OUTPUT_DIR/best_mae.pth"
    echo "========================================="

) &> $LOG_FILE &
# --- FINE CORREZIONE ---


# Messaggio finale nel tuo terminale
echo "✅ Pipeline di addestramento a 3 stadi avviata in background."
echo "Output reindirizzato a: $LOG_FILE"
echo "Puoi chiudere il terminale."
echo "Per monitorare i progressi, esegui:"
echo "tail -f $LOG_FILE"