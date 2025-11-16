#!/bin/bash
# Questo script esegue tutti e 3 gli stadi di addestramento in sequenza.

# === INIZIO MODIFICA ===
# Termina lo script immediatamente se un comando fallisce (es. train.py crasha)
set -e 
# === FINE MODIFICA ===

# Usa il file config.yaml unificato
CONFIG_FILE="config.yaml"

# Legge la output_dir direttamente dal file .yaml per robustezza
OUTPUT_DIR=$(grep 'output_dir:' $CONFIG_FILE | awk '{print $2}' | tr -d \"'\')

# Crea la directory di output e la sottocartella train per i log
LOG_DIR="logs/train"
mkdir -p "$LOG_DIR"

echo "🚀 AVVIO SCRIPT DI ADDESTRAMENTO COMPLETO (3 STADI) 🚀"
echo "Configurazione: $CONFIG_FILE"
echo "Directory di Output: $OUTPUT_DIR"
echo "I log saranno salvati in: $LOG_DIR"
echo "---"

# --- STADIO 1 ---
echo "--- Avvio STADIO 1 (Pre-training PI Head)... ---"
# Esegue in primo piano e salva il log con 'tee' (mostra output e salva)
python train.py --config "$CONFIG_FILE" --stage 1 | tee "$LOG_DIR/stage1.log"
echo "--- ✅ STADIO 1 completato. ---"


# --- STADIO 2 ---
echo "--- Avvio STADIO 2 (Pre-training LAMBDA Head)... ---"
python train.py --config "$CONFIG_FILE" --stage 2 | tee "$LOG_DIR/stage2.log"
echo "--- ✅ STADIO 2 completato. ---"


# --- STADIO 3 ---
echo "--- Avvio STADIO 3 (Joint Fine-tuning)... ---"
python train.py --config "$CONFIG_FILE" --stage 3 | tee "$LOG_DIR/stage3.log"
echo "--- ✅ STADIO 3 completato. ---"

echo "🏁 Addestramento completato! 🏁"