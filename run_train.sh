#!/bin/bash
# Questo script esegue tutti e 3 gli stadi di addestramento in sequenza.
# Lo script attende il completamento di uno stadio prima di avviare il successivo.

#  nohup sh run_train.sh > logs/training.log 2>&1 &
CONFIG_FILE="configs/sha.yaml"
OUTPUT_DIR=$(grep 'output_dir:' $CONFIG_FILE | awk '{print $2}' | tr -d "'")
LOG_DIR="logs/train"

mkdir -p "$LOG_DIR"
set -e # Esce se un comando fallisce
echo "🚀 AVVIO SCRIPT DI ADDESTRAMENTO COMPLETO (3 STADI) 🚀"
echo "Configurazione: $CONFIG_FILE"
echo "Directory di Output: $OUTPUT_DIR"
echo "I log saranno salvati in: $LOG_DIR"

# --- STADIO 1 ---
echo "--- Avvio STADIO 1 (Pre-training PI Head)... ---"
nohup python train.py --config "$CONFIG_FILE" --stage 1 > "$LOG_DIR/stage1.log" 2>&1 &
pid=$!
echo "Stadio 1 in esecuzione con PID: $pid. Log in $LOG_DIR/stage1.log"
wait $pid
echo "--- ✅ STADIO 1 completato. ---"

# --- STADIO 2 ---
echo "--- Avvio STADIO 2 (Pre-training LAMBDA Head)... ---"
nohup python train.py --config "$CONFIG_FILE" --stage 2 > "$LOG_DIR/stage2.log" 2>&1 &
pid=$!
echo "Stadio 2 in esecuzione con PID: $pid. Log in $LOG_DIR/stage2.log"
wait $pid
echo "--- ✅ STADIO 2 completato. ---"

# --- STADIO 3 ---
echo "--- Avvio STADIO 3 (Joint Fine-tuning)... ---"
nohup python train.py --config "$CONFIG_FILE" --stage 3 > "$LOG_DIR/stage3.log" 2>&1 &
pid=$!
echo "Stadio 3 in esecuzione con PID: $pid. Log in $LOG_DIR/stage3.log"
wait $pid
echo "--- ✅ STADIO 3 completato. ---"

echo "🎉 Addestramento completo terminato. 🎉"
echo "Il modello finale (best_mae.pth) è disponibile in $OUTPUT_DIR"
echo "I log completi sono disponibili in $LOG_DIR"
