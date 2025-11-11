#!/bin/bash
# Questo script esegue tutti e 3 gli stadi di addestramento in sequenza.
# Ogni stadio viene eseguito in background con 'nohup' e il suo
# output è reindirizzato a un file di log separato (es. stage1.log).
# Lo script attende il completamento di uno stadio prima di avviare il successivo.

CONFIG_FILE="configs/sha.yaml"
# Legge la output_dir direttamente dal file .yaml per robustezza
OUTPUT_DIR=$(grep 'output_dir:' $CONFIG_FILE | awk '{print $2}' | tr -d "'")

# Crea la directory di output se non esiste
mkdir -p $OUTPUT_DIR

echo "🚀 AVVIO SCRIPT DI ADDESTRAMENTO COMPLETO (3 STADI) 🚀"
echo "Configurazione: $CONFIG_FILE"
echo "Directory di Output: $OUTPUT_DIR"
echo "L'output di ogni stadio sarà nei file .log (es. stage1.log) nella root del progetto."

# --- STADIO 1 ---
echo "--- Avvio STADIO 1 (Pre-training PI Head)... ---"
# Esegue nohup, reindirizza stdout e stderr a stage1.log, e lo manda in background (&)
nohup python train.py --config $CONFIG_FILE --stage 1 > stage1.log 2>&1 &
# Salva il Process ID (PID) dell'ultimo comando in background
pid=$!
echo "Stadio 1 in esecuzione con PID: $pid. Log in stage1.log"
# Mette in pausa lo script finché il processo con quel PID non ha terminato
wait $pid
echo "--- ✅ STADIO 1 completato. ---"


# --- STADIO 2 ---
echo "--- Avvio STADIO 2 (Pre-training LAMBDA Head)... ---"
nohup python train.py --config $CONFIG_FILE --stage 2 > stage2.log 2>&1 &
pid=$!
echo "Stadio 2 in esecuzione con PID: $pid. Log in stage2.log"
wait $pid
echo "--- ✅ STADIO 2 completato. ---"


# --- STADIO 3 ---
echo "--- Avvio STADIO 3 (Joint Fine-tuning)... ---"
nohup python train.py --config $CONFIG_FILE --stage 3 > stage3.log 2>&1 &
pid=$!
echo "Stadio 3 in esecuzione con PID: $pid. Log in stage3.log"
wait $pid
echo "--- ✅ STADIO 3 completato. ---"


echo "🎉 Addestramento completo terminato. 🎉"
echo "Il modello finale (best_mae.pth) è disponibile in $OUTPUT_DIR"