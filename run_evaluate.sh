#!/bin/bash
set -e # Interrompe se un comando fallisce

# --- CONFIGURAZIONE ---

# 1. Percorso ai tuoi pesi
PESI_BASELINE="checkpoints/sha/ebc_b_vit_best/best_mae.pth"

# 2. File di configurazione da cui estrarre il nome del dataset
CONFIG_FILE=${1:-"configs/sha.yaml"}

# 3. Nome del dataset (es. 'sha')
BASENAME=$(basename $CONFIG_FILE .yaml)

# 4. Log
LOG_DIR="logs/test"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/${BASENAME}_eval.log"

# --- CONTROLLO ESISTENZA FILE ---
if [ ! -f "$PESI_BASELINE" ]; then
    echo "❌ Errore: File pesi non trovato in $PESI_BASELINE"
    exit 1
fi
# (Non serve controllare il config file, usiamo solo il suo nome)

# --- ESECUZIONE ---
#
# MODIFICA CHIAVE:
# test.py non usa --config. Passiamo --dataset usando il BASENAME.
#
COMMAND="python3 test.py --weight_path $PESI_BASELINE --dataset $BASENAME"

#
# NOTA IMPORTANTE:
# Altri parametri (es. --split, --input_size, --sliding_window)
# useranno i valori di default definiti in test.py.
# Se i valori nel tuo 'sha.yaml' sono diversi, devi aggiungerli
# manualmente al comando qui sotto.
#
# Esempio se volessi specificare più cose:
# COMMAND="python3 test.py --weight_path $PESI_BASELINE --dataset $BASENAME --split val --input_size 768 --sliding_window"
#

echo "🚀 Avvio valutazione..."
echo "Comando:  $COMMAND"
echo "Log:      $LOG_FILE"

# Esegui in background e salva l'output nel log
nohup $COMMAND > $LOG_FILE 2>&1 &

echo "✅ Valutazione avviata in background."
echo "Per monitorare i risultati (MAE/MSE), esegui:"
echo "  tail -f $LOG_FILE"