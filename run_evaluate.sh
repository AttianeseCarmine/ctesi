#!/bin/bash
set -e # Interrompe se un comando fallisce

# --- CONFIGURAZIONE ---

# 1. Percorso ai tuoi pesi della baseline
#    (Assicurati che sia il percorso corretto!)
PESI_BASELINE="pretrained_weights/nome_del_tuo_file.pth"

# 2. File di configurazione
#    Usa il primo argomento ($1), altrimenti usa sha.yaml
CONFIG_FILE=${1:-"configs/sha.yaml"}

# 3. Log
# Modifica: crea la sottocartella logs/test
LOG_DIR="logs/test"
mkdir -p $LOG_DIR
BASENAME=$(basename $CONFIG_FILE .yaml)
LOG_FILE="$LOG_DIR/${BASENAME}_eval.log"

# --- CONTROLLO ESISTENZA FILE ---
if [ ! -f "$PESI_BASELINE" ]; then
    echo "❌ Errore: File pesi non trovato in $PESI_BASELINE"
    exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Errore: File config non trovato in $CONFIG_FILE"
    exit 1
fi

# --- ESECUZIONE ---

# ATTENZIONE: Questo comando fallirà finché non aggiungi "test.py"
COMMAND="python3 test.py --config $CONFIG_FILE --ckpt_path $PESI_BASELINE"

echo "🚀 Avvio valutazione..."
echo "Config:   $CONFIG_FILE"
echo "Pesi:     $PESI_BASELINE"
echo "Log:      $LOG_FILE"

# Esegui in background e salva l'output nel log
nohup $COMMAND > $LOG_FILE 2>&1 &

echo "✅ Valutazione avviata in background."
echo "Per monitorare i risultati (MAE/MSE), esegui:"
echo "  tail -f $LOG_FILE"