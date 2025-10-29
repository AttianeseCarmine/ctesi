#!/bin/bash

# Termina lo script immediatamente se un comando fallisce
set -e

# --- CONFIGURAZIONE ---

# 1. File di Configurazione:
#    Usa il primo argomento ($1) come file di config.
#    Se non viene fornito nessun argomento, usa "configs/sha.yaml" come default.
CONFIG_FILE=${1:-"configs/sha.yaml"}

# 2. (OPZIONALE) Pesi della Baseline:
#    Togli il commento (rimuovi #) alla riga seguente e metti il percorso
#    corretto per riprendere l'addestramento dalla tua baseline.
RESUME_ARG="--resume pretrained_weights/nome_del_tuo_file.pth"
# RESUME_ARG="" # Lascia vuoto per iniziare da zero

# --- IMPOSTAZIONE LOG ---

# Modifica: crea la sottocartella logs/train
LOG_DIR="logs/train"
mkdir -p $LOG_DIR

# Estrai il nome base del file di config (es. "sha")
BASENAME=$(basename $CONFIG_FILE .yaml)
LOG_FILE="$LOG_DIR/${BASENAME}_$(date +%Y%m%d_%H%M%S).log"

# --- COSTRUZIONE COMANDO ---

# Comando per avviare lo script principale di training
COMMAND="python3 trainer.py --config $CONFIG_FILE $RESUME_ARG"

# --- ESECUZIONE ---

echo "🚀 Avvio addestramento..."
echo "Configurazione: $CONFIG_FILE"
if [ ! -z "$RESUME_ARG" ]; then
  echo "Riprendendo da: $RESUME_ARG"
fi
echo "Log salvati in: $LOG_FILE"

# Avvia il processo in background con nohup
nohup $COMMAND > $LOG_FILE 2>&1 &

# Messaggio di conferma
echo "✅ Addestramento avviato in background."
echo "Per monitorare il progresso, esegui:"
echo "  tail -f $LOG_FILE"