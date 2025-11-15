#!/bin/bash
# Questo script esegue tutti e 3 gli stadi di addestramento in sequenza.
# Ogni stadio viene eseguito in background con 'nohup' e il suo
# output è reindirizzato a un file di log separato (es. stage1.log).
# Lo script attende il completamento di uno stadio prima di avviare il successivo.

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
echo "Ricorda: Loss=0.0 e MAE=nan nello Stadio 1 è NORMALE."
echo "L'addestramento vero inizia nello Stadio 2."
echo "---"

# --- STADIO 1 ---
echo "--- Avvio STADIO 1 (Pre-training PI Head)... ---"
python train.py --config "$CONFIG_FILE" --stage 1 > "$LOG_DIR/stage1.log" 2>&1
echo "--- ✅ STADIO 1 completato. Controlla $LOG_DIR/stage1.log per 'Val MAE: nan' (è normale) ---"


# --- STADIO 2 ---
echo "--- Avvio STADIO 2 (Pre-training LAMBDA Head)... ---"
python train.py --config "$CONFIG_FILE" --stage 2 > "$LOG_DIR/stage2.log" 2>&1
echo "--- ✅ STADIO 2 completato. Controlla $LOG_DIR/stage2.log (ora dovresti vedere MAE e Loss reali) ---"


# --- STADIO 3 ---
echo "--- Avvio STADIO 3 (Joint Fine-tuning)... ---"
python train.py --config "$CONFIG_FILE" --stage 3 > "$LOG_DIR/stage3.log" 2>&1
echo "--- ✅ STADIO 3 completato. Controlla $LOG_DIR/stage3.log per i risultati finali ---"

echo "🏁 Addestramento completato! 🏁"