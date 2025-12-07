import re
import sys
import os
from pathlib import Path

def process_log_file(file_path, output_dir):
    filename = os.path.basename(file_path)
    stem = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{stem}_clean.txt")
    
    print(f"🧹 Elaborazione: {file_path} -> {output_path}")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    cleaned_lines = []
    
    # Variabili di stato per il parsing (serve per Stage 1)
    current_epoch = "0"
    last_train_loss = "N/A"
    
    # Pattern da mantenere così come sono (Intestazioni e Info)
    header_patterns = [
        r"^={10,}",           # Separatori
        r"^🚀",               # Titolo run
        r"^✅",               # Inizializzazioni
        r"^📥",               # Caricamento pesi
        r"^🔓",               # Unfreeze
        r"^ℹ️",               # Info varie
        r"^⚙️",               # Optimizer
        r"^📉 Scheduler",     # Scheduler
        r"^⭐",               # New Best Model
        r"^✅ Stage .* Completato" # Fine
    ]
    
    # Pattern per riconoscere le linee di metrica già formattate (Stage 2 e 3)
    # Es: 📉 Epoch 5: Train Loss=147.1406 | Val MAE=353.93 | RMSE=479.56
    metric_line_pattern = r"^📉 Epoch \d+:"

    for line in lines:
        line = line.strip()
        if not line: continue

        # 1. Se è una riga di metrica già corretta (Stage 2/3), la teniamo
        if re.search(metric_line_pattern, line):
            cleaned_lines.append(line)
            continue

        # 2. Se è un'intestazione importante, la teniamo
        if any(re.search(p, line) for p in header_patterns):
            cleaned_lines.append(line)
            continue

        # --- LOGICA SPECIFICA PER STAGE 1 (da convertire) ---
        
        # A. Cattura il numero dell'epoca: "--- Epoch 1/10 ---"
        epoch_match = re.search(r"--- Epoch (\d+)/", line)
        if epoch_match:
            current_epoch = epoch_match.group(1)
            continue # Non stampiamo questa riga grezza
            
        # B. Cattura la Train Loss dalla progress bar: "... loss=3.3427, ..."
        if "Train Stage 1" in line and "loss=" in line:
            loss_match = re.search(r"loss=([\d\.]+)", line)
            if loss_match:
                last_train_loss = loss_match.group(1)
            # Non stampiamo la barra di progresso
            continue

        # C. Cattura la validazione e crea la riga formattata
        # Es: 🔍 Val: Loss=0.5414 | Avg Pi=0.809 | MAE=12895.56 | RMSE=14207.21
        if "🔍 Val:" in line:
            mae_match = re.search(r"MAE=([\d\.]+)", line)
            rmse_match = re.search(r"RMSE=([\d\.]+)", line)
            
            if mae_match and rmse_match:
                mae = mae_match.group(1)
                # CORREZIONE QUI: Era group(2), ora è group(1) perché è una nuova regex
                rmse = rmse_match.group(1) 
                
                # CREA LA RIGA NEL FORMATO RICHIESTO
                formatted_line = f"📉 Epoch {current_epoch}: Train Loss={last_train_loss} | Val MAE={mae} | RMSE={rmse}"
                cleaned_lines.append(formatted_line)
            continue

    # Scrittura su file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(cleaned_lines))
        f.write("\n")

def main():
    # Creazione cartella output se non esiste
    output_dir = "logs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Argomenti da riga di comando
    files = sys.argv[1:]
    
    if not files:
        # Fallback: cerca file log tipici se non passati argomenti
        defaults = [
            "logs_pipeline/stage1.log", 
            "logs_pipeline/stage2.log", 
            "logs_pipeline/stage3.log",
            "train_stage1.log"
        ]
        files = [f for f in defaults if os.path.exists(f)]
        
    if not files:
        print("❌ Nessun file di log specificato o trovato.")
        print(f"   Uso: python filter_logs.py percorso/mio_log.log")
        return

    for log_file in files:
        process_log_file(log_file, output_dir)
        
    print(f"\n✅ Fatto! I file puliti sono nella cartella '{output_dir}/'")

if __name__ == "__main__":
    main()