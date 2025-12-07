import re
import matplotlib.pyplot as plt
import argparse
import os

def parse_log_file(filepath):
    """
    Legge un file di log e estrae le metriche per Stage 1 e Stage 2.
    """
    if not os.path.exists(filepath):
        print(f"❌ Errore: Il file {filepath} non esiste.")
        return None

    # Dizionari per memorizzare i dati
    stage1_data = {'epochs': [], 'loss': [], 'mae': [], 'rmse': [], 'avg_pi': []}
    stage2_data = {'epochs': [], 'train_loss': [], 'val_mae': [], 'val_rmse': []}
    
    # Regex per i pattern di stampa che abbiamo definito
    # Stage 1: "🔍 Val: Loss=11.8199 | Avg Pi=0.311 | MAE=3425.66 | RMSE=4701.05"
    s1_pattern = re.compile(r"Val: Loss=([\d\.]+) \| Avg Pi=([\d\.]+) \| MAE=([\d\.]+) \| RMSE=([\d\.]+)")
    
    # Stage 2: "📉 Epoch 5: Train Loss=4.2000 | Val MAE=3500.00 | RMSE=4000.00"
    s2_pattern = re.compile(r"Epoch (\d+): Train Loss=([\d\.]+) \| Val MAE=([\d\.]+) \| RMSE=([\d\.]+)")

    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    s1_epoch_counter = 0
    
    for line in lines:
        # Check Stage 1
        m1 = s1_pattern.search(line)
        if m1:
            s1_epoch_counter += 1 # Stage 1 di solito valida ogni N epoche
            stage1_data['epochs'].append(s1_epoch_counter)
            stage1_data['loss'].append(float(m1.group(1)))
            stage1_data['avg_pi'].append(float(m1.group(2)))
            stage1_data['mae'].append(float(m1.group(3)))
            stage1_data['rmse'].append(float(m1.group(4)))
            continue

        # Check Stage 2
        m2 = s2_pattern.search(line)
        if m2:
            epoch = int(m2.group(1))
            stage2_data['epochs'].append(epoch)
            stage2_data['train_loss'].append(float(m2.group(2)))
            stage2_data['val_mae'].append(float(m2.group(3)))
            stage2_data['val_rmse'].append(float(m2.group(4)))

    return stage1_data, stage2_data

def plot_metrics(s1_data, s2_data, save_path=None):
    """Genera i grafici con Matplotlib."""
    
    has_s1 = len(s1_data['loss']) > 0
    has_s2 = len(s2_data['train_loss']) > 0
    
    if not has_s1 and not has_s2:
        print("⚠️ Nessun dato trovato nel log. Assicurati di aver salvato l'output del terminale in un file.")
        return

    # Configurazione Plot
    plt.style.use('ggplot')
    
    # Quanti subplot?
    rows = 0
    if has_s1: rows += 2 # Loss/Pi e MAE/RMSE
    if has_s2: rows += 2 # Loss e MAE/RMSE
    
    fig, axes = plt.subplots(rows, 1, figsize=(10, 4 * rows))
    if rows == 1: axes = [axes]
    
    curr_ax = 0
    
    # --- PLOT STAGE 1 ---
    if has_s1:
        # Plot 1: Loss & Avg Pi
        ax1 = axes[curr_ax]
        color = 'tab:red'
        ax1.set_xlabel('Validation Steps')
        ax1.set_ylabel('Validation Loss', color=color)
        ax1.plot(s1_data['epochs'], s1_data['loss'], color=color, marker='o', label='Val Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_title("Stage 1 (ZIP): Loss & Avg Pi")
        ax1.grid(True)

        # Secondo asse Y per Pi
        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('Avg Pi (Active Blocks)', color=color)
        ax2.plot(s1_data['epochs'], s1_data['avg_pi'], color=color, linestyle='--', marker='x', label='Avg Pi')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 1.0) # Pi è una probabilità
        
        curr_ax += 1
        
        # Plot 2: MAE & RMSE (anche se in Stage 1 sono alti, li plottiamo)
        ax = axes[curr_ax]
        ax.plot(s1_data['epochs'], s1_data['mae'], label='MAE', marker='o')
        ax.plot(s1_data['epochs'], s1_data['rmse'], label='RMSE', linestyle='--')
        ax.set_title("Stage 1 (ZIP): Counting Error (Ignorare se EBC congelato)")
        ax.set_ylabel("Count Error")
        ax.legend()
        ax.grid(True)
        curr_ax += 1

    # --- PLOT STAGE 2 ---
    if has_s2:
        # Plot 3: Train Loss
        ax = axes[curr_ax]
        ax.plot(s2_data['epochs'], s2_data['train_loss'], color='purple', label='Train Loss')
        ax.set_title("Stage 2 (EBC): Training Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss (Cross Entropy)")
        ax.legend()
        ax.grid(True)
        curr_ax += 1
        
        # Plot 4: MAE & RMSE (Il vero risultato)
        ax = axes[curr_ax]
        ax.plot(s2_data['epochs'], s2_data['val_mae'], label='Val MAE', color='green', marker='o')
        ax.plot(s2_data['epochs'], s2_data['val_rmse'], label='Val RMSE', color='orange', linestyle='--')
        ax.set_title("Stage 2 (EBC): Validation Metrics (Crollo atteso!)")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Count Error")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Grafico salvato in: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analizza i log di training ZIP-CLIP-EBC")
    parser.add_argument("logfile", help="Percorso del file di log (txt o .log)")
    parser.add_argument("--out", default="training_plot.png", help="Nome file output immagine")
    args = parser.parse_args()

    s1, s2 = parse_log_file(args.logfile)
    if s1 or s2:
        plot_metrics(s1, s2, args.out)