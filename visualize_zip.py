import torch
import torch.nn.functional as F
from argparse import ArgumentParser
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2 # Per resizing e overlay
import sys

# Assicurati che il percorso del progetto sia nel PYTHONPATH
# per importare 'models' e 'utils'
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    # Importa le funzioni necessarie dal progetto
    from models import get_model
except ImportError:
    print("\nErrore: Impossibile importare 'models'.")
    print("Assicurati di eseguire questo script dalla cartella principale del progetto.")
    print(f"Percorso attuale: {os.getcwd()}\n")
    sys.exit(1)


# Definizione dei transform (Assumendo standard ImageNet, come in molti modelli CLIP)
# Se il tuo modello usa una normalizzazione diversa, questi valori vanno cambiati.
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Creazione cartella output ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 2. Caricamento Modello ---
    print(f"Caricamento modello da: {args.weight_path}")
    # get_model carica sia l'architettura che i pesi
    model = get_model(model_info_path=args.weight_path)
    
    # Gestisce i modelli salvati con DataParallel (DDP)
    if hasattr(model, 'module'):
        model = model.module
        
    model = model.to(device)
    model.eval() # Imposta dropout, batchnorm, etc. in modalità valutazione

    # Controlla se il modello è ZIP (Zero-Inflated)
    if not (hasattr(model, 'zero_inflated') and model.zero_inflated):
        print(f"Errore: Il modello in {args.weight_path} non è un modello ZIP (zero_inflated=True).")
        print("Impossibile estrarre la mappa 'pi' (zero probability map).")
        return
    print("Modello ZIP caricato correttamente.")

    # --- 3. Caricamento e Preprocessing Immagine ---
    print(f"Caricamento immagine da: {args.image_path}")
    img_pil = Image.open(args.image_path).convert("RGB")
    original_size_wh = img_pil.size # (width, height)

    transform = transforms.Compose([
        # Ridimensiona all'input size richiesto dal modello
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
    ])
    
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # --- 4. Esecuzione Inferenza (con "trucco" per ottenere pi_map e den_map) ---
    print("Esecuzione inferenza...")
    with torch.no_grad():
        # Trucco: imposta temporaneamente training=True per forzare
        # il metodo forward di CLIP_EBC a restituire tutti gli output
        # (incluso pi_logit_map) come definito in models/clip_ebc/model.py
        
        # Salva lo stato originale
        original_training_state = model.training
        
        model.training = True
        outputs = model(input_tensor)
        
        # Reimposta lo stato originale
        model.training = original_training_state 

        # Da models/clip_ebc/model.py (quando self.training=True):
        # return pi_logit_map, lambda_logit_map, lambda_map, den_map
        
        # --- RIGA MODIFICATA ---
        # Catturiamo sia pi_logit_map (per lo zero) che den_map (per il conteggio)
        pi_logit_map, den_map = outputs[0], outputs[3]

    # --- 5. Elaborazione Mappa Zero-Probability ---
    print("Elaborazione mappa probabilità zero...")
    # pi_logit_map ha shape (B, 2, H, W)
    # L'indice 0 è la probabilità di "zero", l'indice 1 è "non-zero"
    pi_prob_map = F.softmax(pi_logit_map, dim=1)
    
    # Estrai la mappa di probabilità per la classe "zero"
    # Shape: (1, 1, H, W) -> (H, W)
    zero_prob_map = pi_prob_map[0, 0].cpu().numpy() 
    
    # Genera una maschera binaria basata sulla soglia
    # (H, W)
    zero_mask_binary = (zero_prob_map > args.threshold).astype(np.uint8)

    # --- NUOVA SEZIONE: Calcolo Conteggio ---
    # den_map è la mappa di densità finale (shape [1, 1, H, W])
    # Il conteggio totale è la somma di tutti i valori in questa mappa.
    predicted_count = den_map.sum().item()
    # ------------------------------------

    # --- 6. Visualizzazione e Salvataggio ---
    
    # Prepara i nomi dei file di output
    base_filename = os.path.splitext(os.path.basename(args.image_path))[0]
    output_path_prob = os.path.join(args.output_dir, f"{base_filename}_zip_prob.png")
    output_path_mask = os.path.join(args.output_dir, f"{base_filename}_zip_mask.png")
    output_path_overlay = os.path.join(args.output_dir, f"{base_filename}_zip_overlay.png")

    # Salva la mappa di probabilità (heatmap)
    print(f"Salvataggio mappa probabilità in: {output_path_prob}")
    plt.imsave(output_path_prob, zero_prob_map, cmap='viridis', vmin=0, vmax=1)

    # Carica l'immagine originale con CV2 per l'overlay
    img_cv2 = cv2.imread(args.image_path)
    # Ridimensiona l'immagine originale alla sua stessa dimensione
    # (utile se cv2 la carica con dimensioni diverse, anche se raro)
    img_cv2_resized = cv2.resize(img_cv2, original_size_wh)

    
    # Ridimensiona la maschera alla dimensione originale dell'immagine
    # Usiamo INTER_NEAREST per mantenere i blocchi "pixellati"
    mask_resized = cv2.resize(zero_mask_binary * 255, original_size_wh, interpolation=cv2.INTER_NEAREST)

    # Salva la maschera ingrandita
    print(f"Salvataggio maschera in: {output_path_mask}")
    cv2.imwrite(output_path_mask, mask_resized)

    # Crea un overlay
    # Ridimensiona la mappa di probabilità (non la maschera) per una heatmap più "morbida"
    prob_map_resized_color = cv2.resize(zero_prob_map, original_size_wh, interpolation=cv2.INTER_LINEAR)
    # Normalizza (0-255) e applica colormap
    prob_map_resized_color_norm = cv2.normalize(prob_map_resized_color, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap_color = cv2.applyColorMap(prob_map_resized_color_norm, cv2.COLORMAP_JET)
    
    # Sovrapponi la heatmap all'immagine originale
    overlay = cv2.addWeighted(img_cv2_resized, 0.6, heatmap_color, 0.4, 0)
    
    print(f"Salvataggio overlay in: {output_path_overlay}")
    cv2.imwrite(output_path_overlay, overlay)

    print("\n--- Completato ---")
    
    # --- RIGA AGGIUNTA ---
    # Stampa il conteggio stimato sul terminale
    print(f"\nConteggio persone stimato: {predicted_count:.2f}")
    
    print(f"\nImmagine originale: {args.image_path}")
    print(f"Mappa probabilità (heatmap): {output_path_prob}")
    print(f"Maschera 'zero' (bianco/nero): {output_path_mask}")
    print(f"Overlay (immagine + heatmap): {output_path_overlay}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Visualizza l'output Zero-Inflated (ZIP) di un modello CLIP_EBC.")
    
    # Argomenti richiesti
    parser.add_argument("--weight_path", type=str, required=True, help="Percorso al file dei pesi .pth del modello.")
    parser.add_argument("--image_path", type=str, required=True, help="Percorso all'immagine di input (es. da 'immagini_da_testare').")

    # Argomenti opzionali
    parser.add_argument("--output_dir", type=str, default="visualizzazioni_zip", help="Cartella dove salvare le immagini di output.")
    parser.add_argument("--input_size", type=int, default=224, help="Dimensione di input richiesta dal modello (es. 224).")
    parser.add_argument("--device", type=str, default="cuda", help="Device da usare (es. 'cuda' o 'cpu').")
    parser.add_argument("--threshold", type=float, default=0.5, help="Soglia di probabilità per considerare un blocco come 'zero strutturale'.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.weight_path):
        print(f"Errore: File pesi non trovato in {args.weight_path}")
        sys.exit(1)
    if not os.path.exists(args.image_path):
        print(f"Errore: Immagine non trovata in {args.image_path}")
        sys.exit(1)

    main(args)