# ZIP-CLIP-EBC: Zero-Inflated CLIP-based Crowd Counting

Un framework di crowd counting che combina:
- **Zero-Inflated Architecture** per gestire lo squilibrio spaziale (blocchi vuoti vs pieni)
- **CLIP** per classificazione del conteggio basata su linguaggio

## 🏗️ Architettura

```
Input Image [B, 3, H, W]
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    CLIP Backbone (ViT-B/16)                  │
│         Estrae feature visive per ogni patch 16x16           │
└─────────────────────────────────────────────────────────────┘
         │
         ├───────────────────────────────────────┐
         ▼                                       ▼
┌─────────────────────┐               ┌─────────────────────┐
│     π-Head          │               │    EBC-Head         │
│  (Convoluzionale)   │               │   (CLIP-based)      │
│                     │               │                     │
│ Classifica blocchi  │               │ Conta persone nei   │
│ vuoti vs pieni      │               │ blocchi non-vuoti   │
│                     │               │                     │
│ Output: P(pieno)    │               │ Output: λ (count)   │
└─────────────────────┘               └─────────────────────┘
         │                                       │
         └───────────────┬───────────────────────┘
                         ▼
              Density Map = P(pieno) × λ
```

## 🔑 Caratteristiche Principali

### π-Head (Zero-Inflation)
- **Architettura**: Convoluzionale pura (Conv → BN → ReLU → Conv)
- **Input**: Feature map dal backbone CLIP
- **Output**: Classificazione binaria (vuoto/pieno) per ogni blocco
- **Gating**: Soft gating durante training, hard threshold in inference

### EBC-Head (Expected Bin Counting)
- **Architettura**: CLIP-based similarity matching
- **Input**: Feature visive proiettate nello spazio CLIP
- **Output**: Distribuzione sui bins di conteggio
- **Bins**: 13 classi (1, 2, 3, ..., 10, 11-12, 13-14, 15+)

### Training a 3 Stadi

| Stage | Componenti Addestrati | Obiettivo |
|-------|----------------------|-----------|
| **1** | π-Head + Backbone (leggero) | Imparare dove ci sono persone |
| **2** | EBC-Head | Imparare a contare nei blocchi pieni |
| **3** | Tutto | Fine-tuning congiunto |

## 📁 Struttura del Progetto

```
zip_clip_ebc/
├── configs/
│   └── config_sha.yaml      # Configurazione per ShanghaiTech A
│
├── models/
│   ├── __init__.py
│   ├── clip_backbone.py     # Wrapper per CLIP
│   ├── pi_head.py           # π-Head convoluzionale
│   ├── ebc_head.py          # EBC-Head CLIP-based
│   └── zip_clip_ebc_model.py # Modello completo
│
├── losses/
│   ├── __init__.py
│   └── losses.py            # Loss per tutti gli stage
│
├── train_stage1.py          # Training π-Head
├── train_stage2.py          # Training EBC-Head
├── train_stage3.py          # Joint fine-tuning
│
└── test_pipeline.py         # Script di test
```

## 🚀 Quick Start

### 1. Installazione

```bash
# Requisiti
pip install torch torchvision
pip install open-clip-torch
pip install pyyaml tqdm tensorboard
```

### 2. Configurazione

Modifica `configs/config_sha.yaml`:

```yaml
DATA:
  ROOT: "/path/to/ShanghaiTech/part_A"
  TRAIN_SPLIT: "train_data"
  VAL_SPLIT: "test_data"
```

### 3. Test Pipeline

```bash
python test_pipeline.py
```

### 4. Training

```bash
# Stage 1: π-Head
python train_stage1.py --config configs/config_sha.yaml

# Stage 2: EBC-Head
python train_stage2.py --config configs/config_sha.yaml

# Stage 3: Joint Fine-tuning
python train_stage3.py --config configs/config_sha.yaml
```

## ⚙️ Configurazione Dettagliata

### Modello

```yaml
MODEL:
  BACKBONE: "ViT-B-16"        # CLIP backbone
  CLIP_PRETRAINED: "openai"   # Pesi pre-addestrati
  
  # π-Head
  PI_THRESH: 0.5              # Soglia per inference
  PI_SOFT_GATE: true          # Soft gating per training
  PI_SOFT_MIN: 0.1            # Minimo valore maschera
  
  # Gating
  GATE_MODE: "multiply"       # Come combinare π con feature
```

### Training Stage 1

```yaml
TRAIN_STAGE1:
  EPOCHS: 200
  LR_PI_HEAD: 1.0e-4          # LR per π-head
  LR_BACKBONE: 1.0e-5         # LR per backbone (fine-tuning)
  EARLY_STOPPING_PATIENCE: 50
```

### Loss

```yaml
LOSS_STAGE1:
  POS_WEIGHT: 3.0             # Peso per classe "pieno" (bilanciamento)
  COUNT_WEIGHT: 0.1           # Peso loss conteggio
```

## 📊 Bins di Conteggio

I bins definiscono come discretizzare il conteggio:

```yaml
BINS_CONFIG:
  sha:
    bins: [
      [0, 0],      # Vuoto (gestito da π)
      [1, 1],      # 1 persona
      [2, 2],      # 2 persone
      ...
      [15, 9999]   # 15+ persone
    ]
    bin_centers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11.5, 13.5, 17.0]
```

## 🔧 Integrazione con Dataset

Implementa un dataset loader che restituisce:

```python
def __getitem__(self, idx):
    return {
        "image": torch.Tensor,      # [3, H, W]
        "density": torch.Tensor,    # [1, H, W] 
        "points": torch.Tensor,     # [N, 2] coordinate (x, y)
    }
```

## 📈 Metriche

- **MAE** (Mean Absolute Error): |pred_count - gt_count|
- **RMSE** (Root Mean Square Error): sqrt((pred_count - gt_count)²)
- **Accuracy π**: % blocchi classificati correttamente come vuoti/pieni

## 🎯 Tips per Ottimizzazione

1. **Stage 1**: Usa crop più grandi (384-448) per dare contesto al π-head
2. **Stage 2**: Congela backbone, focus solo su EBC
3. **Stage 3**: LR molto bassi per non distruggere ciò che è stato appreso
4. **Bilanciamento**: Usa `POS_WEIGHT > 1` perché i blocchi vuoti dominano

## 📚 Riferimenti

- [CLIP](https://github.com/openai/CLIP) - Contrastive Language-Image Pre-training
- [P2R](https://arxiv.org/abs/2208.03318) - Point-to-Region Crowd Counting
- [Zero-Inflated Poisson](https://en.wikipedia.org/wiki/Zero-inflated_model) - Modello statistico

## 📝 Note

- Il backbone CLIP usa normalizzazione diversa da ImageNet!
- I text prompts sono cruciali per le performance dell'EBC-head
- Il soft gating durante training aiuta a mantenere i gradienti

---

**TODO**:
- [ ] Implementare dataset loader per SHA/SHB/JHU/UCF-QNRF
- [ ] Aggiungere visualizzazione delle density maps
- [ ] Multi-scale inference
- [ ] Ensemble di prompts
