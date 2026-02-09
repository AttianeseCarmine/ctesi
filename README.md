# Enhancing Crowd Counting in Complex Scenes via Zero-Inflated Vision-Language Models

**Author:** Carmine Attianese  
**Institution:** University of Salerno  
**Course:** Master's Degree in Computer Engineering

## 📜 Abstract

This repository contains the official implementation of the code for the Master's Thesis *"Enhancing Crowd Counting in Complex Scenes via Zero-Inflated Vision-Language Models"*.

The project addresses the problem of **Crowd Counting** in complex scenarios characterized by **Zero-Inflation**—the presence of large, semantically empty regions (background) that cause instability in classical regression models. The proposed solution integrates **Vision-Language Models (CLIP)** with a **Gating mechanism (Zero-Inflated)** to distinguish informative areas from irrelevant ones. This approach significantly improves counting accuracy and reduces model "hallucinations" on complex backgrounds.

---

## 🏗️ Model Architecture (ZIP-CLIP)

The proposed method follows a **Multi-Stage Training** pipeline:

### 1. Stage 1: Gating Network Pre-training
Training of a "Gate" network ($\pi$) for binary classification (**Crowd vs. Background**).
* **Objective:** Maximize **Recall** to ensure no regions of interest are lost.

### 2. Stage 2: Density Estimator Tuning
Adaptation of **CLIP** (via *Visual Prompt Tuning*) for density estimation.
* **Objective:** Accurate counting on crowd regions, temporarily ignoring the background to focus on feature alignment.

### 3. Stage 3: Joint Optimization
End-to-end joint optimization.
* **Objective:** The Gate and the Density Estimator are aligned using a **Consistency Loss**, which forces the density prediction to zero whenever the Gate predicts "background".

---

## ⚙️ Installation

To set up the environment, run the following commands:

```bash

