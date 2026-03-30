<!--
Copyright 2026 MUREX S.A.S. and Université Paris Dauphine - PSL
SPDX-License-Identifier: CC-BY-4.0
-->

# Weak Supervision Labeling

A research-oriented framework for studying **weak supervision**, **label mapping**, and **generative clustering (GMVAE)**, with a strong emphasis on **analysis, robustness, and visualization**.

---

## 📄 Article

This repository accompanies the following article:

👉 [Link to TDS article]

The article is licensed under **CC BY 4.0**.

---

## Overview

This work is motivated by the observation that generative models can discover meaningful structure without labels, raising a fundamental question:

**How much supervision is actually needed to turn these structures into a reliable classifier?**

This project explores how to:
- learn from partially labeled data
- map latent clusters to semantic labels
- compare **hard vs soft decoding strategies**
- analyze performance **per class**
- visualize latent spaces and generative behavior

---

## Key insight

Soft decoding leverages uncertainty in cluster assignments and can significantly outperform hard decoding, especially when cluster-label alignment is imperfect.

This repository provides tools to quantify and visualize this effect.


---


## ⚙️ Installation

### 1. Clone

```bash
git clone git@github.com:murex/gmvae-label-decoding.git
cd gmvae-label-decoding
```

### 2. Create environment

```bash
python -m venv .venv
source .venv/bin/activate
```


### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

---

## ▶️ Usage

### Run main experiment


```bash
python main.py
```

Outputs will be saved in:

 > outputs/runs/

---

### Visualize training (TensorBoard)

To monitor training loss and metrics:

```bash
tensorboard --logdir outputs/runs/
```

Then open in your browser:

`http://localhost:6006/`


## Features

### Weak supervision
- Hard label mapping  
- Soft label mapping  
- Multi-seed evaluation  
- Label fraction sweeps  

### Metrics
- Per-label accuracy  
- Gain (soft vs hard)  
- Stability across seeds  

### Visualization
- Latent space (PCA, UMAP, t-SNE)  
- Per-label performance  
- Gain vs entropy / margin  
- Reconstruction grids  
- Generated samples per cluster  
- Example-level analysis (hard vs soft)  

---

## Key concepts

- **Hard decoding**: assign each sample to a single cluster (via argmax), then map the cluster to a label  
- **Soft decoding**: use the full posterior distribution over clusters to perform a weighted label prediction  
- **Label mapping**: estimate the relationship between clusters and labels from a small labeled subset  
- **GMVAE**: generative clustering model with a latent mixture structure  

---

## 📁 Outputs

The pipeline automatically generates figures such as:

- Figures 1, 2, and 3 from the article
- Additional visualizations:
  - PCA projection of the GMVAE latent space
  - UMAP projection of the latent space
  - t-SNE projection of the latent space
  - Component purity and weight plots

All outputs are stored in:

outputs/runs/

---

## Design principles

- Clear separation between:
  - computation (`per_class`, `weak_supervision`)
  - models (`models/`)
  - visualization (`plotting/`)
  - orchestration (`pipeline.py`)
- Reproducibility via seeds  
- Modular experiments  


---

## License

### Code
This repository code is licensed under the Apache License 2.0.

### Article / Documentation
This work is licensed under **CC BY 4.0**.

You are free to:
- Share — copy and redistribute
- Adapt — remix, transform, and build upon

Under the condition of proper attribution.

See: https://creativecommons.org/licenses/by/4.0/

---
