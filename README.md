# 🔬 GSoC 2026 — ML4SCI E2E | Task 1: Electron vs Photon Classification

**Organization:** [ML4SCI](https://ml4sci.org/)  
**Project:** Sparse Neural Network Pipeline for Particle Collision Event Classification (E2E)  
**Author:** Shaik baba azmath | [@babaazmath1-hub](https://github.com/babaazmath1)

---

## 🎯 Task Overview

Binary classification of **electron** vs **photon** events from 32×32 calorimeter detector images, each with 2 channels:

| Channel | Description |
|---------|-------------|
| Channel 1 | Hit Energy |
| Channel 2 | Hit Time |

**Goal:** Maximize AUC score on the held-out test set.

---

## 🏗️ Model Architecture — ResNet-15

A lightweight custom ResNet inspired by [Andrews et al. (2020)](https://arxiv.org/abs/1807.11916), designed to handle small 32×32 particle physics images.

```
Input (2, 32, 32)
     │
  Stem Conv (32 filters)
     │
  Layer 1 — ResBlock: 32 → 64
  Layer 2 — ResBlock: 64 → 128  [stride 2]
  Layer 3 — ResBlock: 128 → 256 [stride 2]
  Layer 4 — ResBlock: 256 → 256 [stride 2]
     │
  Global Avg Pool → Dropout(0.3) → FC(1)
     │
  BCEWithLogitsLoss
```

- **Total Parameters:** ~1.2M
- **Input channels:** 2 (Energy + Time)
- **Output:** Binary (Electron / Photon)

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Test AUC** | **0.7178** |
| Test Accuracy | ~66% |
| Model | ResNet-15 |
| Epochs | 25 |
| Optimizer | Adam (lr=5e-4, wd=1e-4) |
| Scheduler | CosineAnnealingLR |
| Train Samples | 24,000 |
| Test Samples | 6,000 |

---

## 📁 Dataset

**Source:** [Kaggle — Electron vs Photons (ML4SCI)](https://www.kaggle.com/datasets/vishakkbhat/electron-vs-photons-ml4sci)

| File | Samples |
|------|---------|
| `SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5` | 249,000 |
| `SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5` | 249,000 |

> This task uses 15,000 samples per class (30,000 total) for training efficiency.

---

## ⚙️ Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/babaazmath1-hub/ml4sci-e2e-gsoc2026
.git
cd ml4sci-e2e-gsoc2026

```

### 2. Install dependencies
```bash
pip install numpy h5py torch torchvision scikit-learn matplotlib seaborn
```

### 3. Download dataset
Set your Kaggle credentials as environment variables (do NOT hardcode them):
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```
Then run the download cell in the notebook, or:
```bash
kaggle datasets download -d vishakkbhat/electron-vs-photons-ml4sci
unzip electron-vs-photons-ml4sci.zip
```

### 4. Run the notebook
Open `Task1_Electron_Photon_Classification.ipynb` in Jupyter or Google Colab and run all cells.

> **Google Colab:** Recommended for GPU access (T4 or better).

---

## 📂 Repository Structure

```
gsoc2026-ml4sci-e2e/
├── Task1_Electron_Photon_Classification.ipynb   # Main notebook
├── README.md                                    # This file
└── results/
    ├── task1_results.png                        # Loss, AUC, ROC curves
    └── task1_confusion_matrix.png               # Confusion matrix
```

---

## 🔑 Key Design Choices

- **2-channel input** — Energy and Time channels used together; no channel dropped
- **Residual connections** — Help with gradient flow on small 32×32 images
- **CosineAnnealingLR** — Smooth LR decay over 25 epochs, avoids sharp drops
- **BCEWithLogitsLoss** — Numerically stable binary cross-entropy
- **Dropout(0.3)** — Applied before final FC layer to reduce overfitting

---

## 📈 Training Curves

![Results](results/task1_results.png)
![Confusion Matrix](results/task1_confusion_matrix.png)

---

## 🔗 References

- Andrews et al., "End-to-End Jet Classification" (2020) — [arXiv:1807.11916](https://arxiv.org/abs/1807.11916)
- ML4SCI E2E Deep Learning Project — [ml4sci.org](https://ml4sci.org/)
- He et al., "Deep Residual Learning" (2015) — [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

---

## 👤 Author

**Shaik baba azmath**  
B.Tech AI/ML — Dhanalakshmi Srinivasan University (2023–2027)  
GitHub: [@babaazmath1-hub](https://github.com/babaazmath1-hub)

---

*Submitted as part of GSoC 2026 application to ML4SCI.*
