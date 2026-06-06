#  PneumoVision AI — Chest X-Ray Pneumonia Detection

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![Accuracy](https://img.shields.io/badge/Val%20Accuracy-95.11%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> Système de détection automatique de pneumonie à partir de radiographies
> pulmonaires, basé sur le Transfer Learning avec ResNet-18 et PyTorch.
> Interface médicale interactive développée avec Streamlit.

---

##  Table des matières

- [Contexte clinique](#contexte-clinique)
- [Aperçu du projet](#aperçu-du-projet)
- [Architecture technique](#architecture-technique)
- [Dataset](#dataset)
- [Résultats](#résultats)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)
- [Pistes d'amélioration](#pistes-damélioration)
- [Références scientifiques](#références-scientifiques)

---

## Contexte clinique

La pneumonie est l'une des principales causes de mortalité infantile dans
le monde, responsable de **15% des décès** chez les enfants de moins de
5 ans selon l'OMS (2022). Le diagnostic repose traditionnellement sur
l'interprétation manuelle de radiographies pulmonaires par un radiologue,
un processus long, coûteux et sujet à la variabilité inter-observateurs.

L'automatisation de ce diagnostic par des systèmes d'intelligence
artificielle représente une opportunité majeure pour :

- Accélérer le triage des patients dans les urgences
- Assister les médecins dans les zones à faibles ressources
- Réduire le taux de faux négatifs potentiellement dangereux

Ce projet propose une approche basée sur les **Convolutional Neural
Networks (CNN)** via le **Transfer Learning** pour classifier
automatiquement les radiographies thoraciques.

---

## Aperçu du projet

| Caractéristique     | Détail                            |
|---------------------|-----------------------------------|
| Tâche               | Classification binaire            |
| Classes             | `NORMAL` / `PNEUMONIA`            |
| Modèle              | ResNet-18 (Transfer Learning)     |
| Framework           | PyTorch 2.0                       |
| Interface           | Streamlit                         |
| Val Accuracy        | **95.11%**                        |
| Export              | Rapport PDF automatique           |

---

## Architecture technique

### Pourquoi le Transfer Learning ?

Entraîner un CNN from scratch sur un dataset médical de taille modeste
(~5000 images) conduit généralement à de l'overfitting et des
performances limitées. Le Transfer Learning exploite les représentations
visuelles déjà apprises par ResNet-18 sur ImageNet (1.2M images,
1000 classes) et les réutilise pour notre tâche spécifique.
ResNet-18 pré-entraîné (ImageNet)
│
├── Conv1 → BN → ReLU → MaxPool
├── Layer1 : 2× BasicBlock (64 filtres)
├── Layer2 : 2× BasicBlock (128 filtres)
├── Layer3 : 2× BasicBlock (256 filtres)
├── Layer4 : 2× BasicBlock (512 filtres)
├── AdaptiveAvgPool2d
│
└── FC : 512 → 2         ← seule couche réentraînée
↓
[NORMAL, PNEUMONIA] 
### Stratégie de fine-tuning

- Toutes les couches convolutionnelles sont **gelées** (`requires_grad=False`)
- Seule la couche **Fully Connected finale** est réentraînée
- Optimizer : **Adam** (`lr=0.001`)
- Loss : **CrossEntropyLoss**
- Epochs : **10** avec sauvegarde du meilleur modèle (best val accuracy)

### Data Augmentation

Appliquée uniquement sur le train set pour régulariser l'entraînement :

| Transformation         | Paramètre         | Justification                        |
|------------------------|-------------------|--------------------------------------|
| RandomHorizontalFlip   | p=0.5             | Invariance gauche/droite             |
| RandomRotation         | ±10°              | Variabilité de positionnement        |
| Resize                 | 224×224 px        | Standard ImageNet                    |
| Normalize              | μ ImageNet        | Cohérence avec les poids pré-entraînés|

---

## Dataset

**Chest X-Ray Images (Pneumonia)** — Kaggle  
Guangzhou Women and Children's Medical Center

| Split      | NORMAL  | PNEUMONIA | Total   |
|------------|---------|-----------|---------|
| Train      | 1 341   | 3 875     | 5 216   |
| Test       | 234     | 390       | 624     |
| **Total**  | **1 575**| **4 265** | **5 840**|

> **Déséquilibre de classes** : le dataset présente un ratio
> PNEUMONIA/NORMAL de ~2.9. Ce déséquilibre est partiellement compensé
> par la Data Augmentation sur les images NORMAL.

---

## Résultats

### Courbe d'entraînement

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1     | 0.3084    | 87.01%    | 0.1912   | 94.16%  |
| 2     | 0.1967    | 91.92%    | 0.1625   | 94.35%  |
| 4     | 0.1703    | 93.46%    | 0.1477   | 94.73%  |
| 6     | 0.1372    | 94.65%    | 0.1233   | **95.11%**  |
| 10    | 0.1302    | 94.87%    | 0.1489   | 94.73%  |

> Le meilleur modèle est sauvegardé à l'**Epoch 6** avec
> **Val Accuracy : 95.11%**

### Observations

- Aucun signe d'overfitting significatif — Train Acc et Val Acc restent proches
- La val loss remonte légèrement après l'epoch 6 — arrêt optimal automatique
- Convergence rapide grâce au Transfer Learning (10 epochs suffisent)

---

## Installation

```bash
# 1. Cloner le projet
git clone https://github.com/ton-username/pneumovision-ai.git
cd pneumovision-ai

# 2. Créer l'environnement virtuel
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Télécharger le dataset
# https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
# Placer les images dans data/train/ et data/test/
```

---

## Utilisation

### Entraîner le modèle

```bash
python src/train.py
```

### Tester une image en ligne de commande

```bash
python src/predict.py data/test/PNEUMONIA/person1_virus_11.jpeg
# Résultat    : PNEUMONIA
# Confiance   : 99.96%
```

### Lancer l'interface Streamlit

```bash
streamlit run app/streamlit_app.py
```

Ouvre automatiquement `http://localhost:8501`

---

##  Pistes d'amélioration

| Amélioration                  | Impact attendu                         |
|-------------------------------|----------------------------------------|
| Fine-tuning des dernières couches | +1 à 2% accuracy                  |
| Gestion du déséquilibre (WeightedSampler) | Meilleur rappel sur NORMAL |
| Grad-CAM visualization        | Explicabilité des prédictions          |
| EfficientNet-B0               | Meilleure accuracy, moins de paramètres|
| Cross-validation k-fold       | Évaluation plus robuste                |
| Déploiement Docker            | Portabilité et scalabilité             |
| API REST (FastAPI)            | Intégration dans un système hospitalier|

---

##  Références scientifiques

- **He et al. (2016)** — *Deep Residual Learning for Image Recognition*
  [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

- **Rajpurkar et al. (2017)** — *CheXNet: Radiologist-Level Pneumonia
  Detection on Chest X-Rays with Deep Learning*
  [arXiv:1711.05225](https://arxiv.org/abs/1711.05225)

- **Kermany et al. (2018)** — *Identifying Medical Diagnoses and
  Treatable Diseases by Image-Based Deep Learning* — Cell, 172(5)

- **Yosinski et al. (2014)** — *How transferable are features in deep
  neural networks?* — NeurIPS 2014

---



<p align="center">
  Développé avec : · PyTorch · Streamlit · ResNet-18
</p>
