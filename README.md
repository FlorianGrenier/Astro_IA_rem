# Astro-IA 🌌  

## Description  

Astro-IA est un projet de recherche appliquée en astrophysique moléculaire visant à développer des méthodes d’intelligence artificielle pour l’identification rapide et automatique de raies spectrales issues d’observations (sub)millimétriques complexes.  

L’objectif principal est d’exploiter des techniques de deep learning (CNN 1D, ResNet, etc.) pour reconnaître les signatures de différentes molécules dans des spectres synthétiques et, à terme, dans des spectres observationnels (type ALMA).  

---

## Fonctionnalités principales  

- ⚗️ **Génération de données synthétiques** réalistes (spectres bruités avec bruit gaussien, “pollution” de lignes parasites)  
- 📂 **Pipeline complet de préparation des données** : normalisation, correction d’étiquettes  
- 🧠 **Modèles deep learning pour spectres 1D** : CNN, variantes avec SE-blocks, dilated conv, multi-canaux  
- 📈 **Entraînement et évaluation** avec PyTorch (BCEWithLogitsLoss, pos_weight pour déséquilibre des classes)  
- 🏅 **Métriques avancées** : F1-micro/macro, mAP, Hamming Loss, Exact Match Accuracy, ROC-AUC, PR curves  
- 🔎 **Visualisation et analyse** : courbes d’apprentissage, matrices de confusion, AP par classe  
- 🔄 **Expérimentations extensives** : variation des kernels, profondeur des modèles, blocs résiduels, attention  

---

## Prérequis  

- Python 3.11  
- PyTorch + CUDA (GPU recommandé)  
- Numpy, Pandas, Matplotlib, Scikit-learn  
- Jupyter Notebook (pour les rapports et visualisations)  

---

## Installation  

### Cloner le dépôt  

```bash
git clone https://github.com/FlorianGrenier/Astro_IA_rem.git
cd Astro_IA
